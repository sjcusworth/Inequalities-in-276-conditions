"""
Tests verifying that the streaming calculation methods in IncPrevMethods.py
produce numerically identical results to the original in-memory methods.

Run with:
    cd main
    python -m pytest tests/test_streaming.py -v

Requires: pytest, pandas, numpy, pyarrow, scipy, dateutil
"""

import sys
import os
import datetime

import pytest
import pandas as pd
import numpy as np
from scipy.stats import chi2
from scipy.special import ndtri

_HERE = os.path.dirname(os.path.abspath(__file__))
_METHODS_DIR = os.path.join(
    _HERE, "..", "ANALOGY_SCIENTIFIC",
    "analogy", "study_design", "incidence_prevalence")
sys.path.insert(0, _METHODS_DIR)

from IncPrevMethods import IncPrev  # noqa: E402

# ---------------------------------------------------------------------------
# Study parameters shared by all tests
# ---------------------------------------------------------------------------
STUDY_START = datetime.datetime(2010, 1, 1)
STUDY_END   = datetime.datetime(2012, 1, 1)   # __init__ adds 1 day → 2012-01-02
BASELINE_DATE_LIST = ["BD_COND1"]
DEMOGRAPHY         = ["SEX"]

# Columns required by the class
COLS = [
    "PRACTICE_PATIENT_ID", "PRACTICE_ID", "INDEX_DATE", "START_DATE",
    "END_DATE", "COLLECTION_DATE", "TRANSFER_DATE", "DEATH_DATE",
    "REGISTRATION_STATUS", "BD_COND1", "SEX",
]


# ---------------------------------------------------------------------------
# Synthetic patient data
# ---------------------------------------------------------------------------
def _make_synthetic_df():
    """
    10 patients with controlled dates covering several edge cases:

    Patient | INDEX_DATE   | END_DATE     | BD_COND1     | SEX | notes
    --------|--------------|--------------|--------------|-----|------------------
    0       | 2008-01-01   | 2020-12-31   | 2010-06-15   | M   | event in period 1
    1       | 2008-01-01   | 2020-12-31   | 2011-03-20   | F   | event in period 2
    2       | 2008-01-01   | 2020-12-31   | NaT          | M   | never diagnosed
    3       | 2008-01-01   | 2010-09-01   | NaT          | F   | exits during period 1
    4       | 2011-06-01   | 2020-12-31   | NaT          | M   | enters during period 2
    5       | 2008-01-01   | 2020-12-31   | 2010-02-10   | F   | event early period 1
    6       | 2008-01-01   | 2020-12-31   | 2009-12-31   | M   | diagnosed pre-study
                                                                  (prevalent at start)
    7       | 2008-01-01   | 2020-12-31   | NaT          | F   | never diagnosed
    8       | 2009-06-01   | 2011-06-30   | 2011-04-01   | M   | mid-period entry+exit
    9       | 2008-01-01   | 2020-12-31   | 2011-09-01   | F   | event in period 2
    """
    rows = [
        ("2008-01-01", "2020-12-31", "2010-06-15", "M"),
        ("2008-01-01", "2020-12-31", "2011-03-20", "F"),
        ("2008-01-01", "2020-12-31", None,          "M"),
        ("2008-01-01", "2010-09-01", None,          "F"),
        ("2011-06-01", "2020-12-31", None,          "M"),
        ("2008-01-01", "2020-12-31", "2010-02-10", "F"),
        ("2008-01-01", "2020-12-31", "2009-12-31", "M"),
        ("2008-01-01", "2020-12-31", None,          "F"),
        ("2009-06-01", "2011-06-30", "2011-04-01", "M"),
        ("2008-01-01", "2020-12-31", "2011-09-01", "F"),
    ]
    df = pd.DataFrame(rows, columns=["INDEX_DATE", "END_DATE", "BD_COND1", "SEX"])
    for col in ["INDEX_DATE", "END_DATE", "BD_COND1"]:
        df[col] = pd.to_datetime(df[col])

    df["PRACTICE_PATIENT_ID"] = range(1, 11)
    df["PRACTICE_ID"]          = 1
    df["START_DATE"]           = pd.Timestamp("2000-01-01")
    df["COLLECTION_DATE"]      = pd.Timestamp("2021-01-01")
    df["TRANSFER_DATE"]        = pd.NaT
    df["DEATH_DATE"]           = pd.NaT
    df["REGISTRATION_STATUS"]  = 0
    return df


@pytest.fixture
def parquet_path(tmp_path):
    """Write synthetic data to a temporary Parquet file."""
    df = _make_synthetic_df()
    path = str(tmp_path / "test_data.parquet")
    df.to_parquet(path, index=False)
    return path


def _make_incprev(parquet_path, read_data):
    return IncPrev(
        STUDY_END_DATE=STUDY_END,
        STUDY_START_DATE=STUDY_START,
        FILENAME=parquet_path,
        DATABASE_NAME="GOLD",
        BASELINE_DATE_LIST=BASELINE_DATE_LIST,
        DEMOGRAPHY=DEMOGRAPHY,
        cols=COLS,
        read_data=read_data,
        fileType="parquet",
    )


def _load_csv_sorted(path):
    """Read a CSV and sort rows so order-independent comparison works."""
    df = pd.read_csv(path)
    return df.sort_values(df.columns.tolist()).reset_index(drop=True)


def _compare_output_dirs(dir_inmem, dir_stream, rtol=1e-9):
    """Assert every CSV in dir_inmem has a matching CSV in dir_stream with equal values."""
    inmem_files = sorted(os.listdir(dir_inmem))
    stream_files = sorted(os.listdir(dir_stream))
    assert inmem_files == stream_files, (
        f"File mismatch.\nIn-memory: {inmem_files}\nStreaming: {stream_files}"
    )
    for fname in inmem_files:
        df_i = _load_csv_sorted(os.path.join(dir_inmem, fname))
        df_s = _load_csv_sorted(os.path.join(dir_stream, fname))
        pd.testing.assert_frame_equal(
            df_i, df_s,
            check_exact=False, rtol=rtol,
            obj=f"File: {fname}",
        )


# ===========================================================================
# Unit tests — Byar's confidence interval formula
# ===========================================================================

class TestByarsCI:
    """Verify Byar's CI matches the chi-squared / normal-approximation formula."""

    @pytest.fixture(autouse=True)
    def _obj(self, parquet_path):
        self.obj = _make_incprev(parquet_path, read_data=False)

    @pytest.mark.parametrize("count", [1, 3, 7, 9])
    def test_lower_small_count_uses_chi2(self, count):
        """For count < 10 Byar's lower = chi2.ppf(alpha/2, 2*count) / 2.
        count=0 is excluded because chi2.ppf(0.025, 0) = NaN (degenerate case).
        """
        denom = 100.0
        expected_limit = chi2.ppf(0.025, count * 2) / 2
        assert self.obj.byars_lower(count, denom, return_limits=True) == pytest.approx(
            expected_limit
        )

    @pytest.mark.parametrize("count", [0, 1, 3, 7, 9])
    def test_upper_small_count_uses_chi2(self, count):
        """For count < 10 Byar's upper = chi2.ppf(1-alpha/2, 2*count+2) / 2."""
        denom = 100.0
        expected_limit = chi2.ppf(0.975, 2 * count + 2) / 2
        assert self.obj.byars_higher(count, denom, return_limits=True) == pytest.approx(
            expected_limit
        )

    @pytest.mark.parametrize("count", [10, 25, 100])
    def test_lower_large_count_uses_normal_approx(self, count):
        """For count >= 10 Byar's lower uses normal approximation."""
        denom = 1000.0
        z = ndtri(0.975)
        c = 1 / (9 * count)
        b = 3 * np.sqrt(count)
        expected_limit = count * ((1 - c - (z / b)) ** 3)
        assert self.obj.byars_lower(count, denom, return_limits=True) == pytest.approx(
            expected_limit
        )

    @pytest.mark.parametrize("count", [10, 25, 100])
    def test_upper_large_count_uses_normal_approx(self, count):
        """For count >= 10 Byar's upper uses normal approximation."""
        denom = 1000.0
        z = ndtri(0.975)
        c = 1 / (9 * (count + 1))
        b = 3 * np.sqrt(count + 1)
        expected_limit = (count + 1) * ((1 - c + (z / b)) ** 3)
        assert self.obj.byars_higher(count, denom, return_limits=True) == pytest.approx(
            expected_limit
        )

    def test_lower_ci_rate_equals_limit_over_denominator(self):
        """Rate CI = limit / denominator × PER_PY."""
        denom = 250.0
        count = 5
        rate_ci = self.obj.byars_lower(count, denom, return_limits=False)
        limit   = self.obj.byars_lower(count, denom, return_limits=True)
        assert rate_ci == pytest.approx(limit / denom)

    def test_upper_ci_rate_equals_limit_over_denominator(self):
        denom = 250.0
        count = 5
        rate_ci = self.obj.byars_higher(count, denom, return_limits=False)
        limit   = self.obj.byars_higher(count, denom, return_limits=True)
        assert rate_ci == pytest.approx(limit / denom)

    def test_lower_lt_upper(self):
        """Lower bound must always be strictly below upper bound.
        count=0 is excluded: lower=NaN (chi2.ppf(0.025,0)), comparison undefined.
        """
        for count in [1, 5, 10, 50]:
            denom = max(count, 1) * 10.0
            lo = self.obj.byars_lower(count, denom, return_limits=True)
            hi = self.obj.byars_higher(count, denom, return_limits=True)
            assert lo < hi, f"Lower {lo} >= upper {hi} for count={count}"


# ===========================================================================
# Unit tests — point_incidence and point_prevalence on known mini-datasets
# ===========================================================================

class TestPointCalculations:
    """Verify the per-period calculation functions on a hand-crafted DataFrame."""

    @pytest.fixture(autouse=True)
    def _obj(self, parquet_path):
        self.obj = _make_incprev(parquet_path, read_data=False)
        # Give the object an in-memory DataFrame for the helper methods
        self.obj.raw_data = _make_synthetic_df()
        self.obj.raw_data['INDEX_DATE'] = pd.to_datetime(self.obj.raw_data['INDEX_DATE'])
        self.obj.raw_data['END_DATE']   = pd.to_datetime(self.obj.raw_data['END_DATE'])
        self.obj.raw_data['BD_COND1']   = pd.to_datetime(self.obj.raw_data['BD_COND1'])

    def test_incidence_numerator_period1(self):
        """
        Period 1: 2010-01-01 to 2011-01-01.
        Events in BD_COND1 ∈ [2010-01-01, 2011-01-01) AND > INDEX_DATE:
          Patient 0: BD=2010-06-15 ✓
          Patient 5: BD=2010-02-10 ✓
          Patient 6: BD=2009-12-31 — before start, excluded ✗
        → numerator = 2
        """
        start_yr = datetime.datetime(2010, 1, 1)
        end_yr   = datetime.datetime(2011, 1, 1)
        df = self.obj.raw_data
        mask = self.obj.get_numerator_filter_inc(df, start_yr, end_yr, "BD_COND1")
        assert len(df.iloc[mask]) == 2

    def test_incidence_numerator_period2(self):
        """
        Period 2: 2011-01-01 to 2012-01-02 (STUDY_END+1day).
        Events in BD_COND1 ∈ [2011-01-01, 2012-01-02) AND > INDEX_DATE:
          Patient 1: BD=2011-03-20 ✓
          Patient 8: BD=2011-04-01, ends 2011-06-30, enters 2009-06-01 ✓
          Patient 9: BD=2011-09-01 ✓
        → numerator = 3
        """
        start_yr = datetime.datetime(2011, 1, 1)
        end_yr   = datetime.datetime(2012, 1, 2)  # STUDY_END + 1 day (as stored)
        df = self.obj.raw_data
        mask = self.obj.get_numerator_filter_inc(df, start_yr, end_yr, "BD_COND1")
        assert len(df.iloc[mask]) == 3

    def test_prevalence_numerator_at_study_start(self):
        """
        Prevalence numerator at 2010-01-01:
        Must satisfy INDEX_DATE <= 2010-01-01, END_DATE >= 2010-01-01,
        BD_COND1 <= 2010-01-01.
        Patient 6: BD=2009-12-31 <= 2010-01-01 ✓ → only one prevalent case at start
        """
        study_date = datetime.datetime(2010, 1, 1)
        df = self.obj.raw_data
        mask = self.obj.get_numerator_filter_prev(df, study_date, "BD_COND1")
        assert len(df.iloc[mask]) == 1

    def test_prevalence_denominator_at_study_start(self):
        """
        Prevalence denominator at 2010-01-01:
        Patients with INDEX_DATE <= 2010-01-01 AND END_DATE >= 2010-01-01.
        Excludes patient 4 (INDEX=2011-06-01 > 2010-01-01).
        → 9 patients eligible
        """
        study_date = datetime.datetime(2010, 1, 1)
        df = self.obj.raw_data
        mask = self.obj.get_denominator_filter_prev(df, study_date)
        assert len(df.iloc[mask]) == 9

    def test_point_incidence_returns_six_tuple(self):
        """point_incidence (overall) returns a 6-element tuple."""
        start = datetime.datetime(2010, 1, 1)
        end   = datetime.datetime(2011, 1, 1)
        result = self.obj.point_incidence(self.obj.raw_data, start, end, "BD_COND1")
        assert len(result) == 6
        year, rate, py, num, lo, hi = result
        assert num == 2
        assert py > 0
        assert rate > 0
        assert lo <= rate <= hi

    def test_point_prevalence_returns_six_tuple(self):
        """point_prevalence (overall) returns a 6-element tuple."""
        study_date = datetime.datetime(2010, 1, 1)
        result = self.obj.point_prevalence(self.obj.raw_data, study_date, "BD_COND1")
        assert len(result) == 6
        year, prev, denom, num, lo, hi = result
        assert num == 1
        assert denom >= 9
        assert lo <= prev <= hi

    def test_point_incidence_with_subgroup_returns_seven_tuple(self):
        """Grouped point_incidence returns a 7-element tuple including group label."""
        start = datetime.datetime(2010, 1, 1)
        end   = datetime.datetime(2011, 1, 1)
        males = self.obj.raw_data[self.obj.raw_data["SEX"] == "M"]
        result = self.obj.point_incidence(males, start, end, "BD_COND1", sub_group="M")
        assert len(result) == 7
        assert result[1] == "M"

    def test_person_years_positive(self):
        """Person-years denominator must be positive for any non-empty eligible set."""
        start = datetime.datetime(2010, 1, 1)
        end   = datetime.datetime(2011, 1, 1)
        _, _, py, _, _, _ = self.obj.point_incidence(
            self.obj.raw_data, start, end, "BD_COND1")
        assert py > 0

    def test_incidence_rate_per_100k(self):
        """Incidence rate is expressed per 100 000 person-years."""
        start = datetime.datetime(2010, 1, 1)
        end   = datetime.datetime(2011, 1, 1)
        _, rate, py, num, _, _ = self.obj.point_incidence(
            self.obj.raw_data, start, end, "BD_COND1")
        assert rate == pytest.approx((num / py) * 100_000, rel=1e-9)

    def test_prevalence_rate_per_100k(self):
        """Prevalence is expressed per 100 000 eligible patients.
        The denominator stored in the tuple already includes SMALL_FP_VAL,
        so we use that value directly rather than recomputing (num / denom).
        """
        study_date = datetime.datetime(2010, 1, 1)
        _, prev, denom_stored, num, _, _ = self.obj.point_prevalence(
            self.obj.raw_data, study_date, "BD_COND1")
        # denom_stored = int(denom_with_fp_val); rate = (num / denom_with_fp_val)*PER_PY
        # Verify the rate is consistent with num and stored denominator (within fp_val tolerance)
        expected_approx = (num / denom_stored) * 100_000
        assert prev == pytest.approx(expected_approx, rel=1e-3)


# ===========================================================================
# Regression tests — streaming == in-memory on full synthetic dataset
# ===========================================================================

class TestStreamingEqualsInMemory:
    """
    For each of the four calculate_* methods, run both the original in-memory
    path and the new streaming path on the same synthetic data and assert that
    every output CSV is numerically identical.

    chunk_size=3 forces 4 chunks (10 rows ÷ 3), exercising the accumulator
    across multiple iterations.
    """

    CHUNK_SIZE = 3

    def test_overall_incidence(self, parquet_path, tmp_path):
        out_i = str(tmp_path / "inc_i") + "/"
        out_s = str(tmp_path / "inc_s") + "/"
        os.makedirs(out_i); os.makedirs(out_s)

        obj = _make_incprev(parquet_path, read_data=True)
        obj.calculate_incidence(path_out=out_i)

        obj_s = _make_incprev(parquet_path, read_data=False)
        obj_s.calculate_incidence_streaming(
            cols=COLS, chunk_size=self.CHUNK_SIZE, path_out=out_s)

        _compare_output_dirs(out_i, out_s)

    def test_grouped_incidence(self, parquet_path, tmp_path):
        out_i = str(tmp_path / "ginc_i") + "/"
        out_s = str(tmp_path / "ginc_s") + "/"
        os.makedirs(out_i); os.makedirs(out_s)

        obj = _make_incprev(parquet_path, read_data=True)
        obj.calculate_grouped_incidence(path_out=out_i)

        obj_s = _make_incprev(parquet_path, read_data=False)
        obj_s.calculate_grouped_incidence_streaming(
            cols=COLS, chunk_size=self.CHUNK_SIZE, path_out=out_s)

        _compare_output_dirs(out_i, out_s)

    def test_overall_prevalence(self, parquet_path, tmp_path):
        out_i = str(tmp_path / "prev_i") + "/"
        out_s = str(tmp_path / "prev_s") + "/"
        os.makedirs(out_i); os.makedirs(out_s)

        obj = _make_incprev(parquet_path, read_data=True)
        obj.calculate_prevalence(path_out=out_i)

        obj_s = _make_incprev(parquet_path, read_data=False)
        obj_s.calculate_prevalence_streaming(
            cols=COLS, chunk_size=self.CHUNK_SIZE, path_out=out_s)

        _compare_output_dirs(out_i, out_s)

    def test_grouped_prevalence(self, parquet_path, tmp_path):
        out_i = str(tmp_path / "gprev_i") + "/"
        out_s = str(tmp_path / "gprev_s") + "/"
        os.makedirs(out_i); os.makedirs(out_s)

        obj = _make_incprev(parquet_path, read_data=True)
        obj.calculate_grouped_prevalence(path_out=out_i)

        obj_s = _make_incprev(parquet_path, read_data=False)
        obj_s.calculate_grouped_prevalence_streaming(
            cols=COLS, chunk_size=self.CHUNK_SIZE, path_out=out_s)

        _compare_output_dirs(out_i, out_s)

    def test_streaming_single_chunk_equals_full(self, parquet_path, tmp_path):
        """A chunk_size larger than the dataset should give the same result."""
        out_i = str(tmp_path / "inc_full_i") + "/"
        out_s = str(tmp_path / "inc_full_s") + "/"
        os.makedirs(out_i); os.makedirs(out_s)

        obj = _make_incprev(parquet_path, read_data=True)
        obj.calculate_incidence(path_out=out_i)

        obj_s = _make_incprev(parquet_path, read_data=False)
        obj_s.calculate_incidence_streaming(
            cols=COLS, chunk_size=10_000, path_out=out_s)

        _compare_output_dirs(out_i, out_s)

    def test_merge_eth_other_mixed_streaming_vs_inmemory(self, tmp_path):
        """
        When merge_eth_other_mixed=True the streaming and in-memory paths must
        still produce equal outputs.  Uses ETHNICITY as the demographic.
        """
        # Build a tiny dataset that includes OTHER and MIXED ethnicity values
        rows = [
            ("2008-01-01", "2020-12-31", "2010-06-15", "WHITE"),
            ("2008-01-01", "2020-12-31", "2011-03-20", "OTHER"),
            ("2008-01-01", "2020-12-31", None,          "MIXED"),
            ("2008-01-01", "2020-12-31", None,          "WHITE"),
            ("2008-01-01", "2020-12-31", "2010-11-01", "OTHER"),
        ]
        df = pd.DataFrame(rows, columns=["INDEX_DATE", "END_DATE", "BD_COND1", "ETHNICITY"])
        for col in ["INDEX_DATE", "END_DATE", "BD_COND1"]:
            df[col] = pd.to_datetime(df[col])
        df["PRACTICE_PATIENT_ID"] = range(1, 6)
        df["PRACTICE_ID"]          = 1
        df["START_DATE"]           = pd.Timestamp("2000-01-01")
        df["COLLECTION_DATE"]      = pd.Timestamp("2021-01-01")
        df["TRANSFER_DATE"]        = pd.NaT
        df["DEATH_DATE"]           = pd.NaT
        df["REGISTRATION_STATUS"]  = 0

        path = str(tmp_path / "eth_data.parquet")
        df.to_parquet(path, index=False)

        eth_cols = [c for c in COLS if c != "SEX"] + ["ETHNICITY"]
        out_i = str(tmp_path / "eth_i") + "/"
        out_s = str(tmp_path / "eth_s") + "/"
        os.makedirs(out_i); os.makedirs(out_s)

        obj = IncPrev(
            STUDY_END_DATE=STUDY_END, STUDY_START_DATE=STUDY_START,
            FILENAME=path, DATABASE_NAME="GOLD",
            BASELINE_DATE_LIST=["BD_COND1"], DEMOGRAPHY=["ETHNICITY"],
            cols=eth_cols, read_data=True, fileType="parquet",
        )
        # Apply merge in-memory path manually (mirrors IncPrev.py logic)
        if "ETHNICITY" in obj.raw_data.columns:
            obj.raw_data["ETHNICITY"] = obj.raw_data["ETHNICITY"].apply(
                lambda x: "OTHERS_AND_MIXED" if x in ("OTHER", "MIXED") else x)
        obj.calculate_incidence(path_out=out_i)

        obj_s = IncPrev(
            STUDY_END_DATE=STUDY_END, STUDY_START_DATE=STUDY_START,
            FILENAME=path, DATABASE_NAME="GOLD",
            BASELINE_DATE_LIST=["BD_COND1"], DEMOGRAPHY=["ETHNICITY"],
            cols=eth_cols, read_data=False, fileType="parquet",
        )
        obj_s.calculate_incidence_streaming(
            cols=eth_cols, chunk_size=2,
            path_out=out_s, merge_eth_other_mixed=True)

        _compare_output_dirs(out_i, out_s)


# ===========================================================================
# Edge-case tests
# ===========================================================================

class TestEdgeCases:

    def test_zero_events_incidence(self, tmp_path):
        """If no events occur the numerator must be 0 and rate 0 (or near-0)."""
        rows = [
            ("2008-01-01", "2020-12-31", None, "M"),
            ("2008-01-01", "2020-12-31", None, "F"),
        ]
        df = pd.DataFrame(rows, columns=["INDEX_DATE", "END_DATE", "BD_COND1", "SEX"])
        for col in ["INDEX_DATE", "END_DATE", "BD_COND1"]:
            df[col] = pd.to_datetime(df[col])
        df["PRACTICE_PATIENT_ID"] = [1, 2]
        df["PRACTICE_ID"]          = 1
        df["START_DATE"]           = pd.Timestamp("2000-01-01")
        df["COLLECTION_DATE"]      = pd.Timestamp("2021-01-01")
        df["TRANSFER_DATE"]        = pd.NaT
        df["DEATH_DATE"]           = pd.NaT
        df["REGISTRATION_STATUS"]  = 0

        path = str(tmp_path / "zero.parquet")
        df.to_parquet(path, index=False)

        obj = IncPrev(
            STUDY_END_DATE=STUDY_END, STUDY_START_DATE=STUDY_START,
            FILENAME=path, DATABASE_NAME="GOLD",
            BASELINE_DATE_LIST=["BD_COND1"], DEMOGRAPHY=["SEX"],
            cols=COLS, read_data=True, fileType="parquet")

        result = obj.point_incidence(
            obj.raw_data,
            datetime.datetime(2010, 1, 1),
            datetime.datetime(2011, 1, 1),
            "BD_COND1")
        year, rate, py, num, lo, hi = result
        assert num == 0
        assert rate == pytest.approx(0.0, abs=1.0)   # effectively 0 per 100k

    def test_all_events_prevalent_at_start(self, tmp_path):
        """
        All events occur before study start → prevalence numerator = denominator,
        and no incidence events in either period.
        """
        rows = [
            ("2008-01-01", "2020-12-31", "2005-01-01", "M"),
            ("2008-01-01", "2020-12-31", "2007-06-01", "F"),
        ]
        df = pd.DataFrame(rows, columns=["INDEX_DATE", "END_DATE", "BD_COND1", "SEX"])
        for col in ["INDEX_DATE", "END_DATE", "BD_COND1"]:
            df[col] = pd.to_datetime(df[col])
        df["PRACTICE_PATIENT_ID"] = [1, 2]
        df["PRACTICE_ID"]          = 1
        df["START_DATE"]           = pd.Timestamp("2000-01-01")
        df["COLLECTION_DATE"]      = pd.Timestamp("2021-01-01")
        df["TRANSFER_DATE"]        = pd.NaT
        df["DEATH_DATE"]           = pd.NaT
        df["REGISTRATION_STATUS"]  = 0

        obj = IncPrev(
            STUDY_END_DATE=STUDY_END, STUDY_START_DATE=STUDY_START,
            FILENAME="unused", DATABASE_NAME="GOLD",
            BASELINE_DATE_LIST=["BD_COND1"], DEMOGRAPHY=["SEX"],
            cols=COLS, read_data=False)
        obj.raw_data = df
        for col in ["INDEX_DATE", "END_DATE", "BD_COND1"]:
            obj.raw_data[col] = pd.to_datetime(obj.raw_data[col])

        study_date = datetime.datetime(2010, 1, 1)
        _, _, denom, num, _, _ = obj.point_prevalence(
            obj.raw_data, study_date, "BD_COND1")
        assert num == 2   # both already diagnosed

        start_yr = datetime.datetime(2010, 1, 1)
        end_yr   = datetime.datetime(2011, 1, 1)
        _, _, _, inc_num, _, _ = obj.point_incidence(
            obj.raw_data, start_yr, end_yr, "BD_COND1")
        assert inc_num == 0   # no new events in-study
