# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an epidemiological analysis pipeline for a population-based cohort study examining health inequalities across 276–281 chronic conditions in the UK CPRD Gold database (2001–2021). The pipeline calculates incidence and prevalence rates, applies direct age-sex standardisation, and quantifies subgroup disparities by ethnicity, deprivation, sex, age, and region.

## Environment & Dependencies

Designed to run on a SLURM HPC cluster (University of Birmingham BlueBear). Required modules (loaded in batch scripts):

- Python 3.10.8
- SciPy-bundle/2023.02
- Polars/0.19.12, Arrow/11.0.0
- matplotlib, Seaborn, tqdm, PyYAML, plotly.py, openpyxl

Install locally with:
```bash
pip install polars pyarrow pandas scipy numpy matplotlib seaborn tqdm pyyaml plotly openpyxl
```

## Running the Pipeline

### HPC (SLURM) — intended execution:
```bash
sbatch sBatch_1.sh          # Preprocessing (3h, 40GB, 1 node)
sbatch sBatch_2.sh          # Incidence/prevalence calc (24h, 56GB, 26 array jobs)
sbatch sBatch_3.sh          # Standardisation & analysis (3h, 32GB)
```

### Running scripts individually (local/debug):
```bash
python3 main/preprocessing.py
python3 main/IncPrev.py <BATCH_ID>    # BATCH_ID = 1–26
python3 main/strd.py
python3 main/ratioZscore.py
python3 main/smallNumCens.py
python3 main/formatPublish.py
python3 main/tidyImdCategories.py
python3 main/table1.py
```

## Configuration

All parameters are controlled via `wdir.yml`:

- `PATH`: root directory (default `./`)
- `dir_data`, `dir_main`, `dir_out`: subdirectory paths
- `filename`: input data file (CPRD CSV or Parquet)
- `is_parquet`: `True` if input is already Parquet
- `n_processes` / `batch_size`: parallelisation controls
- `merge_EthOtherMixed`: merge OTHER/MIXED ethnicity categories
- `BD_LIST`: maps batch IDs (1–26) to CPRD condition code lists

## Architecture & Data Flow

```
Raw CPRD CSV
    ↓ preprocessing.py  (DexterProcessing.process_ethImd)
dat_processed.parquet  (ethnicity + IMD mapped)
    ↓ IncPrev.py × 26 parallel batches
out_inc_*.csv, out_prev_*.csv  (crude rates per condition/year/subgroup)
    ↓ strd.py
inc_DSR.csv, prev_DSR.csv  (directly standardised rates)
    ↓ ratioZscore.py
Average_Geometric/[ratio & z-score files]
    ↓ smallNumCens.py  (suppress counts ≤ 10)
    ↓ formatPublish.py
out/Publish/DSR.csv, crude.csv, tableOne.csv, Average_Geometric/[final files]
    ↓ tidyImdCategories.py  (clean IMD labels)
```

### Key Design Decisions

- **Parallelisation**: 281 conditions split into 26 batches (`BD_LIST` in `wdir.yml`), each run as a SLURM array job in `sBatch_2.sh`.
- **Dual implementations**: `ANALOGY_SCIENTIFIC/` provides both a Pandas (`IncPrevMethods.py`) and Polars (`IncPrevMethods_polars.py`) implementation. `IncPrev.py` uses the Pandas version (`usePolars=False`).
- **Memory efficiency**: Polars lazy frames and Parquet are used throughout post-preprocessing steps.
- **External library**: `main/ANALOGY_SCIENTIFIC/` is a git submodule/external repo providing `IncPrev`, `StrdIncPrev`, and related classes. Core incidence/prevalence and standardisation logic lives there, not in the top-level scripts.
- **Privacy suppression**: `smallNumCens.py` and `formatPublish.py` both apply suppression of counts ≤ 10 (rates, CIs, and ratios nulled).

## Key Modules

| File | Role |
|---|---|
| `DexterProcessing.py` | ETL utilities — ethnicity baseline date parsing, IMD quintile mapping, deduplication across data sources |
| `ANALOGY_SCIENTIFIC/` | Core statistical library — `IncPrev` (crude rates), `StrdIncPrev` (DSR), both Pandas and Polars variants |
| `AnalogyGraphing.py` | Visualisation helpers used by `strd.py` and the notebook |
| `Visualisations.ipynb` | Interactive exploration of final outputs |

## Demographic Stratifications

12 grouping combinations are used throughout, including single variables (AGE_CATEGORY, SEX, IMD_pracid, HEALTH_AUTH, ETHNICITY) and composites (e.g. AGE_CATEGORY + SEX + ETHNICITY + IMD). These map to CPRD column names and must match the column structure of the processed Parquet.

## Large-Dataset Memory Optimisations

Two independent optimisations are available for datasets too large to fit in RAM (e.g. 500 GB+). Both are opt-in via `wdir.yml` and are backwards-compatible — the default values leave the pipeline behaviour unchanged.

### Optimisation 1 — Per-batch parquet files (primary fix)

Enable with `create_batch_files: true` under `incprev:` in `wdir.yml`.

During `preprocessing.py`, after `dat_processed.parquet` is written, `DexterProcessing.create_batch_files()` uses Polars `sink_parquet` (streaming, no full load) to write 26 files:

```
data/dat_batch_1.parquet … data/dat_batch_26.parquet
```

Each file contains only: 9 core cohort columns + 5 demographic columns + ~11 BD_ columns for that batch (~25 of ~295 columns total). At 500 GB for the full file, each batch file is roughly 40–50 GB on disk, and ~5–15 GB in pandas memory — well within the 56 GB SLURM node limit.

`IncPrev.py` automatically uses `dat_batch_{ID}.parquet` when it exists, falling back to `dat_processed.parquet` if not.

**Note on sBatch_1.sh wall time**: writing 26 batch files requires 26 streaming passes over the full parquet. For very large files, increase the `sBatch_1.sh` time limit from 3h accordingly (estimate ~30 min per TB on shared HPC storage).

### Optimisation 2 — Single-pass streaming accumulation (safety net)

Enable with `streaming_chunk_size: 500000` (or any integer) under `incprev:` in `wdir.yml`.

`IncPrev.py` passes `read_data=False` to `IncPrevMethods.IncPrev`, bypassing `pd.read_parquet()` entirely. Four new methods on the class (`calculate_incidence_streaming`, `calculate_grouped_incidence_streaming`, `calculate_prevalence_streaming`, `calculate_grouped_prevalence_streaming`) do a single pass through the file via `pyarrow.ParquetFile.iter_batches`, accumulating numerators and denominators (person-years for incidence, head-counts for prevalence) across chunks. Rates are computed from the accumulated totals after the full file is read.

This guarantees that peak memory is proportional to `chunk_size`, not dataset size. Use alongside Optimisation 1 for maximum effect.

**Recommended settings for 500 GB data:**
```yaml
create_batch_files: !!bool True
streaming_chunk_size: 500000
```

## Input Data

The primary input (`data/IncPrev281conditions_Goldv5_fullDB*.csv`) and IMD mapping (`data/imd_mapping.csv`) are not included in the repository (CPRD data sharing restrictions). The condition-to-label mapping is read from `data/aurumGoldLabels_NCQOF.xlsx`.

## Output Files (out/Publish/)

- `crude.csv` — combined crude incidence + prevalence
- `DSR.csv` — directly standardised rates (merged incidence + prevalence)
- `tableOne.csv` — demographic characteristics at baseline and 2019
- `Average_Geometric/281 conditions chi2 z-scores, expected and observed rates.csv` — per-subgroup z-scores, observed/expected rates
- `Average_Geometric/281 conditions yearly subgroup-overall ratios.csv` — year-by-year ratios
- `Average_Geometric/281 conditions 20-year subgroup-overall ratios.csv` — geometric mean ratios over full study period
