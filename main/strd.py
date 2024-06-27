import datetime
import sys
import csv
import multiprocessing as mp
from itertools import repeat
import polars as pl
from pandas import read_excel as pd_read_excel
from pandas import read_csv
import pyarrow.dataset as ds
import yaml
import os
import re

with open("./wdir.yml",
          "r",
          encoding="utf8") as file_config:
    config = yaml.safe_load(file_config)

PATH = config["PATH"]
DIR_DATA = f"{PATH}{config['dir_data']}"
DIR_MAIN = f"{PATH}{config['dir_main']}"
DIR_OUT = f"{PATH}{config['dir_out']}"

sys.path.append(f"{DIR_MAIN}/ANALOGY_SCIENTIFIC/analogy/study_design/incidence_prevalence/")
from IncPrevMethods import StrdIncPrev
import AnalogyGraphing as ag

if not config["incprev"]["usePolars"]:
    ag.organise_wdir(DIR_OUT)

conf_incprev = config["incprev"]

STUDY_START_DATE = datetime.datetime(year=conf_incprev["start_date"]["inc"]["year"],
                                     month=conf_incprev["start_date"]["inc"]["month"],
                                     day=conf_incprev["start_date"]["inc"]["day"])
STUDY_END_DATE = datetime.datetime(year=conf_incprev["end_date"]["inc"]["year"],
                                     month=conf_incprev["end_date"]["inc"]["month"],
                                     day=conf_incprev["end_date"]["inc"]["day"])


data_root = './' #path for crude rates location and crude rate by subgroup subgroups
inc_file = '_Inc.csv'
prev_file = '_Prev.csv'
overall_file = '_OVERALL'

condition_labels = pd_read_excel(
    f'{DIR_DATA}aurumGoldLabels_NCQOF.xlsx',
    header=3).dropna(subset=['Joht Labelling'])

GOLD_labels = condition_labels['GOLD']
JOHT_labels = condition_labels['Joht Labelling']
map_condLabel = dict(zip(GOLD_labels, JOHT_labels))

standardised_data_root = 'Standardised Results 10-05'

OVERALL = 'OVERALL'
overall = 'Overall'

group_col = 'Subgroup'
category_col = 'Group'
condition_col = 'Condition'
col_variables = [condition_col,category_col,group_col] #order important, change names only

standardisation_options = ['Overall','Ethnicity','RegionEth',
                           'DeprivationEth', 'Deprivation', 'Region']

if conf_incprev["usePolars"]:
    standard_breakdowns = dict(
            Overall = "AGE_CATEGORY, SEX",
            Ethnicity = "AGE_CATEGORY, SEX, ETHNICITY",
            RegionEth = "AGE_CATEGORY, SEX, ETHNICITY, HEALTH_AUTH",
            DeprivationEth = "AGE_CATEGORY, SEX, ETHNICITY, IMD_pracid",
            Deprivation = "AGE_CATEGORY, SEX, IMD_pracid",
    )
else:
    standard_breakdowns = dict(
            Overall = "_'AGE_CATEGORY', 'SEX'",
            Ethnicity = "_'AGE_CATEGORY', 'SEX', 'ETHNICITY'",
            RegionEth = "_'AGE_CATEGORY', 'SEX', 'ETHNICITY', 'HEALTH_AUTH'",
            DeprivationEth = "_'AGE_CATEGORY', 'SEX', 'ETHNICITY', 'IMD_pracid'",
            Deprivation = "_'AGE_CATEGORY', 'SEX', 'IMD_pracid'",
            Region = "_'AGE_CATEGORY', 'SEX', 'HEALTH_AUTH'",
    )

DEMOGRAPHY = conf_incprev["DEMOGRAPHY"]

def fmt_data(df):
    df["Condition"] = df["Condition"].map(lambda x: ''.join(letter for letter in x if letter.isalnum()).replace("BDMEDI",""))

    overall_map = (df["Group"] == "Overall")
    df["Subgroup"][overall_map] = ""

    df["std_group"] = df["Subgroup"].map(lambda x: ", ".join("".join([letter for letter in x if letter != " "]).split(",")[0:2]))

    rm_map = df["Subgroup"].apply(lambda x: ''.join(letter for letter in x if letter != " ").split(","))
    #removing intersex
    rm_map = rm_map.apply(lambda x: False if "I" in x else True).to_numpy()
    df = df[(rm_map)]

    return df

incprev = StrdIncPrev(data_root,
            map_condLabel,
            STUDY_END_DATE,
            STUDY_START_DATE,
            FILENAME=f"{DIR_DATA}dat_processed.parquet",
            DEMOGRAPHY=DEMOGRAPHY,
            standardisation_options=standardisation_options,
            standard_breakdowns=standard_breakdowns,
            col_labs = col_variables,
            data_root=DIR_OUT,
            inMemory=conf_incprev["usePolars"],)

if conf_incprev["usePolars"]:
    incprev.raw_data_inc = read_csv(f"{DIR_OUT}out_inc.csv")
    incprev.raw_data_prev = read_csv(f"{DIR_OUT}out_prev.csv")

    incprev.raw_data_prev = fmt_data(incprev.raw_data_prev)
    incprev.raw_data_inc = fmt_data(incprev.raw_data_inc)

incprev.getReference(f"{DIR_DATA}UK Age-Sex Pop Structure.csv")

prev = incprev.standardise_all_conditions()
inc = incprev.standardise_all_conditions(measure="Incidence")

inc.to_csv(f"{DIR_OUT}inc_DSR.csv")
prev.to_csv(f"{DIR_OUT}prev_DSR.csv")

