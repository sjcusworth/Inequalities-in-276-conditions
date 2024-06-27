import datetime
import sys
import csv
import multiprocessing as mp
from itertools import repeat
import polars as pl
import pyarrow.dataset as ds
import yaml
import argparse
#import pip

#if hasattr(pip, 'main'):
#    pip.main(['install', 'openpyxl'])
#else:
#    pip._internal.main(['install', 'openpyxl'])
#print("done install.")

parser = argparse.ArgumentParser()

parser.add_argument("id")

args = parser.parse_args()

ID = args.id

with open("wdir.yml",
          "r",
          encoding="utf8") as file_config:
    config = yaml.safe_load(file_config)

PATH = config["PATH"]
DIR_DATA = f"{PATH}{config['dir_data']}"
DIR_MAIN = f"{PATH}{config['dir_main']}"
DIR_OUT = f"{PATH}{config['dir_out']}"
sys.path.append(f"{DIR_MAIN}/ANALOGY_SCIENTIFIC/analogy/study_design/incidence_prevalence/")
from IncPrevMethods import IncPrev as pdIncPrev
from IncPrevMethods_polars import IncPrev as plIncPrev

conf_incprev = config["incprev"]
FILENAME = f"{DIR_DATA}{conf_incprev['filename']}"

STUDY_START_DATE_inc = datetime.datetime(year=conf_incprev["start_date"]["inc"]["year"],
                                     month=conf_incprev["start_date"]["inc"]["month"],
                                     day=conf_incprev["start_date"]["inc"]["day"])
STUDY_END_DATE_inc = datetime.datetime(year=conf_incprev["end_date"]["inc"]["year"],
                                     month=conf_incprev["end_date"]["inc"]["month"],
                                     day=conf_incprev["end_date"]["inc"]["day"])

STUDY_START_DATE_prev = datetime.datetime(year=conf_incprev["start_date"]["prev"]["year"],
                                     month=conf_incprev["start_date"]["prev"]["month"],
                                     day=conf_incprev["start_date"]["prev"]["day"])
STUDY_END_DATE_prev = datetime.datetime(year=conf_incprev["end_date"]["prev"]["year"],
                                     month=conf_incprev["end_date"]["prev"]["month"],
                                     day=conf_incprev["end_date"]["prev"]["day"])

STUDY_START_DATE = [STUDY_START_DATE_inc, STUDY_START_DATE_prev]
STUDY_END_DATE = [STUDY_END_DATE_inc, STUDY_END_DATE_prev]

#########################################################################

#get condition date columns
if conf_incprev["BD_LIST"] is None:
    if conf_incprev["is_parquet"]:
        dataset = ds.dataset(FILENAME, format="parquet")
        col_head = dataset.head(1).to_pylist()[0].keys()
        del dataset
        BASELINE_DATE_LIST = [col for col in col_head if col.startswith('BD_')]
        del col_head
    else:
        with open(FILENAME,
                  "r",
                  encoding="utf8") as f:
            reader=csv.reader(f)
            col_head = next(reader)
        BASELINE_DATE_LIST = [col for col in col_head if col.startswith('BD_')]
        del col_head
else:
    print("Running for cols: ", conf_incprev["BD_LIST"][ID])
    BASELINE_DATE_LIST = conf_incprev["BD_LIST"][ID]



def processBatch(batch,
                 STUDY_START_DATE,
                 STUDY_END_DATE,
                 FILENAME,
                 DEMOGRAPHY,
                 fileType,
                 usePolars,
                 batchId=0):
    #Get unique categories
    CATGS = list(set([sublist if isinstance(sublist, str) else item \
            for sublist in DEMOGRAPHY for item in sublist]))

    if isinstance(batch, str):
        cols = ['PRACTICE_PATIENT_ID', 'PRACTICE_ID', 'INDEX_DATE', 'START_DATE',
            'END_DATE', 'COLLECTION_DATE', 'TRANSFER_DATE', 'DEATH_DATE',
            'REGISTRATION_STATUS'].append(batch)
    else:
        batch = list(batch)
        cols = ['PRACTICE_PATIENT_ID', 'PRACTICE_ID', 'INDEX_DATE', 'START_DATE',
            'END_DATE', 'COLLECTION_DATE', 'TRANSFER_DATE', 'DEATH_DATE',
            'REGISTRATION_STATUS'] + list(batch)

    if len(CATGS) > 0:
        cols = cols + CATGS

    print("Loading column: ", cols)

    if usePolars:
        #Polars
        dat_incprev = plIncPrev(STUDY_END_DATE[0], #currently this is the value for start/end_date_inc
                                STUDY_START_DATE[0],
                                FILENAME,
                                conf_incprev["database"],
                                batch,
                                DEMOGRAPHY,
                                cols,
                                fileType=fileType)

        if conf_incprev["merge_EthOtherMixed"]:
            dat_incprev.raw_data=(
                    dat_incprev.raw_data
                    .with_columns(
                        pl.when((pl.col("ETHNICITY")=="MIXED") | \
                                (pl.col("ETHNICITY")=="OTHER"))
                            .then(pl.lit('OTHER_OR_MIXED'))
                            .otherwise(pl.col("ETHNICITY"))
                        .alias("ETHNICITY")
                    )
            )
        results = dat_incprev.runAnalysis()
        for result_ in results:
            if "Prevalence" in result_.columns:
                metric = "prev"
            else:
                metric = "inc"
            result_.write_csv(f"{DIR_OUT}out_{metric}_{batchId}.csv")

    else:
        #Pandas
        print("running pandas incprev")
        #Incidence
        dat_incprev = pdIncPrev(STUDY_END_DATE[0],
                                STUDY_START_DATE[0],
                                FILENAME,
                                "GOLD",
                                batch,
                                DEMOGRAPHY,
                                cols,
                                fileType=fileType)
        if conf_incprev["merge_EthOtherMixed"]:
            if 'ETHNICITY' in dat_incprev.raw_data.columns:
                dat_incprev.raw_data['ETHNICITY'] = \
                        dat_incprev.raw_data['ETHNICITY'].apply(
                                lambda x: "OTHERS_AND_MIXED" if x in ('OTHER', 'MIXED') else x)

        dat_incprev.calculate_incidence(path_out=DIR_OUT)
        if conf_incprev["calc_grouped"]:
            dat_incprev.calculate_grouped_incidence(path_out=DIR_OUT)

        #Prevalence
        dat_incprev = pdIncPrev(STUDY_END_DATE[1],
                                STUDY_START_DATE[1],
                                FILENAME,
                                "GOLD",
                                batch,
                                DEMOGRAPHY,
                                cols,
                                fileType=fileType)
        if conf_incprev["merge_EthOtherMixed"]:
            if 'ETHNICITY' in dat_incprev.raw_data.columns:
                dat_incprev.raw_data['ETHNICITY'] = \
                        dat_incprev.raw_data['ETHNICITY'].apply(
                                lambda x: "OTHERS_AND_MIXED" if x in ('OTHER', 'MIXED') else x)

        dat_incprev.calculate_prevalence(path_out=DIR_OUT)
        if conf_incprev["calc_grouped"]:
            dat_incprev.calculate_grouped_prevalence(path_out=DIR_OUT)

if True:#len(BASELINE_DATE_LIST) == 1:
       processBatch(
               BASELINE_DATE_LIST,
               STUDY_START_DATE,
               STUDY_END_DATE,
               FILENAME,
               conf_incprev["DEMOGRAPHY"],
               conf_incprev["fileType"],
               conf_incprev["usePolars"],
        )
else:
    BASELINE_DATE_LIST = \
            [tuple(BASELINE_DATE_LIST[i:i + conf_incprev["batch_size"]]) \
            for i in range(0, len(BASELINE_DATE_LIST), conf_incprev["batch_size"])]
    batches = list(zip(
        BASELINE_DATE_LIST,
        repeat(STUDY_START_DATE),
        repeat(STUDY_END_DATE),
        repeat(FILENAME),
        repeat(conf_incprev["DEMOGRAPHY"]),
        repeat(conf_incprev["fileType"]),
        repeat(conf_incprev["usePolars"]),
        list(range(0, conf_incprev["batch_size"])),#batchId
    ))

    N_PROCESSES = conf_incprev["n_processes"]
    #processBatch(*batches[0])
    #breakpoint()

    if __name__ == '__main__':
        if N_PROCESSES is None:
            pool = mp.Pool(processes = mp.cpu_count() - 2)
        else:
            pool = mp.Pool(processes = N_PROCESSES)
        pool.starmap(processBatch, batches)
#
    from os import listdir
    from re import match, compile

    files_out = listdir(DIR_OUT)
    pattern_inc = compile(r'.*inc_[0-9].*')  # Selects elements starting with 'b'
    pattern_prev = compile(r'.*prev_[0-9].*')  # Selects elements starting with 'b'

    file_names_inc = [x for x in files_out if match(pattern_inc, x)]
    file_names_prev = [x for x in files_out if match(pattern_prev, x)]

    output_file_inc = f"out_inc_{ID}.csv"
    output_file_prev = f"out_prev_{ID}.csv"

    def write_out(file_names, output_file, dir_):
        with open(f"{dir_}{output_file}", 'w') as outfile:
            for file_name in file_names:
                with open(f"{dir_}{file_name}", 'r') as infile:
                #skip header if not 1st out file
                    if file_name.find("_0.csv") > -1:
                        next(infile)
                    else:
                        outfile.write(infile.read())

    write_out(file_names_inc, output_file_inc, DIR_OUT)
    write_out(file_names_prev, output_file_prev, DIR_OUT)

