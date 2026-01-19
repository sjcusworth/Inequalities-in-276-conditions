import sys
import datetime
import csv
import multiprocessing as mp
from itertools import repeat
import logging
from re import sub

import polars as pl
from DexterProcessing import process_imd
from DexterProcessing import rmDup
from DexterProcessing import mergeCols
from DexterProcessing import combineLevels
import pyarrow.dataset as ds
import yaml

## Log
logging.basicConfig(filename="log_sBatch_1Python.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger()
##

with open("wdir.yml",
          "r",
          encoding="utf8") as file_config:
    config = yaml.safe_load(file_config)

PATH = config["PATH"]
DIR_DATA = f"{PATH}{config['dir_data']}"

## Format Null and ColNames ####################################################
logger.info("Formatting null values")

if config['processing']["filename"] is None:
    filesToFormat = [f"{DIR_DATA}{config['processing']['filename_gold']}",
                     f"{DIR_DATA}{config['processing']['filename_aurum']}"]
    config['processing']['filename_gold'] = f"{config['processing']['filename_gold'][:-4]}_formNulls.parquet"
    config['processing']['filename_aurum'] = f"{config['processing']['filename_aurum'][:-4]}_formNulls.parquet"
else:
    filesToFormat = [f"{DIR_DATA}{config['processing']['filename']}"]
    config['processing']["filename"] = filesToFormat[0]

for file_ in filesToFormat:
    dat = (
            pl.scan_csv(file_, infer_schema_length=0)
            .with_columns(
                pl.when(pl.all().str.len_chars() == 0)
                    .then(None)
                    .otherwise(pl.all())
                    .name.keep()
                )
            )
    # remove suffix from col names
    dup_cols = ("BD_MEDI:CPRD_2RY_POLYCYTHAEMIA:266", "BD_MEDI:CPRD_B12_DEF:270",
            "BD_MEDI:CPRDAURUM_ANGIODYSPLASIA_COLON:13",)
    change_colnames = {k:"" for k in dat.columns if k.startswith("BD_MEDI:") and k not in dup_cols}
    change_colnames = {k:sub(r":\d+$", "", k) for k in change_colnames.keys()}

    dat = (
            dat
            # duplicate col to remove
            .select(pl.all().exclude(dup_cols))
            .rename(change_colnames)
            )

    dat.sink_parquet(f"{file_[:-4]}_formNulls.parquet")
logger.info("    Formatting null values finished")
del dat

###LinkingAurumGold############################################################
if config['processing']["filename"] is None:
    print("Linking")
    logger.info("Linking Gold and Aurum")

    dat_a = config['processing']['filename_gold']
    dat_b = config['processing']['filename_aurum']

    outFile="dat_linked.parquet"
    rmDup(
        f"{DIR_DATA}{dat_a}",
        f"{DIR_DATA}{dat_b}",
        A_ind=0,
        B_ind=2,
        map_file=f"{DIR_DATA}{config['processing']['map_file_AtoB']}",
        map_delim=config['processing']['map_delim_AtoB'],
        low_memory=False,
        wdir=DIR_DATA,
        logger=logger,
        outFile=outFile,
        )
    logger.info("    Linking finished")
else:
    outFile = config["processing"]["filename"]

###MergeCols#####################################################################
if outFile.find(".csv") != -1:
    file_type = "csv"
else:
    file_type = "parquet"

if config["processing"]["mergeCols_AtoB"] is not None:
    print("Merging Cols")
    logger.info("Merging columns")

    mergeCols(
            DIR_DATA,
            outFile,
            config["processing"]["mergeCols_AtoB"],
            file_type = file_type,
            low_memory = True, #False,
            logger=logger,
            outFile="condMerged.parquet",
            )
    outFile = "condMerged.parquet"
    logger.info("    Merging Cols finished")

###CombineLevels#####################################################################
if outFile.find(".csv") != -1:
    file_type = "csv"
else:
    file_type = "parquet"

if config["processing"]["combineLevels"] is not None:
    print("Processing Column Levels")
    logger.info("Combining levels")

    combineLevels(DIR_DATA,
                  outFile,
                  config["processing"]["combineLevels"],
                  file_type=file_type,
                  outFile="dat_updatedLevels.parquet",
                  )
    outFile = "dat_updatedLevels.parquet"
    logger.info("    Processing Column Levels finished")

###LinkImd#####################################################################
if outFile.find(".csv") != -1:
    is_parquet = False
else:
    is_parquet = True

if config["processing"]['link_imd']:
    print("Linking IMD")
    logger.info("Linking IMD")

    process_imd(
            outFile,
            DIR_DATA,
            file_map = config["processing"]["imd_map_file"],
            low_memory=False,
            is_parquet=is_parquet,
            logger=logger,
            outFile="dat_processed.parquet",
            )
    outFile="dat_processed.parquet"
    logger.info("    Linking IMD finished")
