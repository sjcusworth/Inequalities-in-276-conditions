import sys
import datetime
import csv
import multiprocessing as mp
from itertools import repeat
#import gc

import polars as pl
from DexterProcessing import process_ethImd, create_batch_files
import pyarrow.dataset as ds
import yaml

with open("wdir.yml",
          "r",
          encoding="utf8") as file_config:
    config = yaml.safe_load(file_config)

PATH = config["PATH"]
DIR_DATA = f"{PATH}{config['dir_data']}"

process_ethImd(
        config["processing"]["filename"],
        DIR_DATA,
        row_group_size=None,
        low_memory=False,
        is_parquet=False,
        toParquet=True,
        file_map = "imd_mapping.csv",
        )

#gc.collect()

# --- Optimisation: write per-batch column-subset parquet files ---
# Enabled via create_batch_files: true in wdir.yml (incprev section).
# Each dat_batch_{ID}.parquet contains only the columns needed by that SLURM
# array job, reducing IncPrev.py memory use from ~full-file to ~25 columns.
if config.get("incprev", {}).get("create_batch_files", False):
    create_batch_files(
        path_dir=DIR_DATA,
        processed_filename=config["incprev"]["filename"],
        bd_list=config["incprev"]["BD_LIST"],
        demography=config["incprev"]["DEMOGRAPHY"],
    )

## Remove duplicate condition
#dat = pl.scan_parquet(f"{DIR_DATA}dat_processed.parquet")
#dat = dat.select(pl.all().exclude(["BD_MEDI:CPRD_2RY_POLYCYTHAEMIA:266", "B_MEDI:CPRD_2RY_POLYCYTHAEMIA:266", "B.M.CPRD_2RY_POLYCYTHAEMIA:266"]))
#dat.collect(streaming=True).write_parquet(f"{DIR_DATA}dat_processed_clean.parquet")
