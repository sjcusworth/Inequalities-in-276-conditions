import logging
import polars as pl
import yaml

## Log
logging.basicConfig(filename="log_sBatch_1Python.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger()
##

## Env vars
with open("wdir.yml",
          "r",
          encoding="utf8") as file_config:
    config = yaml.safe_load(file_config)

PATH = config["PATH"]
DIR_DATA = f"{PATH}{config['dir_data']}"
file_out = f"{DIR_DATA}{config['processing']['filename']}"
##

logger.info("Combining IMD Map Files")
logger.info("    NOTE: aurum imd map has been editted to make Ireland imds Ireland")
def combineImdMapFiles(file_gold, file_aurum, file_out):
    dat_gold = (
            pl.scan_csv(file_gold, infer_schema_length=0,)
            .select(
                pl.col(["pracid", "e2019_imd_10",])
                )
            .rename({
                "e2019_imd_10":"imd",
                })
            )
    dat_aurum = (
            pl.scan_csv(file_aurum, infer_schema_length=0, separator="\t",)
            .select(
                pl.col(["pracid","e2019_imd_10",])
                )
            .rename({
                "e2019_imd_10":"imd",
                })
            )
    dat = pl.concat([dat_aurum, dat_gold], how="vertical")
    checkDups = dat.select(pl.col("pracid")).collect().get_column("pracid")
    if len(checkDups.unique()) != checkDups.shape[0]:
        logger.warning("Warning: duplicate patids across gold and aurum imd mapping files")

    dat.collect().write_csv(file_out)

combineImdMapFiles(
    file_gold=f"{DIR_DATA}imd_map_pracid_gold.csv",
    file_aurum=f"{DIR_DATA}practice_imd_22_002022.txt",
    file_out=f"{DIR_DATA}imd_map.csv",
    )

logger.info("    IMD Map Files Combined")
