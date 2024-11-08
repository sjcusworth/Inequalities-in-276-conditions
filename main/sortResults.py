import os
import shutil
import sys
from re import sub

import polars as pl

PATH_OUT = "out/"
PATH_SAVE = f"{PATH_OUT}dataStore/"
PATH_CRUDE = f"{PATH_OUT}Publish/crude.csv"
PATH_DSR = f"{PATH_OUT}Publish/DSR.csv"

option = 0

if option == 0:
    groups = ["OVERALL", "ETHNICITY", "SEX", "AGE_CATEGORY", "IMD_pracid",]
    (
        pl.scan_csv(PATH_CRUDE, infer_schema_length=0)
        .with_columns(
            pl.col("Year").cast(pl.Int64)
            )
        .filter(pl.col("Year") >= 2006)
        .filter(pl.col("Group").is_in(groups))
        .collect(streaming=True)
        .write_csv(f"{PATH_CRUDE[:-4]}_filtered.csv")
    )

elif option == 1:
    conditions = tuple(
            pl.scan_csv("out/Publish/DSR.csv")
            .select(pl.col("Condition"))
            .collect()
            .get_column("Condition")
            .unique()
            .to_list()
            )

    ## Setup directory structure
    if os.path.isdir(PATH_SAVE):
        shutil.rmtree(PATH_SAVE)
    os.mkdir(PATH_SAVE)

    for cond_ in conditions:
        os.mkdir(f"{PATH_SAVE}{cond_}")

    ## copy corresponding pdfs
    pdf_type = ["crude", "dsr",]
    pdf_type = pdf_type[1]

    path_pdfs = f"{PATH_OUT}{pdf_type}/"
    for pdf_ in [x for x in os.listdir(path_pdfs) if x.endswith(".pdf")]:
        shutil.copy(f"{path_pdfs}{pdf_}",
                      f"{PATH_SAVE}{pdf_[:-4]}/standardised_{pdf_}")


    ## filter and save data to directories

    def filterSaveData(path, dir_out, file_pref, conditions, groups,):
        dat = (
                pl.scan_csv(path)
                .filter(
                    pl.col("Group").is_in(groups)
                    )
            )

        for cond_ in conditions:
            (
                dat
                .filter(pl.col("Condition") == cond_)
                .collect()
                .write_csv(f"{dir_out}{cond_}/{file_pref}_{cond_}.csv")
            )

    #crude
    filterSaveData(PATH_CRUDE,
                   PATH_SAVE,
                   "crude",
                   conditions,
                   ["OVERALL", "ETHNICITY", "SEX", "AGE_CATEGORY", "IMD_pracid",])

    #dsr
    filterSaveData(PATH_DSR,
                   PATH_SAVE,
                   "standardised",
                   conditions,
                   ["Overall", "Ethnicity", "Deprivation",])
