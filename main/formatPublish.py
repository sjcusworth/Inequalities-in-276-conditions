from os import listdir, makedirs, remove
from os.path import isdir, exists
from shutil import copy
from re import match, compile
import polars as pl
from tqdm import tqdm
from distutils.dir_util import copy_tree

### Combine incprev data #####################################################
def combineIncPrev(path_inc, path_prev, path_out,
        rename_inc, rename_prev, rmOld=False):
    dat_inc = pl.read_csv(path_inc)
    dat_prev = pl.read_csv(path_prev)

    #get year
    dat_inc = dat_inc.with_columns(
            pl.col("Year").str.slice(0, 4)
            )
    dat_prev = dat_prev.with_columns(
            pl.col("Year").str.slice(0, 4)
            )

    dat_inc = dat_inc.rename(rename_inc)
    dat_prev = dat_prev.rename(rename_prev)

    join_cols = [x for x in dat_inc.columns if x not in [
        "Denominator_inc",
        "Numerator_inc",
        "Incidence",
        "Prevalence",
        "LowerCI_inc",
        "UpperCI_inc",
        ]]
    (
        dat_inc
        .join(
            dat_prev,
            how="outer",
            on=join_cols,
            )
        .write_csv(path_out)
    )

    if rmOld:
        remove(path_inc)
        remove(path_prev)

combineIncPrev("out/Publish/inc_crude.csv", "out/Publish/prev_crude.csv",
        "out/Publish/crude.csv",
        rename_inc={
            "Numerator":"Numerator_inc",
            "Denominator":"Denominator_inc",
            "LowerCI":"LowerCI_inc",
            "UpperCI":"UpperCI_inc",
            },
        rename_prev={
            "Numerator":"Numerator_prev",
            "Denominator":"Denominator_prev",
            "LowerCI":"LowerCI_prev",
            "UpperCI":"UpperCI_prev",
            },
        rmOld=True)

combineIncPrev("out/Publish/inc_DSR.csv", "out/Publish/prev_DSR.csv",
        "out/Publish/DSR.csv",
        rename_inc={
            "LowerCI":"LowerCI_inc",
            "UpperCI":"UpperCI_inc",
            },
        rename_prev={
            "LowerCI":"LowerCI_prev",
            "UpperCI":"UpperCI_prev",
            },
        rmOld=True)


### Ratio zscoring ############################################################
#format date in ratioZscoring
dat = pl.read_csv("out/Publish/Average_Geometric/281 conditions chi2 z-scores, expected and observed rates.csv", infer_schema_length=0)
dat = (
        dat
        .with_columns(
            pl.col("Date").str.slice(0, 4)
            )
        .write_csv("out/Publish/Average_Geometric/281 conditions chi2 z-scores, expected and observed rates.csv")
        )

dat = pl.read_csv("out/Publish/Average_Geometric/281 conditions yearly subgroup-overall ratios.csv", infer_schema_length=0)
cols_date = [x for x in dat.columns if x.startswith("2")]
cols_re = [x[0:4] for x in cols_date]
renameDict = dict(zip(cols_date, cols_re))
dat = (
        dat
        .rename(renameDict)
        .write_csv("out/Publish/Average_Geometric/281 conditions yearly subgroup-overall ratios.csv")
        )


