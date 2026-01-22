from os import listdir, makedirs, remove
import gc
from os.path import isdir, exists
from shutil import copy
from re import match, compile
import polars as pl
from tqdm import tqdm
from distutils.dir_util import copy_tree
from pandas import read_excel as pd_read_excel

### Make publish dir #########################################################

if not exists("out/Publish/Average_Geometric"):
    makedirs("out/Publish/Average_Geometric")

copy_tree("out/Average_Geometric", "out/Publish/Average_Geometric")

### Censor small counts ######################################################

def getCrudeMap(filePath,):
    dat_ = (
            pl.read_csv(filePath, infer_schema_length=0,)
            .with_columns(
                pl.col("Numerator").cast(pl.Int64)
                )
            .select(pl.col(["Subgroup", "Year", "Condition", "Numerator", "Group",]))
            .with_columns(
                pl.col("Subgroup").apply(lambda x: "'" + "', '".join(["".join([char for char in label.strip() if char not in ["'", "(", ")", '"']]) for label in x.split(",")[2:]]) + "'"),
                )
            .filter(pl.col("Subgroup")!="''")
            .filter(pl.col("Group").str.starts_with("'AGE_CATEGORY', 'SEX'"))
            .select(pl.all().exclude("Group"))
            .groupby(pl.col(["Subgroup", "Year", "Condition"])).sum()
            )
    dat_overall_ = (
            pl.read_csv(filePath, infer_schema_length=0,)
            .select(pl.col(["Subgroup", "Year", "Condition", "Numerator",]))
            .filter(pl.col("Subgroup")=="OVERALL")
            .with_columns(
                pl.col("Subgroup").map_dict({"OVERALL":"Overall"}, default=pl.first())
                )
            .with_columns(
                pl.col("Numerator").cast(pl.Int64)
                )
            )
    dat_ = pl.concat([dat_, dat_overall_])

    return dat_

dat_crude_inc = getCrudeMap("out/Publish/inc_crude.csv")
dat_crude_prev = getCrudeMap("out/Publish/prev_crude.csv")


## Combine dsr files with numerators from crude (need to define values to censor)
dat_dsr_inc = pl.read_csv("out/inc_DSR.csv", infer_schema_length=0,)
dat_dsr_prev = pl.read_csv("out/prev_DSR.csv", infer_schema_length=0,)

dat_dsr_inc = (
        dat_dsr_inc
        .join(
            dat_crude_inc,
            on=["Subgroup", "Year", "Condition"],
            how="left",
            )
        .write_csv("out/Publish/inc_DSR.csv")
        )
dat_dsr_prev = (
        dat_dsr_prev
        .join(
            dat_crude_prev,
            on=["Subgroup", "Year", "Condition"],
            how="left",
            )
        .write_csv("out/Publish/prev_DSR.csv")
        )

del dat_dsr_inc
del dat_dsr_prev
gc.collect()

## Combine geo files with numerators from crude (need to define values to censor)
dat_geo = pl.read_csv("out/Publish/Average_Geometric/281 conditions chi2 z-scores, expected and observed rates.csv",
        infer_schema_length=0,)

dat_geo = (
        dat_geo
        .with_columns(
            pl.col("Date").str.slice(offset=0, length=4)
            )
        )

dat_geo = (
        dat_geo
        .join(
            (
                dat_crude_inc
                .rename({"Subgroup": "Group", "Year": "Date",})
                .with_columns(
                    pl.col("Date").str.slice(offset=0, length=4)
                    )
                ),
            on=["Group", "Date", "Condition"],
            how="left",
            )
        .rename({"Numerator": "Numerator_inc"})
        )
dat_geo = (
        dat_geo
        .join(
            (
                dat_crude_prev
                .rename({"Subgroup": "Group", "Year": "Date",})
                .with_columns(
                    pl.col("Date").str.slice(offset=0, length=4)
                    )
                ),
            on=["Group", "Date", "Condition"],
            how="left",
            )
        .rename({"Numerator": "Numerator_prev"})
        .write_csv("out/Publish/Average_Geometric/281 conditions chi2 z-scores, expected and observed rates.csv")
        )

del dat_geo
gc.collect()


## Setting small counts and corresponding incprev to null
def smallCountsCens(path_dat, cols, metric=None, upperCI="UpperCI", lowerCI="LowerCI"):
    dat = pl.read_csv(path_dat, infer_schema_length=0,)
    dat = (
            dat
            .with_columns(
                #float for compatibility when e.g. "11.0"
                pl.col(cols).cast(pl.Float64)
                )
            )
    for col_ in cols:
        censor = (
               dat
               .with_columns(
                   pl.col(col_).fill_null(0) #will be set to null at next line; needed for compatibility with apply
                   )
               .with_columns(
                   pl.col(col_).apply(lambda x: False if isinstance(x, pl.Null) or x <= 10 else True).alias("censor")
                   )
               .get_column("censor")
                )
        dat = (
                dat
                .with_columns(
                    dat.get_column(col_).zip_with(
                        censor,
                        pl.Series([None]*censor.shape[0]),
                        ).alias(col_)
                    )
                )
        if metric is not None:
            for metric in metric:
                dat = (
                        dat
                        .with_columns(
                            dat.get_column(metric).zip_with(
                                censor,
                                pl.Series([None]*censor.shape[0]),
                                ).alias(metric)
                            )
                        )
        if upperCI is not None:
            dat = (
                    dat
                    .with_columns(
                        dat.get_column(upperCI).zip_with(
                            censor,
                            pl.Series([None]*censor.shape[0]),
                            ).alias(upperCI)
                        )
                    )

        if lowerCI is not None:
            dat = (
                    dat
                    .with_columns(
                        dat.get_column(lowerCI).zip_with(
                            censor,
                            pl.Series([None]*censor.shape[0]),
                            ).alias(lowerCI)
                        )
                    )

    dat.write_csv(path_dat)


smallCountsCens("out/Publish/inc_DSR.csv", ["Numerator"], metric=["Incidence"])
pl.read_csv("out/Publish/inc_DSR.csv", infer_schema_length=0).select(pl.all().exclude("Numerator")).write_csv("out/Publish/inc_DSR.csv")

smallCountsCens("out/Publish/prev_DSR.csv", ["Numerator"], metric=["Prevalence"])
pl.read_csv("out/Publish/prev_DSR.csv", infer_schema_length=0).select(pl.all().exclude("Numerator")).write_csv("out/Publish/prev_DSR.csv")

smallCountsCens("out/Publish/prev_crude.csv", ["Numerator"], metric=["Prevalence"])
smallCountsCens("out/Publish/inc_crude.csv", ["Numerator"], metric=["Incidence"])

smallCountsCens("out/Publish/Average_Geometric/281 conditions chi2 z-scores, expected and observed rates.csv", ["Numerator_prev"], metric=["Prevalence", "Prevalence Ratio", "Expected Prevalence", "Prevalence Z-Score",], upperCI=None, lowerCI=None,)
smallCountsCens("out/Publish/Average_Geometric/281 conditions chi2 z-scores, expected and observed rates.csv", ["Numerator_inc"], metric=["Incidence", "Incidence Ratio", "Expected Incidence", "Incidence Z-Score",], upperCI=None, lowerCI=None,)
pl.read_csv("out/Publish/Average_Geometric/281 conditions chi2 z-scores, expected and observed rates.csv", infer_schema_length=0).select(pl.all().exclude(["Numerator_prev", "Numerator_inc"])).write_csv("out/Publish/Average_Geometric/281 conditions chi2 z-scores, expected and observed rates.csv")
