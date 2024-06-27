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
if not exists("out/Publish"):
    makedirs("out/Publish")

if not exists("out/Publish/Average_Geometric"):
    makedirs("out/Publish/Average_Geometric")

copy_tree("out/Average_Geometric", "out/Publish/Average_Geometric")

### Format crude files #######################################################
output_file_inc = f"out/Publish/inc_crude.csv"
output_file_prev = f"out/Publish/prev_crude.csv"

conds_out = [x for x in listdir("out/") if (isdir(f"out/{x}") and not x.startswith("Average") and not x.startswith("inc") and not x.startswith("prev") and not x.startswith("Publish"))]
pattern_inc = compile(r'.*Inc.*')  # Selects elements starting with 'b'
pattern_prev = compile(r'.*Prev.*')  # Selects elements starting with 'b'

schema_inc = {
        "Condition": pl.Utf8,
        "Subgroup": pl.Utf8,
        "Year": pl.Utf8,
        "Group": pl.Utf8,
        "Denominator": pl.Utf8,
        "Numerator": pl.Utf8,
        "Incidence": pl.Utf8,
        "LowerCI": pl.Utf8,
        "UpperCI": pl.Utf8,
        }
schema_prev = {
        "Condition": pl.Utf8,
        "Subgroup": pl.Utf8,
        "Year": pl.Utf8,
        "Group": pl.Utf8,
        "Denominator": pl.Utf8,
        "Numerator": pl.Utf8,
        "Prevalence": pl.Utf8,
        "LowerCI": pl.Utf8,
        "UpperCI": pl.Utf8,
        }

toAdd_inc = [] #pl.DataFrame(schema=schema_inc)
toAdd_prev = [] #pl.DataFrame(schema=schema_prev)

for cond_ in tqdm(conds_out):
    files_out = listdir(f"out/{cond_}")

    file_names_inc = [x for x in files_out if match(pattern_inc, x)]
    file_names_prev = [x for x in files_out if match(pattern_prev, x)]

    for metric_, files_ in {"Incidence":file_names_inc,"Prevalence":file_names_prev}.items():
        for file_ in files_:
            file_ = f"out/{cond_}/{file_}"
            dat_temp = pl.read_csv(file_, infer_schema_length=0,)
            dat_temp_subgroupCol = dat_temp.columns[1]
            dat_temp = dat_temp.rename({
                dat_temp.columns[1]:"Subgroup",
                "Lower":"LowerCI",
                "Upper":"UpperCI",
                })
            if metric_ == "Incidence":
                dat_temp = dat_temp.rename({"PersonYears":"Denominator"})

            dat_temp = (
                    dat_temp
                    .with_columns(
                        pl.lit(cond_).alias("Condition"),
                        pl.lit(dat_temp_subgroupCol).alias("Group"),
                        )
                    .select(
                        pl.col([
                            "Condition",
                            "Subgroup",
                            "Year",
                            "Group",
                            "Denominator",
                            "Numerator",
                            metric_,
                            "LowerCI",
                            "UpperCI",
                            ])
                        )
                    )
            if metric_ == "Incidence":
                toAdd_inc.append(dat_temp) #pl.concat([toAdd_inc, dat_temp])
            elif metric_ == "Prevalence":
                toAdd_prev.append(dat_temp) #pl.concat([toAdd_prev, dat_temp])

toAdd_inc = pl.concat(toAdd_inc)
toAdd_prev = pl.concat(toAdd_prev)

## Tidy condition names
condition_labels = pd_read_excel(
    f'data/aurumGoldLabels_NCQOF.xlsx',
    header=3).dropna(subset=['Joht Labelling'])

GOLD_labels = condition_labels['GOLD']
JOHT_labels = condition_labels['Joht Labelling']
map_condLabel = dict(zip(GOLD_labels, JOHT_labels))

toAdd_inc = (
        toAdd_inc
        .with_columns(
            pl.col("Condition").map_dict(map_condLabel)
            )
        )
toAdd_prev = (
        toAdd_prev
        .with_columns(
            pl.col("Condition").map_dict(map_condLabel)
            )
        )

toAdd_inc.write_csv(output_file_inc)
toAdd_prev.write_csv(output_file_prev)

del toAdd_inc
del toAdd_prev
gc.collect()

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
        .join(
            dat_crude_inc.rename({"Subgroup": "Group", "Year": "Date",}),
            on=["Group", "Date", "Condition"],
            how="left",
            )
        .rename({"Numerator": "Numerator_inc"})
        )
dat_geo = (
        dat_geo
        .join(
            dat_crude_prev.rename({"Subgroup": "Group", "Year": "Date",}),
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

smallCountsCens("out/Publish/Average_Geometric/281 conditions chi2 z-scores, expected and observed rates.csv", ["Numerator_prev"], metric=["Prevalence", "Prevalence Ratio"], upperCI=None, lowerCI=None,)
smallCountsCens("out/Publish/Average_Geometric/281 conditions chi2 z-scores, expected and observed rates.csv", ["Numerator_inc"], metric=["Incidence", "Incidence Ratio"], upperCI=None, lowerCI=None,)
#pl.read_csv("out/Publish/Average_Geometric/281 conditions chi2 z-scores, expected and observed rates.csv", infer_schema_length=0).select(pl.all().exclude(["Numerator_prev", "Numerator_inc"])).write_csv("out/Publish/Average_Geometric/281 conditions chi2 z-scores, expected and observed rates.csv")
