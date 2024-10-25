import polars as pl
from tqdm import tqdm

# Must check for surrounding ''. Need to search and replace 'Ireland' (IMD) but not replace 'Northern Ireland' (region).
    #If surrounding '' can just search and replace "'Ireland'"
files = {
        #NOTE: Need more memory to run crude.csv
        #'out/Publish/crude.csv': {
        #    "colsEdit": ['Subgroup'],
            #"group_surrounding_'": True,
            #"single_surrounding_'": False,
        #    },
        'out/Publish/DSR.csv': {
            "colsEdit": ['Subgroup'],
            #"group_surrounding_'": True,
            #"single_surrounding_'": True,
            },
        'out/Publish/Average_Geometric/281 conditions yearly subgroup-overall ratios.csv': {
                    "colsEdit": ['Group'],
                    #"group_surrounding_'": None,
                    #"single_surrounding_'": True,
            },
        'out/Publish/Average_Geometric/281 conditions chi2 z-scores, expected and observed rates.csv': {
                    "colsEdit": ['Group'],
                    #"group_surrounding_'": None,
                    #"single_surrounding_'": True,
            },
        }

for file_, values_ in tqdm(files.items()):
    query = pl.scan_csv(file_, infer_schema_length=0,)

    query = (
            query
            .with_columns(
                pl.col(values_["colsEdit"]).str.replace_all("'", "")\
                        .str.strip_chars("()")\
                        .str.split(by=",")
                )
            .with_columns(
                pl.col(values_["colsEdit"]).map_elements(
                    lambda x: [x_.strip() if x_.strip()!="Ireland" else "MissingImd" for x_ in x]
                    )
                )
            .with_columns(
                pl.col(values_["colsEdit"]).list.join(",")
                )
            )
    #query.collect().write_csv(f"{file_[:-4]}_test.csv")
    query.collect().write_csv(file_)

