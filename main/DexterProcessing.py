from csv import DictWriter, reader
from polars import scan_csv, col, concat, Series, concat_list
from polars import scan_parquet
from polars import min as plmin
from re import sub, split
from polars import Utf8 as plUtf8
from polars import Categorical as plCategorical
from polars import Date as plDate
import gc
import pyarrow.dataset as ds
from pyarrow.csv import CSVWriter

def rmDup(
        A_raw:str,
        B_raw:str,
        A_ind:int=2,
        B_ind:int=0,
        map_file:str="./VisionToEmisMigrators.txt",
        map_delim:str="\t",
        matching_cols:bool=True,
        low_memory:bool=True,
        wdir = "./",
        ):
    """
    Link datasets A and B by PracticeID
    Links the two datasets, removing duplicate practices from A.

    Compares the practices in the practice map_file across A and
    B datasets.
    Writes a csv summary, of number of practices from the mapping present in A
    or B, and whether practices exisiting in one dataset also exist in the
    other.

    Parameters:
        A_raw (str): File A path
        B_raw (str): File B path
        A_ind (int): Index of A practiceIDs column in map_file
        B_ind (int): Index of A practiceIDs column in map_file
        map_file (str): File containing mapping information
        map_delim (str): Delimeter used in mapping file
        matching_cols (bool): Do the columns match across A and B?

    Returns:
        None
    """
    #Get names for naming of output csv
    A_name = sub("^.*/(.*)\.csv$", r"\1", A_raw)
    B_name = sub("^.*/(.*)\.csv$", r"\1", B_raw)
    #Read in data
        #infer_schema_length=0 reads all cols as utf8 (str)
        #preventing type errors when concat
    A_raw = scan_csv(A_raw, infer_schema_length=0, low_memory=low_memory)
    B_raw = scan_csv(B_raw, infer_schema_length=0, low_memory=low_memory)

    def practiceMatches(
            dict_pracMap:dict,
            uniquePracticeID_A:list,
            uniquePracticeID_B:list
            ):
        """
        Compares the practices in the practice map_file across A and
        B datasets.

        Parameters:
            dict_pracMap (dict): Mapping of practice IDs across A and B
            uniquePracticeID_A (list): Unique practice IDs in A
            uniquePracticeID_B (list): Unique practice IDs in B

        Returns:
            (dict): Counts of duplicate practices across A and B
            (list[list]): Each element defines practice ID in A, if practice\
                exists in A, and if practice exists in B
        """
        results = {}

        #Find total count of practices in the practice map for A and B
        f_exists = lambda x,y:True if x in y else False
        results['n_pracMap_inA'] = sum([f_exists(x, list(dict_pracMap.keys()))\
                for x in uniquePracticeID_A])
        results['n_pracMap_inB'] = sum([f_exists(x, \
                list(dict_pracMap.values())) for x in uniquePracticeID_B])

        def check_dictMatch(x_A, x_B, dictionary, f_exists):
            """
            Find duplicate practices across A and B

            Parameters:
                x_A (list): Unique practice IDs in A
                x_B (list): Unique practice IDs in B
                dictionary (dict): Mapping dict where key=A_Ids, value=B_Ids
                f_exists (callable): Function to check occurance of A in B

            Returns:
                (list): List, where each element is a list of\
                    Practice ID in A, if practice exists in A, and if \
                    practice exists in B.
            """
            checks = []
            for key, value in dictionary.items():
                check = [key, False, False] #Initialise list where practice ID\
                    #is as in A
                if f_exists(key, x_A):
                    check[1] = True #Is practice in A
                if f_exists(value, x_B):
                    check[2] = True #Is practice in B
                checks.append(check)
            return(checks)

        checkMatches = check_dictMatch(uniquePracticeID_A,
                uniquePracticeID_B, dict_pracMap, f_exists)

        f_compare = lambda x,y,in_A, in_B:True if x == in_A and y == \
                in_B else False
        #Get counts of practices across A and B
        results['n_inA_inB'] = sum([f_compare(x, y, True, True) for _,x,y \
                in checkMatches]) #in A and in B
        results['n_inA_notB'] = sum([f_compare(x, y, True, False) for _,x,y \
                in checkMatches]) #in A and not in B
        results['n_notA_inB'] = sum([f_compare(x, y, False, True) for _,x,y \
                in checkMatches]) #not in A and not in B
        results['n_notA_notB'] = sum([f_compare(x, y, False, False) for \
                _,x,y in checkMatches]) #not in A and not in B

        return(results, checkMatches)

    #construct the practice mapping dictionary
    dict_pracMap = {}
    with open(map_file) as f:
        csv_reader = reader(f, delimiter=map_delim)
        for i, practice in enumerate(csv_reader):
            if i != 0: #skip header
                dict_pracMap[f"p{practice[A_ind]}"] = f"p{practice[B_ind]}"

    gc.collect()

    #Run practiceMatches()
    mapping_info, dict_mappings = practiceMatches(dict_pracMap,
            (A_raw.select(
                    col('PRACTICE_ID').unique().alias("PRACTICE_ID")
                ).collect()
                .get_column("PRACTICE_ID").to_list()
            ), #Gets list of unique IDs in A
            (B_raw.select(
                    col('PRACTICE_ID').unique().alias("PRACTICE_ID")
                ).collect()
                .get_column("PRACTICE_ID").to_list()
            ) #Gets list of unique IDs in B
            )
    #Save summary counts found in practiceMatches()
    with open(f"{wdir}LINK_STATS_{A_name}_{B_name}.csv", "w") as f:
        w = DictWriter(f, mapping_info.keys())
        w.writeheader()
        w.writerow(mapping_info)
    gc.collect()

    ##Remove PRACTICE_IDs found in B and in A, from A (duplicates)
        #i.e. keep practices in A that are in the practice mapping, but do not\
        #appear in B (remove the rest from A)
    removePractice_Ids = [x for x,a,b in dict_mappings if a == True and b == True]

    #Find rows to include in A
        #Include if PATIENT_ID not in removePractice_Ids
    f_map_rmDup = lambda x,y:True if x not in y else False
    A_practiceID = (
        A_raw.select(col("PRACTICE_ID"))
        .collect().get_column("PRACTICE_ID")
        .to_list()
    ) #Get list of PRACTICE_ID in A

    #Gets a bool mapping of rows to include in A
    AExclude_map = [f_map_rmDup(x, y) for y in [removePractice_Ids] for x in A_practiceID]

    #Bool subset of A using AExclude map
        #Workaround using polars LazyFrame
    A_deDup = (
        A_raw
        .with_columns(
            Series("map", AExclude_map)
        ) #Add bool map as a column
        .filter(col("map") == True) #Filter for True in map
        .select(
            col("*").exclude("map") #Remove map column
        )
    )
    del A_raw
    del AExclude_map
    gc.collect()

    if low_memory:
        del B_raw
        gc.collect()
        print("Saving A dedup for joining to B (low memory mode)")
        print(f"Run in shell: export IFS=","; join -t, -1 N -2 N -a 1 -a 2 <(sort -k N {B_name}) <(sort -k N Dedup_{A_name}.csv) > linked_rmAurumPracs.csv")
        print("(Where N is column index of a unique identifier (PATIENT_ID))")
        A_deDup.collect(streaming=True).write_csv(f"{wdir}Dedup_{A_name}.csv")
    else:
        print("Saving combined data")
        if matching_cols is True:
            combo_raw = concat([A_deDup, B_raw],
                how="vertical", parallel=low_memory)
        else:
            combo_raw = concat([A_deDup, B_raw],
                how="diagonal", parallel=low_memory)

        combo_raw.collect().write_csv(f"{wdir}LINKED_{A_name}_{B_name}.csv")


def process_ethImd(
    file_dat,
    path_dir = "./",
    file_map = "imd_mapping.csv",
    merge_imd=None, #deprecated
    imd_delim=",",
    i_imd_key:list[int]=[0],
    i_imd_value=1,
    row_group_size=None,
    low_memory=False,
    is_parquet=True,
    calcEth=False,
    toParquet=True,
    ):
    """

    """
    if is_parquet:
        dataset = ds.dataset(f"{path_dir}{file_dat}", format="parquet")
        cols = list(dataset.head(1).to_pylist()[0].keys())
        del dataset
    else:
        with open(f"{path_dir}{file_dat}") as f:
            cols = f.readline().strip('\n')
            cols = cols.split(",")
    cols = [x for x in cols if x.find("BD") != -1 and x.find("ETH") != -1]
    colLabs = [split(":", x)[1] for x in cols.copy()]
    colLabs = [sub("_ETH.*$", "", x) for x in colLabs]

    meta = ["PATIENT_ID", "PRACTICE_ID"]

    if calcEth:
        #Missing column required to be before all other ethnicity columns
        i_missing = colLabs.index("MISSING")
        x = colLabs.pop(i_missing)
        colLabs.insert(0, x)
        x = cols.pop(i_missing)
        cols.insert(0, x)

        meta.append("ETHNICITY")

    imd_dict = dict()

    for i_key in i_imd_key:
        with open(f"{path_dir}{file_map}", "r") as f:
            r = reader(f, delimiter=imd_delim)
            label = next(r, None)[i_key]
            label = f"IMD_{label}"
            imd_dict[label] = dict()
            for line in r:
                value = str(line[i_imd_value])
                #Ensure all values contain no special characters
                value = sub(r'[^\w]', '', value)

                if label == "IMD_pracid":
                    imd_dict[label][f"p{line[i_key]}"] = value
                else:
                    imd_dict[label][f"{line[i_key]}"] = str(line[i_imd_value])

    if is_parquet:
        q1 = (
            scan_parquet(f"{path_dir}{file_dat}", low_memory=True)
        )
    else:
        q1 = (
            scan_csv(f"{path_dir}{file_dat}", low_memory=True)
        )

    q1 = (
        q1
        .select(cols+meta)
        .with_columns(
            col(["PATIENT_ID"]).cast(plUtf8),
            col(["PRACTICE_ID"]).cast(plUtf8)
        )
    )
    if calcEth:
        q1 = (
            q1
            .with_columns(
                col(["ETHNICITY"]).cast(plCategorical)
            )
            .with_columns(
                col("^BD.*$").str.strptime(plDate, format="%Y-%m-%d", strict=False)
            )
            .with_columns(
                concat_list(col("^BD.*$")).alias("dates")
            )
            .with_columns(
                    col("dates").apply(lambda x: \
                        colLabs[x.arg_max()]
                    ).alias("ETHNICITY_REVISED")
            )
    )

    # calculating imd is now the purpose of this function (no conditional)
    for imd_type in imd_dict.keys():
        if imd_type == "IMD_patid":
            lab_col = "PATIENT_ID"
        elif imd_type == "IMD_pracid":
            lab_col = "PRACTICE_ID"
        else:
            print("IMD script not working")
            break

        map_imd = imd_dict[imd_type]
        q1 = (
            q1
            .with_columns(
                col(lab_col).map_dict(map_imd).alias(imd_type)
            )
        )
    if not calcEth:
        q1 = (
            q1
            .select(["PATIENT_ID"]+list(imd_dict.keys()))
            .collect().write_parquet(f"{path_dir}dat_imd.parquet")
        )
    else:
        q1 = (
            q1
            .select(["PATIENT_ID", "ETHNICITY_REVISED"]+list(imd_dict.keys()))
            .collect().write_parquet(f"{path_dir}dat_imd.parquet")
        )

    q1
    del q1
    gc.collect()

    file_1 = f"{path_dir}{file_dat}"
    file_2 = f"{path_dir}dat_imd.parquet"
    joinCol = "PATIENT_ID"

    toAdd = (
        scan_parquet(file_2, low_memory=low_memory)
        .with_columns(col("*").cast(plUtf8))
    )

    if is_parquet:
        Combine = (
            scan_parquet(file_1,
                     low_memory=low_memory)
            .with_columns(col("*").cast(plUtf8))
        )
    else:
        Combine = (
            scan_csv(file_1,
                     infer_schema_length=0,
                     low_memory=low_memory)
        )
    Combine = (
        Combine
        .join(toAdd, on=joinCol, how="left") #how="left" supports .sink_parquet and should be same as outer
    )
    if toParquet:
        Combine.sink_parquet(f"{path_dir}dat_processed.parquet",
                             row_group_size=row_group_size)
    else:
        Combine.collect().write_csv(f"{path_dir}dat_processed.csv")


def mergeCols(path_dat: str,
    file_dat: str,
    dict_merge: dict,
    low_memory = False,
    file_type = "parquet"):
    if file_type == "parquet":
        q1 = scan_parquet(f"{path_dat}{file_dat}")#, infer_schema_length=0)
    else:
        q1 = scan_csv(f"{path_dat}{file_dat}", infer_schema_length=0)

    for out_col, merge_cols in dict_merge.copy().items():
        if len(merge_cols) > 1:
            q1 = (
                q1
                .with_columns(
                    col(merge_cols).str.strptime(plDate, "%Y-%m-%d"),
                    plmin(merge_cols).alias(out_col)
                )
            )
        else:
            del dict_merge[out_col] #prevents deleting of unmerged cols

    if low_memory:
        q1 = (
            q1
            .select(col(["PATIENT_ID"] + list(dict_merge.keys())))
            .collect().write_parquet(f"{path_dat}condMerged.parquet")
        )
        q1

        del q1

        file_1 = f"{path_dat}{file_dat}"
        file_2 = f"{path_dat}condMerged.parquet"
        joinCol = "PATIENT_ID"

        toAdd = (
            scan_parquet(file_2)
        )
        Combine = (
            scan_parquet(file_1)
            .join(toAdd, on=joinCol, how="left") #how="left" supports .sink_parquet and should be same as outer
            .sink_parquet(f"{path_dat}mergedDat.parquet")
        )
        Combine

    else:
        rm_cols = list(dict_merge.values())
        rm_cols = [x for sublist in rm_cols for x in sublist]
        q1 = (
            q1
            .select(col("*").exclude(rm_cols))
            .collect().write_parquet(f"{path_dat}condMerged.parquet")
        )
        q1
    return "finished merging"

def par_to_csv(file_noExtension):
    dataset = ds.dataset(f"{file_noExtension}.parquet", format="parquet")
    schema = dataset.schema
    writer = CSVWriter(f"{file_noExtension}.csv", schema)
    for batch in dataset.to_batches():
        writer.write_batch(batch)
    print("Finished")
