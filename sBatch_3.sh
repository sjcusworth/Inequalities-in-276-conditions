#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --time=3:0:0
#SBATCH --mem=32G
#SBATCH --qos=bbdefault
#SBATCH --mail-type=NONE

#Due to need for openpyxl in these scripts, older version of BlueBear apps used (2022a vs 2022b)
module purge; module load bluebear;
module load bear-apps/2022a;

module load Python/3.10.4-GCCcore-11.3.0;
module load SciPy-bundle/2022.05-foss-2022a;
module load matplotlib/3.5.2-foss-2022a;
module load Seaborn/0.12.1-foss-2022a;
module load tqdm/4.64.0-GCCcore-11.3.0;
module load zstd/1.5.2-GCCcore-11.3.0;
module load Arrow/8.0.0-foss-2022a;
module load polars/0.17.12-foss-2022a;
module load PyYAML/6.0-GCCcore-11.3.0;
module load plotly.py/5.12.0-GCCcore-11.3.0;
module load openpyxl/3.0.10-GCCcore-11.3.0;
echo "Modules Loaded";

# init out/Publish
rm -r out/Publish
mkdir out/Publish
# get combined crude inc/prev files
for metric in "inc" "prev";
do
    files_combo=( $(ls out | grep out_${metric}_) );
    file_out=out/out_${metric}.csv;

    for i in ${!files_combo[@]};
    do
        # if 1st file to write, include header
        if (( i -e 0 ));
        then
            cat out/${files_combo[$i]} > $file_out;
        else # remove header
            tail -n +2 out/${files_combo[$i]} >> $file_out;
        fi;
    done;
done;

echo "Standardising started"
python3 main/strd.py
echo "Standardising complete"

echo "ratioZscore started"
python3 main/ratioZscore.py
echo "Finished ratioZscore"

# init out/Publish
rm -r out/Publish
mkdir out/Publish
# NOTE: these are not publishable until after smallNumCens.py
cp out/out_inc.csv out/Publish/inc_crude.csv
cp out/out_prev.csv out/Publish/prev_crude.csv

echo "Censor small numbers started"
python3 main/smallNumCens.py
echo "Censor small numbers complete"

echo "format started"
python3 main/formatPublish.py
echo "format complete"

echo "tidyImdCategories started"
python3 main/tidyImdCategories.py
echo "tidyImdCategories finished"
