#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --time=24:0:0
#SBATCH --mem=56G
#SBATCH --qos=castles
#SBATCH --mail-type=NONE
#SBATCH --array=1-26

module purge; module load bluebear;
module load bear-apps/2022b;

module load Python/3.10.8-GCCcore-12.2.0;
module load SciPy-bundle/2023.02-gfbf-2022b;
module load matplotlib/3.7.0-gfbf-2022b;
module load Seaborn/0.12.2-foss-2022b;
module load tqdm/4.64.1-GCCcore-12.2.0;
module load zstd/1.5.2-GCCcore-12.2.0;
module load Arrow/11.0.0-gfbf-2022b;
module load polars/0.19.12-foss-2022b;
module load PyYAML/6.0-GCCcore-12.2.0;
module load plotly.py/5.13.1-GCCcore-12.2.0;
module load openpyxl/3.0.10-GCCcore-11.3.0
echo "Modules Loaded";

echo "IncPrev started"
python3 main/IncPrev.py "${SLURM_ARRAY_TASK_ID}"
echo "IncPrev complete"

