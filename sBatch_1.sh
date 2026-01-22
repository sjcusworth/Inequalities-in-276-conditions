#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --time=24:0:0
#SBATCH --mem=500G
#SBATCH --qos=bbdefault
#SBATCH --mail-type=NONE

module purge; module load bluebear;
module load bear-apps/2024a

module load Python/3.12.3-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load matplotlib/3.9.2-gfbf-2024a
module load Seaborn/0.13.2-gfbf-2024a
module load tqdm/4.66.5-GCCcore-13.3.0
module load zstd/1.5.6-GCCcore-13.3.0
module load Arrow/17.0.0-gfbf-2024a
module load polars/1.31.0-gfbf-2024a
module load PyYAML/6.0.2-GCCcore-13.3.0
module load plotly.py/5.24.1-GCCcore-13.3.0

echo "Modules Loaded";

echo "Preprocessing started"
python3 main/preprocessing.py
echo "Preprocessing complete"
