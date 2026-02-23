#!/bin/bash
#SBATCH --job-name=baseline_KGMgT_train
#SBATCH -A mrasynthesis
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --gpus=2
#SBATCH -o logs/%x_%j.out

module purge
module load gcc/13.4.0
module load cuda/13.0.0
module load conda

conda activate KGMgT_old

python -m kg_net.train -opt "options/train/train_restoration.yml"