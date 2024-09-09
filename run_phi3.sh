#!/bin/bash
#SBATCH --partition="gpu-ms,gpu-troja"
#SBATCH -J phi3
#SBATCH -o phi3_%a.out
#SBATCH -e phi3_%a.err
#SBATCH -D .  
#SBATCH -N 1
#SBATCH --mem=45G
#SBATCH --gpus=6
#SBATCH -c 6
#SBATCH --array=1-132%6 # 120 jobs, 5 at the time
set -e

source .venv/bin/activate

which python
date +"%Y-%m-%d %H:%M:%S"

./phi3.py
