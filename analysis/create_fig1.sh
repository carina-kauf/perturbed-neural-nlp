#!/bin/bash

#SBATCH --job-name=create_match2brain_PlotAndStats
#SBATCH --mem=20G
#SBATCH -t 00:30:00
#SBATCH -p evlab

cd /om2/user/ckauf/perturbed-neural-nlp/analysis
source /om2/user/ckauf/anaconda39/etc/profile.d/conda.sh
conda activate perturbed3.8

python create_match2brain_PlotAndStats.py --model_identifier gpt2-xl