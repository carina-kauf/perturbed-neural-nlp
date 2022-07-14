#!/bin/bash
#
#SBATCH --job-name=get_surp_distil
#SBATCH --output=get_surp_distil_%j.out
#SBATCH --error=get_surp_distil_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH -t 03:00:00
#SBATCH -p evlab

timestamp() {
  date +"%T"
}

echo 'Executing get_surp'
timestamp

filename="get_surp_distil_$(date '+%Y%m%d%T').txt"

cd $PWD
source /om2/user/ckauf/anaconda39/etc/profile.d/conda.sh
conda activate perturbed3.8

python get_surprisals.py > "$PWD/$filename"

timestamp

echo 'All complete!'
