#!/bin/bash
#
#SBATCH --job-name=grabngrams
#SBATCH --output=grabngrams_%j.out
#SBATCH --error=grabngrams_%j.err
#SBATCH --nodes=1
#SBATCH --mem=15G
#SBATCH -t 20:00:00
timestamp() {
  date +"%T"
}

module load openmind/miniconda/4.0.5-python3
cd /om/user/ckauf/neural-nlp/neural_nlp/analyze/neural-scrambled/metric-validation/add_random_lowPMI_condition

echo 'Executing run grabngrams.py'
timestamp
filename="grabngrams_$(date '+%Y%m%d%T').txt"

python GrabNGrams.py > $filename
echo 'Finished!'
timestamp
