#!/bin/bash
#
#SBATCH --job-name=pPMI
#SBATCH --output=pPMI_%j.out
#SBATCH --error=pPMI_%j.err
#SBATCH --nodes=1
#SBATCH --mem=15G
#SBATCH -t 02:00:00
timestamp() {
  date +"%T"
}
echo 'Executing run pPMI.py'
timestamp

module load openmind/miniconda/4.0.5-python3
cd /om/user/ckauf/neural-nlp/neural_nlp/analyze/neural-scrambled/metric-validation/add_random_lowPMI_condition

filename="pPMI_$(date '+%Y%m%d%T').txt"

python pPMI.py > $filename
echo 'Finished!'
timestamp
