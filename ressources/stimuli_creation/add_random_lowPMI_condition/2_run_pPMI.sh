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
cd /om2/user/ckauf/perturbed-neural-nlp/ressources/stimuli_creation/add_random_lowPMI_condition

filename="3_pPMI_$(date '+%Y%m%d%T').txt"

python pPMI.py > $filename
echo 'Finished!'
timestamp
