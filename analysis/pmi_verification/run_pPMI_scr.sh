#!/bin/bash
#
#SBATCH --job-name=pPMI_scr
#SBATCH --output=bash_output/pPMI_scr_%j.out
#SBATCH --error=bash_output/pPMI_scr_%j.err
#SBATCH --nodes=1
#SBATCH --mem=15G
#SBATCH -t 00:10:00
#SBATCH -p cpl

timestamp() {
  date +"%T"
}

module load openmind/miniconda/4.0.5-python3
cd /om2/user/ckauf/perturbed-neural-nlp/analysis/pmi_verification

echo "Executing run pPMI.py for condition: ${1}"

timestamp
filename="bash_output/pPMI_${1}_$(date '+%Y%m%d%T').txt"

python 2_pPMI.py ${1} > $filename
echo 'Finished!'
timestamp

#RUN for cond in Original Scr1 Scr3 Scr5 Scr7 lowPMI lowPMI_random backward random random_poscontrolled random_withreplacement; do sbatch run_pPMI_scr.sh $cond; done

#TEST
#RUN for cond in lowPMI_random backward; do sbatch run_pPMI_scr.sh $cond; done