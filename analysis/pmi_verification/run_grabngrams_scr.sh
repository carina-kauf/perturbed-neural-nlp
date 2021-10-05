#!/bin/bash
#
#SBATCH --job-name=grabngrams_scr
#SBATCH --output=bash_output/grabngrams_scr_%j.out
#SBATCH --error=bash_output/grabngrams_scr_%j.err
#SBATCH --nodes=1
#SBATCH --mem=15G
#SBATCH -t 06:00:00
#SBATCH -p cpl

timestamp() {
  date +"%T"
}

module load openmind/miniconda/4.0.5-python3
cd /om2/user/ckauf/perturbed-neural-nlp/analysis/pmi_verification

echo "Executing run grabngrams.py for condition ${1}"
timestamp
filename="bash_output/grabngrams_${1}_$(date '+%Y%m%d%T').txt"

python 1_GrabNGrams.py ${1} > $filename
echo 'Finished!'
timestamp

#RUN for cond in Scr1 Scr3 Scr5 Scr7 lowPMI lowPMI_random backward random random_poscontrolled random_withreplacement; do sbatch run_grabngrams_scr.sh $cond; done

#TEST
#RUN for cond in lowPMI_random; do sbatch run_grabngrams_scr.sh $cond; done
