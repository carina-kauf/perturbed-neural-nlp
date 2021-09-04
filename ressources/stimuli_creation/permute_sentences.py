import re
from random import randint

from Permutations import *


def generate_stable_pmi_conditions(F):
    
    output = []
    output_name = "_".join(F.split("_")[:-1])
    print(output_name)

    for l in open(F, 'r'):
        l = l.strip().lower()

        words = re.split(r'\s+', l)
        perm = make_permutation_with_distance(0, len(words))

        for level in range(8): #8

            perm = make_permutation_with_distance(level, len(words))
            assert kendall_distance(perm, range(len(words))) == level # Make sure we're not an unreachable perm

            outstring = str(level)+"\t"+" ".join([ words[i] for i in perm])
            print(outstring)
            output.append(outstring)
    
    with open(f"{output_name}_scrambled.txt", "w") as f:
        for item in output:
            f.write("%s\n" % item)
    
    return output

if __name__ == "__main__":
    stimuli_files = ["stim_243sentences_nopunct.txt", "stim_384sentences_nopunct.txt"]
    for F in stimuli_files:
        generate_stable_pmi_conditions(F)


	
