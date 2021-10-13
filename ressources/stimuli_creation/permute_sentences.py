import re
from random import randint

from Permutations import *


def generate_stable_pmi_conditions(F):
    
    output = []

    for l in open(F, 'r'):
        l = l.strip().lower()

        words = re.split(r'\s+', l)
        words = [elm.strip(".") for elm in words]
        perm = make_permutation_with_distance(0, len(words))

        for level in range(8): #8

            perm = make_permutation_with_distance(level, len(words))
            assert kendall_distance(perm, range(len(words))) == level # Make sure we're not an unreachable perm

            outstring = str(level)+"\t"+" ".join([ words[i] for i in perm])
            print(outstring)
            output.append(outstring)
    
    with open("Pereira2018_scrambled.txt", "w") as f:
        for item in output:
            f.write("%s\n" % item)
    
    return output

if __name__ == "__main__":
    stimuli_file = "Pereira2018_sentences.txt"
    generate_stable_pmi_conditions(stimuli_file)
