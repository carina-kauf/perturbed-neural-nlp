from zs import ZS
import pickle
import argparse
import os
from os.path import abspath

parser = argparse.ArgumentParser(description='get ngrams')
parser.add_argument('condition',type=str,nargs="+",help="should be one of: Original Scr1 Scr3 Scr5 Scr7 lowPMI lowPMI_random backward random random_poscontrolled random_withreplacement")
args = parser.parse_args()

print('*'*30, '\n LOADING STIMULI! \n','*'*30)

cond = args.condition[0]
print(cond)
savedir = abspath('./stimuli_csvs')
fname = "{}/stimuli_{}.csv".format(savedir,cond)
print(fname)

sentences = []
sample = []
with open(fname) as f:
    for line in f.readlines():
        entry = line.split(',')
        sentences.append(entry)
        sample.append(entry[2])
print("File opened!")

google1 = ZS('/om/data/public/corpora/google-books-v2/eng-us-all/google-books-eng-us-all-20120701-1gram.zs')
google2 = ZS('/om/data/public/corpora/google-books-v2/eng-us-all/google-books-eng-us-all-20120701-2gram.zs')

#  break sentences into strings
def populate(sentences):
    ngra = dict()
    nm1gra = dict()
    for sentence in sentences:
        print('S', sentence)
        tokens = sentence.lower().split()
        tokens = ['_START_'] + tokens + ['_END_']
        for t in range(0, len(tokens) - 1):
            ngra[(tokens[t], tokens[t + 1])] = 0
            print(0, (tokens[t], tokens[t + 1]))
            nm1gra[tokens[t]] = 0
        for t in range(0, len(tokens) - 2):
            ngra[(tokens[t], tokens[t + 2])] = 0
            print(1, (tokens[t], tokens[t + 2]))
        for t in range(0, len(tokens) - 3):
            ngra[(tokens[t], tokens[t + 3])] = 0
            print(2, (tokens[t], tokens[t + 3]))
        nm1gra[tokens[len(tokens) - 1]] = 0
    for t1, t2 in list(ngra.keys()):
        ngra[(t2, t1)] = 0
    return ngra, nm1gra

print('*'*30, '\n GETTING NGRAMS! \n','*'*30)
ngrams, nm1grams = populate(sample)

#  fetch ngram and n-1gram
def fetch(ngra, z=google2, zm1=google1):
    ngram_c = 0
    ngram_str = " ".join(ngra).encode()
    for record in z.search(prefix=ngram_str):
        record = record.decode("utf-8")
        entry = record.split()
        if entry[1] == ngra[1]:
            ngram_c += int(entry[3])
    if nm1grams[ngra[0]] > 0:
        nm1gram_c = nm1grams[ngra[0]]
    else:
        nm1gram_c = 0
        for record in zm1.search(prefix=ngra[0].encode()):
            record = record.decode("utf-8")
            entry = record.split()
            if entry[0] == ngra[0]:
                nm1gram_c += int(entry[2])
    return ngram_c, nm1gram_c

print('*'*30, '\n FETCHING NGRAMS! \n','*'*30)

surprisals = dict()
for ngram in list(ngrams.keys()):
    print(ngram)
    ngrams[ngram], nm1grams[ngram[0]] = fetch(ngram)

print('*'*30, '\n SAVING DICTIONARIES! \n','*'*30)

with open("{}/1_ngrams_{}.pkl".format(savedir, cond), 'wb') as f:
    pickle.dump(ngrams, f)

with open("{}/1_nm1grams_{}.pkl".format(savedir, cond), 'wb') as f:
    pickle.dump(nm1grams, f)