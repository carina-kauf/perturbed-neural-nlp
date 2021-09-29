from math import log
import pickle


print('*'*30, '\n LOADING STIMULI! \n','*'*30)

sentences = []
sample = []
with open('0_into_calculatePMI_10randompermutations.csv') as f:
    for line in f.readlines():
        entry = line.split(',')
        sentences.append(entry)
        sample.append(entry[2])


with open('1_ngrams_10rand.pkl', 'rb') as f:
    ngrams = pickle.load(f)


with open('1_nm1grams_10rand.pkl', 'rb') as f:
    nm1grams = pickle.load(f)

N = 356033418959 # US american english v2 google ngrams
nm1grams['_START_'] = float(sum([ ngrams[w] for w in list(ngrams.keys()) if w[0] == '_START_']))


def calc_prob(sentences, ngra=ngrams, nm1gra=nm1grams, ALPHA=0.1, lag=0):
    assert lag <= 2, 'impossible lag'
    results = []
    Z = len(ngrams.keys())*ALPHA + N
    for sent in sentences:
        string = sent[2]
        tokens = string.lower().split()
        mi = 0
        # No lag
        for t in range(0, len(tokens) - 1):
            joint_c = log(ngra[(tokens[t], tokens[t + 1])] + ngra[(tokens[t + 1], tokens[t])] + ALPHA)
            x_c = log(nm1gra[tokens[t]] + ALPHA * len(list(ngrams.keys())))
            y_c = log(nm1gra[tokens[t + 1]] + ALPHA * len(list(ngrams.keys())))
            pmi = max([0, (joint_c + log(Z) - x_c - y_c) / log(2)])
            mi += pmi
        # 1 word lag
        if lag >= 1:
            for t in range(0, len(tokens) - 2):
                joint_c = log(ngra[(tokens[t], tokens[t + 2])] + ngra[(tokens[t + 2], tokens[t])] + ALPHA)
                x_c = log(nm1gra[tokens[t]] + ALPHA * len(list(ngrams.keys())))
                y_c = log(nm1gra[tokens[t + 2]] + ALPHA * len(list(ngrams.keys())))
                pmi = max([0, (joint_c + log(Z) - x_c - y_c) / log(2)])
                mi += pmi
        # 2 word lag
        if lag >= 2:
            for t in range(0, len(tokens) - 3):
                joint_c = log(ngra[(tokens[t], tokens[t + 3])] + ngra[(tokens[t + 3], tokens[t])] + ALPHA)
                x_c = log(nm1gra[tokens[t]] + ALPHA * len(list(ngrams.keys())))
                y_c = log(nm1gra[tokens[t + 3]] + ALPHA * len(list(ngrams.keys())))
                pmi = max([0,(joint_c + log(Z) - x_c - y_c) / log(2)])
                mi += pmi
            mi = mi/len(tokens) #needs to be normalized still
        results.append(','.join([str(sent[0]), sent[1], sent[2].strip('\n'), str(mi)]))
    return results


print('*'*30, '\n CALCULATING LAG 0 PPMIS! \n','*'*30)

result = calc_prob(sentences, lag=0)
printstring = "\n".join(result)

with open('2_pPMI_lag0_10rand.csv', 'w') as f:
    f.write(printstring)

print('*'*30, '\n CALCULATING LAG 1 PPMIS! \n','*'*30)

result = calc_prob(sentences, lag=1)
printstring = "\n".join(result)

with open('2_pPMI_lag1_10rand.csv', 'w') as f:
    f.write(printstring)

print('*'*30, '\n CALCULATING LAG 2 PPMIS! \n','*'*30)

result = calc_prob(sentences, lag=2)
printstring = "\n".join(result)

with open('2_pPMI_lag2_10rand.csv', 'w') as f:
    f.write(printstring)
