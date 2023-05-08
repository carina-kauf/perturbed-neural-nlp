import pandas as pd
import numpy as np
import os
import os.path
import pickle
from tqdm import tqdm
import argparse

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import math
            

def score(model, tokenizer, sentence):
    tokenize_input = tokenizer.tokenize(tokenizer.eos_token + sentence + tokenizer.eos_token)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss = model(tensor_input, labels=tensor_input)[0]
    
    sent_score = loss.item() * len(tokenize_input)
    sent_scoreAvgTokenScore = loss.item()
    sent_scoreAvgWordScore = loss.item() * len(tokenize_input) / len(sentence.split())
    return sent_score, sent_scoreAvgTokenScore, sent_scoreAvgWordScore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='gpt2-xl')
    args = parser.parse_args()
    
    assert "gpt2" in args.version, "Invalid version specified! Currently only implemented for GPT models"

    out_dir = f'files/'
    os.makedirs(out_dir, exist_ok=True)

    print("Loading model", flush=True)
    # Load pre-trained model (weights)
    model = GPT2LMHeadModel.from_pretrained(args.version)
    model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained(args.version)
    

    # define conditions
    conditions = [('Original', 'original'),
                    ('Scr1', 'scrambled'),
                    ('Scr3', 'scrambled'),
                    ('Scr5', 'scrambled'),
                    ('Scr7', 'scrambled'),
                    ('backward', 'scrambled'),
                    ('lowPMI', 'scrambled'),
                    ('lowPMI_random', 'scrambled'),
                    ('random', 'scrambled'),
                    ('nouns', 'perturbed'),
                    ('randomnouns', 'perturbed'),
                    ('nounsverbs', 'perturbed'),
                    ('nounsverbsadj', 'perturbed'),
                    ('contentwords', 'perturbed'),
                    ('functionwords', 'perturbed'),
                        #
                    ('sentenceshuffle-random', 'perturbed'),
                    ('sentenceshuffle-withinpassage', 'perturbed'),
                    ('sentenceshuffle-withintopic', 'perturbed')
    ]

    # Load pickles and get sentences for all conditions
    working_dir = "/om2/user/ckauf/perturbed-neural-nlp/ressources/scrambled_stimuli_dfs/"
    COND2SENT = {}
    for ind, (cond, scr_perturb) in enumerate(conditions):
        key = f"{scr_perturb}+{cond}"
        for filename in os.listdir(working_dir):
            if filename == f'stimuli_{cond}.pkl':
                condition = "_".join(filename.split(".pkl")[0].split("_")[1:])
                if condition == "random":
                    condition = "random-wl"

                with open(os.path.join(working_dir,filename), 'rb') as f:
                    df = pickle.load(f)
                sentences = list(df["sentence"])

                COND2SENT[key] = sentences

    # Get scores per condition and write to file
    for key, sentences in COND2SENT.items():
        out_name = os.path.join(f'{args.version}.{key}.sentence_surp.txt')
        out_nameTokAverage = out_name.rstrip(".txt") + ".average.txt"
        out_nameWordAverage = out_name.rstrip(".txt") + ".average_sentlength.txt"

        print(f"Getting {args.version} probabilities for condition {key}", flush=True)
        scores = []
        AvgTokenScores = []
        AvgWordScores = []
        
        for sent in tqdm(sentences):
            sent_score, sent_scoreAvgTokenScore, sent_scoreAvgWordScore = score(model, tokenizer, sent)
            scores.append(sent_score)
            AvgTokenScores.append(sent_scoreAvgTokenScore)
            AvgWordScores.append(sent_scoreAvgWordScore)

        with open(out_name, "w") as fout:
            fout.write(f'Index\tSentence\tSurprisal\tPPL\n') #add header
            for i, sent, sent_score in zip(range(len(sentences)), sentences, scores):
                fout.write(f'{i}\t{sent}\t{sent_score}\t{math.exp(sent_score)}\n')
                
        with open(out_nameTokAverage, "w") as fout:
            fout.write(f'Index\tSentence\tSurprisal\tPPL\n') #add header
            for i, sent, sent_score in zip(range(len(sentences)), sentences, AvgTokenScores):
                fout.write(f'{i}\t{sent}\t{sent_score}\t{math.exp(sent_score)}\n')
                
        with open(out_nameWordAverage, "w") as fout:
            fout.write(f'Index\tSentence\tSurprisal\tPPL\n') #add header
            for i, sent, sent_score in zip(range(len(sentences)), sentences, AvgWordScores):
                fout.write(f'{i}\t{sent}\t{sent_score}\t{math.exp(sent_score)}\n')
                
if __name__ == "__main__":
    main()
