import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

import scipy.stats as stats

import statsmodels.api as sm
import statsmodels.stats.multicomp

from statistics import mean, stdev
from math import sqrt

import scikit_posthocs as sp

#import from own script
import plot_utils as pu

import warnings; warnings.simplefilter('ignore')

working_dir = "/om2/user/ckauf/.result_caching/neural_nlp.score"

def get_all_layers(model_identifier):
    """
    input: model_identifier of model of which we want to find the layers
    output: np.array of all unique layer identifiers, ordered by position
    """
    for ind,filename in enumerate(os.listdir(working_dir)):
        if "model=" + model_identifier in filename:
            file = os.path.join(working_dir,filename)
            with open(file, 'rb') as f:
                result = pickle.load(f)
            result = result['data']
            layer_list = np.unique(result.layer)
            #order double-digit layers at end of list
            double_digits = [elm for elm in layer_list if 'encoder.h.' in elm and len(elm.split('.h.')[-1]) > 1]
            layers = [e for e in layer_list if e not in double_digits] + double_digits
            return layers
            break

def get_best_layer(matrix):
    """
    input: result = out['data'].values matrix (e.g. for distilgpt2 a matrix of dimensions 7x2)
    output: index of the best-performing layer
    """
    max_score, error, best_layernr = 0, 0, 0
    for i in range(len(matrix)):
        if matrix[i][0] > max_score:
            max_score = matrix[i][0]
            error = matrix[i][1]
            best_layernr = i
    return best_layernr


def get_passage_identifier(filename):
    """
    get passage identifier to be used as key for the dictionary.
    fill the identifier with 0s for single-digit passage numbers
    """
    passage = filename.split("-")[-1].split(".")[0]
    number = passage.split("sentences")[-1]
    if len(number) == 1:
        passage_identifier = passage[:-1] + number.zfill(2)
    else:
        passage_identifier = passage
    return passage_identifier


def get_stats_df(model_identifier, emb_context="Passage", split_coord="Sentence", testonperturbed=False,
                randomnouns=False,length_control=False):
    """
    output: dataframe of subject-median'ed predicted scores of best layer
    """
    conditions, categories = [], []
    raw_scores = []
    subject_scores = []

    subdict = {}

    layers = get_all_layers(model_identifier)
    #print(layers)

    CAT2COND, COND2CAT = pu.get_conditions(testonperturbed=testonperturbed,
                                           randomnouns=randomnouns,
                                           length_control=length_control)

    condition_order = list(COND2CAT.keys())

    for filename in os.listdir(working_dir):
        if os.path.isdir(os.path.join(working_dir,filename)):
            continue

        if not f"emb_context={emb_context},split_coord={split_coord}" in filename:
            continue

        if not testonperturbed:
            if "teston:" in filename:
                continue
        else:
            if not "teston:" in filename:
                continue

        exclude_list = ["-control", "random-nouns"]
        if randomnouns:
            exclude_list = ["-control"]
        if length_control:
            include_list = ["original", "length-control", "random-wl"]
            
        if length_control:
            if all(x not in filename for x in include_list):
                continue
        else:
            if any(x in filename for x in exclude_list):
                continue

        model_name = filename.split(",")[1]
        bm = filename.split(",")[0]

        if bm == "benchmark=Pereira2018-encoding": #exclude old orignial bm in different format
            continue

        if "model=" + model_identifier == model_name:

            condition = bm.split("benchmark=Pereira2018-encoding-")[-1]

            #clean name
            condition = re.sub("perturb-","",condition)
            if not any(x in condition for x in ["1", "3", "5", "7"]):
                condition = re.sub("scrambled-","",condition)

            if testonperturbed:
                condition = re.sub("scrambled","scr",condition)

            file = os.path.join(working_dir,filename)
            with open(file, 'rb') as f:
                out = pickle.load(f)
            result_all = out['data']
            result = out['data'].values

            best_layernr = get_best_layer(result)
            best_layer = layers[best_layernr]

            raw_score = result_all.raw.raw
            best_layer_raw = raw_score[{"layer": [layer == best_layer for layer in raw_score["layer"]]}]
            raw_scores.append(best_layer_raw.values)

            subject_score = best_layer_raw.groupby('subject').median().values
            subject_scores.append(subject_score)

            # append to dict
            subdict[condition] = subject_score #keeping variance across subjects here

            conditions.append(condition)
            categories.append(COND2CAT[condition])

    # Transform subdict to statsmodel api form:
    x = subdict.copy()
    subdf = pd.DataFrame(x)
    subdf = subdf.melt(var_name='condition', value_name='values')

    subdf['condition'] = pd.Categorical(subdf['condition'], categories=condition_order, ordered=True)
    subdf = subdf.sort_values('condition')

    subdf['category'] = [COND2CAT[elm] for elm in list(subdf['condition'])]

    #clean names
    subdf['condition'] = subdf['condition'].str.replace("teston:","")
    subdf['condition'] = subdf['condition'].replace(
        {'sentenceshuffle_random': 'sent_random',
    'sentenceshuffle_passage': 'sent_passage',
    'sentenceshuffle_topic': 'sent_topic',
    'scr1': 'scrambled1',
    'scr3': 'scrambled3',
    'scr5': 'scrambled5',
    'scr7': 'scrambled7'}
    )

    return subdf


def cohens_d(v1, v2):
    return (mean(v1) - mean(v2)) / (sqrt((stdev(v1) ** 2 + stdev(v2) ** 2) / 2))


def assign_significance_labels(pvals):
    levels = []
    for pval in pvals:
        if pval < 0.001:
            levels.append("***")
        elif pval < 0.01:
            levels.append("**")
        elif pval < 0.05:
            levels.append("*")
        else:
            levels.append("n.s.")
    return levels


def barplot_annotate_brackets(num1, num2, data, center, height, updown="up",yerr=None, dh=.05, barh=.05, fs=None, maxasterix=5):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2] 

    ax_y0, ax_y1 = plt.gca().get_ylim()
    #dh *= (ax_y1 - ax_y0)/2
    barh *= (ax_y1 - ax_y0)/2
    
    if updown == "down":
        y = max(ly, ry) - dh
    else:
        y = max(ly, ry) + dh
    

    barx = [lx, lx, rx, rx]
    
    if updown == "down":
        bary = [y+barh, y, y, y+barh]
        mid = ((lx+rx)/2, y)
        
    else:
        bary = [y, y+barh, y+barh, y]
        mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black', linewidth=1)

    kwargs = dict(ha='center', va='bottom')
    
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)