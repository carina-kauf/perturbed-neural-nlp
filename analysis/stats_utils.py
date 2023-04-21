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
    
    
import scipy
import statsmodels

# Benjamini/Hochberg corrected (NOL SUBMISSION)
# Bonferroni corrected (REVISION)
def get_ttest_results(model_identifier, emb_context="Passage", split_coord="Sentence",
                testonperturbed=False, category=None, randomnouns=False, length_control=False):

    stats_df = pu.get_best_scores_df(model_identifier=model_identifier,
                                              emb_context=emb_context,
                                              split_coord=split_coord,
                                              testonperturbed=testonperturbed,
                                              randomnouns=randomnouns,
                                              length_control=length_control,
                                              nr_of_splits=5,
                                              which_df='stats') # output stats df
    #stats_df.to_csv(f'{savedir}/approach={approach}_individual_scores.csv')

    pvals2original, pvals2random = [], []
    ttest2original, ttest2random = [], []
    cohensd2original, cohensd2random = [], []
    conds = []
    
    CAT2COND, COND2CAT = pu.get_conditions(testonperturbed=False,
                                                   randomnouns = randomnouns,
                                                   length_control=length_control)
    for cond in CAT2COND[category]:
        subdf = stats_df.loc[stats_df['category_group'] == category]
        if length_control and "random-wl" in cond:
            continue
        
        #adjust names for consistency
        if cond == 'sentenceshuffle_random':
            cond = 'sent_random'
        elif cond == 'sentenceshuffle_passage':
            cond = 'sent_passage'
        elif cond == 'sentenceshuffle_topic':
            cond = 'sent_topic'

        #get subject scores
        original_scores = list(subdf[subdf['condition'] == 'original']["values"])
        cond_scores = list(subdf[subdf['condition'] == cond]["values"])
        random_scores = list(subdf[subdf['condition'] == 'random-wl']["values"])
        
        #get ttest
        ttest2orig, pval2orig = scipy.stats.ttest_rel(original_scores,cond_scores)#, alternative="greater")
        ttest2rand, pval2rand = scipy.stats.ttest_rel(random_scores,cond_scores)#, alternative="less")
        
        # get effect size
        cohensd2orig = cohens_d(original_scores, cond_scores)
        cohensd2rand = cohens_d(random_scores, cond_scores)
        
        conds.append(cond)
        pvals2original.append(pval2orig)
        ttest2original.append(ttest2orig)
        pvals2random.append(pval2rand)
        ttest2random.append(ttest2rand)
        cohensd2original.append(cohensd2orig)
        cohensd2random.append(cohensd2rand)
    
    #
    _, adjusted_pvals2original, _, _ = statsmodels.stats.multitest.multipletests(pvals2original, method='bonferroni')
    _, adjusted_pvals2random, _, _ = statsmodels.stats.multitest.multipletests(pvals2random, method='bonferroni')
    
    #assign significance levels
    significance2original = assign_significance_labels(adjusted_pvals2original)
    significance2random = assign_significance_labels(adjusted_pvals2random)
    
    
    stats_df = pd.DataFrame({
        "condition": conds,
        "ttest2original" : ttest2original,
        "ttest2random" : ttest2random,
        "adjusted_pvals2original" : adjusted_pvals2original,
        "adjusted_pvals2random" : adjusted_pvals2random,
        "cohensd2original" : cohensd2original,
        "cohensd2random" : cohensd2random,
        "significance2original" : significance2original,
        "significance2random" : significance2random,
        "pvals2original" : pvals2original,
        "pvals2random" : pvals2random
    })
    
    return stats_df


import pandas as pd
from scipy.stats import ttest_rel
import statsmodels
import pandas as pd

def posthoc_ttest_dep(df, val_col=None, group_col=None, p_adjust=None):
    """
    Performs dependent t-tests between all pairs of columns in a pandas DataFrame,
    stratified by a grouping column.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame containing the values to be tested.
    val_col : str, optional
        The name of the column containing the values to be tested. If None, assumes the values are in all columns.
        Default is None.
    group_col : str, optional
        The name of the column containing the group labels. If None, assumes all data are from the same group.
        Default is None.

    Returns:
    --------
    pandas DataFrame
        A DataFrame containing the p-values of the dependent t-tests between all pairs of columns,
        stratified by the grouping column.
    """

    # Initialize an empty DataFrame to store p-values
    conditions = data[group_col].unique()
    k = conditions.size
    vs = np.zeros((k, k), dtype=float)
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)

    # Loop over all pairs of conditions
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            if i >= j:
                continue
            cond1_scores = list(data[data['condition'] == cond1][val_col])
            cond2_scores = list(data[data['condition'] == cond2][val_col])
            # Perform a dependent t-test between the two columns, stratified by the grouping column
            t, p = ttest_rel(cond1_scores, cond2_scores)
            # Store the minimum p-value for the pair of columns
            vs[i, j] = p
            vs[j, i] = p
    
    # Apply Bonferroni correction to unique comparisons
    if p_adjust:
        vs[tri_upper] = statsmodels.stats.multitest.multipletests(vs[tri_upper], method=p_adjust)[1]
    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)
    
    return pd.DataFrame(vs, index=conditions, columns=conditions)
