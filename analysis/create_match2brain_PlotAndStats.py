# general
import os
import numpy as np
import pandas as pd
import json
import argparse
# plotting
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# stats
import scipy
import scikit_posthocs as sp
import statsmodels
import math
# functions from other scripts
import plot_utils
import stats_utils

out_dir = os.path.abspath(os.path.join(os.getcwd(), 'results_paper_revision/match2brain'))
add_stats = True


##########################
### BRAIN PREDICTIVITY DF
##########################
def get_plot_df(approach, model_identifier, emb_context, split_coord, testonperturbed, randomnouns):
    """
    get sample stimuli for each condition (for figure legend)
    get brain predictivity scores & MAD errors for the best layer for each condition
    """
    stimuli_df = plot_utils.get_sample_stimuli(randomnouns=randomnouns)
    scores_df = plot_utils.get_best_scores_df(model_identifier=model_identifier,
                                              emb_context=emb_context,
                                              split_coord=split_coord,
                                              testonperturbed=testonperturbed,
                                              randomnouns=randomnouns)
    # merge dataframes on condition
    full_df = scores_df.merge(stimuli_df, on='condition', how='inner')
    # rename conditions according to new names:
    full_df['condition'] = full_df['condition'].map(plot_utils.COND2LABEL)
    # create legend label column
    full_df["labelname"] = [': '.join(i) for i in zip(full_df["condition"].map(str), full_df["stimulus"])]

    full_df.to_csv(f"{out_dir}/{approach}_brainscores.csv", index=False)

    return full_df


##########################
### STATS
##########################
def get_ttest_results(approach, model_identifier, emb_context, split_coord, testonperturbed, randomnouns):
    """
    get pairwise, independent two-sided ttest results for each manipulation class to i) Original and ii) RandWordList
    Pvals are corrected for multiple comparison using the Benjamini-Hochberg procedure
    Also output cohens's d

    returns full stats df for all conditions
    """
    category_stats_frames = []

    subdf = stats_utils.get_stats_df(model_identifier=model_identifier, emb_context=emb_context,
                                     split_coord=split_coord, testonperturbed=testonperturbed,
                                     randomnouns=randomnouns)

    CAT2COND, COND2CAT = plot_utils.get_conditions()  # always use default settings because "teston:" is stripped in different part of script

    for cat in CAT2COND.keys():
        if cat in ["original", "random-wl"]:
            continue

        pvals2original, pvals2random = [], []
        ttest2original, ttest2random = [], []
        cohensd2original, cohensd2random = [], []
        conds = []

        for cond in CAT2COND[cat]:
            # adjust names for consistency
            if cond == 'sentenceshuffle_random':
                cond = 'sent_random'
            elif cond == 'sentenceshuffle_passage':
                cond = 'sent_passage'
            elif cond == 'sentenceshuffle_topic':
                cond = 'sent_topic'

            # get subject scores
            original_scores = list(subdf[subdf['condition'] == 'original']["values"])
            cond_scores = list(subdf[subdf['condition'] == cond]["values"])
            random_scores = list(subdf[subdf['condition'] == 'random-wl']["values"])
    
            # get ttest
            ttest2orig, pval2orig = scipy.stats.ttest_rel(original_scores, cond_scores)
            ttest2rand, pval2rand = scipy.stats.ttest_rel(random_scores, cond_scores)

            # get effect size
            cohensd2orig = stats_utils.cohens_d(original_scores, cond_scores)
            cohensd2rand = stats_utils.cohens_d(random_scores, cond_scores)

            conds.append(cond)
            pvals2original.append(pval2orig)
            ttest2original.append(ttest2orig)
            pvals2random.append(pval2rand)
            ttest2random.append(ttest2rand)
            cohensd2original.append(cohensd2orig)
            cohensd2random.append(cohensd2rand)

        # correct for multiple comparisons
        # statsmodels.stats.multitest.fdrcorrection(pvals) = statsmodels.stats.multitest.multipletests(pvals, method='fdr_bh')
        # first output is list of Booleans indicating whether to reject null hypothesis or not
#         _, adjusted_pvals2original = statsmodels.stats.multitest.fdrcorrection(pvals2original)
#         _, adjusted_pvals2random = statsmodels.stats.multitest.fdrcorrection(pvals2random)
        _, adjusted_pvals2original, _, _ = statsmodels.stats.multitest.multipletests(pvals2original, method='bonferroni')
        _, adjusted_pvals2random, _, _ = statsmodels.stats.multitest.multipletests(pvals2random, method='bonferroni')

        # assign significance levels
        significance2original = stats_utils.assign_significance_labels(adjusted_pvals2original)
        significance2random = stats_utils.assign_significance_labels(adjusted_pvals2random)

        stats_df = pd.DataFrame({
            "condition": conds,
            "ttest2original": ttest2original,
            "ttest2random": ttest2random,
            "adjusted_pvals2original": adjusted_pvals2original,
            "adjusted_pvals2random": adjusted_pvals2random,
            "cohensd2original": cohensd2original,
            "cohensd2random": cohensd2random,
            "significance2original": significance2original,
            "significance2random": significance2random,
            "pvals2original": pvals2original,
            "pvals2random": pvals2random
        })

        category_stats_frames.append(stats_df)
    full_stats_df = pd.concat(category_stats_frames)

    # rename conditions according to new names:
    full_stats_df['condition'] = full_stats_df['condition'].map(plot_utils.COND2LABEL)

    # define stats dataframe to be saved to file
#     to_save = full_stats_df[
#         ['condition', 'ttest2original', 'adjusted_pvals2original', 'cohensd2original', 'ttest2random',
#          'adjusted_pvals2random', 'cohensd2random']].T
#     to_save, to_save.columns = to_save[1:], to_save.iloc[0]
    to_save = full_stats_df[
        ['condition', 'ttest2original', 'adjusted_pvals2original', 'cohensd2original', 'ttest2random',
         'adjusted_pvals2random', 'cohensd2random']]
    to_save.to_csv(f"{out_dir}/{approach}_tteststats2orig+rand.csv", index=False)

    if randomnouns:
        randomnouns_scores = list(subdf[subdf['condition'] == 'random-nouns']["values"])
        functionword_scores = list(subdf[subdf['condition'] == 'functionwords']["values"])
        # get ttest
        ttest, pval = scipy.stats.ttest_rel(randomnouns_scores, functionword_scores)
        # get effect size
        cohensd = stats_utils.cohens_d(randomnouns_scores, functionword_scores)
        print(f"RANDOM NOUNS vs. FUNCTIONWORDS \nttest: {ttest} \npval: {pval} \ncohen's d: {cohensd}", flush=True)

    return full_stats_df


##########################
### PLOTTING
##########################

def get_colors(randomnouns):
    # define colors
    CAT2COLOR = {
        "original": "dimgray",
        #
        "word-order": sns.cubehelix_palette(7, start=.2, rot=-.25, dark=0.2, light=.9, reverse=True),
        "information-loss": sns.cubehelix_palette(5, start=2, rot=0, dark=0.2, light=.85, reverse=True),
        # cut off as last gradient color is similar across colors
        "semantic-distance": sns.light_palette("maroon", 5, reverse=True)[:4], #5 here s.th. the last bar is visible!
        #
        "control": "lightgray"
    }

    if randomnouns:
        CAT2COLOR["information-loss"] = sns.cubehelix_palette(6, start=2, rot=0, dark=0.2, light=.85, reverse=True)

    return CAT2COLOR

def adjust_name(cond):
    #adjust names for consistency
    if cond == 'sentenceshuffle_random':
        cond = 'sent_random'
    elif cond == 'sentenceshuffle_passage':
        cond = 'sent_passage'
    elif cond == 'sentenceshuffle_topic':
        cond = 'sent_topic'
    return cond


def main_barplot(config, approach, model_identifier, full_df, full_stats_df, randomnouns, vertical=False):
    """
    vertical determines if subplots are stacked vertically. Default is horizontal alignment.
    """
    CAT2COLOR = get_colors(randomnouns)
    # define global figure settings
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'

    custom_params = {"axes.spines.right": False,
                     "axes.spines.top": False,
                     'ytick.left': True,
                     'xtick.bottom': True,
                     'grid.linestyle': ""  # gets rid of horizontal lines
                     }
    sns.set_theme(font_scale=1.4, style="white", rc=custom_params)

    categories = ["word-order", "information-loss", "semantic-distance"]
    if randomnouns:
        categories = ["information-loss"]

    num_bars = [len(full_df[full_df["category"] == cat]) for cat in categories]
    max_bars = max(num_bars)
    props = [1 + 0.2 * x for x in num_bars]

    if vertical:
        nrows = len(categories)
        ncols = 1
        figsize = (15, 7 * nrows)
        height_ratios = props
    else:
        nrows = 1
        ncols = len(categories)
        if not add_stats:
            figsize = (8 * ncols, 11)
        else:
            figsize = (7 * ncols, 15)
        height_ratios = None

    fig2 = plt.figure(constrained_layout=True, figsize=figsize, facecolor='white')
    spec2 = GridSpec(ncols=ncols, nrows=nrows, figure=fig2, height_ratios=height_ratios)
    f2_ax = []

    to_add = 0

    for ind, current_category in enumerate(categories):
        # set up subplot
        if vertical:
            f2_ax.append(fig2.add_subplot(spec2[ind, 0]))
        else:
            if ind == 0:
                f2_ax.append(fig2.add_subplot(spec2[0, ind]))
            else:
                f2_ax.append(fig2.add_subplot(spec2[0, ind], sharey=f2_ax[0]))

        categories = ["original", current_category, "control"]
        plot_df = full_df[full_df["category"].isin(categories)]

        colors = [CAT2COLOR["original"]] + CAT2COLOR[current_category] + [CAT2COLOR["control"]]

        x_pos = np.arange(len(plot_df))
        scores = list(plot_df['score'])
        errors = list(plot_df['error'])
        conditions = list(plot_df['condition'])

        stimuli = [list(plot_df.loc[plot_df["category"] == cat]["labelname"]) for cat in categories]
        stimuli = plot_utils.flatten_list(stimuli)
        from textwrap import fill
        stimuli = [fill(l, 75) for l in stimuli]

        # add empty slots for same bar width
        if len(x_pos) != max_bars + 2:  # +2 because of original and random-wl
            to_add = max_bars - num_bars[ind]
            x_pos = np.arange(len(plot_df) + to_add)
            multiplier1 = math.floor(to_add / 2)
            multiplier2 = math.ceil(to_add / 2)

            insert_at1 = 1
            insert_at2 = -1

            insert_elements1 = [np.nan] * multiplier1
            insert_elements2 = [np.nan] * multiplier2
            #
            scores[insert_at1:insert_at1] = insert_elements1
            scores[insert_at2:insert_at2] = insert_elements2
            #
            errors[insert_at1:insert_at1] = insert_elements1
            errors[insert_at2:insert_at2] = insert_elements2

            insert_elements1 = [""] * multiplier1
            insert_elements2 = [""] * multiplier2
            #
            conditions[insert_at1:insert_at1] = insert_elements1
            conditions[insert_at2:insert_at2] = insert_elements2
            #
            stimuli[insert_at1:insert_at1] = insert_elements1
            stimuli[insert_at2:insert_at2] = insert_elements2

            insert_elements1 = ["white"] * multiplier1
            insert_elements2 = ["white"] * multiplier2

            colors[insert_at1:insert_at1] = insert_elements1
            colors[insert_at2:insert_at2] = insert_elements2
            
            
        if vertical:
    
            f2_ax[-1].bar(x_pos, scores,
            yerr=errors,
            align='center',
            alpha=0.9, #color intensity
            ecolor='black',
            capsize=5, #error-bar width
            color=colors,
            error_kw={'elinewidth': 2})
        
            # add scatter points for individual subject data
            emb_context = config[approach]["emb_context"]
            split_coord = config[approach]["split_coord"]
            testonperturbed = config[approach]["testonperturbed"]
            individual_subject_data_df = stats_utils.get_stats_df(model_identifier=model_identifier,
                                                 emb_context=emb_context,
                                                 split_coord=split_coord,
                                                 testonperturbed=testonperturbed)
            individual_subject_data_df['condition'] = individual_subject_data_df['condition'].map(plot_utils.COND2LABEL)
            
            
            curr_subj_df = individual_subject_data_df.loc[individual_subject_data_df['category'] == current_category]
            indiv_original_scores = list(individual_subject_data_df.loc[individual_subject_data_df['condition'] == 'Original']['values'])
            indiv_randwordlist_scores = list(individual_subject_data_df.loc[individual_subject_data_df['condition'] == 'RandWordList']['values'])
            indiv_category_scores = [list(curr_subj_df.loc[curr_subj_df['condition'] == cond]['values']) for cond in conditions
                                    if cond not in ['Original', 'RandWordList']]
            # merge score lists
            indiv_category_scores.insert(0, indiv_original_scores)
            indiv_category_scores.append(indiv_randwordlist_scores)
            # add scatters at correct positions
            scatter_xpos = []
            scatter_ys = []
            scatter_colors = []
            positions = [i for i, score in enumerate(scores) if not np.isnan(score)]
            for elm in positions:
                for s in indiv_category_scores[elm]:
                    scatter_xpos.append(elm)
                    scatter_ys.append(s)
                    scatter_colors.append(colors[elm])

            # generate random jitter values within (-0.05, 0.05)
            jitter = np.random.uniform(low=-0.05, high=0.05, size=np.shape(scatter_xpos))
            # add the jitter to x position array
            scatter_xpos += jitter
            # scatter plot
            # f2_ax[-1].scatter(scatter_xpos, scatter_ys, color=scatter_colors, s=20, linewidth=0.2, edgecolors='black')
            f2_ax[-1].scatter(scatter_xpos, scatter_ys, color='dimgray', s=20, linewidth=0.2, edgecolors='black')

        else:
            f2_ax[-1].bar(x_pos, scores,
                          yerr=errors,
                          align='center',
                          alpha=0.9,  # color intensity
                          ecolor='black',
                          capsize=5,  # error-bar width
                          color=colors)

        CAT2COND, COND2CAT = plot_utils.get_conditions()

        # annotate stats 2 original
        for ind_c, cond in enumerate(CAT2COND[current_category]):
            # adjust names for consistency
            cond = adjust_name(cond)
            cond = plot_utils.COND2LABEL[cond]

            if cond in ["Original", "RandWordList"]:
                continue

            idx_random = conditions.index("RandWordList") - to_add

            positions = [i for i, score in enumerate(scores) if not np.isnan(score)]

            if add_stats:
                # add stats annotations for comparison with original score
                heights = [scores[0]] * len(positions)
                # add height offset to annotations
                height_offset = [0.05 * i for i in range(len(heights))]
                heights = [sum(x) for x in zip(heights, height_offset)]
                label = full_stats_df.loc[full_stats_df["condition"] == cond]["significance2original"].item()
                stats_utils.barplot_annotate_brackets(0, ind_c + 1, data=label, center=positions, height=heights, fs=10)

                # add stats annotations for comparison with random-wl score
                heights = [0] * len(positions)
                # add height offset to annotations
                height_offset = [-0.05 * i for i in range(len(heights))]
                heights = [sum(x) for x in zip(heights, height_offset)]
                label = full_stats_df.loc[full_stats_df["condition"] == cond]["significance2random"].item()
                stats_utils.barplot_annotate_brackets(ind_c + 1, idx_random, data=label, center=positions, height=heights,
                                                      updown="down", fs=10)

        # add horizontal lines for original and random-wl
        orig_score = full_df[full_df["category"] == "original"]["score"].item()
        random_score = full_df[full_df["category"] == "control"]["score"].item()
        f2_ax[-1].axhline(y=orig_score, color=CAT2COLOR["original"], linestyle=':', dashes=(5, 3), linewidth=1)
        f2_ax[-1].axhline(y=random_score, color=CAT2COLOR["control"], linestyle=":", dashes=(5, 3), linewidth=1)

        f2_ax[-1].axhline(y=0, color="black")

        # add legend (sample stimuli)
        # map names to colors
        cmap = dict(zip(conditions, colors))
        cmap = {k: v for k, v in cmap.items() if k}
        # create the rectangles for the legend
        from matplotlib.patches import Patch
        patches = [Patch(color=v, label=k, alpha=0.8) for k, v in cmap.items()]
        # remove empty strings from labels
        stimuli = [x for x in stimuli if x]

        # add the legend
        if vertical:
            f2_ax[-1].legend(title=f'{current_category} manipulations', labels=stimuli, handles=patches,
                             bbox_to_anchor=(1.05, 0.5), loc='center left', title_fontsize=15, prop={'size': 13.5})
        else:
            f2_ax[-1].legend(title='', labels=stimuli, handles=patches,
                             loc='upper center', bbox_to_anchor=(0.5, -0.7),
                             title_fontsize=15, prop={'size': 12})             

        # TICKS
        ## to get current ones: f2_ax[-1].get_yticks()
        if ind == 0:
            yticks = [0, 0.1, 0.2, 0.3, 0.4] 
            f2_ax[-1].set_yticks(yticks)
        else:
            if vertical:
                f2_ax[-1].set_yticks([])
        # set xticks
        f2_ax[-1].set_xticks(positions)
        xticknames = ["\nFrom".join(elm.split("From")) for elm in conditions]
        f2_ax[-1].set_xticklabels([x for x in xticknames if x], rotation=60, ha="right", rotation_mode="anchor")

        f2_ax[-1].set_ylabel('Brain predictivity')

        if not vertical:
            f2_ax[-1].set_title(f"{current_category} manipulations", pad=0.5)
            
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{approach}.svg', dpi=180, bbox_inches="tight")
    plt.savefig(f'{out_dir}/{approach}.png', dpi=180, bbox_inches="tight")

    if vertical:
        plt.savefig(f'{out_dir}/figure2.svg', dpi=180, bbox_inches="tight")
        plt.savefig(f'{out_dir}/figure2.png', dpi=180, bbox_inches="tight")

    plt.show()


def within_category_stats_and_plots(approach, model_identifier, emb_context, split_coord, testonperturbed, randomnouns):
    
    stats_df = stats_utils.get_stats_df(model_identifier=model_identifier, emb_context=emb_context,
                                        split_coord=split_coord, testonperturbed=testonperturbed,
                                        randomnouns=randomnouns)
    
    CAT2COND, _ = plot_utils.get_conditions()
    categories = [x for x in CAT2COND.keys() if x not in ['original', 'control']]
    
    #get dataframes with ttest values
    for ind, current_category in enumerate(categories):

        already_tested = [] #list to keep track of which combos were already tested (for correct number of multiple comparisons)
        # NOTE: only done for half of the comparisons, i.e., upper triangle of comparison matrix
        # https://github.com/maximtrp/scikit-posthocs/blob/master/scikit_posthocs/_posthocs.py#L1653

        ttests, pvals = [], []
        compare1, compare2 = [], []
        cohens_d = []

        for cond1 in CAT2COND[current_category]:
            cond1 = adjust_name(cond1)
            cond1_scores = list(stats_df[stats_df['condition'] == cond1]["values"])

            for cond2 in CAT2COND[current_category]:
                cond2 = adjust_name(cond2)

                if cond1 == cond2:
                    continue

                if set([cond1, cond2]) in already_tested:
                    #print(f"Already tested the comparison {cond1} & {cond2}")
                    continue
                else:
                    #print(f"Testing the comparison {cond1} & {cond2}")
                    cond2_scores = list(stats_df[stats_df['condition'] == cond2]["values"])

                    #get ttest
                    ttest, pval = scipy.stats.ttest_rel(cond1_scores, cond2_scores)
                    #get cohens d
                    cohensd = stats_utils.cohens_d(cond1_scores, cond2_scores)

                    ttests.append(round(ttest,3))
                    pvals.append(pval)
                    compare1.append(cond1)
                    compare2.append(cond2)
                    cohens_d.append(cohensd)

                    already_tested.append(set([cond1, cond2]))

        _, adjusted_pvals, _, _ = statsmodels.stats.multitest.multipletests(pvals, method='bonferroni')
        # _, adjusted_pvals = statsmodels.stats.multitest.fdrcorrection(pvals)
        adjusted_pvals = [round(x,3) for x in adjusted_pvals]
        pvals = [round(x,3) for x in pvals]
        cohens_d = [round(x,3) for x in cohens_d]
        significances = stats_utils.assign_significance_labels(adjusted_pvals)


        ttest_df = pd.DataFrame({
            "Condition 1" : compare1,
            "Condition 2" : compare2,
            "T-test statistic" : ttests,
            "Adjusted pvalue" : adjusted_pvals,
            "Significance" : significances,
            "Cohen's d" : cohens_d,
            "Unadjusted p value" : pvals
        })
        ttest_df["Manipulation"] = current_category
        ttest_df["Computational design"] = approach

        # rename conditions according to new names:
        ttest_df['Condition 1'] = ttest_df['Condition 1'].map(plot_utils.COND2LABEL)
        ttest_df['Condition 2'] = ttest_df['Condition 2'].map(plot_utils.COND2LABEL)

        print(ttest_df)
        ttest_df.to_csv(f'{out_dir}/stats_{approach}_within_condition={current_category}_with_ttest.csv',index=False)
 
    return ttest_df


def run(config, approach, model_identifier, randomnouns):
    print(f"******* APPROACH: {approach} *******", flush=True)
    emb_context = config[approach]["emb_context"]
    split_coord = config[approach]["split_coord"]
    testonperturbed = config[approach]["testonperturbed"]

    scores_stim_df = get_plot_df(approach=approach, model_identifier=model_identifier,
                                 emb_context=emb_context, split_coord=split_coord,
                                 testonperturbed=testonperturbed, randomnouns=randomnouns)
    stats_df = get_ttest_results(approach=approach, model_identifier=model_identifier,
                                 emb_context=emb_context, split_coord=split_coord,
                                 testonperturbed=testonperturbed, randomnouns=randomnouns)

    if approach == "TrainIntact-TestPerturbed:contextualized":
        main_barplot(config=config, approach=approach, model_identifier=model_identifier,
                     full_df=scores_stim_df, full_stats_df=stats_df, randomnouns=randomnouns, vertical=True)
    main_barplot(config=config, approach=approach, model_identifier=model_identifier,
                 full_df=scores_stim_df, full_stats_df=stats_df, randomnouns=randomnouns, vertical=False)
    
    ttest_df = within_category_stats_and_plots(approach=approach, model_identifier=model_identifier, emb_context=emb_context,
                                        split_coord=split_coord, testonperturbed=testonperturbed,
                                        randomnouns=randomnouns)


def main():
    ##########################
    ## CONFIGUATION SETUP
    ##########################
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', type=str, default="all")
    parser.add_argument('--model_identifier', type=str, default='gpt2-xl')
    parser.add_argument('--randomnouns', action='store_true')
    args = parser.parse_args()

    model_identifier = args.model_identifier
    randomnouns = args.randomnouns  # whether or not to include control condition random nouns (used for SI)

    # load configuration
    with open('configurations.json', 'r') as f:
        config = json.load(f)

    if args.approach == "all":
        for approach in list(config.keys()):
            if approach != "TrainPerturbed-TestPerturbed:cv-by-passage":
                run(config, approach, model_identifier, randomnouns)
    else:
        run(config, args.approach, model_identifier, randomnouns)


if __name__ == "__main__":
    main()
