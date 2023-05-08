import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import re
import scipy as sp

from scipy import stats
from tqdm import tqdm
import json

# functions from other scripts
import plot_utils
import stats_utils

out_dir = 'results_paper_revision'


def get_activations_dictionary(activations_dir, model_identifier, emb_context):
    """
    input:
        * activations_dir : directory where cached model activations live
        * model identifier
        * emb_context %in% {"Passage", "Sentence"}
    output: populated model dictionary with data of all layers
    dictionary structure: passage_identifier --> condition identifier --> data
    """

    model_dictionary = {}
    layer_names = stats_utils.get_all_layers(model_identifier)

    for filename in tqdm(os.listdir(activations_dir)):
        if not f"emb_context={emb_context}" in filename:
            continue
        if not model_identifier in filename:
            continue
        if not "-lasttoken" in filename:
            continue

        passage_identifier = stats_utils.get_passage_identifier(filename)

        condition = filename.split("Pereira2018-")[1]
        condition = condition.split("-lasttoken")[0]

        file = os.path.join(activations_dir, filename)
        with open(file, 'rb') as f:
            out = pickle.load(f)
        result = out['data']

        activations = []
        for l in layer_names:
            data = result[{"neuroid": [layer == l for layer in result["layer"].values]}].data  # activations * nrlayers
            activations.append(data)

        if not condition in model_dictionary:
            model_dictionary[condition] = {}
        model_dictionary[condition][passage_identifier] = activations
    #         print(model_dictionary)
    #         print(np.shape(model_dictionary[condition][passage_identifier])) #(49, len_passage, nr_units)

    return model_dictionary


def run_get_model_activations(model_identifier, emb_context):
    activations_dir = "/om2/user/ckauf/.result_caching/neural_nlp.models.wrapper.core.ActivationsExtractorHelper._from_sentences_stored"
    model_dictionary = get_activations_dictionary(activations_dir, model_identifier=model_identifier,
                                                  emb_context=emb_context)

    with open(os.path.abspath(
            f"model_activation_dictionary,model={model_identifier},emb_context={emb_context},lasttoken.pkl"),
              "wb") as f:
        pickle.dump(model_dictionary, f)

    return model_dictionary

def get_nlayers(activations_dictionary):
    for key in activations_dictionary.keys():
        for passage_id in activations_dictionary[key].keys():
            nlayers = np.shape(activations_dictionary[key][passage_id])[0]
            print(nlayers)
            break
        break
    return nlayers


def get_correlations(activations_dictionary, approach, testonperturbed, randomnouns):

    if os.path.isfile(f'activation_spearman_correlations_{approach}.pkl'):
        print("Loading activations correlations dict from cache!")
        with open(f'activation_spearman_correlations_{approach}.pkl', 'rb') as handle:
            correlations_dict = pickle.load(handle)
    else:
        print("Generating activations correlations dict!")

        # want: 1 list of correlations per condition, 49 values for each layer
        nlayers = get_nlayers(activations_dictionary)

        correlations_dict = {}

        for cond in activations_dictionary.keys():
            print(f"Original vs. {cond}")
            condition_correlations = []

            for layer in tqdm(range(nlayers)):
                layer_correlations = []

                for passage_id in list(activations_dictionary['Original'].keys()):
                    n_sentences_in_passage = np.shape(activations_dictionary['Original'][passage_id])[1]

                    for sentence_ind in range(n_sentences_in_passage):
                        original_act = activations_dictionary['Original'][passage_id][layer][sentence_ind]
                        cond_act = activations_dictionary[cond][passage_id][layer][sentence_ind]
                        corr = stats.spearmanr(original_act, cond_act)[0]
                        layer_correlations.append(corr)

                avg_correlation_for_layer = np.mean(layer_correlations)
                condition_correlations.append(avg_correlation_for_layer)

            correlations_dict[cond] = condition_correlations

        with open(f'activation_spearman_correlations_{approach}.pkl', 'wb') as handle:
            pickle.dump(correlations_dict, handle)

    correlations_dict = {k: v for k, v in correlations_dict.items() if not "-control" in k}

    # assimilate names between activations and brainscores
    if testonperturbed:
        to_prepend = "teston:"
    else:
        to_prepend = ""

    CONDMAP = {
        'Original': f'{to_prepend}original',
        #
        'Scr1': f'{to_prepend}scrambled1',  # different for teston, there teston:scr1
        'Scr3': f'{to_prepend}scrambled3',
        'Scr5': f'{to_prepend}scrambled5',
        'Scr7': f'{to_prepend}scrambled7',
        'backward': f'{to_prepend}backward',
        'lowPMI': f'{to_prepend}lowpmi',
        'lowPMI-random': f'{to_prepend}lowpmi-random',
        #
        'nouns': f'{to_prepend}nouns',
        'randomnouns': f'{to_prepend}random-nouns',
        'nounsverbs': f'{to_prepend}nounsverbs',
        'nounsverbsadj': f'{to_prepend}nounsverbsadj',
        'contentwords': f'{to_prepend}contentwords',
        'functionwords': f'{to_prepend}functionwords',
        #
        'chatgpt': f'{to_prepend}chatgpt',
        'sentenceshuffle_passage': f'{to_prepend}sent_passage',
        'sentenceshuffle_topic': f'{to_prepend}sent_topic',
        'sentenceshuffle_random': f'{to_prepend}sent_random',
        #
        'random-wl': f'{to_prepend}random-wl'
    }

    if testonperturbed:
        for key, value in CONDMAP.items():
            CONDMAP[key] = re.sub("scrambled", "scr", value)

    correlations_df = pd.DataFrame.from_dict(correlations_dict)
    correlations_df = correlations_df.rename(columns={col: CONDMAP[col] for col in correlations_df.columns})

    _, COND2CAT = plot_utils.get_conditions(testonperturbed=testonperturbed, randomnouns=randomnouns)
    condition_order = [re.sub("enceshuffle", "", elm) for elm in list(COND2CAT.keys())]

    correlations_df = correlations_df[condition_order]

    long_correlations = pd.melt(correlations_df)
    long_correlations = long_correlations.rename(columns={"value": f"correlation"})
    long_correlations["index"] = long_correlations.index

    return long_correlations


# TODO adapt function in plot_utils.py to output either best or all scores

def get_scores_dictionary(model_identifier, emb_context, split_coord, testonperturbed, randomnouns):
    """
    input: model_identifier, embedding context, split_coordinate & whether to test on perturbed sentence
    output: dataframe containing the maximum score and associated error per condition.
    """

    print_cnt = 0

    score_dict = {}
    scores_dir = "/om2/user/ckauf/.result_caching/neural_nlp.score"

    for filename in os.listdir(scores_dir):

        if os.path.isdir(os.path.join(scores_dir, filename)):
            continue

        if not testonperturbed:
            if "teston:" in filename:
                continue
        else:
            if not "teston:" in filename:
                continue

        if not f"emb_context={emb_context}" in filename:
            continue

        if not f"split_coord={split_coord}" in filename:
            continue

        exclude_list = ["-control", "random-nouns"]
        if randomnouns:
            exclude_list = ["-control"]

        if any(x in filename for x in exclude_list):
            continue

        model_name = filename.split(",")[1]
        bm = filename.split(",")[0]

        if bm == "benchmark=Pereira2018-encoding":  # exclude old orignial bm in different format
            continue

        if "model=" + model_identifier == model_name:

            print_cnt += 1

            condition = bm.split("benchmark=Pereira2018-encoding-")[-1]

            # clean name
            condition = re.sub("perturb-", "", condition)
            if not any(x in condition for x in ["1", "3", "5", "7"]):
                condition = re.sub("scrambled-", "", condition)

            if testonperturbed:
                condition = re.sub("scrambled", "scr", condition)

            # load scores
            file = os.path.join(scores_dir, filename)
            with open(file, 'rb') as f:
                out = pickle.load(f)
                
            scores, errors = [], []
            frames = []
            result = out['data']
            for l in result.raw.raw.layer.data:
                subject_scores = []
                layerwise_scores = result.raw.raw.sel(layer=l)
                subject_score = layerwise_scores.groupby('subject').median()
                df = pd.DataFrame({
                'subject_id' : subject_score.subject,
                'subject_score' : subject_score,
                'layer' : [l] * len(subject_score)
                })
                frames.append(df)
            raw_brainscore_df = pd.concat(frames)
            # per voxel, get median per layer per subject
            brainscore_df = raw_brainscore_df.groupby(['subject_id', 'layer']).median().reset_index()

            brainscore_df_with_medians = brainscore_df.copy(deep=True)
            for l in result.raw.raw.layer.data:
                # get median over subjects
                score = brainscore_df_with_medians.loc[brainscore_df_with_medians['layer'] == l]['subject_score'].median()
                subject_scores = list(brainscore_df_with_medians.loc[brainscore_df_with_medians['layer'] == l]['subject_score'])
                error = stats.median_abs_deviation(subject_scores, scale='normal')
                scores.append(score)
                errors.append(error)
                

            score_dict[condition] = {}
            score_dict[condition]["score"] = scores
            score_dict[condition]["error"] = errors

    scores_only = {k: score_dict[k]["score"] for k in list(score_dict.keys())}

    df = pd.DataFrame.from_dict(scores_only)

    _, COND2CAT = plot_utils.get_conditions(testonperturbed=testonperturbed, randomnouns=randomnouns)
    condition_order = list(COND2CAT.keys())
    df = df[condition_order]
    df = df.rename(columns={col: re.sub("enceshuffle", "", col) for col in df.columns})

    long_brainscores = pd.melt(df)
    long_brainscores = long_brainscores.rename(columns={"value": f"brainscore"})
    long_brainscores["index"] = long_brainscores.index

    return long_brainscores

##########################
### PLOTTING
##########################

mypalette = "Spectral_r"  # "cubehelix_r" # "viridis_r" # "magma_r"


def main_scatterplot(plot_df, approach, model_identifier,
                     emb_context, split_coord, testonperturbed, randomnouns):
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'

    custom_params = {"axes.spines.right": False,
                     "axes.spines.top": False,
                     'ytick.left': True,
                     'xtick.bottom': True,
                     'grid.linestyle': "",  # gets rid of horizontal lines
                     # set tick width
                     'xtick.major.size': 20,
                     'xtick.major.width': 4,
                     'xtick.minor.size': 10,
                     'xtick.minor.width': 2,
                     'ytick.major.size': 20,
                     'ytick.major.width': 4,
                     'ytick.minor.size': 10,
                     'ytick.minor.width': 2
                     }
    sns.set_theme(font_scale=2, style="white", rc=custom_params)

    fig = plt.figure(figsize=(15, 15))
    ax = sns.scatterplot(data=plot_df, x="correlation", y="brainscore", hue="layer", style="condition",
                         palette=mypalette,
                         edgecolor="none", s=200, legend="auto")

    norm = plt.Normalize(plot_df['layer'].min(), plot_df['layer'].max())
    sm = plt.cm.ScalarMappable(cmap=mypalette, norm=norm)
    sm.set_array([])

    y0 = -0.05
    y1 = 0.38
    lims = [y0, y1]
    ax.plot(lims, lims, '--',color="lightgray")

    ax = sns.scatterplot(data=plot_df, x="correlation", y="brainscore", hue="layer", style="condition",
                         palette=mypalette,
                         edgecolor="none", s=200, legend=None)

    len_layers = len(plot_df["layer"].unique())

    # Remove the legend and add a colorbar
    ax.get_legend().remove()
    bar = fig.colorbar(sm, ticks=list(range(len_layers))[::5], aspect=30, orientation='vertical', ax=ax,
                       label='layer index',
                       fraction=0.04)

    # Put a legend below current axis (only for styles)
    handles, labels = ax.get_legend_handles_labels()
    handles_labels = list(zip(handles, labels))

    handles_labels = [elm for elm in handles_labels if not elm[1].isdigit()]
    handles_labels = [elm for elm in handles_labels if elm[1] not in ["condition", "layer"]]
    handles = [elm[0] for elm in handles_labels]
    labels = [elm[1] for elm in handles_labels]

    ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.18),
              fancybox=True, shadow=False, ncol=4, title="perturbation manipulation condition", markerscale=2,
              title_fontsize=25)


    # change fontsize of labels
    ax.set(xlabel="Degree of similarity in the embedding space to Original\n(Spearman’s rho)",
           ylabel='Normalized predictivity')
    ax.set_ylim(y0, y1)
    ax.xaxis.get_label().set_fontsize(30)
    ax.yaxis.get_label().set_fontsize(30)
    ax.tick_params(labelsize=30)

    # add stats
    r, p = sp.stats.pearsonr(plot_df['correlation'], plot_df['brainscore'])

    if p < 0.0001:
        p_text = "p<<.001"
    elif p < 0.001:
        p_text = "p<.001"
    elif p < 0.01:
        p_text = "p<.01"
    elif p < 0.05:
        p_text = "p<.05"
    else:
        p_text = "p={:.2g}, n.s.".format(p)

    annotation_x = plot_df["correlation"].min() + 0.1
    annotation_y = plot_df["brainscore"].max() + 0.01
    plt.text(annotation_x, annotation_y, 'R={:.2f}, {}'.format(r, p_text),
             horizontalalignment='right', color='black', weight='semibold', fontsize=25)

    savename = f'{out_dir}/correlation2brainscore,model_identifier={model_identifier},approach={approach}'

    plt.savefig(f'{savename}.svg', dpi=180, bbox_inches='tight')
    plt.savefig(f'{savename}.png', dpi=180, bbox_inches='tight')
    plt.show()

    return y1

##########################
### RUN
##########################


def run(config, approach, model_identifier, randomnouns):
    print(f"******* APPROACH: {approach} *******", flush=True)
    emb_context = config[approach]["emb_context"]
    split_coord = config[approach]["split_coord"]
    testonperturbed = config[approach]["testonperturbed"]

    if os.path.isfile(os.path.abspath(
            f"model_activation_dictionary,model={model_identifier},emb_context={emb_context},lasttoken.pkl")):
        print("Loading model activations dictionary from cache!", flush=True)
        with open(os.path.abspath(
                f"model_activation_dictionary,model={model_identifier},emb_context={emb_context},lasttoken.pkl"),
                  "rb") as f:
            activations_dictionary = pickle.load(f)
    else:
        print("Getting model activations dictionary!", flush=True)
        activations_dictionary = run_get_model_activations(model_identifier, emb_context)

    if not randomnouns:
        activations_dictionary = {k: v for k, v in activations_dictionary.items() if k != "random-nouns"}


    long_correlations = get_correlations(activations_dictionary, approach, testonperturbed, randomnouns)
    long_brainscores = get_scores_dictionary(model_identifier, emb_context, split_coord, testonperturbed, randomnouns)

    # merge
    plot_df = pd.merge(long_brainscores, long_correlations, how='inner', on='index')
    plot_df = plot_df.rename(columns={"variable_x": "condition"})

    if testonperturbed:
        to_prepend = "teston:"
    else:
        to_prepend = ""

    # add layer indices
    nlayers = len(plot_df.loc[plot_df["condition"] == f"{to_prepend}original"])
    nconds = len(list(plot_df["condition"].unique()))
    layers = list(range(nlayers)) * nconds
    plot_df["layer"] = layers

    plot_df = plot_df[["index", "condition", "brainscore", "correlation", "layer"]]
    # rename conditions according to new names:
    if testonperturbed:
        plot_df['condition'] = [re.sub("teston:", "", elm) for elm in list(plot_df['condition'])]
        plot_df['condition'] = [re.sub("scr", "scrambled", elm) for elm in list(plot_df['condition'])]
    plot_df['condition'] = plot_df['condition'].map(plot_utils.COND2LABEL)
    
    y1 = main_scatterplot(plot_df, approach=approach, model_identifier=model_identifier,
                                 emb_context=emb_context, split_coord=split_coord,
                                 testonperturbed=testonperturbed, randomnouns=randomnouns)



def main():
    ##########################
    ## CONFIGUATION SETUP
    ##########################
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', type=str, default="all")
    parser.add_argument('--model_identifier', type=str, default='gpt2-xl')
    parser.add_argument('--randomnouns', action='store_true')
    args = parser.parse_args()
    print(args, flush=True)

    model_identifier = args.model_identifier
    randomnouns = args.randomnouns  # whether or not to include control condition random nouns (used for SI)

    # load configuration
    with open('configurations.json', 'r') as f:
        config = json.load(f)

    if args.approach == "all":
        for approach in list(config.keys()):
            run(config, approach, model_identifier, randomnouns)
    else:
        run(config, args.approach, model_identifier, randomnouns)
        

if __name__ == "__main__":
    main()