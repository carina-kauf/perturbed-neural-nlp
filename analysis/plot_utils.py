import os, sys
import re
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


def get_colors(randomnouns=False):
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


COND2LABEL = {
        "original" : "Original",
        #
        "scrambled1" : "1LocalWordSwap",
        "scrambled3" : "3LocalWordSwaps",
        "scrambled5" : "5LocalWordSwaps",
        "scrambled7" : "7LocalWordSwaps",
        "backward" : "ReverseOrder",
        "lowpmi" : "LowPMI",
        "lowpmi-random" : "LowPMIRand",
        #
        "contentwords" : "KeepContentW",
        "nounsverbsadj" : "KeepNVAdj",
        "nounsverbs" : "KeepNV",
        "nouns" : "KeepN",
        "functionwords" : "KeepFunctionW",
        #
        "chatgpt" : "Paraphrase",
        "sent_passage" : "RandSentFromPassage",
        "sent_topic" : "RandSentFromTopic",
        "sent_random" : "RandSent",
        #
        "random-wl" : "RandWordList",
        #
        "random-nouns" : "RandN",
        "length-control" : "LengthControl",
        "concatenated-control" : "ConcatenatedControl"
    }


def figure_setup():
    #define global figure settings
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'

    custom_params = {"axes.spines.right": False,
                     "axes.spines.top": False,
                     'ytick.left': True,
                     'xtick.bottom': True,
                    'grid.linestyle': "" #gets rid of horizontal lines
                    }
    sns.set_theme(font_scale=1.4, style="white", rc=custom_params)


def flatten_list(xss):
    return [x for xs in xss for x in xs]


def get_conditions(testonperturbed=False, randomnouns=False, length_control=False):
    
    if testonperturbed:
        to_prepend = "teston:"
    else:
        to_prepend = ""
        
    original = [f'{to_prepend}original']
    
    conditions_control = [f'{to_prepend}random-wl']

    conditions_scrambled = [f'{to_prepend}scrambled1',
                            f'{to_prepend}scrambled3',
                            f'{to_prepend}scrambled5',
                            f'{to_prepend}scrambled7',
                            f'{to_prepend}backward',
                            f'{to_prepend}lowpmi',
                            f'{to_prepend}lowpmi-random']
    
    if testonperturbed:
        conditions_scrambled = [re.sub("scrambled","scr",elm) for elm in conditions_scrambled]

    conditions_perturb_loss = [f'{to_prepend}contentwords',
                               f'{to_prepend}nounsverbsadj',
                               f'{to_prepend}nounsverbs',
                               f'{to_prepend}nouns',
                               f'{to_prepend}functionwords']
    
    if randomnouns:
        conditions_perturb_loss += [f'{to_prepend}random-nouns']
        
    if length_control:
        conditions_control = [f'{to_prepend}length-control'] + conditions_control

    conditions_perturb_meaning = [f'{to_prepend}chatgpt',
                                  f'{to_prepend}sentenceshuffle_passage',
                                  f'{to_prepend}sentenceshuffle_topic',
                                  f'{to_prepend}sentenceshuffle_random']


    #create CAT2COND
    conditions = [original, conditions_scrambled, conditions_perturb_loss, conditions_perturb_meaning, conditions_control]
    categories_unique = ["original", "word-order", "information-loss", "semantic-distance", "control"]
    
    CAT2COND = dict(zip(categories_unique, conditions)) #dictionary from manipulation category to condition
    
    #create COND2CAT
    categories = [["original"] * len(original),
                  ["word-order"] * len(conditions_scrambled),
                  ["information-loss"] * len(conditions_perturb_loss),
                  ["semantic-distance"] * len(conditions_perturb_meaning),
                  ["control"] * len(conditions_control)]
    
    conditions = flatten_list(conditions)
    categories = flatten_list(categories)

    COND2CAT = dict(zip(conditions, categories))

    return CAT2COND, COND2CAT


def get_max_score_ceiled(matrix):
    """
    input: result = out['data'] matrix
    output: maximum score and associated error for this matrix.
    """
    matrix = matrix.values #values: (e.g. for distilgpt2 a matrix of dimensions 7x2)
    max_score, error = 0,0
    for i in range(len(matrix)):
        if matrix[i][0] > max_score:
            max_score = matrix[i][0]
            error = matrix[i][1]
    return max_score, error


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


def get_all_layers(model_identifier):
    """
    input: model_identifier of model of which we want to find the layers
    output: np.array of all unique layer identifiers, ordered by position
    """
    working_dir = "/om2/user/ckauf/.result_caching/neural_nlp.score"
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


def get_file(cond, testonperturbed=True, model_identifier="gpt2-xl", emb_context="Passage", split_coord="Sentence", randomnouns=False,
             length_control=False, nr_of_splits=5, working_dir="/om2/user/ckauf/.result_caching/neural_nlp.score", return_best_layer=False):
    for filename in os.listdir(working_dir):
        if os.path.isdir(filename):
            continue
            
        if not testonperturbed and "teston:" in filename:
            continue
        if testonperturbed and "teston:" not in filename:
            continue
        
        if not any(filename.startswith(x) for x in [f"benchmark=Pereira2018-encoding-{cond}",
                                                   f"benchmark=Pereira2018-encoding-scrambled-{cond}",
                                                   f"benchmark=Pereira2018-encoding-perturb-{cond}"]):
            continue
            
        if not f"emb_context={emb_context},split_coord={split_coord}" in filename:
            continue
        
        
        exclude_list = []
        include_list = []
        
        exclude_list = ["-control", "random-nouns"]
        if randomnouns:
            exclude_list = ["-control"]
        if length_control:
            include_list = ["original", "length-control", "random-wl"]
            
            
        # factor in number of splits!
        if nr_of_splits == 5:
            exclude_list.append("nr_of_splits=2")
            
        if length_control:
            if all(x not in filename for x in include_list):
                continue 
            if "nr_of_splits=2" in filename:
                continue 
        elif nr_of_splits == 2:
            if "nr_of_splits=2" not in filename:
                continue
        else:
            if any(x in filename for x in exclude_list):
                continue
                
        if return_best_layer:
            if not "original" in filename:
                continue
                
        model_name = filename.split(",")[1]
        if model_name != f"model={model_identifier}":
            continue
            
        file_path = os.path.join(working_dir, filename)
        return file_path


def get_best_scores_df(model_identifier, emb_context="Passage", split_coord="Sentence", testonperturbed=False, randomnouns=False, length_control=False, nr_of_splits=5, return_best_layer=False, return_best_layer_score=False, best_layer=None, which_df='plot'):
    """
    input: model_identifier, embedding context, split_coordinate & whether to test on perturbed sentence
    output: dataframe containing the maximum score and associated error per condition.
    """
    
    working_dir = "/om2/user/ckauf/.result_caching/neural_nlp.score"
    
    CAT2COND, COND2CAT = get_conditions(testonperturbed=testonperturbed,
                                        randomnouns=randomnouns, length_control=length_control)
    
    if not length_control:
        category_groups = {cat : CAT2COND['original'] + CAT2COND[cat] + CAT2COND['control']
                           for cat in CAT2COND.keys() if cat not in ['original', 'control']}
    else:
        category_groups = {'control' : CAT2COND['original'] + CAT2COND['control']}
    
    print(category_groups)
    
    if nr_of_splits == 2:
        # create a new dictionary with only the 'name' key-value pair (exclude chatgpt here)
        value = [elm for elm in category_groups['semantic-distance'] if not 'chatgpt' in elm]
        category_groups = {'semantic-distance': value}

        
    frames = []
    for cat_name in category_groups.keys():
        no_files = False
        cat_frames = []
        for cond in category_groups[cat_name]:
            file = get_file(cond=cond, testonperturbed=testonperturbed,
                            model_identifier=model_identifier, emb_context=emb_context,
                            split_coord=split_coord, randomnouns=randomnouns,
                            length_control=length_control, nr_of_splits=nr_of_splits, working_dir=working_dir,
                           return_best_layer=return_best_layer)
            if file:
                with open(file, 'rb') as f:
                    out = pickle.load(f)
            else:
                no_files = True
                break
            result = out['data']
            # get error (noise-corrected for within-category variance)
            best_layernr = get_best_layer(result) #note2self: can keep this after moving to reporting raw scores, because the best layer remains the same
            layers = get_all_layers(model_identifier)
            best_layer_name = layers[best_layernr]
            if return_best_layer:
                return best_layer_name
            raw_score = result.raw.raw
            
            if return_best_layer_score:
                assert best_layer
                print(best_layer)
                best_layer_raw = raw_score[{"layer": [layer == best_layer for layer in raw_score["layer"]]}]
            else:
                best_layer_raw = raw_score[{"layer": [layer == best_layer_name for layer in raw_score["layer"]]}]

            subject_score_with_index = best_layer_raw.groupby('subject').median()
            subject_score = subject_score_with_index.values
            subject_index = subject_score_with_index.subject.values
            df = pd.DataFrame({
                'values' : subject_score,
                'subject_index' : subject_index,
            })
            df['condition'] = cond
            cat_frames.append(df)
        
        if no_files:
            break
        cat_df = pd.concat(cat_frames)
        if which_df == 'plot':
            piv = cat_df.pivot_table(index='subject_index', columns='condition', values='values')
            conds = category_groups[cat_name]
            categories = [COND2CAT[cond] for cond in conds]
            # reindex according to condition order
            piv_reindexed = piv[conds]
            # get subject scores
            score = piv_reindexed.median(axis=0)
            # we subtract the mean across conditions within a subject
            demeaned = piv_reindexed.subtract(piv_reindexed.mean(axis=1).values, axis=0)
            yerr = stats.median_abs_deviation(demeaned.values.T, axis=1, scale='normal')
            cat_df = pd.DataFrame({
                'score' : score,
                'error' : yerr,
                'condition' : conds,
                'category' : categories
            })
            cat_df['category_group'] = cat_name
        elif which_df == 'stats': #if stats_df
            cat_df['category'] = cat_df['condition'].map(COND2CAT)
            cat_df['category_group'] = cat_name
        else:
            raise NotImplementedError
            
        frames.append(cat_df)
            
    out_df = pd.concat(frames)
    # clean names
    out_df['condition'] = out_df['condition'].str.replace("teston:","")
    out_df['condition'] = out_df['condition'].replace(
    {'sentenceshuffle_random': 'sent_random',
    'sentenceshuffle_passage': 'sent_passage',
    'sentenceshuffle_topic': 'sent_topic',
    'scr1': 'scrambled1',
    'scr3': 'scrambled3',
    'scr5': 'scrambled5',
    'scr7': 'scrambled7'})
    # get rid of categories as index
    if which_df == 'plot':
        out_df = out_df.reset_index(drop=True)
    return out_df


def get_sample_stimuli(getall=False, randomnouns=False, length_control=False):
    working_dir = "/om2/user/ckauf/perturbed-neural-nlp/ressources/scrambled_stimuli_dfs/"
    
    # Translate between names for stimuli dataframes and condition names used for plotting
    conditions = [
    ('Original', 'original'),
        #
    ('Scr1', 'scrambled1'),
    ('Scr3', 'scrambled3'),
    ('Scr5', 'scrambled5'),
    ('Scr7', 'scrambled7'),
    ('backward', 'backward'),
    ('lowPMI', 'lowpmi'),
    ('lowPMI_random', 'lowpmi-random'),
        #
    ('contentwords', 'contentwords'),
    ('nounsverbsadj', 'nounsverbsadj'),
    ('nounsverbs', 'nounsverbs'),
    ('nouns', 'nouns'),
    ('functionwords', 'functionwords'),
        #
    ('chatGPT', 'chatgpt'),
    ('sentenceshuffle-withinpassage', 'sent_passage'),
    ('sentenceshuffle-withintopic', 'sent_topic'),
    ('sentenceshuffle-random', 'sent_random'),
        #
    ('random', 'random-wl')
    ]
    
    if randomnouns:
        conditions.append(('randomnouns', 'random-nouns'))
    if length_control:
        conditions.append(('length_control', 'length-control'))
    
    # set up empty lists
    conds, sentences = [], []
    
    # populate lists
    for (stimuli_name, condition) in conditions:
        print(f"{stimuli_name} | {condition}")
        for filename in os.listdir(working_dir):
            if filename == f'stimuli_{stimuli_name}.pkl':
                with open(os.path.join(working_dir,filename), 'rb') as f:
                    df = pickle.load(f)
                    
                if getall:
                    sentences += list(df["sentence"])
                    conds += [condition] * len(list(df["sentence"]))
                else:
                    sentences += list(df["sentence"])[:1]
                    conds += [condition]
                
    sample_stim_df = pd.DataFrame({
    'condition' : conds,
    'stimulus': sentences
    })     
    
    return sample_stim_df