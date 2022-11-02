import os, sys
import re
import pickle
import numpy as np
import pandas as pd


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
        "nouns" : "Nouns",
        "nounsverbs" : "NounsVerbs",
        "nounsverbsadj" : "NounsVerbsAdj",
        "contentwords" : "NounsVerbsAdjAdv",
        "functionwords" : "FunctionWords",
        #
        "sent_passage" : "RandSentFromPassage",
        "sent_topic" : "RandSentFromTopic",
        "sent_random" : "RandSent",
        #
        "random-wl" : "RandWordList",
        #
        "random-nouns" : "RandNouns",
        "length-control" : "LengthControl",
        "concatenated-control" : "ConcatenatedControl"
    }


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

    conditions_perturb_loss = [f'{to_prepend}nouns',
                               f'{to_prepend}nounsverbs',
                               f'{to_prepend}nounsverbsadj',
                               f'{to_prepend}contentwords',
                               f'{to_prepend}functionwords']
    
    if randomnouns:
        conditions_perturb_loss += [f'{to_prepend}random-nouns']
        
    if length_control:
        conditions_control = [f'{to_prepend}length-control'] + conditions_control

    conditions_perturb_meaning = [f'{to_prepend}sentenceshuffle_passage',
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


def get_max_score(matrix):
    """
    input: result = out['data'].values matrix (e.g. for distilgpt2 a matrix of dimensions 7x2)
    output: maximum score and associated error for this matrix.
    """
    max_score, error = 0,0
    for i in range(len(matrix)):
        if matrix[i][0] > max_score:
            max_score = matrix[i][0]
            error = matrix[i][1]
    return max_score, error

# from brainio_base.assemblies import DataAssembly, merge_data_arrays
# from brainscore.metrics import Score

# from scipy.stats import median_abs_deviation

# def aggregate_neuroid_scores(neuroid_scores, subject_column):
#     subject_scores = neuroid_scores.groupby(subject_column).median()
#     center = subject_scores.median(subject_column)
#     subject_values = np.nan_to_num(subject_scores.values, nan=0)  # mad cannot deal with all-nan in one axis, treat as 0
#     subject_axis = subject_scores.dims.index(subject_scores[subject_column].dims[0])
#     error = median_abs_deviation(subject_values, axis=subject_axis, scale=1)
#     score = Score([center, error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
#     score.attrs['raw'] = neuroid_scores
#     score.attrs['description'] = "score aggregated by taking median of neuroids per subject, " \
#                                  "then median of subject scores"
#     return score

#aggregation from pipeline
# from brainscore.metrics.transformations import  apply_aggregate

# def get_lang_score(content_of_picklefile):
#     lang_score_matrix = []
#     all_raw_scores = content_of_picklefile["data"].raw.raw.raw
    
#     # same as https://github.com/carina-kauf/perturbed-neural-nlp/blob/master/neural_nlp/benchmarks/neural.py#L173
#     raw_neuroids = apply_aggregate(lambda values: values.mean('split').mean('experiment'), all_raw_scores)
#     lang_neuroids = raw_neuroids.sel(atlas='language', _apply_raw=False)
#     layers = list(lang_neuroids.layer.data)
    
#     for layer in layers:
#         lang_neuroids.sel(layer=layer)
#         score = aggregate_neuroid_scores(lang_neuroids.sel(layer=layer), "subject")
# #         print(f"{layer} | Score: {score.data}")
#         lang_score_matrix.append(score.data)
#     return lang_score_matrix

def get_best_scores_df(model_identifier, emb_context="Passage", split_coord="Sentence", testonperturbed=False, randomnouns=False, length_control=False):
    """
    input: model_identifier, embedding context, split_coordinate & whether to test on perturbed sentence
    output: dataframe containing the maximum score and associated error per condition.
    """
    
    working_dir = "/om2/user/ckauf/.result_caching/neural_nlp.score"
    
    CAT2COND, COND2CAT = get_conditions(testonperturbed=testonperturbed, randomnouns=randomnouns, length_control= length_control)
    
    conditions, categories = [], []
    max_scores, errors = [], []
    
    for filename in os.listdir(working_dir):
        
        if os.path.isdir(os.path.join(working_dir,filename)):
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
                
#             print(filename, sys.stdout.flush())
            
            #clean name
            condition = re.sub("perturb-","",condition)
            if not any(x in condition for x in ["1", "3", "5", "7"]):
                condition = re.sub("scrambled-","",condition)
            
            if testonperturbed:
                condition = re.sub("scrambled","scr",condition)
                
#             print(condition)

            #load scores
            file = os.path.join(working_dir,filename)
            with open(file, 'rb') as f:
                out = pickle.load(f)
#                 lang_score_matrix = get_lang_score(out)
#                 max_score, error = get_max_score(lang_score_matrix)
                result = out['data'].values
#                 #print(result, '\n\n')
                max_score, error = get_max_score(result)

            conditions.append(condition)
            categories.append(COND2CAT[condition])
            max_scores.append(max_score)
            errors.append(error)

                
    import pandas as pd
    
    index = conditions
    condition_order = list(COND2CAT.keys())
    
    df = pd.DataFrame({
        'score': max_scores,
        'error': errors,
        'condition': conditions,
        'category': categories})
    
    df['condition'] = pd.Categorical(df['condition'], categories=condition_order, ordered=True)
    scores_df = df.sort_values(by='condition').reset_index(drop=True)
    
    #clean names
    scores_df['condition'] = scores_df['condition'].str.replace("teston:","")
    scores_df['condition'] = scores_df['condition'].replace(
    {'sentenceshuffle_random': 'sent_random',
    'sentenceshuffle_passage': 'sent_passage',
    'sentenceshuffle_topic': 'sent_topic',
    'scr1': 'scrambled1',
    'scr3': 'scrambled3',
    'scr5': 'scrambled5',
    'scr7': 'scrambled7'}
    )
    
    return scores_df

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
    ('nouns', 'nouns'),
    ('nounsverbs', 'nounsverbs'),
    ('nounsverbsadj', 'nounsverbsadj'),
    ('contentwords', 'contentwords'),
    ('functionwords', 'functionwords'),
        #
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