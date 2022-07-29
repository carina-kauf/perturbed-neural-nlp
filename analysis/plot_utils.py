import os, sys
import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm


def flatten_list(xss):
    return [x for xs in xss for x in xs]


def get_conditions(testonperturbed=False):
    
    if testonperturbed:
        to_prepend = "teston:"
    else:
        to_prepend = ""
        
    original = [f'{to_prepend}original']
    
    conditions_control = [f'{to_prepend}length-control',
                         f'{to_prepend}random-wl']

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
                               f'{to_prepend}functionwords',
                               f'{to_prepend}random-nouns']

    conditions_perturb_meaning = [f'{to_prepend}sentenceshuffle_passage',
                                  f'{to_prepend}sentenceshuffle_topic',
                                  f'{to_prepend}sentenceshuffle_random']


    #create CAT2COND
    conditions = [original, conditions_control, conditions_perturb_loss, conditions_perturb_meaning,conditions_scrambled]
    categories_unique = ["original", "controls", "information-loss", "sentence-meaning", "word-order"]
    
    CAT2COND = dict(zip(categories_unique, conditions)) #dictionary from manipulation category to condition
    
    #create COND2CAT
    categories = [["original"] * 1,
            ["controls"]*len(conditions_control),
            ["information-loss"]*len(conditions_perturb_loss),
            ["sentence-meaning"]*len(conditions_perturb_meaning),
            ["word-order"]*len(conditions_scrambled)]
    
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

def get_best_scores_df(model_identifier, emb_context="Passage", split_coord="Sentence", testonperturbed=False):
    """
    input: model_identifier, embedding context, split_coordinate & whether to test on perturbed sentence
    output: dataframe containing the maximum score and associated error per condition.
    """
    
    working_dir = "/om2/user/ckauf/.result_caching/neural_nlp.score"
    
    CAT2COND, COND2CAT = get_conditions(testonperturbed=testonperturbed)
    
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
            
        if "constant-control" in filename:
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