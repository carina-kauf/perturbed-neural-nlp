import os
import numpy as np
import torch.cuda
if not os.getenv('USER') == 'ckauf':
    import pickle5 as pickle #for local debugging
else:
    import pickle


def load_activations_into_matrix(identifier,
                                 stimuli_identifier,
                                 agg,
                                 expt,
                                 layer_identifier,
                                 path,
                                 verbose=True):
    """
    Load pickled activations into a matrix of shape [num items; emb dim]

    Args:
        identifier: str, the model identifier, e.g., 'distilgpt2'
        stimuli_identifier: str, the benchmark identifier, e.g., 'Pereira2018-Original'
        agg: str, aggregation method, e.g., 'lasttoken'
        expt: str, experiment name (Pereira), e.g., '243'
        layer_identifier: str, name of source layer, e.g., 'drop'
        path: str, name of directory where activations live

    Returns
        activations_matrix: np.array, shape [num items; emb dim]
        flat_sentence_array: list, loaded sentences (items)
        flat_sentence_num_array: list, loaded sentences and their respective index within a passage


    """

    d_expt_passage = {'243sentences': np.arange(1, 72 + 1),
                      '384sentences': np.arange(1, 96 + 1)}

    d_expt_sentence = {'243sentences': np.arange(243),
                       '384sentences': np.arange(384)}

    activations = []
    sentence_array = []  # for sanity checking sentences
    sentence_num_array = []

    if os.getenv('DECONTEXTUALIZED_EMB', '0') == '1':
        print("I'm loading the activations with DECONTEXTUALIZED_EMB=1, hence loading individual sentence activations.")
        activations_iterator_dict = d_expt_sentence
    else:
        activations_iterator_dict = d_expt_passage
        print("I'm loading the activations with DECONTEXTUALIZED_EMB=None, hence loading contextualized sentence activations.")

    for idx in activations_iterator_dict[expt]:

        if os.getenv('DECONTEXTUALIZED_EMB', '0') == '1':
            # identifier=distilgpt2,stimuli_identifier=Pereira2018,emb_context=Sentence-384sentences.99.pkl
            # NOTE: difference is dot in between experiment and idx!
            intended_filename = f'identifier={identifier},stimuli_identifier={stimuli_identifier}-{agg},emb_context=Sentence-{expt}.{idx}.pkl'
        else:
            intended_filename = f'identifier={identifier},stimuli_identifier={stimuli_identifier}-{agg},emb_context=Passage-{expt}{idx}.pkl'

        if verbose:
            print(f'Loading {intended_filename}')

        file = os.path.join(path, intended_filename)

        with open(file, 'rb') as f:
            out = pickle.load(f)
        result = out['data']
        data = result[{"neuroid": [layer == layer_identifier for layer in result["layer"].values]}]

        activations.append(data.values)
        sentence_array.append(list(data.stimulus_sentence.values))
        sentence_num_array.append(list(data.sentence_num.values))

    print(f"Dimensions BEFORE sublist flattening!")
    print(f'Sentence array dimensions: {np.shape(sentence_array)}')
    print(f'Sentence number array dimensions: {np.shape(sentence_num_array)}')
    print(f'Activations array dimensions: {np.shape(activations)}')

    flat_sentence_array = [item for sublist in sentence_array for item in sublist]
    flat_sentence_num_array = [item for sublist in sentence_num_array for item in sublist]
    flat_activations = [item for sublist in activations for item in sublist]

    print(f"Dimensions AFTER sublist flattening!")
    print(f'Sentence array dimensions: {np.shape(flat_sentence_array)}')
    print(f'Sentence number array dimensions: {np.shape(flat_sentence_num_array)}')
    print(f'Activations array dimensions: {np.shape(flat_activations)}')

    activations_matrix = np.array(flat_activations)

    exp_num = expt.strip("sentences")

    assert (len(flat_sentence_array)) == int(exp_num)
    assert (len(flat_sentence_num_array)) == int(exp_num)
    assert (activations_matrix.shape[0] == int(exp_num))
    # assert (activations_matrix.shape[1] == 768)  # Todo don't hardcode the num of units if this assertion is kept..

    return activations_matrix, flat_sentence_array, flat_sentence_num_array


###################
## OLD FUNCTION!
###################

# def load_activations_into_matrix(identifier,
#                                  stimuli_identifier,
#                                  agg,
#                                  expt,
#                                  layer_identifier,
#                                  path,
#                                  d_expt,
#                                  verbose=True):
#     """
#     Load pickled activations into a matrix of shape [num items; emb dim]
#
#     Args:
#         identifier: str, the model identifier, e.g., 'distilgpt2'
#         stimuli_identifier: str, the benchmark identifier, e.g., 'Pereira2018-Original'
#         agg: str, aggregation method, e.g., 'lasttoken'
#         expt: str, experiment name (Pereira), e.g., '243'
#         layer_identifier: str, name of source layer, e.g., 'drop'
#         path: str, name of directory where activations live
#         d_expt: dictionary, defines dimensions
#
#     Returns
#         activations_matrix: np.array, shape [num items; emb dim]
#         flat_sentence_array: list, loaded sentences (items)
#         flat_sentence_num_array: list, loaded sentences and their respective index within a passage
#
#
#     """
#
#     activations = []
#     sentence_array = []  # for sanity checking sentences
#     sentence_num_array = []
#
#     for passage_idx in d_expt[expt]:
#
#         #identifier=distilgpt2,stimuli_identifier=Pereira2018,emb_context=Sentence-384sentences.99.pkl
#         intended_filename = f'identifier={identifier},stimuli_identifier={stimuli_identifier}-{agg},emb_context=Passage-{expt}{passage_idx}.pkl'
#
#         if verbose:
#             print(f'Loading {intended_filename}')
#
#         file = os.path.join(path, intended_filename)
#
#         with open(file, 'rb') as f:
#             out = pickle.load(f)
#         result = out['data']
#         data = result[{"neuroid": [layer == layer_identifier for layer in result["layer"].values]}]
#
#         activations.append(data.values)
#         sentence_array.append(list(data.stimulus_sentence.values))
#         sentence_num_array.append(list(data.sentence_num.values))
#
#     flat_sentence_array = [item for sublist in sentence_array for item in sublist]
#     flat_sentence_num_array = [item for sublist in sentence_num_array for item in sublist]
#     flat_activations = [item for sublist in activations for item in sublist]
#
#     activations_matrix = np.array(flat_activations)
#
#     exp_num = expt.strip("sentences")
#
#     assert (len(flat_sentence_array)) == int(exp_num)
#     assert (len(flat_sentence_num_array)) == int(exp_num)
#     assert (activations_matrix.shape[0] == int(exp_num))
#     # assert (activations_matrix.shape[1] == 768)  # Todo don't hardcode the num of units if this assertion is kept..
#
#     return activations_matrix, flat_sentence_array, flat_sentence_num_array
