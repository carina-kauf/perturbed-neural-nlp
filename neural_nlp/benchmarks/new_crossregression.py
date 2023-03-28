
from neural_nlp.benchmarks.new_split import SplitNew as Split #used for new split_by_passage_id functionality
from brainscore.metrics.transformations import Transformation #, Split
from brainio_collection.transform import subset
from brainscore.metrics import Score
from tqdm import tqdm
import logging
from brainscore.utils import fullname

import numpy as np
import torch
import random
import re

import os
import pickle

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


class CrossRegressedCorrelationPerturbed:
    def __init__(self, regression, correlation, crossvalidation_kwargs=None, scrambled_version=None):
        crossvalidation_defaults = dict(train_size=.9, test_size=None)
        crossvalidation_kwargs = {**crossvalidation_defaults, **(crossvalidation_kwargs or {})}
        
        if scrambled_version and re.match("sentenceshuffle_", scrambled_version):
            self.cross_validation = CrossValidationPerturbed(scrambled_version=scrambled_version, **crossvalidation_kwargs)
        else:
            self.cross_validation = CrossValidationPerturbed(**crossvalidation_kwargs)
        self.regression = regression
        self.correlation = correlation
        self.scrambled_version = scrambled_version

    def __call__(self, source_train_emb, source_test_emb, target):
        """"
        source_train_emb = model activations used for training the regression model
        source_test_emb = model activations used for evaluating the regression model (from perturbed sentences)
        NOTE: In the case of the semantic distance benchmarks, we shuffle the source_train_emb directly and ignore the source_test_emb
        """
        if self.scrambled_version and re.match("sentenceshuffle_", self.scrambled_version):
            return self.cross_validation(source_train_emb, source_test_emb, target, scrambled_version=self.scrambled_version, apply=self.apply, aggregate=self.aggregate)
        else:
            return self.cross_validation(source_train_emb, source_test_emb, target, apply=self.apply, aggregate=self.aggregate)

    def apply(self, source_emb_train, target_train, source_emb_test, target_test):
        self.regression.fit(source_emb_train, target_train)
        prediction = self.regression.predict(source_emb_test)
        score = self.correlation(prediction, target_test)
        return score

    def aggregate(self, scores):
        return scores.median(dim='neuroid')


class CrossValidationPerturbed(Transformation):
    """
    Performs multiple splits over a source and target assembly.
    No guarantees are given for data-alignment, use the metadata.
    """

    def __init__(self, *args, split_coord=Split.Defaults.split_coord,
                 stratification_coord=Split.Defaults.stratification_coord, scrambled_version=None, **kwargs):
        self._split_coord = split_coord
        self._stratification_coord = stratification_coord
        self._split = Split(*args, split_coord=split_coord, stratification_coord=stratification_coord, **kwargs)
        self._scrambled_version = scrambled_version

        
    def pipe(self, source_train_emb, source_test_emb, target_assembly, scrambled_version=None):
        # check only for equal values, alignment is given by metadata
        assert sorted(source_train_emb[self._split_coord].values) == sorted(target_assembly[self._split_coord].values)
        cross_validation_values, splits = self._split.build_splits(target_assembly)

        split_scores = []
        for split_iterator, (train_indices, test_indices), done \
                in tqdm(enumerate_done(splits), total=len(splits), desc='cross-validation'):
            train_values, test_values = cross_validation_values[train_indices], cross_validation_values[test_indices]
            train_source = subset(source_train_emb, train_values, dims_must_match=False)
            train_target = subset(target_assembly, train_values, dims_must_match=False)
            assert len(train_source[self._split_coord]) == len(train_target[self._split_coord])
            
            expt = np.unique(target_assembly.experiment.data)[0]
            layer_identifier = np.unique(source_train_emb.layer.data)[0]

            if scrambled_version and scrambled_version.startswith("sentenceshuffle_"):
                print(f"#{scrambled_version}#")
                print(f"EXPERIMENT #{expt}# | LAYER #{layer_identifier}#")

                if not os.getenv('DECONTEXTUALIZED_EMB', '0'):
                    print("Note that contextualization is different than in other contextualized perturbed benchmarks! We're loading contextualized activations for intact sentences and are shuffling them!")

                # STEP 1: Pick out test set (same indices as for original condition)
                print('Getting the activations that would be in the test set for this split in the original benchmark')
                test_source_orig = subset(source_train_emb, test_values, dims_must_match=False)
                # set up perturbed test source assembly
                test_source_initial = test_source_orig.copy(deep=True)
                # get activations
                test_source_orig_array = test_source_orig.values
                test_source_perturbed_array = test_source_initial.values

                # STEP 2: Make sure that the sentences are mismatched with the fMRI data
                print('Making sure that the sentences are mismatched with the fMRI data')
                
                if scrambled_version == "sentenceshuffle_random":
                    
                    # assert all activations for the sentences are different, else leave in place
                    # (some activations are similar in the "drop" layer only because it's the embedding of the final period, combined with sentence length information):
                    check_diff_rows = [(test_source_perturbed_array[i] == test_source_perturbed_array[j]).all() for i in range(np.shape(test_source_perturbed_array)[0]) for j in range(np.shape(test_source_perturbed_array)[0]) if i != j]
                    if True in check_diff_rows:
                        print(f"Not all sentence activations in layer #{layer_identifier}# are different (happens for drop layer), so randomly shuffling once, but not asserting that ALL activations end up in different spots!")
                        np.random.shuffle(test_source_perturbed_array)
                    else:
                        all_different = False
                        attempt = 0
                        while not all_different:
                            attempt += 1
                            if attempt % 100000 == 0:
                                print(f"Attempt number {attempt}", flush=True)
                            for ind, row in enumerate(test_source_perturbed_array):
                                if (row == test_source_orig_array[ind]).all():
                                    all_different = False
                                    break
                                else:
                                    all_different = True
                                    continue
                            # if not all different, shuffle for next attempt
                            if not all_different:
                                np.random.shuffle(test_source_perturbed_array)
                        # once all rows are mismatched, output the test_source_perturbed_array and run checks
                        print(f"All activations successfully mismatched after {attempt} attempts!", flush=True)

                        # CHECKS
                        # check: assert no test activations in the same spot as for teston:original benchmark
                        check_diff_rows = [(test_source_perturbed_array[i] == test_source_orig_array[i]).all() for i in range(len(test_source_orig_array))]
                        assert not True in check_diff_rows
                        print("CHECK 3 OK: no test activations in the same spot as for teston:original benchmark", flush=True)
                    
                    # CHECKS
                    # check: assert shape of test activations same as for teston:original benchmark
                    assert np.shape(test_source_perturbed_array) == np.shape(test_source_orig_array)
                    print("CHECK 1 OK: shape of test activations same as for teston:original benchmark", flush=True)
                    # check 2: assert same set of activations in test set as for teston:original benchmark
                    print(f"Shape of first row: {np.shape(test_source_orig_array[0])}", flush=True)
                    test_source_orig_array_rows = [test_source_orig_array[i] for i in range(len(test_source_orig_array))]
                    print(f"Shape of test_source_orig_array_rows: {np.shape(test_source_orig_array_rows)}", flush=True)
                    for ind, row in enumerate(test_source_perturbed_array):
                        assert np.any(np.all(row == test_source_orig_array_rows, axis=1))
                    print("CHECK 2 OK: assert same set of activations in test set as for teston:original benchmark", flush=True)

                    # set activations as values in test_source Assembly
                    print("Setting shuffled activations as values in test_source assembly!", flush=True)
                    print(np.shape(test_source_perturbed_array))
                    test_source = test_source_orig.copy(deep=True)
                    test_source.values = test_source_perturbed_array
                
                elif scrambled_version in ["sentenceshuffle_passage", "sentenceshuffle_topic"]:
                    
                    # get topicIDs (NOTE: left topicIDs throughout as an identifier, even though this works with passages too!)
                    if scrambled_version == "sentenceshuffle_topic":
                        orig_topic_ids = list(test_source_orig.passage_category.values)
                        print_name = "topic"
                    elif scrambled_version == "sentenceshuffle_passage":
                        orig_topic_ids = list(test_source_orig.passage_index.values)
                        print_name = "passage"
                    else:
                        raise NotImplementedError
                        
                    print(f'{print_name.upper()} IDS:\n{orig_topic_ids}')
                    print(f'Shape of activations:\n{np.shape(test_source_perturbed_array)}')
                    print(f'SENTENCES:\n{list(test_source_orig.sentence.values)}')
                    
                    # zip activations with topicIDs
                    test_source_orig_array_with_topicIDs = list(zip(test_source_orig_array, orig_topic_ids))
                    test_source_perturbed_array_with_topicIDs = list(zip(test_source_perturbed_array, orig_topic_ids))
                    # turn into list of lists to enable assignment
                    test_source_orig_array_with_topicIDs = [list(elm) for elm in test_source_orig_array_with_topicIDs]
                    test_source_perturbed_array_with_topicIDs = [list(elm) for elm in test_source_perturbed_array_with_topicIDs]
                    
                    # count number of sentences belonging to each topic/passage in the current test split
                    from collections import Counter
                    occurrence_dict = Counter(orig_topic_ids)
                    print(f"Counter dictionary of number of sentences from each {print_name} in the test set:")
                    print(occurrence_dict)
                    list_of_single_topic_ids = [k for k in list(occurrence_dict.keys()) if occurrence_dict[k] == 1]
                    print(f"List of {print_name}s of which only one sentence is in the test set:")
                    print(list_of_single_topic_ids)
                    
                    list_of_topic_ids_multiple_sents = [k for k in list(occurrence_dict.keys()) if occurrence_dict[k] > 1]
                    leave_in_place_counter = 0
                    
                    for topicID in list(occurrence_dict.keys()):
                        print(f"******\n{print_name.upper()} #{topicID}#")
                        print(f"Number of sentences in this {print_name}: {occurrence_dict[topicID]}")
                        
                        if topicID in list_of_single_topic_ids:
                            print(f"{print_name.capitalize()} {topicID} only has one sentence in the test set, so leaving the activations in their place!")
                            leave_in_place_counter += 1

                        else:
                            # get activations belonging to the current topic
                            topic_act_indexes = [i for i in range(len(test_source_perturbed_array_with_topicIDs)) if test_source_perturbed_array_with_topicIDs[i][1] == topicID]
                            print(f"Activations for current {print_name} found at indices: {topic_act_indexes}")
                            
                            # get original activation order for this topic
                            original_topic_activations = [test_source_orig_array_with_topicIDs[i][0] for i in topic_act_indexes]
                            shuffled_topic_activations = [test_source_perturbed_array_with_topicIDs[i][0] for i in topic_act_indexes]
                            # before shuffling, the topic activations are the same
                            np.testing.assert_array_equal(shuffled_topic_activations, original_topic_activations)
                            
                            # assert all activations for the sentences are different, else leave in place (happens for drop layer only):
                            
                            check_diff_rows = [(shuffled_topic_activations[i] == shuffled_topic_activations[j]).all() for i in range(np.shape(shuffled_topic_activations)[0]) for j in range(np.shape(shuffled_topic_activations)[0]) if i != j]
#                             assert not True in check_diff_rows
                            if True in check_diff_rows:
                                print(f"Not all sentence activations in {print_name} {topicID} are different (happens for drop layer), so leaving the {len(shuffled_topic_activations)} activations in their place!")
                                leave_in_place_counter += len(shuffled_topic_activations)
                                continue
                                
                            else: # if all activations for the sentences are different
                                all_different_but_same_topic = False
                                attempt = 0
                                while not all_different_but_same_topic:
                                    attempt += 1
                                    if attempt % 1000000 == 0:
                                        print(f"Attempt number {attempt}", flush=True)

                                    # shuffle perturbed activations array
                                    np.random.shuffle(shuffled_topic_activations)

                                    check_diff_rows = [(shuffled_topic_activations[i][0] == original_topic_activations[i][0]).all() for i in range(len(original_topic_activations))]
                                    
#                                     #CONTROL_SEM_DISTANCE_BM# leaves 2 or more sentences from the same ID in place
#                                     if os.getenv('CONTROL_SEM_DISTANCE_BM', '0') == '1': #TODO TAKE OUT AGAIN!
#                                         if attempt % 1000000 == 0:
#                                             print(f'!!! Using control version of the semantic distance benchmark!')
#                                         if occurrence_dict[topicID] > 3:
#                                             if check_diff_rows.count(True) == 2:
#                                                 all_different_but_same_topic = True
#                                             else:
#                                                 all_different_but_same_topic = False
#                                         else:
#                                             print(f'{print_name} #{topicID}# only contains 3 sentences, leaving them all in place! Cannot set only 2 of them, so leaving all 3 in place')
#                                             leave_in_place_counter += 3
#                                             all_different_but_same_topic = True
                                        
#                                     else:
                                    if True in check_diff_rows:
                                        all_different_but_same_topic = False
                                    else:
                                        all_different_but_same_topic = True

#                                 if os.getenv('CONTROL_SEM_DISTANCE_BM', '0') == '1': #TODO TAKE OUT AGAIN!
#                                     print(f"All activations #EXCEPT 2# (if possible) successfully mismatched within {print_name} {topicID} after {attempt} attempts!")
#                                 else:
                                print(f"All activations successfully mismatched within {print_name} {topicID} after {attempt} attempts!")
                                print(f"Setting activations for the current {print_name} to the shuffled activations")
                                # once all are different within the same topic replace activations for current topic in place
                                for (index, replacement) in zip(topic_act_indexes, shuffled_topic_activations):
                                    test_source_perturbed_array_with_topicIDs[index][0] = replacement

#                     if not os.getenv('CONTROL_SEM_DISTANCE_BM', '0'): #TODO TAKE OUT AGAIN!
                        # final check after shuffling within a topic is done for all topics!
                    if (np.corrcoef(test_source_perturbed_array)==1).sum()==np.shape(test_source_perturbed_array)[0]:
                        print("Checking final activations assignment for layers other than drop!")
                        all_different_but_same_topic_final = False
                        for ind, (row, topicID) in enumerate(test_source_perturbed_array_with_topicIDs):
                            if (row == test_source_orig_array_with_topicIDs[ind][0]).all() and topicID not in list_of_single_topic_ids:
                                all_different_but_same_topic_final = False
                                break
                            else:
                                if topicID != test_source_orig_array_with_topicIDs[ind][1] and topicID not in list_of_single_topic_ids:
                                    all_different_but_same_topic_final = False
                                    break
                                else:
                                    all_different_but_same_topic_final = True
                                    continue
                        assert all_different_but_same_topic_final == True
                        print("All activations successfully shuffled where possible!")

                    print(f"!!! LEFT {leave_in_place_counter} our of {len(test_source_perturbed_array_with_topicIDs)} sentence activations in their original spot!")
                    
                    topic_ids_after_shuffling = np.array([elm[1] for elm in test_source_perturbed_array_with_topicIDs])
                    test_source_perturbed_array_after_shuffling = np.array([elm[0] for elm in test_source_perturbed_array_with_topicIDs])
                    # set activations as values in test_source Assembly
                    print("Setting shuffled activations as values in test_source assembly!")
                    test_source = test_source_orig.copy(deep=True)
                    test_source.values = test_source_perturbed_array_after_shuffling


            else: #if not sem. distance benchmark
                test_source = subset(source_test_emb, test_values, dims_must_match=False)

            test_target = subset(target_assembly, test_values, dims_must_match=False)
            assert len(test_source[self._split_coord]) == len(test_target[self._split_coord])
            
#             # CHECK: STORING CV ACTIVATIONS & SPLITS
#             assert np.unique(source_train_emb.layer.data)[0] == np.unique(source_test_emb.layer.data)[0], "Layers are not the same!"
#             layers_of_interest = ['drop']#, 'encoder.h.5', 'encoder.h.44']
            
#             if layer_identifier in layers_of_interest:
#                 store_path = '/om2/user/ckauf/perturbed-neural-nlp/analysis/checks/activations_storage/slurm_job={}/'.format(os.getenv('SLURM_JOB_ID'))
#                 os.makedirs(store_path, exist_ok=True)
#                 train_store_name = 'CrossValidationPerturbed_train_source_expt={}_layer={}_decontextualized={}_splitnr={}_{}.pkl'.format(expt, layer_identifier, os.getenv('DECONTEXTUALIZED_EMB'), split_iterator, os.getenv('SLURM_JOB_ID'))
#                 test_store_name = 'CrossValidationPerturbed_test_source_expt={}_layer={}_decontextualized={}_splitnr={}_{}.pkl'.format(expt, layer_identifier, os.getenv('DECONTEXTUALIZED_EMB'), split_iterator, os.getenv('SLURM_JOB_ID'))
#                 split_train_store_name = 'Splits_TrainIndices_expt={}_layer={}_decontextualized={}_splitnr={}_{}.pkl'.format(expt, layer_identifier, os.getenv('DECONTEXTUALIZED_EMB'), split_iterator, os.getenv('SLURM_JOB_ID'))
#                 split_test_store_name = 'Splits_TestIndices_expt={}_layer={}_decontextualized={}_splitnr={}_{}.pkl'.format(expt, layer_identifier, os.getenv('DECONTEXTUALIZED_EMB'), split_iterator, os.getenv('SLURM_JOB_ID'))

#                 with open(os.path.join(store_path, train_store_name), 'wb') as file:
#                     pickle.dump(train_source.values, file)
#                 with open(os.path.join(store_path, test_store_name), 'wb') as file:
#                     pickle.dump(test_source.values, file)
#                 with open(os.path.join(store_path, split_train_store_name), 'wb') as file:
#                     pickle.dump(train_indices, file)
#                 with open(os.path.join(store_path, split_test_store_name), 'wb') as file:
#                     pickle.dump(test_indices, file)

            split_score = yield from self._get_result(train_source, train_target, test_source, test_target, done=done)
            split_score = split_score.expand_dims('split')
            split_score['split'] = [split_iterator]
            split_scores.append(split_score)

        split_scores = Score.merge(*split_scores)
        yield split_scores

    def aggregate(self, score):
        return self._split.aggregate(score)


class CrossRegressedCorrelation:
    def __init__(self, regression, correlation, crossvalidation_kwargs=None):
        regression = regression
        crossvalidation_defaults = dict(train_size=.9, test_size=None)
        crossvalidation_kwargs = {**crossvalidation_defaults, **(crossvalidation_kwargs or {})}

        self.cross_validation = CrossValidation(**crossvalidation_kwargs)
        self.regression = regression
        self.correlation = correlation

    def __call__(self, source, target):
        return self.cross_validation(source, target, apply=self.apply, aggregate=self.aggregate)

    def apply(self, source_train, target_train, source_test, target_test):
        self.regression.fit(source_train, target_train)
        prediction = self.regression.predict(source_test)
        score = self.correlation(prediction, target_test)
        return score

    def aggregate(self, scores):
        return scores.median(dim='neuroid')


class CrossValidation(Transformation):
    """
    Performs multiple splits over a source and target assembly.
    No guarantees are given for data-alignment, use the metadata.
    """

    def __init__(self, *args, split_coord=Split.Defaults.split_coord,
                 stratification_coord=Split.Defaults.stratification_coord, **kwargs):
        self._logger = logging.getLogger(fullname(self))
        self._split_coord = split_coord
        print(f"Running with split_coord {self._split_coord}!")
        self._stratification_coord = stratification_coord
        print(f"Running with stratification_coord {self._stratification_coord}!")
        self._split = Split(*args, split_coord=split_coord, stratification_coord=stratification_coord, **kwargs)

    def pipe(self, source_assembly, target_assembly):
        # check only for equal values, alignment is given by metadata
        assert sorted(source_assembly[self._split_coord].values) == sorted(target_assembly[self._split_coord].values)
        if self._split.do_stratify:
            assert hasattr(source_assembly, self._stratification_coord)
            assert sorted(source_assembly[self._stratification_coord].values) == \
                   sorted(target_assembly[self._stratification_coord].values)
        cross_validation_values, splits = self._split.build_splits(target_assembly)

        split_scores = []
        for split_iterator, (train_indices, test_indices), done \
                in tqdm(enumerate_done(splits), total=len(splits), desc='cross-validation'): #problem is that splits is [] at this point!
            train_values, test_values = cross_validation_values[train_indices], cross_validation_values[test_indices]
            train_source = subset(source_assembly, train_values, dims_must_match=False)
            train_target = subset(target_assembly, train_values, dims_must_match=False)
            assert len(train_source[self._split_coord]) == len(train_target[self._split_coord])
            test_source = subset(source_assembly, test_values, dims_must_match=False)
            test_target = subset(target_assembly, test_values, dims_must_match=False)
            assert len(test_source[self._split_coord]) == len(test_target[self._split_coord])
            
#             CHECK: STORING CV ACTIVATIONS & SPLITS (TrainPerturbed_TestPerturbed)
#             expt = np.unique(target_assembly.experiment.data)[0]
#             layer_identifier = np.unique(train_source.layer.data)[0]
#             store_path = '/om2/user/ckauf/perturbed-neural-nlp/analysis/checks/activations_storage/slurm_job={}/'.format(os.getenv('SLURM_JOB_ID'))
#             os.makedirs(store_path, exist_ok=True)
#             train_store_name = 'CrossValidation_TrainPTestP_train_source_expt={}_layer={}_decontextualized={}_splitnr={}_{}.pkl'.format(expt, layer_identifier, os.getenv('DECONTEXTUALIZED_EMB'), split_iterator, os.getenv('SLURM_JOB_ID'))
#             test_store_name = 'CrossValidation_TrainPTestP_test_source_expt={}_layer={}_decontextualized={}_splitnr={}_{}.pkl'.format(expt, layer_identifier, os.getenv('DECONTEXTUALIZED_EMB'), split_iterator, os.getenv('SLURM_JOB_ID'))
#             split_train_store_name = 'Splits_TrainIndices_expt={}_layer={}_decontextualized={}_splitnr={}_{}.pkl'.format(expt, layer_identifier, os.getenv('DECONTEXTUALIZED_EMB'), split_iterator, os.getenv('SLURM_JOB_ID'))
#             split_test_store_name = 'Splits_TestIndices_expt={}_layer={}_decontextualized={}_splitnr={}_{}.pkl'.format(expt, layer_identifier, os.getenv('DECONTEXTUALIZED_EMB'), split_iterator, os.getenv('SLURM_JOB_ID'))
            
#             with open(os.path.join(store_path, train_store_name), 'wb') as file:
#                 pickle.dump(train_source.values, file)
#             with open(os.path.join(store_path, test_store_name), 'wb') as file:
#                 pickle.dump(test_source.values, file)
#             with open(os.path.join(store_path, split_train_store_name), 'wb') as file:
#                 pickle.dump(train_indices, file)
#             with open(os.path.join(store_path, split_test_store_name), 'wb') as file:
#                 pickle.dump(test_indices, file)

            split_score = yield from self._get_result(train_source, train_target, test_source, test_target,
                                                      done=done)
            split_score = split_score.expand_dims('split')
            split_score['split'] = [split_iterator]
            split_scores.append(split_score)

        split_scores = Score.merge(*split_scores)
        yield split_scores

    def aggregate(self, score):
        return self._split.aggregate(score)




def enumerate_done(values):
    for i, val in enumerate(values):
        done = i == len(values) - 1
        yield i, val, done

