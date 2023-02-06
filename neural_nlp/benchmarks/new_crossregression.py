
from neural_nlp.benchmarks.new_split import SplitNew as Split #used for new split_by_passage_id functionality
from brainscore.metrics.transformations import Transformation #, Split
from brainio_collection.transform import subset
from brainscore.metrics import Score
from tqdm import tqdm
import logging
from brainscore.utils import fullname

print("I AM USING THE NEW CROSSVALIDATION SCRIPT")

import numpy as np
import torch
import random

import os
import pickle

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


class CrossRegressedCorrelationPerturbed:
    def __init__(self, regression, correlation, crossvalidation_kwargs=None):
        crossvalidation_defaults = dict(train_size=.9, test_size=None)
        crossvalidation_kwargs = {**crossvalidation_defaults, **(crossvalidation_kwargs or {})}

        self.cross_validation = CrossValidationPerturbed(**crossvalidation_kwargs)
        self.regression = regression
        self.correlation = correlation

    def __call__(self, source_train_emb, source_test_emb, target):
        """"
        source_train_emb = model activations used for training the regression model
        source_test_emb = model activations used for evaluating the regression model (from perturbed sentences)
        """
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
                 stratification_coord=Split.Defaults.stratification_coord, **kwargs):
        self._split_coord = split_coord
        self._stratification_coord = stratification_coord
        self._split = Split(*args, split_coord=split_coord, stratification_coord=stratification_coord, **kwargs)

    def pipe(self, source_train_emb, source_test_emb, target_assembly):
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

            if os.getenv('NEW_SEM_DISTANCE_BM', '0') == '1':
                # STEP 1: Pick out test set (same indices as for original condition)
                if not os.getenv('DECONTEXTUALIZED_EMB', '0'):
                    raise NotImplementedError("The new way of doing TrainIntact-TestPerturbed semantic-distance benchmarks"
                                              "is only implemented for decontextualized embeddings! Otherwise the activations"
                                              "differ from the original case in the current setup > check if we want this"
                                              "approach for contextualized benchmarks, too.")
                ## Get the embeddings that would be in the test set for this split in the original benchmark
                test_source_orig = subset(source_train_emb, test_values, dims_must_match=False)
                test_source_orig_rows = [test_source_orig[ind] for ind in range(np.shape(test_source_orig)[0])]
                test_source_acts = []
                for ind, row in enumerate(source_test_emb): #NOTE: we could also just work with the orig here without loading the test_source_embs
                    if np.any(np.all(row == test_source_orig_rows, axis=1)):
                        test_source_acts.append(row)
                test_source_acts_shuffled = np.array(test_source_acts)

                # STEP 2: Make sure that the sentences are mismatched with the fMRI data
                print("THIS ONLY WORKS FOR THE teston:sentenceshuffle_random benchmark RIGHT NOW, if you want this to work for"
                      "teston:sentenceshuffle_topic as well, output the topic_ids as an array in the utils.py script and"
                      "add another condition to the while loop below!")
                all_different = False
                attempt = 0
                while not all_different:
                    print(f"Attempt number {attempt}")
                    for ind, row in enumerate(test_source_acts_shuffled):
                        print(row, test_source_orig[ind])
                        if (row == test_source_orig[ind]).all():
                            print(f'rows at index {ind} are the same, shuffling matrix and retrying\n')
                            all_different = False
                            break
                        else:
                            all_different = True
                            continue
                    attempt += 1
                    np.random.shuffle(test_source_acts_shuffled)
                test_source = test_source_acts_shuffled
                print(test_source)
                # check 1: assert same set of activations in test set as for teston:original benchmark
                assert set([test_source[i] for i in range(np.shape(test_source)[0])]) == \
                       set([test_source_orig[i] for i in range(np.shape(test_source_orig)[0])])
                # check 2: assert shape of test activations same as for teston:original benchmark
                assert np.shape(test_source) == np.shape(test_source_orig)
                # check 3: assert no test activations in the same spot as for teston:original benchmark
                check_diff_rows = [(test_source[i] == test_source_orig[i]).all() for i in range(len(test_orig_acts))]
                assert not True in check_diff_rows

            else:
                test_source = subset(source_test_emb, test_values, dims_must_match=False)

            test_target = subset(target_assembly, test_values, dims_must_match=False)
            assert len(test_source[self._split_coord]) == len(test_target[self._split_coord])
            
            # FIXME: take out again
            print('I AM IN THE FUNCTION THAT STORES THE ACTIVATION SPLITS!')
            expt = np.unique(target_assembly.experiment.data)[0]
            layer_identifier = np.unique(source_train_emb.layer.data)[0]
            
            assert np.unique(source_train_emb.layer.data)[0] == np.unique(source_test_emb.layer.data)[0], "Layers are not the same!"
            layers_of_interest = ['drop', 'encoder.h.5', 'encoder.h.44']
            
            if layer_identifier in layers_of_interest:
                store_path = '/om2/user/ckauf/perturbed-neural-nlp/analysis/checks/activations_storage/slurm_job={}/'.format(os.getenv('SLURM_JOB_ID'))
                os.makedirs(store_path, exist_ok=True)
                train_store_name = 'CrossValidationPerturbed_train_source_expt={}_layer={}_decontextualized={}_splitnr={}_{}.pkl'.format(expt, layer_identifier, os.getenv('DECONTEXTUALIZED_EMB'), split_iterator, os.getenv('SLURM_JOB_ID'))
                test_store_name = 'CrossValidationPerturbed_test_source_expt={}_layer={}_decontextualized={}_splitnr={}_{}.pkl'.format(expt, layer_identifier, os.getenv('DECONTEXTUALIZED_EMB'), split_iterator, os.getenv('SLURM_JOB_ID'))
                split_train_store_name = 'Splits_TrainIndices_expt={}_layer={}_decontextualized={}_splitnr={}_{}.pkl'.format(expt, layer_identifier, os.getenv('DECONTEXTUALIZED_EMB'), split_iterator, os.getenv('SLURM_JOB_ID'))
                split_test_store_name = 'Splits_TestIndices_expt={}_layer={}_decontextualized={}_splitnr={}_{}.pkl'.format(expt, layer_identifier, os.getenv('DECONTEXTUALIZED_EMB'), split_iterator, os.getenv('SLURM_JOB_ID'))

                with open(os.path.join(store_path, train_store_name), 'wb') as file:
                    pickle.dump(train_source.values, file)
                with open(os.path.join(store_path, test_store_name), 'wb') as file:
                    pickle.dump(test_source.values, file)
                with open(os.path.join(store_path, split_train_store_name), 'wb') as file:
                    pickle.dump(train_indices, file)
                with open(os.path.join(store_path, split_test_store_name), 'wb') as file:
                    pickle.dump(test_indices, file)

            split_score = yield from self._get_result(train_source, train_target, test_source, test_target,
                                                      done=done)
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
        self._stratification_coord = stratification_coord
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
                in tqdm(enumerate_done(splits), total=len(splits), desc='cross-validation'):
            train_values, test_values = cross_validation_values[train_indices], cross_validation_values[test_indices]
            train_source = subset(source_assembly, train_values, dims_must_match=False)
            train_target = subset(target_assembly, train_values, dims_must_match=False)
            assert len(train_source[self._split_coord]) == len(train_target[self._split_coord])
            test_source = subset(source_assembly, test_values, dims_must_match=False)
            test_target = subset(target_assembly, test_values, dims_must_match=False)
            assert len(test_source[self._split_coord]) == len(test_target[self._split_coord])
            
            # FIXME: take out again
            print('I AM IN THE FUNCTION THAT STORES THE ACTIVATION SPLITS!')
            expt = np.unique(target_assembly.experiment.data)[0]
            layer_identifier = np.unique(train_source.layer.data)[0]
            store_path = '/om2/user/ckauf/perturbed-neural-nlp/analysis/checks/activations_storage/slurm_job={}/'.format(os.getenv('SLURM_JOB_ID'))
            os.makedirs(store_path, exist_ok=True)
            train_store_name = 'CrossValidation_TrainPTestP_train_source_expt={}_layer={}_decontextualized={}_splitnr={}_{}.pkl'.format(expt, layer_identifier, os.getenv('DECONTEXTUALIZED_EMB'), split_iterator, os.getenv('SLURM_JOB_ID'))
            test_store_name = 'CrossValidation_TrainPTestP_test_source_expt={}_layer={}_decontextualized={}_splitnr={}_{}.pkl'.format(expt, layer_identifier, os.getenv('DECONTEXTUALIZED_EMB'), split_iterator, os.getenv('SLURM_JOB_ID'))
            split_train_store_name = 'Splits_TrainIndices_expt={}_layer={}_decontextualized={}_splitnr={}_{}.pkl'.format(expt, layer_identifier, os.getenv('DECONTEXTUALIZED_EMB'), split_iterator, os.getenv('SLURM_JOB_ID'))
            split_test_store_name = 'Splits_TestIndices_expt={}_layer={}_decontextualized={}_splitnr={}_{}.pkl'.format(expt, layer_identifier, os.getenv('DECONTEXTUALIZED_EMB'), split_iterator, os.getenv('SLURM_JOB_ID'))
            
            with open(os.path.join(store_path, train_store_name), 'wb') as file:
                pickle.dump(train_source.values, file)
            with open(os.path.join(store_path, test_store_name), 'wb') as file:
                pickle.dump(test_source.values, file)
            with open(os.path.join(store_path, split_train_store_name), 'wb') as file:
                pickle.dump(train_indices, file)
            with open(os.path.join(store_path, split_test_store_name), 'wb') as file:
                pickle.dump(test_indices, file)

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

