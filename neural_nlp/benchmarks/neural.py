"""
Neural benchmarks to probe match of model internals against human internals.
"""
import warnings

import itertools
import logging
import numpy as np
from brainio_base.assemblies import DataAssembly, walk_coords, merge_data_arrays, array_is_element
from numpy.random.mtrand import RandomState
from scipy.stats import median_absolute_deviation

import xarray as xr

from pathlib import Path
import os
import pandas as pd

from brainscore.benchmarks import Benchmark
from brainscore.metrics import Score
from brainscore.metrics.rdm import RDM, RDMSimilarity, RDMCrossValidated
from brainscore.metrics.regression import linear_regression, pearsonr_correlation#, CrossRegressedCorrelation
from brainscore.metrics.transformations import CartesianProduct, apply_aggregate#CrossValidation
from brainscore.utils import LazyLoad
from neural_nlp.benchmarks.ceiling import ExtrapolationCeiling, HoldoutSubjectCeiling
from neural_nlp.benchmarks.s3 import load_s3
from neural_nlp.neural_data.fmri import load_voxels, load_rdm_sentences, \
    load_Pereira2018_Blank
from neural_nlp.stimuli import load_stimuli, StimulusSet
from neural_nlp.utils import ordered_set
from result_caching import store
from neural_nlp.benchmarks.new_crossregression import CrossRegressedCorrelationPerturbed, CrossRegressedCorrelation, CrossValidation

ressources_dir = os.path.abspath("../../ressources")

_logger = logging.getLogger(__name__)

import pickle

import numpy as np
import torch
import random

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


class Invert:
    def __init__(self, metric):
        self._metric = metric

    def __call__(self, source, target):
        source, target = target, source
        return self._metric(source, target)



class RDMSimilarityCrossValidated:
    # adapted from
    # https://github.com/brain-score/brain-score/blob/3d59d7a841fca63a5d346e599143f547560b5082/brainscore/metrics/rdm.py#L8

    class LeaveOneOutWrapper:
        def __init__(self, metric):
            self._metric = metric

        def __call__(self, train_source, train_target, test_source, test_target):
            # compare assemblies for a single split. we ignore the 10% train ("leave-one-out") and only use test.
            score = self._metric(test_source, test_target)
            return DataAssembly(score)

    def __init__(self, stimulus_coord='stimulus_sentence'):
        self._rdm = RDM()
        self._similarity = RDMSimilarity(comparison_coord=stimulus_coord)
        self._cross_validation = CrossValidation(test_size=.9,  # leave 10% out
                                                 split_coord=stimulus_coord, stratification_coord=None)

    def __call__(self, model_activations, target_rdm):
        model_activations = align(model_activations, target_rdm, on='stimulus_sentence')
        model_rdm = self._rdm(model_activations)
        values = model_rdm.values
        if np.isnan(values.flatten()).any():
            warnings.warn(f"{np.isnan(values.flatten()).sum()} nan values found in model rdm - setting to 0")
            values[np.isnan(values)] = 0
            model_rdm = type(model_rdm)(values, coords={coord: (dims, vals) for coord, dims, vals in
                                                        walk_coords(model_rdm)}, dims=model_rdm.dims)
        leave_one_out = self.LeaveOneOutWrapper(self._similarity)
        # multi-dimensional coords with repeated dimensions not yet supported in CrossValidation
        drop_coords = [coord for coord, dims, value in walk_coords(target_rdm) if dims == ('stimulus', 'stimulus')]
        target_rdm = target_rdm.drop(drop_coords)
        return self._cross_validation(model_rdm, target_rdm, apply=leave_one_out)


def align(source, target, on):
    source_values, target_values = source[on].values.tolist(), target[on].values
    indices = [source_values.index(value) for value in target_values]
    assert len(source[on].dims) == 1, "multi-dimensional coordinates not implemented"
    dim = source[on].dims[0]
    dim_indices = {_dim: slice(None) if _dim != dim else indices for _dim in source.dims}
    aligned = source.isel(**dim_indices)
    return aligned


class _PereiraBenchmark(Benchmark):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4
    """

    def __init__(self, identifier, metric, data_version='base'):
        self._identifier = identifier
        self._data_version = data_version
        self._target_assembly = LazyLoad(lambda: self._load_assembly(version=self._data_version))

        stimuli = self._target_assembly.attrs['stimulus_set']
        stimuli.name = self._target_assembly.attrs['stimulus_set_name']

        if os.getenv('DECONTEXTUALIZED_EMB', '0') == '1':
            stimuli.name += ",emb_context=Sentence"
        elif os.getenv('PAPER_GROUPING', '0') == '1':
            stimuli.name += ",emb_context=Topic"
        else:
            stimuli.name += ",emb_context=Passage"

        self._target_assembly.attrs['stimulus_set'] = stimuli
        self._target_assembly.attrs['stimulus_set_name'] = stimuli.name #CK 2021-08-05, doesn't get reset otherwise as "Pereira2018" is stores as stimulus_set_name in the stored assembly
        _logger.debug(f"THIS IS THE STIMULUS SET NAME: {self._target_assembly.attrs['stimulus_set'].name}") #e.g., Stimulus set name: Pereira2018-Original-lasttoken

        self._single_metric = metric
        print(self._single_metric)
        self._ceiler = self.PereiraExtrapolationCeiling(subject_column='subject', num_bootstraps=100)
        self._cross = CartesianProduct(dividers=['experiment', 'atlas'])


    @property
    def identifier(self):
        return self._identifier #no need to change the identifier since the store name is set in __main__

    def _metric(self, source_assembly, target_assembly):
        """ for ceiling compute """
        cross_scores = self._cross(target_assembly, apply=
        lambda cross_assembly: self._apply_cross(source_assembly, cross_assembly))
        score = self._average_cross_scores(cross_scores)
        return score

    def _average_cross_scores(self, cross_scores):
        return cross_scores.mean(['experiment', 'atlas'])

    @load_s3(key='Pereira2018')
    def _load_assembly(self, version='base'):
        assembly = load_Pereira2018_Blank(version=version)
        assembly = assembly.sel(atlas_selection_lower=90)
        assembly = assembly[{'neuroid': [filter_strategy in [np.nan, 'HminusE', 'FIXminusH']
                                         for filter_strategy in assembly['filter_strategy'].values]}]
        return assembly

    def __call__(self, candidate):
        stimulus_set = self._target_assembly.attrs['stimulus_set']

        if os.getenv('PAPER_GROUPING', '0') == '1':
            model_activations = listen_to(candidate, stimulus_set)  # in this case using default: reset_column='story'
        elif os.getenv('DECONTEXTUALIZED_EMB', '0') == '1':
            model_activations = listen_to(candidate, stimulus_set, reset_column='stimulus_id')
        else:
            stimulus_set.loc[:, 'passage_id'] = stimulus_set['experiment'] + stimulus_set['passage_index'].astype(str)
            model_activations = listen_to(candidate, stimulus_set, reset_column='passage_id')

        assert set(model_activations['stimulus_id'].values) == set(self._target_assembly['stimulus_id'].values)

        _logger.info('Scoring across experiments & atlases')
        cross_scores = self._cross(self._target_assembly,
                                   apply=lambda cross_assembly: self._apply_cross(model_activations, cross_assembly))
        raw_scores = cross_scores.raw
        raw_neuroids = apply_aggregate(lambda values: values.mean('split').mean('experiment'), raw_scores)

        # normally we would ceil every single neuroid here. To estimate the strongest ceiling possible (i.e. make it as
        # hard as possible on the models), we used experiment-overlapping neuroids from as many subjects as possible
        # which means some neuroids got excluded. Since median(r/c) is the same as median(r)/median(c), we just
        # normalize the neuroid aggregate by the overall ceiling aggregate.
        # Additionally, the Pereira data also has voxels from DMN, visual etc. but we care about language here.
        language_neuroids = raw_neuroids.sel(atlas='language', _apply_raw=False)
        score = aggregate_ceiling(language_neuroids, ceiling=self.ceiling, subject_column='subject')
        return score

    def _apply_cross(self, source_assembly, cross_assembly):
        cross_assembly = cross_assembly.dropna('neuroid')  # some subjects have only done one experiment
        source_assembly = source_assembly.dropna('neuroid')  # only relevant when running audio-visual self as "model"
        assert len(cross_assembly['presentation']) in [243, 384]
        assert not np.isnan(cross_assembly).any()
        source_assembly = source_assembly[{'presentation': [stimulus_id in cross_assembly['stimulus_id'].values
                                                            for stimulus_id in source_assembly['stimulus_id'].values]}]
        return self._single_metric(source_assembly, cross_assembly)

    @property
    def ceiling(self):
        return self._ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)

    class PereiraExtrapolationCeiling(ExtrapolationCeiling):
        def __init__(self, subject_column, *args, **kwargs):
            super(_PereiraBenchmark.PereiraExtrapolationCeiling, self).__init__(
                subject_column, *args, **kwargs)
            self._num_subsamples = 10
            self.holdout_ceiling = _PereiraBenchmark.PereiraHoldoutSubjectCeiling(subject_column=subject_column)
            self._rng = RandomState(0)

        def iterate_subsets(self, assembly, num_subjects):
            # cross experiment to obtain more subjects to extrapolate.
            # don't worry about atlases here, cross-metric will take care of it.
            experiments = set(assembly['experiment'].values)
            for experiment in sorted(experiments):
                experiment_assembly = assembly[{'presentation': [
                    experiment_value == experiment for experiment_value in assembly['experiment'].values]}]
                experiment_assembly = experiment_assembly.dropna('neuroid')  # drop subjects that haven't done this exp
                if len(set(experiment_assembly[self.subject_column].values)) < num_subjects:
                    continue  # not enough subjects
                for sub_subjects in self._random_combinations(
                        subjects=set(experiment_assembly[self.subject_column].values),
                        num_subjects=num_subjects, choice=self._num_subsamples, rng=self._rng):
                    sub_assembly = assembly[{'neuroid': [subject in sub_subjects
                                                         for subject in assembly[self.subject_column].values]}]
                    yield {self.subject_column: sub_subjects, 'experiment': experiment}, sub_assembly

        def _random_combinations(self, subjects, num_subjects, choice, rng):
            # following https://stackoverflow.com/a/55929159/2225200. Also see similar method in `behavioral.py`.
            subjects = np.array(list(subjects))
            combinations = set()
            while len(combinations) < choice:
                elements = rng.choice(subjects, size=num_subjects, replace=False)
                combinations.add(tuple(elements))
            return combinations

        def extrapolate(self, ceilings):
            ceiling = super(_PereiraBenchmark.PereiraExtrapolationCeiling, self).extrapolate(ceilings)
            # compute aggregate ceiling only for language neuroids
            neuroid_ceilings = ceiling.raw
            language_ceilings = neuroid_ceilings.sel(atlas='language')
            ceiling = self.aggregate_neuroid_ceilings(language_ceilings)
            ceiling.attrs['raw'] = neuroid_ceilings  # reset to all neuroids
            return ceiling

        def fit(self, subject_subsamples, bootstrapped_scores):
            valid = ~np.isnan(bootstrapped_scores)
            if sum(valid) < 1:
                raise RuntimeError("No valid scores in sample")
            return super(_PereiraBenchmark.PereiraExtrapolationCeiling, self).fit(
                np.array(subject_subsamples)[valid], np.array(bootstrapped_scores)[valid])

        def post_process(self, scores):
            scores = apply_aggregate(lambda values: values.mean('sub_experiment').mean('experiment'), scores)
            return scores

    class PereiraHoldoutSubjectCeiling(HoldoutSubjectCeiling):
        def __init__(self, *args, **kwargs):
            super(_PereiraBenchmark.PereiraHoldoutSubjectCeiling, self).__init__(*args, **kwargs)
            self._rng = RandomState(0)
            self._num_bootstraps = 5

        def get_subject_iterations(self, subjects):
            # use only a subset of subjects
            return self._rng.choice(list(subjects), size=self._num_bootstraps)


def listen_to(candidate, stimulus_set, reset_column='story', average_sentence=True):
    """
    Pass a `stimulus_set` through a model `candidate`.
    Operates on a sentence-based `stimulus_set`.
    """
    activations = []
    for story in ordered_set(stimulus_set[reset_column].values):
        story_stimuli = stimulus_set[stimulus_set[reset_column] == story]
        story_stimuli.name = f"{stimulus_set.name}-{story}"
        story_activations = candidate(stimuli=story_stimuli, average_sentence=average_sentence)
        # if np.sum(np.isnan(story_activations)) > 0:
        #     print(f'Story with nans: {story_stimuli} and number of nans: {np.sum(np.isnan(story_activations))}')
        #     story_activations = np.nan_to_num(story_activations, nan=0.0)
        activations.append(story_activations)
    # model_activations = merge_data_arrays(activations) #old
    model_activations = xr.concat(activations, dim='presentation')
    # merging does not maintain stimulus order. the following orders again
    idx = [model_activations['stimulus_id'].values.tolist().index(stimulus_id) for stimulus_id in
           itertools.chain.from_iterable(s['stimulus_id'].values for s in activations)]
    assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
    model_activations = model_activations[{'presentation': idx}]
    return model_activations


def read_words(candidate, stimulus_set, reset_column='sentence_id', copy_columns=(), average_sentence=False):
    """
    Pass a `stimulus_set` through a model `candidate`.
    In contrast to the `listen_to` function, this function operates on a word-based `stimulus_set`.
    """
    # Input: stimulus_set = pandas df, col 1 with sentence ID and 2nd col as word.
    activations = []
    for i, reset_id in enumerate(ordered_set(stimulus_set[reset_column].values)):
        part_stimuli = stimulus_set[stimulus_set[reset_column] == reset_id]
        # stimulus_ids = part_stimuli['stimulus_id']
        sentence_stimuli = StimulusSet({'sentence': ' '.join(part_stimuli['word']),
                                        reset_column: list(set(part_stimuli[reset_column]))})
        sentence_stimuli.name = f"{stimulus_set.name}-{reset_id}"
        sentence_activations = candidate(stimuli=sentence_stimuli, average_sentence=average_sentence)
        for column in copy_columns:
            sentence_activations[column] = ('presentation', part_stimuli[column])
        activations.append(sentence_activations)
    model_activations = merge_data_arrays(activations)
    # merging does not maintain stimulus order. the following orders again
    idx = [model_activations['stimulus_id'].values.tolist().index(stimulus_id) for stimulus_id in
           itertools.chain.from_iterable(s['stimulus_id'].values for s in activations)]
    assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
    model_activations = model_activations[{'presentation': idx}]

    return model_activations


class _PereiraSubjectWise(_PereiraBenchmark):
    def __init__(self, **kwargs):
        super(_PereiraSubjectWise, self).__init__(**kwargs)
        self._cross = CartesianProduct(dividers=['experiment', 'atlas', 'subject'])
        self._ceiler = self.PereiraSubjectWiseExtrapolationCeiling(
            extrapolation_dimension='subject', subject_column='subject', num_bootstraps=self._ceiler.num_bootstraps)

    def _apply_cross(self, source_assembly, cross_assembly):
        # some subjects have only done one experiment which leads to nans
        cross_assembly = cross_assembly.dropna('neuroid')
        if len(cross_assembly['neuroid']) == 0:
            return Score([np.nan, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        return super(_PereiraSubjectWise, self)._apply_cross(
            source_assembly=source_assembly, cross_assembly=cross_assembly)

    def _average_cross_scores(self, cross_scores):
        return super(_PereiraSubjectWise, self)._average_cross_scores(cross_scores).median('subject')

    class PereiraSubjectWiseExtrapolationCeiling(_PereiraBenchmark.PereiraExtrapolationCeiling):
        def post_process(self, scores):
            return scores.mean('sub_experiment').sel(aggregation='center')

        def extrapolate(self, ceilings):
            # skip parent implementation, go straight to parent's parent
            return super(_PereiraBenchmark.PereiraExtrapolationCeiling, self).extrapolate(ceilings)


class PereiraDecoding(_PereiraSubjectWise):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4?fbclid=IwAR0W7EZrnIFFO1kvANgeOEICaoDG5fhmdHipazy6n-APUJ6lMY98PkvuTyU
    """

    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord=None))
        metric = Invert(metric)
        super(PereiraDecoding, self).__init__(metric=metric, **kwargs)


class PereiraRDM(_PereiraSubjectWise):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4?fbclid=IwAR0W7EZrnIFFO1kvANgeOEICaoDG5fhmdHipazy6n-APUJ6lMY98PkvuTyU
    """

    def __init__(self, **kwargs):
        metric = RDMCrossValidated(
            comparison_coord='stimulus_id',
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord=None, splits=5,
                                        kfold=True, test_size=None))
        super(PereiraRDM, self).__init__(metric=metric, **kwargs)

    @property
    @load_s3(key='Pereira2018-rdm-ceiling')
    def ceiling(self):
        return super(PereiraRDM, self).ceiling


# class PereiraCKA(_PereiraSubjectWise):
#     """
#     data source:
#         Pereira et al., nature communications 2018
#         https://www.nature.com/articles/s41467-018-03068-4?fbclid=IwAR0W7EZrnIFFO1kvANgeOEICaoDG5fhmdHipazy6n-APUJ6lMY98PkvuTyU
#     """
#
#     def __init__(self, **kwargs):
#         metric = CKACrossValidated(
#             comparison_coord='stimulus_id',
#             crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord=None, splits=5,
#                                         kfold=True, test_size=None))
#         super(PereiraCKA, self).__init__(metric=metric, **kwargs)
#
#     @property
#     def ceiling(self):
#         return super(_PereiraSubjectWise, self).ceiling


def aggregate(score, combine_layers=True):
    if hasattr(score, 'experiment') and score['experiment'].ndim > 0:
        score = score.mean('experiment')
    if hasattr(score, 'atlas') and score['atlas'].ndim > 0:
        score = score.mean('atlas')
    if hasattr(score, 'layer') and score['layer'].ndim > 0 and combine_layers:
        score_core = score.sel(aggregation='center') if hasattr(score, 'aggregation') else score
        max_score = score_core.max()
        max_score = score[{'layer': (score_core == max_score).values}]
        if len(max_score['layer']) > 1:  # multiple layers achieved exactly the same score
            layer_index = max_score['layer'].values[0].tolist().index(max_score['layer'].values[0])  # choose first one
            max_score = max_score.isel(layer=[layer_index])
        max_score = max_score.squeeze('layer', drop=True)
        max_score.attrs['raw'] = score.copy()
        score = max_score
    return score


def ceil_neuroids(raw_neuroids, ceiling, subject_column='subject'):
    ceiled_neuroids = consistency_neuroids(raw_neuroids, ceiling.raw)
    ceiled_neuroids.attrs['raw'] = raw_neuroids
    ceiled_neuroids.attrs['ceiling'] = ceiling.raw
    score = aggregate_neuroid_scores(ceiled_neuroids, subject_column)
    score.attrs['ceiling'] = ceiling
    score.attrs['description'] = "per-neuroid ceiling-normalized score"
    return score


def aggregate_neuroid_scores(neuroid_scores, subject_column):
    subject_scores = neuroid_scores.groupby(subject_column).median()
    center = subject_scores.median(subject_column)
    subject_values = np.nan_to_num(subject_scores.values, nan=0)  # mad cannot deal with all-nan in one axis, treat as 0
    subject_axis = subject_scores.dims.index(subject_scores[subject_column].dims[0])
    error = median_absolute_deviation(subject_values, axis=subject_axis)
    score = Score([center, error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
    score.attrs['raw'] = neuroid_scores
    score.attrs['description'] = "score aggregated by taking median of neuroids per subject, " \
                                 "then median of subject scores"
    return score


def consistency_neuroids(neuroids, ceiling_neuroids):
    # assert set(neuroids['neuroid_id'].values) == set(ceiling_neuroids['neuroid_id'].values)
    if 'neuroid_id' in ceiling_neuroids.dims:
        assert set(neuroids['neuroid_id'].values) == set(ceiling_neuroids['neuroid_id'].values)
    elif 'neuroid' in ceiling_neuroids.dims:
        assert set(neuroids['neuroid_id'].values) == set(ceiling_neuroids['neuroid'].values)

    ceiling_neuroids = ceiling_neuroids[{'neuroid': [neuroids['neuroid_id'].values.tolist().index(neuroid_id)
                                                     for neuroid_id in neuroids['neuroid_id'].values]}]  # align
    ceiling_neuroids = ceiling_neuroids.sel(aggregation='center')
    values = consistency(neuroids.values, ceiling_neuroids.values)
    neuroids = type(neuroids)(values, coords={coord: (dims, values) for coord, dims, values in walk_coords(neuroids)},
                              dims=neuroids.dims)
    return neuroids


def aggregate_ceiling(neuroid_scores, ceiling, subject_column='subject'):
    aggregate_raw = aggregate_neuroid_scores(neuroid_scores, subject_column=subject_column)
    score = consistency(aggregate_raw, ceiling.sel(aggregation='center'))
    score.attrs['raw'] = aggregate_raw
    score.attrs['ceiling'] = ceiling
    score.attrs['description'] = "ceiling-normalized score"
    return score


def consistency(score, ceiling):
    return score / ceiling



class _PereiraBenchmarkScrambled(Benchmark):

    def __init__(self, identifier, metric, scrambled_version, data_version='base'):
        self._identifier = identifier
        self._data_version = data_version
        self._target_assembly = LazyLoad(lambda: self._load_assembly(version=self._data_version))

        scrambled_data_dir = os.path.join(ressources_dir, "scrambled_stimuli_dfs/")

        STIMULI_TO_PKL_MAP = {'Original': os.path.join(scrambled_data_dir, 'stimuli_Original.pkl'),
                              # control conditions
                              'length-control': os.path.join(scrambled_data_dir, 'stimuli_length_control.pkl'),
                              # scrambling | word order manipulations
                              'Scr1': os.path.join(scrambled_data_dir, 'stimuli_Scr1.pkl'),
                              'Scr3': os.path.join(scrambled_data_dir, 'stimuli_Scr3.pkl'),
                              'Scr5': os.path.join(scrambled_data_dir, 'stimuli_Scr5.pkl'),
                              'Scr7': os.path.join(scrambled_data_dir, 'stimuli_Scr7.pkl'),
                              'backward': os.path.join(scrambled_data_dir, 'stimuli_backward.pkl'),
                              'random-wl': os.path.join(scrambled_data_dir, 'stimuli_random.pkl'),
                              'lowPMI': os.path.join(scrambled_data_dir, 'stimuli_lowPMI.pkl'),
                              'lowPMI-random': os.path.join(scrambled_data_dir, 'stimuli_lowPMI_random.pkl'),
                              # perturbation | information loss manipulations
                              'contentwords': os.path.join(scrambled_data_dir, 'stimuli_contentwords.pkl'),
                              'nouns': os.path.join(scrambled_data_dir, 'stimuli_nouns.pkl'),
                              'random-nouns': os.path.join(scrambled_data_dir, 'stimuli_randomnouns.pkl'),
                              'verbs': os.path.join(scrambled_data_dir, 'stimuli_verbs.pkl'),
                              'nounsverbs': os.path.join(scrambled_data_dir, 'stimuli_nounsverbs.pkl'),
                              'nounsverbsadj': os.path.join(scrambled_data_dir, 'stimuli_nounsverbsadj.pkl'),
                              'functionwords': os.path.join(scrambled_data_dir, 'stimuli_functionwords.pkl'),
                              # perturbation | sentence meaning manipulations
                              'sentenceshuffle_random': os.path.join(scrambled_data_dir, 'stimuli_sentenceshuffle-random.pkl'),
                              'sentenceshuffle_passage': os.path.join(scrambled_data_dir, 'stimuli_sentenceshuffle-withinpassage.pkl'),
                              'sentenceshuffle_topic': os.path.join(scrambled_data_dir, 'stimuli_sentenceshuffle-withintopic.pkl'),
                              'chatgpt' : os.path.join(scrambled_data_dir, 'stimuli_chatGPT.pkl')
                              }


        for key in STIMULI_TO_PKL_MAP.keys():
            if scrambled_version == key:
                stimuli = pd.read_pickle(STIMULI_TO_PKL_MAP[key])
                if os.getenv('AVG_TOKEN_TRANSFORMERS', '0') == '1':
                    stimuli.name = f"Pereira2018-{scrambled_version}-avgtoken" #added this
                else:
                    stimuli.name = f"Pereira2018-{scrambled_version}-lasttoken"  # added this

                if os.getenv('DECONTEXTUALIZED_EMB', '0') == '1':
                    stimuli.name += ",emb_context=Sentence"
                elif os.getenv('PAPER_GROUPING', '0') == '1':
                    stimuli.name += ",emb_context=Topic"
                else:
                    stimuli.name += ",emb_context=Passage"

        self._target_assembly.attrs['stimulus_set'] = stimuli
        self._target_assembly.attrs['stimulus_set_name'] = stimuli.name

        self._single_metric = metric
        self._ceiler = self.PereiraExtrapolationCeiling(subject_column='subject', num_bootstraps=100)
        self._cross = CartesianProduct(dividers=['experiment', 'atlas'])

    @property
    def identifier(self):
        return self._identifier

    def _metric(self, source_assembly, target_assembly):
        cross_scores = self._cross(target_assembly, apply=
        lambda cross_assembly: self._apply_cross(source_assembly, cross_assembly))
        score = self._average_cross_scores(cross_scores)
        return score

    def _average_cross_scores(self, cross_scores):
        return cross_scores.mean(['experiment', 'atlas'])

    @load_s3(key='Pereira2018')
    def _load_assembly(self, version='base'):
        assembly = load_Pereira2018_scrambled(version=version) #Note that this is actually superfluous right now.
        #This could also be load_Pereira2018_Blank as this function isn't called; Instead, we load the assembly from s3. Note that if we comment out line 521 ("@load_s3(key='Pereira2018')"), then we get a key error ('z') and 0 assemblies are being tried to be merged.
        assembly = assembly.sel(atlas_selection_lower=90)
        assembly = assembly[{'neuroid': [filter_strategy in [np.nan, 'HminusE', 'FIXminusH']
                                         for filter_strategy in assembly['filter_strategy'].values]}]
        return assembly

    def __call__(self, candidate):
        stimulus_set = self._target_assembly.attrs['stimulus_set']

        if os.getenv('PAPER_GROUPING', '0') == '1':
            model_activations = listen_to(candidate, stimulus_set) #this resets with column story/topic
        elif os.getenv('DECONTEXTUALIZED_EMB', '0') == '1':
            model_activations = listen_to(candidate, stimulus_set, reset_column='stimulus_id')
        else:
            stimulus_set.loc[:, 'passage_id'] = stimulus_set['experiment'] + stimulus_set['passage_index'].astype(str)
            model_activations = listen_to(candidate, stimulus_set, reset_column='passage_id')

        assert set(model_activations['stimulus_id'].values) == set(self._target_assembly['stimulus_id'].values)

        _logger.info('Scoring across experiments & atlases')
        cross_scores = self._cross(self._target_assembly, apply=
        lambda cross_assembly: self._apply_cross(model_activations, cross_assembly))
        raw_scores = cross_scores.raw
        raw_neuroids = apply_aggregate(lambda values: values.mean('split').mean('experiment'), raw_scores)

        # normally we would ceil every single neuroid here. To estimate the strongest ceiling possible (i.e. make it as
        # hard as possible on the models), we used experiment-overlapping neuroids from as many subjects as possible
        # which means some neuroids got excluded. Since median(r/c) is the same as median(r)/median(c), we just
        # normalize the neuroid aggregate by the overall ceiling aggregate.
        # Additionally, the Pereira data also has voxels from DMN, visual etc. but we care about language here.
        language_neuroids = raw_neuroids.sel(atlas='language', _apply_raw=False)
        score = aggregate_ceiling(language_neuroids, ceiling=self.ceiling, subject_column='subject')
        return score

    def _apply_cross(self, source_assembly, cross_assembly):
        cross_assembly = cross_assembly.dropna('neuroid')  # some subjects have only done one experiment
        source_assembly = source_assembly.dropna('neuroid')  # only relevant when running audio-visual self as "model"
        assert len(cross_assembly['presentation']) in [243, 384]
        assert not np.isnan(cross_assembly).any()
        source_assembly = source_assembly[{'presentation': [stimulus_id in cross_assembly['stimulus_id'].values
                                                            for stimulus_id in source_assembly['stimulus_id'].values]}]
        return self._single_metric(source_assembly, cross_assembly)

    @property
    def ceiling(self):
        return self._ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)

    class PereiraExtrapolationCeiling(ExtrapolationCeiling):
        def __init__(self, subject_column, *args, **kwargs):
            super(_PereiraBenchmarkScrambled.PereiraExtrapolationCeiling, self).__init__(
                subject_column, *args, **kwargs)
            self._num_subsamples = 10
            self.holdout_ceiling = _PereiraBenchmarkScrambled.PereiraHoldoutSubjectCeiling(subject_column=subject_column)
            self._rng = RandomState(0)

        def iterate_subsets(self, assembly, num_subjects):
            # cross experiment to obtain more subjects to extrapolate.
            # don't worry about atlases here, cross-metric will take care of it.
            experiments = set(assembly['experiment'].values)
            for experiment in sorted(experiments):
                experiment_assembly = assembly[{'presentation': [
                    experiment_value == experiment for experiment_value in assembly['experiment'].values]}]
                experiment_assembly = experiment_assembly.dropna('neuroid')  # drop subjects that haven't done this exp
                if len(set(experiment_assembly[self.subject_column].values)) < num_subjects:
                    continue  # not enough subjects
                for sub_subjects in self._random_combinations(
                        subjects=set(experiment_assembly[self.subject_column].values),
                        num_subjects=num_subjects, choice=self._num_subsamples, rng=self._rng):
                    sub_assembly = assembly[{'neuroid': [subject in sub_subjects
                                                         for subject in assembly[self.subject_column].values]}]
                    yield {self.subject_column: sub_subjects, 'experiment': experiment}, sub_assembly

        def _random_combinations(self, subjects, num_subjects, choice, rng):
            # following https://stackoverflow.com/a/55929159/2225200. Also see similar method in `behavioral.py`.
            subjects = np.array(list(subjects))
            combinations = set()
            while len(combinations) < choice:
                elements = rng.choice(subjects, size=num_subjects, replace=False)
                combinations.add(tuple(elements))
            return combinations

        def extrapolate(self, ceilings):
            ceiling = super(_PereiraBenchmarkScrambled.PereiraExtrapolationCeiling, self).extrapolate(ceilings)
            # compute aggregate ceiling only for language neuroids
            neuroid_ceilings = ceiling.raw
            language_ceilings = neuroid_ceilings.sel(atlas='language')
            ceiling = self.aggregate_neuroid_ceilings(language_ceilings)
            ceiling.attrs['raw'] = neuroid_ceilings  # reset to all neuroids
            return ceiling

        def fit(self, subject_subsamples, bootstrapped_scores):
            valid = ~np.isnan(bootstrapped_scores)
            if sum(valid) < 1:
                raise RuntimeError("No valid scores in sample")
            return super(_PereiraBenchmarkScrambled.PereiraExtrapolationCeiling, self).fit(
                np.array(subject_subsamples)[valid], np.array(bootstrapped_scores)[valid])

        def post_process(self, scores):
            scores = apply_aggregate(lambda values: values.mean('sub_experiment').mean('experiment'), scores)
            return scores

    class PereiraHoldoutSubjectCeiling(HoldoutSubjectCeiling):
        def __init__(self, *args, **kwargs):
            super(_PereiraBenchmarkScrambled.PereiraHoldoutSubjectCeiling, self).__init__(*args, **kwargs)
            self._rng = RandomState(0)
            self._num_bootstraps = 5

        def get_subject_iterations(self, subjects):
            # use only a subset of subjects
            return self._rng.choice(list(subjects), size=self._num_bootstraps)


#specify split coordinate for cross-validation
if os.getenv('SPLIT_AT_PASSAGE', '0') == '1':
    pereira_split_coord = 'passage_index'
elif os.getenv('SPLIT_AT_TOPIC', '0') == '1':
    pereira_split_coord = 'passage_category'
else:
    pereira_split_coord = 'stimulus_id'

_logger.info(f"\nCross validation split coordinate is {pereira_split_coord}\n")

#specify number of splits (can be used for teston:original, teston:random-wl, teston:sentenceshuffle_*)
if os.getenv('TWO_SPLITS', '0') == '1':
    nr_of_splits = 2
else:
    nr_of_splits = 5

###################################
##### ORIGINAL BENCHMARK
###################################

class PereiraEncoding(_PereiraBenchmark):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4?fbclid=IwAR0W7EZrnIFFO1kvANgeOEICaoDG5fhmdHipazy6n-APUJ6lMY98PkvuTyU
    """

    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding, self).__init__(metric=metric, **kwargs)

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding, self).ceiling

###################################
##### SCRAMBLING BENCHMARKS > word order manpulations
###################################

class PereiraEncoding_ScrOriginal(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="Original", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_ScrOriginal, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled-original'

    # @property
    # @load_s3(key='Pereira2018-encoding-ceiling')
    # def ceiling(self):
    #     return super(PereiraEncoding_ScrOriginal, self).ceiling

    if os.getenv('COMPUTE_CEILING', '0') == '1': #TODO DELETE
        @property
        def ceiling(self):
            print("I AM COMPUTING THE CEILING!")
            computed_ceiling = super(PereiraEncoding_ScrOriginal, self).ceiling
            outfile = f"language-ceiling_split_coord={pereira_split_coord}.pkl"
            with open(outfile, "wb") as f:
                pickle.dump(computed_ceiling, f)
            return computed_ceiling
    else:
        @property
        @load_s3(key='Pereira2018-encoding-ceiling')
        def ceiling(self):
            return super(PereiraEncoding_ScrOriginal, self).ceiling


class PereiraEncoding_Scr1(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="Scr1", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_Scr1, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled1'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_Scr1, self).ceiling

class PereiraEncoding_Scr3(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="Scr3", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_Scr3, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled3'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_Scr3, self).ceiling

class PereiraEncoding_Scr5(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="Scr5", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_Scr5, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled5'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_Scr5, self).ceiling

class PereiraEncoding_Scr7(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="Scr7", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_Scr7, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled7'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_Scr7, self).ceiling


class PereiraEncoding_ScrLowPMI(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="lowPMI", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_ScrLowPMI, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled-lowpmi'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_ScrLowPMI, self).ceiling

class PereiraEncoding_ScrLowPMIRandom(_PereiraBenchmarkScrambled): #within sentence random lowPMI condition

    def __init__(self, scrambled_version="lowPMI-random", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_ScrLowPMIRandom, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled-lowpmi-random'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_ScrLowPMIRandom, self).ceiling

class PereiraEncoding_ScrBackwardSent(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="backward", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_ScrBackwardSent, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled-backward'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_ScrBackwardSent, self).ceiling

class PereiraEncoding_ScrWordlistRandom(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="random-wl", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_ScrWordlistRandom, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled-random-wl'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_ScrWordlistRandom, self).ceiling


###################################
##### PERTURBATION BENCHMARKS > information loss manipulations
###################################


class PereiraEncoding_PerturbedContentWords(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="contentwords", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_PerturbedContentWords, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-contentwords'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_PerturbedContentWords, self).ceiling

class PereiraEncoding_PerturbedN(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="nouns", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_PerturbedN, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-nouns'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_PerturbedN, self).ceiling
    
    
class PereiraEncoding_PerturbedRandomN(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="random-nouns", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_PerturbedRandomN, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-random-nouns'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_PerturbedRandomN, self).ceiling

    
    
class PereiraEncoding_PerturbedV(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="verbs", **kwargs):
        raise NotImplementedError() #TODO can't be run as not all sentences have verbs according to SpaCy
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_PerturbedV, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-verbs'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_PerturbedV, self).ceiling


class PereiraEncoding_PerturbedNV(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="nounsverbs", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_PerturbedNV, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-nounsverbs'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_PerturbedNV, self).ceiling


class PereiraEncoding_PerturbedNVA(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="nounsverbsadj", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_PerturbedNVA, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-nounsverbsadj'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_PerturbedNVA, self).ceiling
    

class PereiraEncoding_PerturbedFN(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="functionwords", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_PerturbedFN, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-functionwords'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_PerturbedFN, self).ceiling


###################################
##### PERTURBATION BENCHMARKS > sentence meaning manipulations
###################################

class PereiraEncoding_PerturbedRandomSentenceShuffle(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="sentenceshuffle_random", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_PerturbedRandomSentenceShuffle, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-sentenceshuffle_random'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_PerturbedRandomSentenceShuffle, self).ceiling


class PereiraEncoding_PerturbedShuffleWithinPassage(_PereiraBenchmarkScrambled): #Sentences are shuffled within a passage (i.e., sentences with the same passageID and same experiment)

    def __init__(self, scrambled_version="sentenceshuffle_passage", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_PerturbedShuffleWithinPassage, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-sentenceshuffle_passage'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_PerturbedShuffleWithinPassage, self).ceiling


class PereiraEncoding_PerturbedShuffleWithinTopic(_PereiraBenchmarkScrambled): #Sentences are shuffled within a topic

    def __init__(self, scrambled_version="sentenceshuffle_topic", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_PerturbedShuffleWithinTopic, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-sentenceshuffle_topic'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_PerturbedShuffleWithinTopic, self).ceiling
    

class PereiraEncoding_PerturbedChatGPT(_PereiraBenchmarkScrambled): #Sentences are shuffled within a topic

    def __init__(self, scrambled_version="chatgpt", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_PerturbedChatGPT, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-chatgpt

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_PerturbedChatGPT, self).ceiling
    

###################################
##### CONTROL BENCHMARKS
###################################
    
class PereiraEncoding_LengthControl(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="length-control", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_LengthControl, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-length-control'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_LengthControl, self).ceiling
    


###################################
##### END PERTURBATION BENCHMARKS
###################################


##########################################
##### TRAIN ON ORIGINAL, TEST ON PERTURBED
##########################################


class _PereiraBenchmarkTestOnPerturbed(Benchmark):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4
    """

    def __init__(self, identifier, metric, scrambled_version, data_version='base'):
        self._identifier = identifier
        self._data_version = data_version
        self._target_assembly = LazyLoad(lambda: self._load_assembly(version=self._data_version))

        self.scrambled_version = scrambled_version

        # load the scrambled-original stimuli as train dataset
        scrambled_data_dir = os.path.join(ressources_dir, "scrambled_stimuli_dfs/")
        stimuli = pd.read_pickle(os.path.join(scrambled_data_dir, 'stimuli_Original.pkl'))

        if os.getenv('AVG_TOKEN_TRANSFORMERS', '0') == '1':  # CK
            stimuli.name = f"Pereira2018-Original-avgtoken"  # added this
        else:
            stimuli.name = f"Pereira2018-Original-lasttoken"  # added this

        if os.getenv('DECONTEXTUALIZED_EMB', '0') == '1':
            stimuli.name += ",emb_context=Sentence"
        elif os.getenv('PAPER_GROUPING', '0') == '1':
            stimuli.name += ",emb_context=Topic"
        else:
            stimuli.name += ",emb_context=Passage"

        self._target_assembly.attrs['stimulus_set'] = stimuli
        self._target_assembly.attrs['stimulus_set_name'] = stimuli.name
        _logger.debug(f"WE'RE TRAINING ON ACTIVATIONS FROM THIS STIMULUS SET: {self._target_assembly.attrs['stimulus_set'].name}")
        _logger.debug(f"WE'RE TESTING ON ACTIVATIONS FROM SCRAMBLED VERSION: {self.scrambled_version}")

        self._single_metric = metric
        self._ceiler = self.PereiraExtrapolationCeiling(subject_column='subject', num_bootstraps=100)
        self._cross = CartesianProduct(dividers=['experiment', 'atlas'])


    @property
    def identifier(self):
        return self._identifier

    def _metric(self, source_assembly, target_assembly):
        """ for ceiling compute """
        cross_scores = self._cross(target_assembly, apply=
        lambda cross_assembly: self._apply_cross(source_assembly, cross_assembly))
        score = self._average_cross_scores(cross_scores)
        return score

    def _average_cross_scores(self, cross_scores):
        return cross_scores.mean(['experiment', 'atlas'])

    @load_s3(key='Pereira2018')
    def _load_assembly(self, version='base'):
        assembly = load_Pereira2018_Blank(version=version)
        assembly = assembly.sel(atlas_selection_lower=90)
        assembly = assembly[{'neuroid': [filter_strategy in [np.nan, 'HminusE', 'FIXminusH']
                                         for filter_strategy in assembly['filter_strategy'].values]}]
        return assembly

    def __call__(self, candidate):
        stimulus_set = self._target_assembly.attrs['stimulus_set']

        if os.getenv('PAPER_GROUPING', '0') == '1':
            model_activations = listen_to(candidate, stimulus_set)  # this resets with column story/topic
        elif os.getenv('DECONTEXTUALIZED_EMB', '0') == '1':
            model_activations = listen_to(candidate, stimulus_set, reset_column='stimulus_id')
        else:
            stimulus_set.loc[:, 'passage_id'] = stimulus_set['experiment'] + stimulus_set['passage_index'].astype(
                str)
            model_activations = listen_to(candidate, stimulus_set, reset_column='passage_id')

        assert set(model_activations['stimulus_id'].values) == set(self._target_assembly['stimulus_id'].values)

        _logger.info('Scoring across experiments & atlases')
        cross_scores = self._cross(self._target_assembly,
                                   apply=lambda cross_assembly: self._apply_cross(model_activations, cross_assembly))
        raw_scores = cross_scores.raw
        raw_neuroids = apply_aggregate(lambda values: values.mean('split').mean('experiment'), raw_scores)

        # normally we would ceil every single neuroid here. To estimate the strongest ceiling possible (i.e. make it as
        # hard as possible on the models), we used experiment-overlapping neuroids from as many subjects as possible
        # which means some neuroids got excluded. Since median(r/c) is the same as median(r)/median(c), we just
        # normalize the neuroid aggregate by the overall ceiling aggregate.
        # Additionally, the Pereira data also has voxels from DMN, visual etc. but we care about language here.
        language_neuroids = raw_neuroids.sel(atlas='language', _apply_raw=False)
        score = aggregate_ceiling(language_neuroids, ceiling=self.ceiling, subject_column='subject')
        return score

    def _apply_cross(self, source_assembly, cross_assembly):
        cross_assembly = cross_assembly.dropna('neuroid')  # some subjects have only done one experiment
        source_assembly = source_assembly.dropna('neuroid')  # only relevant when running audio-visual self as "model"
        assert len(cross_assembly['presentation']) in [243, 384]
        assert not np.isnan(cross_assembly).any()
        source_assembly_train = source_assembly[{'presentation': [stimulus_id in cross_assembly['stimulus_id'].values
                                                            for stimulus_id in source_assembly['stimulus_id'].values]}]
        source_assembly_test = source_assembly_train.copy(deep=True)

        from neural_nlp.benchmarks.utils import load_activations_into_matrix
        path = '/om2/user/ckauf/.result_caching/neural_nlp.models.wrapper.core.ActivationsExtractorHelper._from_sentences_stored'
        layer_identifier = np.unique(source_assembly_test.layer.data)
        assert len(layer_identifier) == 1
        layer_identifier = layer_identifier[0]
        stimuli_identifier = f'Pereira2018-{self.scrambled_version}'
        if os.getenv('AVG_TOKEN_TRANSFORMERS', '0') == '1':
            agg = 'avgtoken'
        else:
            agg = 'lasttoken'
        expt = np.unique(source_assembly_test.experiment.data)
        assert len(expt) == 1
        expt = expt[0]
        identifier = np.unique(source_assembly_test.model.data)[0]

        if "sentenceshuffle" in self.scrambled_version:
            verbose = False
        else:
            verbose = True
        activations_matrix, flat_sentence_array, flat_sentence_num_array = load_activations_into_matrix(
            identifier=identifier,
            stimuli_identifier=stimuli_identifier,
            agg=agg,
            expt=expt,
            layer_identifier=layer_identifier,
            path=path, verbose=verbose)

        sentencelist_source = list(source_assembly_train.sentence.data)

        if self.scrambled_version == "Original":
            assert all([a == b for a, b in zip(sentencelist_source, flat_sentence_array)])

        source_assembly_test.values = activations_matrix

        return self._single_metric(source_train_emb=source_assembly_train, source_test_emb=source_assembly_test, target=cross_assembly)

    @property
    def ceiling(self):
        return self._ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)

    class PereiraExtrapolationCeiling(ExtrapolationCeiling):
        def __init__(self, subject_column, *args, **kwargs):
            super(_PereiraBenchmarkTestOnPerturbed.PereiraExtrapolationCeiling, self).__init__(
                subject_column, *args, **kwargs)
            self._num_subsamples = 10
            self.holdout_ceiling = _PereiraBenchmarkTestOnPerturbed.PereiraHoldoutSubjectCeiling(subject_column=subject_column)
            self._rng = RandomState(0)

        def iterate_subsets(self, assembly, num_subjects):
            # cross experiment to obtain more subjects to extrapolate.
            # don't worry about atlases here, cross-metric will take care of it.
            experiments = set(assembly['experiment'].values)
            for experiment in sorted(experiments):
                experiment_assembly = assembly[{'presentation': [
                    experiment_value == experiment for experiment_value in assembly['experiment'].values]}]
                experiment_assembly = experiment_assembly.dropna('neuroid')  # drop subjects that haven't done this exp
                if len(set(experiment_assembly[self.subject_column].values)) < num_subjects:
                    continue  # not enough subjects
                for sub_subjects in self._random_combinations(
                        subjects=set(experiment_assembly[self.subject_column].values),
                        num_subjects=num_subjects, choice=self._num_subsamples, rng=self._rng):
                    sub_assembly = assembly[{'neuroid': [subject in sub_subjects
                                                         for subject in assembly[self.subject_column].values]}]
                    yield {self.subject_column: sub_subjects, 'experiment': experiment}, sub_assembly

        def _random_combinations(self, subjects, num_subjects, choice, rng):
            # following https://stackoverflow.com/a/55929159/2225200. Also see similar method in `behavioral.py`.
            subjects = np.array(list(subjects))
            combinations = set()
            while len(combinations) < choice:
                elements = rng.choice(subjects, size=num_subjects, replace=False)
                combinations.add(tuple(elements))
            return combinations

        def extrapolate(self, ceilings):
            ceiling = super(_PereiraBenchmark.PereiraExtrapolationCeiling, self).extrapolate(ceilings)
            # compute aggregate ceiling only for language neuroids
            neuroid_ceilings = ceiling.raw
            language_ceilings = neuroid_ceilings.sel(atlas='language')
            ceiling = self.aggregate_neuroid_ceilings(language_ceilings)
            ceiling.attrs['raw'] = neuroid_ceilings  # reset to all neuroids
            return ceiling

        def fit(self, subject_subsamples, bootstrapped_scores):
            valid = ~np.isnan(bootstrapped_scores)
            if sum(valid) < 1:
                raise RuntimeError("No valid scores in sample")
            return super(_PereiraBenchmark.PereiraExtrapolationCeiling, self).fit(
                np.array(subject_subsamples)[valid], np.array(bootstrapped_scores)[valid])

        def post_process(self, scores):
            scores = apply_aggregate(lambda values: values.mean('sub_experiment').mean('experiment'), scores)
            return scores

    class PereiraHoldoutSubjectCeiling(HoldoutSubjectCeiling):
        def __init__(self, *args, **kwargs):
            super(_PereiraBenchmarkTestOnPerturbed.PereiraHoldoutSubjectCeiling, self).__init__(*args, **kwargs)
            self._rng = RandomState(0)
            self._num_bootstraps = 5

        def get_subject_iterations(self, subjects):
            # use only a subset of subjects
            return self._rng.choice(list(subjects), size=self._num_bootstraps)

class PereiraEncoding_Original_TestOnOriginal(_PereiraBenchmarkTestOnPerturbed):

    def __init__(self, scrambled_version="Original", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=nr_of_splits, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_Original_TestOnOriginal, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-teston:original'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_Original_TestOnOriginal, self).ceiling

class PereiraEncoding_Original_TestOnScr1(_PereiraBenchmarkTestOnPerturbed):

    def __init__(self, scrambled_version="Scr1", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_Original_TestOnScr1, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-teston:scr1'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_Original_TestOnScr1, self).ceiling

class PereiraEncoding_Original_TestOnScr3(_PereiraBenchmarkTestOnPerturbed):

    def __init__(self, scrambled_version="Scr3", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_Original_TestOnScr3, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-teston:scr3'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_Original_TestOnScr3, self).ceiling

class PereiraEncoding_Original_TestOnScr5(_PereiraBenchmarkTestOnPerturbed):

    def __init__(self, scrambled_version="Scr5", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_Original_TestOnScr5, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-teston:scr5'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_Original_TestOnScr5, self).ceiling

class PereiraEncoding_Original_TestOnScr7(_PereiraBenchmarkTestOnPerturbed):

    def __init__(self, scrambled_version="Scr7", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(PereiraEncoding_Original_TestOnScr7, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-teston:scr7'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_Original_TestOnScr7, self).ceiling


class PereiraEncoding_TestOnLowPMI(_PereiraBenchmarkTestOnPerturbed):

    def __init__(self, scrambled_version="lowPMI", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord,
                                        stratification_coord=None))
        super(PereiraEncoding_TestOnLowPMI, self).__init__(metric=metric, scrambled_version=scrambled_version,
                                                        **kwargs)  # identifier='Pereira2018-encoding-teston:lowpmi'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_TestOnLowPMI, self).ceiling


class PereiraEncoding_TestOnLowPMIRandom(_PereiraBenchmarkTestOnPerturbed):  # within sentence random lowPMI condition

    def __init__(self, scrambled_version="lowPMI-random", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord,
                                        stratification_coord=None))
        super(PereiraEncoding_TestOnLowPMIRandom, self).__init__(metric=metric, scrambled_version=scrambled_version,
                                                              **kwargs)  # identifier='Pereira2018-encoding-teston:lowpmi-random'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_TestOnLowPMIRandom, self).ceiling


class PereiraEncoding_TestOnBackward(_PereiraBenchmarkTestOnPerturbed):

    def __init__(self, scrambled_version="backward", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord,
                                        stratification_coord=None))
        super(PereiraEncoding_TestOnBackward, self).__init__(metric=metric, scrambled_version=scrambled_version,
                                                              **kwargs)  # identifier='Pereira2018-encoding-teston:backward'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_TestOnBackward, self).ceiling


class PereiraEncoding_TestOnWordlistRandom(_PereiraBenchmarkTestOnPerturbed):

    def __init__(self, scrambled_version="random-wl", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=nr_of_splits, kfold=True, split_coord=pereira_split_coord,
                                        stratification_coord=None))
        super(PereiraEncoding_TestOnWordlistRandom, self).__init__(metric=metric, scrambled_version=scrambled_version,
                                                                **kwargs)  # identifier='Pereira2018-encoding-teston:random-wl'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_TestOnWordlistRandom, self).ceiling



class PereiraEncoding_TestOnContentWords(_PereiraBenchmarkTestOnPerturbed):

    def __init__(self, scrambled_version="contentwords", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord,
                                        stratification_coord=None))
        super(PereiraEncoding_TestOnContentWords, self).__init__(metric=metric, scrambled_version=scrambled_version,
                                                                    **kwargs)  # identifier='Pereira2018-encoding-teston:contentwords'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_TestOnContentWords, self).ceiling


class PereiraEncoding_TestOnN(_PereiraBenchmarkTestOnPerturbed):

    def __init__(self, scrambled_version="nouns", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord,
                                        stratification_coord=None))
        super(PereiraEncoding_TestOnN, self).__init__(metric=metric, scrambled_version=scrambled_version,
                                                         **kwargs)  # identifier='Pereira2018-encoding-teston:nouns'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_TestOnN, self).ceiling


class PereiraEncoding_TestOnRandomN(_PereiraBenchmarkTestOnPerturbed):

    def __init__(self, scrambled_version="random-nouns", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord,
                                        stratification_coord=None))
        super(PereiraEncoding_TestOnRandomN, self).__init__(metric=metric, scrambled_version=scrambled_version,
                                                               **kwargs)  # identifier='Pereira2018-encoding-teston:random-nouns'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_TestOnRandomN, self).ceiling


class PereiraEncoding_TestOnNV(_PereiraBenchmarkTestOnPerturbed):

    def __init__(self, scrambled_version="nounsverbs", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord,
                                        stratification_coord=None))
        super(PereiraEncoding_TestOnNV, self).__init__(metric=metric, scrambled_version=scrambled_version,
                                                          **kwargs)  # identifier='Pereira2018-encoding-teston:nounsverbs'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_TestOnNV, self).ceiling


class PereiraEncoding_TestOnNVA(_PereiraBenchmarkTestOnPerturbed):

    def __init__(self, scrambled_version="nounsverbsadj", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord,
                                        stratification_coord=None))
        super(PereiraEncoding_TestOnNVA, self).__init__(metric=metric, scrambled_version=scrambled_version,
                                                           **kwargs)  # identifier='Pereira2018-encoding-teston:nounsverbsadj'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_TestOnNVA, self).ceiling


class PereiraEncoding_TestOnFN(_PereiraBenchmarkTestOnPerturbed):

    def __init__(self, scrambled_version="functionwords", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord,
                                        stratification_coord=None))
        super(PereiraEncoding_TestOnFN, self).__init__(metric=metric, scrambled_version=scrambled_version,
                                                          **kwargs)  # identifier='Pereira2018-encoding-teston:functionwords'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_TestOnFN, self).ceiling


class PereiraEncoding_TestOnRandomSentence(_PereiraBenchmarkTestOnPerturbed):

    def __init__(self, scrambled_version="sentenceshuffle_random", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=nr_of_splits, kfold=True, split_coord=pereira_split_coord,
                                        stratification_coord=None), scrambled_version=scrambled_version)
        
        super(PereiraEncoding_TestOnRandomSentence, self).__init__(metric=metric,
                                                                             scrambled_version=scrambled_version,
                                                                             **kwargs)  # identifier='Pereira2018-encoding-teston:sentenceshuffle_random'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_TestOnRandomSentence, self).ceiling



class PereiraEncoding_TestOnShuffleWithinPassage(
    _PereiraBenchmarkTestOnPerturbed):  # Sentences are shuffled within a passage (i.e., sentences with the same passageID and same experiment)

    def __init__(self, scrambled_version="sentenceshuffle_passage", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=nr_of_splits, kfold=True, split_coord=pereira_split_coord,
                                        stratification_coord=None), scrambled_version=scrambled_version)
        super(PereiraEncoding_TestOnShuffleWithinPassage, self).__init__(metric=metric,
                                                                            scrambled_version=scrambled_version,
                                                                            **kwargs)  # identifier='Pereira2018-encoding-teston:sentenceshuffle_passage'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_TestOnShuffleWithinPassage, self).ceiling


class PereiraEncoding_TestOnShuffleWithinTopic(_PereiraBenchmarkTestOnPerturbed):  # Sentences are shuffled within a topic

    def __init__(self, scrambled_version="sentenceshuffle_topic", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=nr_of_splits, kfold=True, split_coord=pereira_split_coord,
                                        stratification_coord=None), scrambled_version=scrambled_version)
            
        super(PereiraEncoding_TestOnShuffleWithinTopic, self).__init__(metric=metric,
                                                                          scrambled_version=scrambled_version,
                                                                          **kwargs)  # identifier='Pereira2018-encoding-teston:sentenceshuffle_topic'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_TestOnShuffleWithinTopic, self).ceiling
    
    
class PereiraEncoding_TestOnChatGPT(_PereiraBenchmarkTestOnPerturbed):

    def __init__(self, scrambled_version="chatgpt", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord,
                                        stratification_coord=None))
        super(PereiraEncoding_TestOnChatGPT, self).__init__(metric=metric, scrambled_version=scrambled_version,
                                                           **kwargs)  # identifier='Pereira2018-encoding-teston:chatgpt'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_TestOnChatGPT, self).ceiling


class PereiraEncoding_TestOnLengthControl(_PereiraBenchmarkTestOnPerturbed):

    def __init__(self, scrambled_version="length-control", **kwargs):
        metric = CrossRegressedCorrelationPerturbed(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord,
                                        stratification_coord=None))
        super(PereiraEncoding_TestOnLengthControl, self).__init__(metric=metric, scrambled_version=scrambled_version,
                                                            **kwargs)  # identifier='Pereira2018-teston:length-control'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding_TestOnLengthControl, self).ceiling



#########################################

benchmark_pool = [
    # primary benchmarks
    ('Pereira2018-encoding', PereiraEncoding),
    #control benchmarks
    ('Pereira2018-encoding-length-control', PereiraEncoding_LengthControl), #length control (#words-many 'the's), no sentence-internal punctuation but final period.
    ('Pereira2018-encoding-scrambled-original', PereiraEncoding_ScrOriginal), #lower-cased, no sentence-internal punctuation but final period. (keeps hyphens, apostrophe, currency & units)
    ('Pereira2018-encoding-scrambled1', PereiraEncoding_Scr1),
    ('Pereira2018-encoding-scrambled3', PereiraEncoding_Scr3),
    ('Pereira2018-encoding-scrambled5', PereiraEncoding_Scr5),
    ('Pereira2018-encoding-scrambled7', PereiraEncoding_Scr7),
    ('Pereira2018-encoding-scrambled-lowpmi', PereiraEncoding_ScrLowPMI),
    ('Pereira2018-encoding-scrambled-lowpmi-random', PereiraEncoding_ScrLowPMIRandom), #lowPMI random word shuffling within sentence
    ('Pereira2018-encoding-scrambled-backward', PereiraEncoding_ScrBackwardSent),
    ('Pereira2018-encoding-scrambled-random-wl', PereiraEncoding_ScrWordlistRandom),
    #perturbation benchmarks > information loss manipulations
    ('Pereira2018-encoding-perturb-nouns', PereiraEncoding_PerturbedN), #keep only nouns
    ('Pereira2018-encoding-perturb-random-nouns', PereiraEncoding_PerturbedRandomN), #nouns in each sentence replaced by random nouns
    ('Pereira2018-encoding-perturb-verbs', PereiraEncoding_PerturbedV), #currently not running
    ('Pereira2018-encoding-perturb-nounsverbs', PereiraEncoding_PerturbedNV),
    ('Pereira2018-encoding-perturb-nounsverbsadj', PereiraEncoding_PerturbedNVA),
    ('Pereira2018-encoding-perturb-contentwords', PereiraEncoding_PerturbedContentWords), #keep only content words (nouns, verbs, adj, adv)
    ('Pereira2018-encoding-perturb-functionwords', PereiraEncoding_PerturbedFN),
    #perturbation benchmarks > sentence meaning manipulations
    ('Pereira2018-encoding-perturb-sentenceshuffle_random', PereiraEncoding_PerturbedRandomSentenceShuffle), #randomly shuffle sentences across datasets/experiments
    ('Pereira2018-encoding-perturb-sentenceshuffle_passage', PereiraEncoding_PerturbedShuffleWithinPassage), #shuffle sentences within passage
    ('Pereira2018-encoding-perturb-sentenceshuffle_topic', PereiraEncoding_PerturbedShuffleWithinTopic), #shuffle sentences within topic
    ('Pereira2018-encoding-chatgpt', PereiraEncoding_PerturbedChatGPT),
    # train regression on scrambled-original, test it on activations for perturbed stimuli
    ('Pereira2018-encoding-teston:original', PereiraEncoding_Original_TestOnOriginal),
    ('Pereira2018-encoding-teston:scr1', PereiraEncoding_Original_TestOnScr1),
    ('Pereira2018-encoding-teston:scr3', PereiraEncoding_Original_TestOnScr3),
    ('Pereira2018-encoding-teston:scr5', PereiraEncoding_Original_TestOnScr5),
    ('Pereira2018-encoding-teston:scr7', PereiraEncoding_Original_TestOnScr7),
    ('Pereira2018-encoding-teston:lowpmi', PereiraEncoding_TestOnLowPMI),
    ('Pereira2018-encoding-teston:lowpmi-random', PereiraEncoding_TestOnLowPMIRandom),
    ('Pereira2018-encoding-teston:backward', PereiraEncoding_TestOnBackward),
    ('Pereira2018-encoding-teston:random-wl', PereiraEncoding_TestOnWordlistRandom),
    ('Pereira2018-encoding-teston:nouns', PereiraEncoding_TestOnN),
    ('Pereira2018-encoding-teston:random-nouns', PereiraEncoding_TestOnRandomN),
    ('Pereira2018-encoding-teston:nounsverbs', PereiraEncoding_TestOnNV),
    ('Pereira2018-encoding-teston:nounsverbsadj', PereiraEncoding_TestOnNVA),
    ('Pereira2018-encoding-teston:contentwords', PereiraEncoding_TestOnContentWords),
    ('Pereira2018-encoding-teston:functionwords', PereiraEncoding_TestOnFN),
    ('Pereira2018-encoding-teston:sentenceshuffle_random', PereiraEncoding_TestOnRandomSentence),
    ('Pereira2018-encoding-teston:sentenceshuffle_passage', PereiraEncoding_TestOnShuffleWithinPassage),
    ('Pereira2018-encoding-teston:sentenceshuffle_topic', PereiraEncoding_TestOnShuffleWithinTopic),
    ('Pereira2018-encoding-teston:chatgpt', PereiraEncoding_TestOnChatGPT),
    ('Pereira2018-encoding-teston:length-control', PereiraEncoding_TestOnLengthControl),
]

benchmark_pool = {identifier: LazyLoad(lambda identifier=identifier, ctr=ctr: ctr(identifier=identifier))
                  for identifier, ctr in benchmark_pool}
