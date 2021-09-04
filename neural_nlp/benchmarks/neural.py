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
from tqdm import tqdm

from brainscore.benchmarks import Benchmark
from brainscore.metrics import Score
from brainscore.metrics.rdm import RDM, RDMSimilarity, RDMCrossValidated
#from brainscore.metrics.cka import CKACrossValidated
from brainscore.metrics.regression import linear_regression, pearsonr_correlation, CrossRegressedCorrelation
from brainscore.metrics.transformations import CartesianProduct, CrossValidation, apply_aggregate
from brainscore.utils import LazyLoad
from neural_nlp.benchmarks.ceiling import ExtrapolationCeiling, HoldoutSubjectCeiling
from neural_nlp.benchmarks.s3 import load_s3
from neural_nlp.neural_data.fmri import load_voxels, load_rdm_sentences, \
    load_Pereira2018_Blank
from neural_nlp.stimuli import load_stimuli, StimulusSet
from neural_nlp.utils import ordered_set
from result_caching import store

_logger = logging.getLogger(__name__)


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
        activations.append(story_activations)
    model_activations = merge_data_arrays(activations)
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
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncoding, self).__init__(metric=metric, **kwargs)

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding, self).ceiling


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


class PereiraCKA(_PereiraSubjectWise):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4?fbclid=IwAR0W7EZrnIFFO1kvANgeOEICaoDG5fhmdHipazy6n-APUJ6lMY98PkvuTyU
    """

    def __init__(self, **kwargs):
        metric = CKACrossValidated(
            comparison_coord='stimulus_id',
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord=None, splits=5,
                                        kfold=True, test_size=None))
        super(PereiraCKA, self).__init__(metric=metric, **kwargs)

    @property
    def ceiling(self):
        return super(_PereiraSubjectWise, self).ceiling


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
    assert set(neuroids['neuroid_id'].values) == set(ceiling_neuroids['neuroid_id'].values)
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

        scrambled_data_dir = os.path.join(ressources_dir, "scrambled-stimuli-dfs/")

        STIMULI_TO_PKL_MAP = {'lowPMI': os.path.join(scrambled_data_dir, 'stimuli_lowPMI.pkl'),
                              'Original': os.path.join(scrambled_data_dir, 'stimuli_Original.pkl'),
                              'Scr1': os.path.join(scrambled_data_dir, 'stimuli_Scr1.pkl'),
                              'Scr3': os.path.join(scrambled_data_dir, 'stimuli_Scr3.pkl'),
                              'Scr5': os.path.join(scrambled_data_dir, 'stimuli_Scr5.pkl'),
                              'Scr7': os.path.join(scrambled_data_dir, 'stimuli_Scr7.pkl'),
                              'backward': os.path.join(scrambled_data_dir, 'stimuli_backward.pkl'),
                              'random-wl': os.path.join(scrambled_data_dir, 'stimuli_random.pkl'),
                              'random-lowPMI': os.path.join(scrambled_data_dir, 'stimuli_random_lowPMI.pkl'),
                              #perturb pkls
                              'contentwords': os.path.join(scrambled_data_dir, 'stimuli_contentwords.pkl'),
                              'nouns': os.path.join(scrambled_data_dir, 'stimuli_nouns.pkl'),
                              'nouns-delete50percent': os.path.join(scrambled_data_dir, 'stimuli_nouns_delete50percent.pkl'),
                              'random-nouns': os.path.join(scrambled_data_dir, 'stimuli_randomnouns.pkl'),
                              'verbs': os.path.join(scrambled_data_dir, 'stimuli_verbs.pkl'),
                              'nounsverbs': os.path.join(scrambled_data_dir, 'stimuli_nounsverbs.pkl'),
                              'nounsverbsadj': os.path.join(scrambled_data_dir, 'stimuli_nounsverbsadj.pkl'),
                              'functionwords': os.path.join(scrambled_data_dir, 'stimuli_functionwords.pkl'),
                              # perturb pkl (control)
                              'randomsentenceshuffle': os.path.join(scrambled_data_dir, 'stimuli_sentenceshuffle-random.pkl'),
                              'passageshuffle': os.path.join(scrambled_data_dir, 'stimuli_sentenceshuffle-withinpassage.pkl')
                              # perturb pkls (checks) > not currently generated by stimuli generation script
                              #'nopunctuation': os.path.join(scrambled_data_dir, 'stimuli_nopunctuation.pkl'),
                              #'lowPMI-finalperiod': os.path.join(scrambled_data_dir, 'stimuli_lowpmi_finalperiod.pkl')
                              }


        for key in STIMULI_TO_PKL_MAP.keys():
            if scrambled_version == key:
                _logger.debug(f"I AM USING THIS DATA VERSION: {key}")
                stimuli = pd.read_pickle(STIMULI_TO_PKL_MAP[key])
                if os.getenv('AVG_TOKEN_TRANSFORMERS', '0') == '1': #CK
                    stimuli.name = f"Pereira2018-{scrambled_version}-avgtoken" #added this
                else:
                    stimuli.name = f"Pereira2018-{scrambled_version}-lasttoken"  # added this

        self._target_assembly.attrs['stimulus_set'] = stimuli
        self._target_assembly.attrs['stimulus_set_name'] = stimuli.name #CK 2021-08-05, doesn't get reset otherwise as "Pereira2018" is stores as stimulus_set_name in the stored assembly
        _logger.debug(f"THIS IS THE STIMULUS SET NAME: {self._target_assembly.attrs['stimulus_set'].name}") #e.g., Stimulus set name: Pereira2018-Original-lasttoken

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
        activations.append(story_activations)
    model_activations = merge_data_arrays(activations)
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
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncoding, self).__init__(metric=metric, **kwargs)

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding, self).ceiling


class PereiraEncodingScrambledOriginal(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="Original", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingScrambledOriginal, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled-original'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingScrambledOriginal, self).ceiling

class PereiraEncodingScrambled1(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="Scr1", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingScrambled1, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled1'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingScrambled1, self).ceiling

class PereiraEncodingScrambled3(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="Scr3", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingScrambled3, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled3'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingScrambled3, self).ceiling

class PereiraEncodingScrambled5(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="Scr5", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingScrambled5, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled5'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingScrambled5, self).ceiling

class PereiraEncodingScrambled7(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="Scr7", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingScrambled7, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled7'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingScrambled7, self).ceiling

class PereiraEncodingScrambledLowPMI(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="lowPMI", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingScrambledLowPMI, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled-lowpmi'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingScrambledLowPMI, self).ceiling

class PereiraEncodingScrambledBackward(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="backward", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingScrambledBackward, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled-backward'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingScrambledBackward, self).ceiling

class PereiraEncodingScrambledRandomWL(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="random-wl", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingScrambledRandomWL, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled-random'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingScrambledRandomWL, self).ceiling

class PereiraEncodingScrambledRandomLowPMI(_PereiraBenchmarkScrambled): #within sentence random lowPMI condition

    def __init__(self, scrambled_version="random-lowPMI", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingScrambledRandomLowPMI, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled-random-lowPMI'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingScrambledRandomLowPMI, self).ceiling
    

###################################
##### PERTURBATION BENCHMARKS
###################################

class PereiraEncodingPerturbNoPunctuation(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="nopunctuation", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingPerturbNoPunctuation, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-nopunctuation'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingPerturbNoPunctuation, self).ceiling
    

class PereiraEncodingScrambledLowPMIFinalPeriod(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="lowPMI-finalperiod", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingScrambledLowPMIFinalPeriod, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-scrambled-lowpmi-finalperiod'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingScrambledLowPMIFinalPeriod, self).ceiling


class PereiraEncodingPerturbContentWords(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="contentwords", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingPerturbContentWords, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-contentwords'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingPerturbContentWords, self).ceiling

class PereiraEncodingPerturbN(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="nouns", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingPerturbN, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-nouns'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingPerturbN, self).ceiling
    
class PereiraEncodingPerturbN50Percent(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="nouns-delete50percent", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingPerturbN50Percent, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-nouns-delete50percent'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingPerturbN50Percent, self).ceiling
    
    
class PereiraEncodingPerturbRandomN(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="random-nouns", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingPerturbRandomN, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-random-nouns'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingPerturbN50Percent, self).ceiling
    
    
class PereiraEncodingPerturbV(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="verbs", **kwargs):
        raise NotImplementedError() #TODO can't be run as not all sentences have verbs according to SpaCy
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingPerturbV, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-verbs'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingPerturbV, self).ceiling


class PereiraEncodingPerturbNV(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="nounsverbs", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingPerturbNV, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-nounsverbs'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingPerturbNV, self).ceiling


class PereiraEncodingPerturbNVA(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="nounsverbsadj", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingPerturbNVA, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-nounsverbsadj'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingPerturbNVA, self).ceiling

class PereiraEncodingPerturbFN(_PereiraBenchmarkScrambled):

    def __init__(self, scrambled_version="functionwords", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingPerturbFN, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-functionwords'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingPerturbFN, self).ceiling
    

class PereiraEncodingPerturbRandomSentenceShuffle(_PereiraBenchmarkScrambled): #Sentences are shuffled within a passage (i.e., sentences with the same passageID and same experiment)

    def __init__(self, scrambled_version="randomsentenceshuffle", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingPerturbRandomSentenceShuffle, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-randomsentenceshuffle'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingPerturbRandomSentenceShuffle, self).ceiling


class PereiraEncodingPerturbPassageShuffle(_PereiraBenchmarkScrambled): #Sentences are shuffled within a passage (i.e., sentences with the same passageID and same experiment)

    def __init__(self, scrambled_version="passageshuffle", **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncodingPerturbPassageShuffle, self).__init__(metric=metric, scrambled_version=scrambled_version, **kwargs) # identifier='Pereira2018-encoding-perturb-passageshuffle'

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncodingPerturbPassageShuffle, self).ceiling
    
    

###################################
##### END PERTURBATION BENCHMARKS
###################################


benchmark_pool = [
    # primary benchmarks
    ('Pereira2018-encoding', PereiraEncoding),
    # secondary benchmarks
    ('Pereira2018-rdm', PereiraRDM),
    ('Pereira2018-cka', PereiraCKA),
    #scrambling benchmarks
    ('Pereira2018-encoding-scrambled-original', PereiraEncodingScrambledOriginal), #lower-cased, no sentence-internal punctuation but final period.
    ('Pereira2018-encoding-scrambled1', PereiraEncodingScrambled1),
    ('Pereira2018-encoding-scrambled3', PereiraEncodingScrambled3),
    ('Pereira2018-encoding-scrambled5', PereiraEncodingScrambled5),
    ('Pereira2018-encoding-scrambled7', PereiraEncodingScrambled7),
    ('Pereira2018-encoding-scrambled-lowpmi', PereiraEncodingScrambledLowPMI),
    ('Pereira2018-encoding-scrambled-random-lowPMI', PereiraEncodingScrambledRandomLowPMI), #lowPMI random word shuffling within sentence
    ('Pereira2018-encoding-scrambled-backward', PereiraEncodingScrambledBackward),
    ('Pereira2018-encoding-scrambled-random-wl', PereiraEncodingScrambledRandomWL),
    #perturb benchmarks
    ('Pereira2018-encoding-perturb-nouns', PereiraEncodingPerturbN), #keep only nouns
    ('Pereira2018-encoding-perturb-nouns-delete50percent', PereiraEncodingPerturbN50Percent), #keep only 50% (randomly selected) of nouns
    ('Pereira2018-encoding-perturb-random-nouns', PereiraEncodingPerturbRandomN), #nouns in each sentence replaced by random nouns
    ('Pereira2018-encoding-perturb-verbs', PereiraEncodingPerturbV), #currently not running
    ('Pereira2018-encoding-perturb-nounsverbs', PereiraEncodingPerturbNV),
    ('Pereira2018-encoding-perturb-nounsverbsadj', PereiraEncodingPerturbNVA),
    ('Pereira2018-encoding-perturb-contentwords', PereiraEncodingPerturbContentWords), #keep only content words (nouns, verbs, adj, adv)
    ('Pereira2018-encoding-perturb-functionwords', PereiraEncodingPerturbFN),
    #control benchmarks
    ('Pereira2018-encoding-perturb-randomsentenceshuffle', PereiraEncodingPerturbRandomSentenceShuffle), #randomly shuffle sentences across datasets/experiments
    ('Pereira2018-encoding-perturb-passageshuffle', PereiraEncodingPerturbPassageShuffle), #shuffle sentences within passage
    #TEST benchmarks
    ('Pereira2018-encoding-perturb-nopunctuation', PereiraEncodingPerturbNoPunctuation),
    ('Pereira2018-encoding-scrambled-lowpmi-finalperiod', PereiraEncodingScrambledLowPMIFinalPeriod)
]

benchmark_pool = {identifier: LazyLoad(lambda identifier=identifier, ctr=ctr: ctr(identifier=identifier))
                  for identifier, ctr in benchmark_pool}
