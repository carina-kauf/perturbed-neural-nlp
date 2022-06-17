from brainscore.benchmarks import Benchmark
from brainscore.metrics.transformations import CartesianProduct, CrossValidation, apply_aggregate
from brainscore.utils import LazyLoad
from neural_nlp.benchmarks.ceiling import ExtrapolationCeiling, HoldoutSubjectCeiling
import numpy as np
from neural_nlp.benchmarks.s3 import load_s3
from numpy.random.mtrand import RandomState
from brainscore.metrics.regression import linear_regression, pearsonr_correlation, CrossRegressedCorrelation
import os
import sys
import logging


_logger = logging.getLogger(__name__)

#
# #specify split coordinate for cross-validation
# if os.getenv('SPLIT_AT_PASSAGE', '0') == '1':
#     pereira_split_coord = 'passage_index'
# elif os.getenv('SPLIT_AT_TOPIC', '0') == '1':
#     pereira_split_coord = 'passage_category'
# else:
#     pereira_split_coord = 'stimulus_id'

metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=2, kfold=True, split_coord='stimulus_id', stratification_coord=None))
            #todo splits should be 5

identifier = "Pereira-encoding"

class _PereiraMDCeiling(Benchmark):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4
    """

    def _average_cross_scores(self, cross_scores):
        return cross_scores.mean(['experiment', 'atlas'])

    def __init__(self, data_version='base'):
        self._data_version = data_version
        self._target_assembly = LazyLoad(lambda: self._load_assembly(version=self._data_version))
        self._ceiler = self.PereiraExtrapolationCeiling(subject_column='subject', num_bootstraps=2) #todo should be 100
        self._cross = CartesianProduct(dividers=['experiment', 'atlas'])
        self._single_metric = metric

    @load_s3(key='Pereira2018')
    def _load_assembly(self, version='base'):
        assembly = load_Pereira2018_Blank(version=version)
        assembly = assembly.sel(atlas_selection_lower=90)
        assembly = assembly[{'neuroid': [filter_strategy in [np.nan, 'HminusE', 'FIXminusH']
                                         for filter_strategy in assembly['filter_strategy'].values]}]
        return assembly

    def _apply_cross(self, source_assembly, cross_assembly):
        cross_assembly = cross_assembly.dropna('neuroid')  # some subjects have only done one experiment
        source_assembly = source_assembly.dropna('neuroid')  # only relevant when running audio-visual self as "model"
        assert len(cross_assembly['presentation']) in [243, 384]
        assert not np.isnan(cross_assembly).any()
        source_assembly = source_assembly[{'presentation': [stimulus_id in cross_assembly['stimulus_id'].values
                                                            for stimulus_id in source_assembly['stimulus_id'].values]}]
        return self._single_metric(source_assembly, cross_assembly)

    def _metric(self, source_assembly, target_assembly):
        cross_scores = self._cross(target_assembly, apply=
        lambda cross_assembly: self._apply_cross(source_assembly, cross_assembly))
        score = self._average_cross_scores(cross_scores)
        return score


    class PereiraExtrapolationCeiling(ExtrapolationCeiling):
        def __init__(self, subject_column, *args, **kwargs):
            super(_PereiraMDCeiling.PereiraExtrapolationCeiling, self).__init__(
                subject_column, *args, **kwargs)
            self._num_subsamples = 2 #todo should be 10
            self.holdout_ceiling = _PereiraMDCeiling.PereiraHoldoutSubjectCeiling(subject_column=subject_column)
            self._rng = RandomState(0)

        def iterate_subsets(self, assembly, num_subjects):
            # cross experiment to obtain more subjects to extrapolate.
            # don't worry about atlases here, cross-metric will take care of it.
            experiments = set(assembly['experiment'].values)
            for experiment in sorted(experiments):
                _logger.info(f"Experiment: {experiment}")
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
            ceiling = super(_PereiraMDCeiling.PereiraExtrapolationCeiling, self).extrapolate(ceilings) #extrapolate from ExtrapolationCeiling in transformations.py
            #FIXME Don't extrapolate here, but just take last one and take the median? i.e., use all subjects!

            # compute aggregate ceiling only for MD neuroids
            neuroid_ceilings = ceiling.raw
            md_ceilings = neuroid_ceilings.sel(atlas='MD')
            ceiling = self.aggregate_neuroid_ceilings(md_ceilings)
            ceiling.attrs['raw'] = neuroid_ceilings  # reset to all neuroids
            return ceiling

        def fit(self, subject_subsamples, bootstrapped_scores):
            valid = ~np.isnan(bootstrapped_scores)
            if sum(valid) < 1:
                raise RuntimeError("No valid scores in sample")
            return super(_PereiraMDCeiling.PereiraExtrapolationCeiling, self).fit(
                np.array(subject_subsamples)[valid], np.array(bootstrapped_scores)[valid])

        def post_process(self, scores):
            scores = apply_aggregate(lambda values: values.mean('sub_experiment').mean('experiment'), scores)
            return scores

    class PereiraHoldoutSubjectCeiling(HoldoutSubjectCeiling):
        def __init__(self, *args, **kwargs):
            super(_PereiraMDCeiling.PereiraHoldoutSubjectCeiling, self).__init__(*args, **kwargs)
            self._rng = RandomState(0)
            self._num_bootstraps = 2 #todo should be 5

        def get_subject_iterations(self, subjects):
            # use only a subset of subjects
            return self._rng.choice(list(subjects), size=self._num_bootstraps)


if __name__ == '__main__':
    PC = _PereiraMDCeiling()
    ceiling = PC._ceiler(identifier=identifier, assembly=PC._target_assembly, metric=PC._metric)
    print(ceiling)

