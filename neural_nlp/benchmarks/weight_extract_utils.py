import numpy as np
from brainio_base.assemblies import NeuroidAssembly, array_is_element, walk_coords
from sklearn.linear_model import LinearRegression
from brainscore.metrics import Score, _logger
from scipy.stats import median_absolute_deviation
from brainscore.metrics.regression import pls_regression
from brainscore.metrics.transformations import Split, enumerate_done, apply_aggregate
from tqdm import tqdm
import logging
import warnings
from brainscore.utils import fullname
from brainio_collection.transform import subset
from brainio_base.assemblies import DataAssembly, walk_coords, merge_data_arrays
class Defaults:
    expected_dims = ('presentation', 'neuroid')
    stimulus_coord = 'image_id'
    neuroid_dim = 'neuroid'
    neuroid_coord = 'neuroid_id'

class TransformationWeight(object):
    """
    Transforms an incoming assembly into parts/combinations thereof,
    yields them for further processing,
    and packages the results back together.
    """

    def __call__(self, *args, apply, aggregate=None, **kwargs):
        values_weights = self._run_pipe(*args, apply=apply, **kwargs)
        values=values_weights[0]
        weights=values_weights[1]
        score = apply_aggregate(aggregate, values) if aggregate is not None else values
        score = apply_aggregate(self.aggregate, score)
        return score, weights

    def _run_pipe(self, *args, apply, **kwargs):
        generator = self.pipe(*args, **kwargs)
        for vals in generator:
            y = apply(*vals)
            done = generator.send(y)
            if done:
                break
        result = next(generator)
        return result

    def pipe(self, *args, **kwargs):
        raise NotImplementedError()

    def _get_result(self, *args, done):
        """
        Yields the `*args` for further processing by coroutines
        and waits for the result to be sent back.
        :param args: transformed values
        :param bool done: whether this is the last transformation and the next `yield` is the combined result
        :return: the result from processing by the coroutine
        """
        result = yield args  # yield the values to coroutine
        yield done  # wait for coroutine to send back similarity and inform whether result is ready to be returned
        return result

    def aggregate(self, score):
        return Score(score)

class XarrayRegressionWithWeights():
    """
    Adds alignment-checking, un- and re-packaging, and comparison functionality to a regression.
    """

    def __init__(self, regression, expected_dims=Defaults.expected_dims, neuroid_dim=Defaults.neuroid_dim,
                 neuroid_coord=Defaults.neuroid_coord, stimulus_coord=Defaults.stimulus_coord):
        self._regression = regression
        self._expected_dims = expected_dims
        self._neuroid_dim = neuroid_dim
        self._neuroid_coord = neuroid_coord
        self._stimulus_coord = stimulus_coord
        self._target_neuroid_values = None

    def fit(self, source, target):
        source, target = self._align(source), self._align(target)
        source, target = source.sortby(self._stimulus_coord), target.sortby(self._stimulus_coord)

        self._regression.fit(source, target)

        self._target_neuroid_values = {}
        for name, dims, values in walk_coords(target):
            if self._neuroid_dim in dims:
                assert array_is_element(dims, self._neuroid_dim)
                self._target_neuroid_values[name] = values

    def predict(self, source):
        source = self._align(source)
        predicted_values = self._regression.predict(source)
        prediction = self._package_prediction(predicted_values, source=source)
        return prediction
    def _package_prediction(self, predicted_values, source):
        coords = {coord: (dims, values) for coord, dims, values in walk_coords(source)
                  if not array_is_element(dims, self._neuroid_dim)}
        # re-package neuroid coords
        dims = source.dims
        # if there is only one neuroid coordinate, it would get discarded and the dimension would be used as coordinate.
        # to avoid this, we can build the assembly first and then stack on the neuroid dimension.
        neuroid_level_dim = None
        if len(self._target_neuroid_values) == 1:  # extract single key: https://stackoverflow.com/a/20145927/2225200
            (neuroid_level_dim, _), = self._target_neuroid_values.items()
            dims = [dim if dim != self._neuroid_dim else neuroid_level_dim for dim in dims]
        for target_coord, target_value in self._target_neuroid_values.items():
            # this might overwrite values which is okay
            coords[target_coord] = (neuroid_level_dim or self._neuroid_dim), target_value
        prediction = NeuroidAssembly(predicted_values, coords=coords, dims=dims)
        if neuroid_level_dim:
            prediction = prediction.stack(**{self._neuroid_dim: [neuroid_level_dim]})

        return prediction

    def extract_coeff(self, source):
        source = self._align(source)
        coef_values = self._regression.coef_
        coef_values = self._package_coeffs(coef_values, source=source)
        return coef_values
    def _package_coeffs(self, coefficents, source):
        # re-package neuroid coords
        #dims =('electrode',source.dims[1])
        dims = ('neuroid_id', source.dims[1])
        coords={}
        coords[source.dims[1]]=source[dims[1]].values
        coords[dims[0]]=self._target_neuroid_values[dims[0]]
        coef_values = NeuroidAssembly(coefficents, coords=coords, dims=dims)

        return coef_values

    def _align(self, assembly):
        assert set(assembly.dims) == set(self._expected_dims), \
            f"Expected {set(self._expected_dims)}, but got {set(assembly.dims)}"
        return assembly.transpose(*self._expected_dims)

def linear_regression_with_weights(xarray_kwargs=None):
    regression = LinearRegression()
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegressionWithWeights(regression, **xarray_kwargs)
    return regression


class CrossRegressedCorrelationWithWeights:
    def __init__(self, regression, correlation,return_weights=False, crossvalidation_kwargs=None):
        regression = regression or pls_regression()
        crossvalidation_defaults = dict(train_size=.9, test_size=None)
        self.return_weights=return_weights
        crossvalidation_kwargs = {**crossvalidation_defaults, **(crossvalidation_kwargs or {})}

        self.cross_validation = CrossValidationWithWeight(**crossvalidation_kwargs)
        self.regression = regression
        self.correlation = correlation

    def __call__(self, source, target):
        return self.cross_validation(source, target, apply=self.apply, aggregate=self.aggregate)

    def apply(self, source_train, target_train, source_test, target_test):
        self.regression.fit(source_train, target_train)
        prediction = self.regression.predict(source_test)
        weights = self.regression.extract_coeff(source_train)
        score = self.correlation(prediction, target_test)
        if self.return_weights:
            return score , weights
        else:
            return score ,[]

    def aggregate(self, scores):
        return scores.median(dim='neuroid')

class CrossValidationWithWeight(TransformationWeight):
    """
    Performs multiple splits over a source and target assembly.
    No guarantees are given for data-alignment, use the metadata.
    """

    def __init__(self, *args, split_coord=Split.Defaults.split_coord,
                 stratification_coord=Split.Defaults.stratification_coord, **kwargs):
        self._split_coord = split_coord
        self._stratification_coord = stratification_coord
        self._split = Split(*args, split_coord=split_coord, stratification_coord=stratification_coord, **kwargs)
        self._logger = logging.getLogger(fullname(self))

    def pipe(self, source_assembly, target_assembly):
        # check only for equal values, alignment is given by metadata
        assert sorted(source_assembly[self._split_coord].values) == sorted(target_assembly[self._split_coord].values)
        if self._split.do_stratify:
            assert hasattr(source_assembly, self._stratification_coord)
            assert sorted(source_assembly[self._stratification_coord].values) == \
                   sorted(target_assembly[self._stratification_coord].values)
        cross_validation_values, splits = self._split.build_splits(target_assembly)

        split_scores = []
        split_weights = []
        for split_iterator, (train_indices, test_indices), done \
                in tqdm(enumerate_done(splits), total=len(splits), desc='cross-validation'):
            train_values, test_values = cross_validation_values[train_indices], cross_validation_values[test_indices]
            train_source = subset(source_assembly, train_values, dims_must_match=False)
            train_target = subset(target_assembly, train_values, dims_must_match=False)
            assert len(train_source[self._split_coord]) == len(train_target[self._split_coord])
            test_source = subset(source_assembly, test_values, dims_must_match=False)
            test_target = subset(target_assembly, test_values, dims_must_match=False)
            assert len(test_source[self._split_coord]) == len(test_target[self._split_coord])

            split_score,split_weight = yield from self._get_result(train_source, train_target, test_source, test_target,
                                                      done=done)
            split_score = split_score.expand_dims('split')
            split_weight = split_weight.expand_dims('split')
            split_score['split'] = [split_iterator]
            split_weight['split'] = [split_iterator]
            split_scores.append(split_score)
            split_weights.append(split_weight)

        split_scores = Score.merge(*split_scores)
        split_weights=merge_data_arrays(split_weights)

        yield split_scores, split_weights

    def aggregate(self, score):
        return self._split.aggregate(score)


