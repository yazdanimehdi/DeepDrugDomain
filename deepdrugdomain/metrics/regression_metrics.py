from .base_metrics import BaseMetric
from .factory import MetricFactory

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error,
    mean_poisson_deviance,
    mean_gamma_deviance,
    mean_tweedie_deviance,
)


@MetricFactory.register('mean_absolute_error')
class MeanAbsoluteErrorMetric(BaseMetric):
    def compute(self, prediction, target):
        return mean_absolute_error(target, prediction)


@MetricFactory.register('mean_squared_error')
class MeanSquaredErrorMetric(BaseMetric):
    def compute(self, prediction, target):
        return mean_squared_error(target, prediction)


@MetricFactory.register('mean_squared_log_error')
class MeanSquaredLogErrorMetric(BaseMetric):
    def compute(self, prediction, target):
        return mean_squared_log_error(target, prediction)


@MetricFactory.register('median_absolute_error')
class MedianAbsoluteErrorMetric(BaseMetric):
    def compute(self, prediction, target):
        return median_absolute_error(target, prediction)


@MetricFactory.register('r2_score')
class R2ScoreMetric(BaseMetric):
    def compute(self, prediction, target):
        return r2_score(target, prediction)


@MetricFactory.register('mean_absolute_percentage_error')
class MeanAbsolutePercentageErrorMetric(BaseMetric):
    def compute(self, prediction, target):
        return mean_absolute_percentage_error(target, prediction)


@MetricFactory.register('explained_variance_score')
class ExplainedVarianceScoreMetric(BaseMetric):
    def compute(self, prediction, target):
        return explained_variance_score(target, prediction)


@MetricFactory.register('max_error')
class MaxErrorMetric(BaseMetric):
    def compute(self, prediction, target):
        return max_error(target, prediction)


@MetricFactory.register('mean_poisson_deviance')
class MeanPoissonDevianceMetric(BaseMetric):
    def compute(self, prediction, target):
        return mean_poisson_deviance(target, prediction)


@MetricFactory.register('mean_gamma_deviance')
class MeanGammaDevianceMetric(BaseMetric):
    def compute(self, prediction, target):
        return mean_gamma_deviance(target, prediction)


@MetricFactory.register('mean_tweedie_deviance')
class MeanTweedieDevianceMetric(BaseMetric):
    def compute(self, prediction, target):
        return mean_tweedie_deviance(target, prediction)
