from lifelines.utils import concordance_index
from .factory import MetricFactory
from .base_metrics import BaseMetric


@MetricFactory.register('concordance_index')
class ConcordanceIndexMetric(BaseMetric):
    def compute(self, prediction, target):
        return concordance_index(target, prediction)
