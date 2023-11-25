from typing import Any, List
from .base_metrics import BaseMetric
from .factory import MetricFactory
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    classification_report, cohen_kappa_score, confusion_matrix,
    f1_score, fbeta_score, hamming_loss, hinge_loss,
    jaccard_score, log_loss, matthews_corrcoef,
    multilabel_confusion_matrix, precision_recall_fscore_support,
    precision_score, recall_score, zero_one_loss, roc_auc_score
)
import numpy as np


@MetricFactory.register('auc')
class AUCMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return roc_auc_score(target, prediction)


@MetricFactory.register('accuracy_score')
class AccuracyScoreMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return accuracy_score(target, prediction)


@MetricFactory.register('balanced_accuracy_score')
class BalancedAccuracyScoreMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return balanced_accuracy_score(target, prediction)


@MetricFactory.register('brier_score_loss')
class BrierScoreLossMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return brier_score_loss(target, prediction)


@MetricFactory.register('classification_report')
class ClassificationReportMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray) -> str:
        return classification_report(target, prediction)


@MetricFactory.register('cohen_kappa_score')
class CohenKappaScoreMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return cohen_kappa_score(target, prediction)


@MetricFactory.register('confusion_matrix')
class ConfusionMatrixMetric(BaseMetric):
    def compute(self,  prediction: np.ndarray, target: np.ndarray) -> List[List[int]]:
        return confusion_matrix(target, prediction).tolist()


@MetricFactory.register('f1_score')
class F1ScoreMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return f1_score(target, prediction)


@MetricFactory.register('fbeta_score')
class FbetaScoreMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray, beta=1.0) -> float:
        return fbeta_score(target, prediction, beta)


@MetricFactory.register('hamming_loss')
class HammingLossMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return hamming_loss(target, prediction)


@MetricFactory.register('hinge_loss')
class HingeLossMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return hinge_loss(target, prediction)


@MetricFactory.register('jaccard_score')
class JaccardScoreMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return jaccard_score(target, prediction)


@MetricFactory.register('log_loss')
class LogLossMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return log_loss(target, prediction)


@MetricFactory.register('matthews_corrcoef')
class MatthewsCorrcoefMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return matthews_corrcoef(target, prediction)


@MetricFactory.register('multilabel_confusion_matrix')
class MultilabelConfusionMatrixMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray) -> List[List[List[int]]]:
        return multilabel_confusion_matrix(target, prediction).tolist()


@MetricFactory.register('precision_recall_fscore_support')
class PrecisionRecallFscoreSupportMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray) -> dict:
        precision, recall, fscore, support = precision_recall_fscore_support(
            target, prediction)
        return {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "fscore": fscore.tolist(),
            "support": support.tolist()
        }


@MetricFactory.register('precision_score')
class PrecisionScoreMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return precision_score(target, prediction)


@MetricFactory.register('recall_score')
class RecallScoreMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return recall_score(target, prediction)


@MetricFactory.register('zero_one_loss')
class ZeroOneLossMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return zero_one_loss(target, prediction)


@MetricFactory.register('roc_enrichment_0.5')
class ROCEnrichmentMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray, enrichment_percent: float = 0.005) -> float:
        # Determine the number of true positives at the enrichment percentage
        num_total = len(target)
        num_positives = sum(target)
        num_to_select = int(enrichment_percent * num_total)
        num_selected_positives = sum(sorted(
            [(p, t) for p, t in zip(prediction, target)], reverse=True)[:num_to_select])

        # Compute the TPR at the enrichment percentage
        tpr_at_x = num_selected_positives / num_positives

        # Calculate ROC Enrichment
        re_at_x = tpr_at_x / enrichment_percent
        return re_at_x


@MetricFactory.register('roc_enrichment_1')
class ROCEnrichmentMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray, enrichment_percent: float = 0.01) -> float:
        # Determine the number of true positives at the enrichment percentage
        num_total = len(target)
        num_positives = sum(target)
        num_to_select = int(enrichment_percent * num_total)
        num_selected_positives = sum(sorted(
            [(p, t) for p, t in zip(prediction, target)], reverse=True)[:num_to_select])

        # Compute the TPR at the enrichment percentage
        tpr_at_x = num_selected_positives / num_positives

        # Calculate ROC Enrichment
        re_at_x = tpr_at_x / enrichment_percent
        return re_at_x


@MetricFactory.register('roc_enrichment_2')
class ROCEnrichmentMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray, enrichment_percent: float = 0.02) -> float:
        # Determine the number of true positives at the enrichment percentage
        num_total = len(target)
        num_positives = sum(target)
        num_to_select = int(enrichment_percent * num_total)
        num_selected_positives = sum(sorted(
            [(p, t) for p, t in zip(prediction, target)], reverse=True)[:num_to_select])

        # Compute the TPR at the enrichment percentage
        tpr_at_x = num_selected_positives / num_positives

        # Calculate ROC Enrichment
        re_at_x = tpr_at_x / enrichment_percent
        return re_at_x


@MetricFactory.register('roc_enrichment_3')
class ROCEnrichmentMetric(BaseMetric):

    def compute(self, prediction: np.ndarray, target: np.ndarray, enrichment_percent: float = 0.03) -> float:
        # Determine the number of true positives at the enrichment percentage
        num_total = len(target)
        num_positives = sum(target)
        num_to_select = int(enrichment_percent * num_total)
        num_selected_positives = sum(sorted(
            [(p, t) for p, t in zip(prediction, target)], reverse=True)[:num_to_select])

        # Compute the TPR at the enrichment percentage
        tpr_at_x = num_selected_positives / num_positives

        # Calculate ROC Enrichment
        re_at_x = tpr_at_x / enrichment_percent
        return re_at_x


@MetricFactory.register('roc_enrichment_5')
class ROCEnrichmentMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray, enrichment_percent: float = 0.05) -> float:
        # Determine the number of true positives at the enrichment percentage
        num_total = len(target)
        num_positives = sum(target)
        num_to_select = int(enrichment_percent * num_total)
        num_selected_positives = sum(sorted(
            [(p, t) for p, t in zip(prediction, target)], reverse=True)[:num_to_select])

        # Compute the TPR at the enrichment percentage
        tpr_at_x = num_selected_positives / num_positives

        # Calculate ROC Enrichment
        re_at_x = tpr_at_x / enrichment_percent
        return re_at_x


@MetricFactory.register('roc_enrichment_10')
class ROCEnrichmentMetric(BaseMetric):
    def compute(self, prediction: np.ndarray, target: np.ndarray, enrichment_percent: float = 0.10) -> float:
        # Determine the number of true positives at the enrichment percentage
        num_total = len(target)
        num_positives = sum(target)
        num_to_select = int(enrichment_percent * num_total)
        num_selected_positives = sum(sorted(
            [(p, t) for p, t in zip(prediction, target)], reverse=True)[:num_to_select])

        # Compute the TPR at the enrichment percentage
        tpr_at_x = num_selected_positives / num_positives

        # Calculate ROC Enrichment
        re_at_x = tpr_at_x / enrichment_percent
        return re_at_x
