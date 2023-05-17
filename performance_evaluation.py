# =============================================================================
# PERFORMANCE EVALUATION
#
# last major rev. 2020/04
#
# Filippo Maria Castelli
# LENS Biophotonics Group
# castelli@lens.unifi.it
# =============================================================================

import logging

import numpy as np
import sklearn.metrics as skmetrics

class PerformanceMetrics:
    """general class for performance metrics evaluation"""
    def __init__(self, y_true, y_pred, thr):
        self.y_true = np.copy(y_true)
        self.y_pred = np.copy(y_pred)
        self.thr = thr
        
        self.y_pred_fuzzy = y_pred
        self.y_true_fuzzy = y_true
        
        self.fuzzy_intersection = np.sum(self.y_pred_fuzzy.flatten() * self.y_true_fuzzy.flatten())
        self.fuzzy_summation = np.sum(self.y_pred_fuzzy.flatten()) + np.sum(self.y_true_fuzzy.flatten())
        self.fuzzy_union = self.fuzzy_summation - self.fuzzy_intersection
        
        self.y_pred = self.threshold_array(self.y_pred, thr, to_bool=True)
        self.y_true = self.threshold_array(self.y_true, thr, to_bool=True)

        self.tp, self.fp, self.tn, self.fn = self.cardinal_metrics()

        self.specificity = self.crisp_specificity()
        self.recall = self.crisp_recall()
        self.precision = self.crisp_precision()
        self.false_negative_ratio = self.crisp_false_negative_ratio()
        self.fallout = self.crisp_fallout()

        self.measure_dict = self.create_dict()

    def cardinal_metrics(self, sum_elems=True):
        # TP
        true_positive = np.logical_and(self.y_pred, self.y_true)
        # TN
        true_negative = np.logical_and(
            np.logical_not(self.y_pred), np.logical_not(self.y_true)
        )
        # FP
        false_positive = np.logical_and(self.y_pred == True, self.y_true == False)
        # FN
        false_negative = np.logical_and(self.y_pred == False, self.y_true == True)

        if sum_elems == True:
            return (
                np.sum(true_positive),
                np.sum(false_positive),
                np.sum(true_negative),
                np.sum(false_negative),
            )
        else:
            return true_positive, false_positive, true_negative, false_negative
        
    def fuzzy_dice(self):
        """Dice coefficient for fuzzy segmentations.
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (6)
        """
        return 2 * self.fuzzy_intersection / self.fuzzy_summation
    
    def fuzzy_jaccard(self):
        """Jaccard index for fuzzy segmentations.
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (7)
        """
        return self.fuzzy_intersection / self.fuzzy_union
    
    @staticmethod
    def threshold_array(arr, thr=0.5, to_bool=False):
        arr[arr >= thr] = 1
        arr[arr < thr] = 0

        if to_bool == True:
            return np.array(arr, dtype=bool)
        else:
            return arr

    def crisp_jaccard(self):
        """Jaccard index for crisp segmentations
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (7)
        """
        return self.tp / (self.tp + self.fp + self.fn)

    def crisp_dice(self):
        """Dice coefficient for crisp segmentations.
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (6)
        """
        return (2 * self.tp) / (2 * self.tp + self.fp + self.fn)

    # recall/sensitivity
    def crisp_recall(self):
        """Recall or Sensitivity or TPR for crisp segmentations.
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (10)
        """
        return (self.tp) / (self.tp + self.fn)

    # specificity/true negative ratio
    def crisp_specificity(self):
        """Specificity or TNR for crisp segmentations.
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (11)
        """
        return (self.tn) / (self.tn + self.fp)

    # fallout/false positive ratio
    def crisp_fallout(self):
        """Fallout or FPR for crisp segmentations
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (12)
        """
        return 1 - self.specificity

    def crisp_false_negative_ratio(self):
        """FNR for crisp segmentations
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (13)
        """
        return 1 - self.recall

    # precision/positive predictive ratio
    def crisp_precision(self):
        """Precision or PPV for crisp segmentations
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (14)
        """
        return (self.tp) / (self.tp + self.fp)

    def crisp_auc(self):
        """Estimator of Area under ROC 
        ROC is defined as TPR vs FPR plot, AUC here is calculated as area of the 
        trapezoid defined by the measurement opint of the lines TPR=0 nad FPR=1.
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (47)
        """
        return 1 - ((self.fallout + self.false_negative_ratio) / 2)

    def crisp_f1_score(self):
        """F1-score or FMS1 for crisp segmentations
        Implemented from Taha et al. - Metrics from Evaluating 3D Medical Image Segmentation (16)
        """
        return (2 * self.precision * self.recall) / (self.precision + self.recall)
    
    def sk_f1_score(self):
        """F1-score implemented by scikit-learn"""
        return skmetrics.f1_score(self.y_true.flatten(), self.y_pred.flatten())
    
    def sk_roc_curve(self):
        """ROC curve
        Implemented by scikit-learn"""
        logging.info("Calculating ROC curve")
        fpr, tpr, thresholds = skmetrics.roc_curve(y_true=self.y_true_fuzzy.flatten(),
                                                   y_score=self.y_pred_fuzzy.flatten())
        
        roc_dict = {"fpr": fpr,
                    "tpr": tpr,
                    "thresholds": thresholds}
        return roc_dict
    
    def sk_roc_auc_score(self):
        """Area under ROC curve
        Implemented by scikit-learn"""
        logging.info("Calculating PR curve")
        return skmetrics.roc_auc_score(y_true=self.y_true_fuzzy.flatten(),
                                       y_score=self.y_pred_fuzzy.flatten())
    
    def sk_pr_curve(self):
        """Precision Recall curve
        Implemented by scikit-learn"""
        precision, recall, threhsolds = skmetrics.precision_recall_curve(y_true=self.y_true_fuzzy.flatten(),
                                                                         probas_pred=self.y_pred_fuzzy.flatten())
        
        pr_dict = {"precision": precision,
                   "recall": recall,
                   "thresholds": threhsolds}
        return pr_dict

    def create_dict(self):
        dictionary = {
            "crisp_jaccard": self.crisp_jaccard(),
            "crisp_dice": self.crisp_dice(),
            "crisp_recall": self.crisp_recall(),
            "crisp_specificity": self.crisp_specificity(),
            "crisp_fallout": self.crisp_fallout(),
            "crisp_false_negative_ratio": self.crisp_false_negative_ratio(),
            "crisp_precision": self.crisp_precision(),
            "crisp_auc": self.crisp_auc(),
            "crisp_f1_score": self.crisp_f1_score(),
            "sk_f1_score": self.sk_f1_score(),
            "fuzzy_dice" : self.fuzzy_dice(),
            "fuzzy_jaccard": self.fuzzy_jaccard(),
            "roc_curve": self.sk_roc_curve(),
            "pr_curve": self.sk_pr_curve(),
            "sk_roc_auc_score": self.sk_roc_auc_score()
        }

        return dictionary

    def get_metrics(self):
        return self.measure_dict
