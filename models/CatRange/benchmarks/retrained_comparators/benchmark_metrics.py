"""Standalone metric computation mirroring CatRange/src/evaluation.py."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    classes: Optional[np.ndarray] = None,
    average: str = "weighted",
) -> Tuple[float, float, float, float, float, float]:
    """Calculate the same classification metrics used by CatRange."""
    if classes is None:
        classes = np.unique(y_true)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    if y_proba is not None:
        y_true_bin = label_binarize(y_true, classes=classes)
        ap_scores = []
        supports = []
        for j in range(len(classes)):
            ap_scores.append(average_precision_score(y_true_bin[:, j], y_proba[:, j]))
            supports.append((y_true == classes[j]).sum())
        ap_scores = np.asarray(ap_scores, dtype=float)
        supports = np.asarray(supports, dtype=float)
        auc_pr = np.nanmean(ap_scores) if average == "macro" else np.nansum(ap_scores * supports / supports.sum())
    else:
        auc_pr = np.nan

    return accuracy, precision, recall, f1, mcc, auc_pr


def calculate_e_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.abs(y_pred - y_true) <= 1).mean())


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
):
    accuracy, precision, recall, f1, mcc, auc_pr = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        average="weighted",
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "auc_pr": auc_pr,
        "e_accuracy": calculate_e_accuracy(y_true, y_pred),
        "n_samples": int(len(y_true)),
    }


def confusion_matrix_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_labels = [f"true_bin_{label}" for label in labels]
    col_labels = [f"pred_bin_{label}" for label in labels]
    return pd.DataFrame(matrix, index=row_labels, columns=col_labels)
