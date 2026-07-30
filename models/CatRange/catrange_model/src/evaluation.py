"""
Evaluation Module for CatRange

Handles:
- Classification metrics (accuracy, precision, recall, F1, MCC, AUC-PR)
- e-accuracy (within ±1 bin)
- Cross-validation result aggregation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, average_precision_score,
)
from sklearn.preprocessing import label_binarize
from typing import Tuple, Dict, Optional, List
from pathlib import Path


# ============================================================================
# METRIC COMPUTATION
# ============================================================================
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    classes: Optional[np.ndarray] = None,
    average: str = "macro",
) -> Tuple[float, float, float, float, float, float]:
    """Calculate (accuracy, precision, recall, f1, mcc, auc_pr)."""
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
            ap = average_precision_score(y_true_bin[:, j], y_proba[:, j])
            ap_scores.append(ap)
            supports.append((y_true == classes[j]).sum())
        ap_scores = np.array(ap_scores)
        supports = np.array(supports)
        auc_pr = np.nanmean(ap_scores) if average == "macro" else \
                 np.nansum(ap_scores * supports / supports.sum())
    else:
        auc_pr = np.nan

    return accuracy, precision, recall, f1, mcc, auc_pr


def calculate_e_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy within ±1 bin."""
    return float((np.abs(y_pred - y_true) <= 1).mean())


# ============================================================================
# FOLD EVALUATION
# ============================================================================
def evaluate_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    fold_idx: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate a single fold and return summary metrics dict."""
    accuracy, precision, recall, f1, mcc, auc_pr = calculate_metrics(
        y_true, y_pred, y_proba=y_proba, average="weighted"
    )
    e_accuracy = calculate_e_accuracy(y_true, y_pred)

    results = {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "mcc": mcc, "auc_pr": auc_pr,
        "e_accuracy": e_accuracy, "n_samples": len(y_true),
    }

    if verbose and fold_idx:
        print(f"\nFold {fold_idx} Results:")
        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  e-Accuracy:  {e_accuracy:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")
        print(f"  F1:          {f1:.4f}")
        print(f"  MCC:         {mcc:.4f}")
        if not np.isnan(auc_pr):
            print(f"  AUC-PR:      {auc_pr:.4f}")

    return results


# ============================================================================
# CV AGGREGATION
# ============================================================================
def aggregate_cv_results(
    fold_results: List[Dict],
    output_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Aggregate cross-validation results across folds into a summary DataFrame."""
    df = pd.DataFrame(fold_results)
    summary = {"Metric": [], "Mean": [], "Std": [], "Min": [], "Max": []}
    for col in df.columns:
        if col.startswith("fold"):
            continue
        summary["Metric"].append(col)
        summary["Mean"].append(df[col].mean())
        summary["Std"].append(df[col].std())
        summary["Min"].append(df[col].min())
        summary["Max"].append(df[col].max())
    df_summary = pd.DataFrame(summary)

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Saved CV results to {output_file}")

    return df_summary
