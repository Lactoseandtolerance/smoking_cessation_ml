"""Utility functions: metrics and small helpers."""
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
