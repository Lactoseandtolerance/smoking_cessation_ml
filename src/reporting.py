"""Reporting utilities for summarizing model comparison and dataset statistics.

This module centralizes formatting and summarization logic so notebooks and
scripts can generate consistent textual and Markdown reports.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional
import pandas as pd
import datetime


@dataclass
class ModelSummary:
    model_name: str
    model_index: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None  # e.g. output of evaluate_model


@dataclass
class DataSplitSummary:
    n_train: Optional[int] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    quit_rate_train: Optional[float] = None
    quit_rate_val: Optional[float] = None
    quit_rate_test: Optional[float] = None
    n_features: Optional[int] = None
    n_transitions: Optional[int] = None


def _safe_len(x) -> Optional[int]:
    try:
        return len(x)
    except Exception:
        return None


def summarize_results(
    comparison_df: Optional[pd.DataFrame] = None,
    best_model: Optional[ModelSummary] = None,
    splits: Optional[Dict[str, Any]] = None,
    feature_cols: Optional[Iterable[str]] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a structured summary of model comparisons and data splits.

    Args:
        comparison_df: DataFrame with model comparison metrics.
        best_model: ModelSummary instance describing best model.
        splits: Dict optionally containing X_train, X_val, X_test, y_train, y_val, y_test.
        feature_cols: Iterable of feature column names.
        timestamp: Optional timestamp override (ISO string). If None, current UTC time.

    Returns:
        Dictionary with structured summary sections.
    """
    ts = timestamp or datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # Gather dataset sizes & quit rates if available
    ds_summary = DataSplitSummary()
    if splits:
        X_train = splits.get("X_train")
        X_val = splits.get("X_val")
        X_test = splits.get("X_test")
        y_train = splits.get("y_train")
        y_val = splits.get("y_val")
        y_test = splits.get("y_test")
        ds_summary.n_train = _safe_len(X_train)
        ds_summary.n_val = _safe_len(X_val)
        ds_summary.n_test = _safe_len(X_test)
        if y_train is not None:
            try: ds_summary.quit_rate_train = float(pd.Series(y_train).mean())
            except Exception: pass
        if y_val is not None:
            try: ds_summary.quit_rate_val = float(pd.Series(y_val).mean())
            except Exception: pass
        if y_test is not None:
            try: ds_summary.quit_rate_test = float(pd.Series(y_test).mean())
            except Exception: pass

    if feature_cols is not None:
        ds_summary.n_features = _safe_len(feature_cols)

    # Attempt to infer transitions count from comparison_df if present
    if comparison_df is not None:
        if "n_transitions" in comparison_df.columns:
            try:
                ds_summary.n_transitions = int(comparison_df["n_transitions"].max())
            except Exception:
                pass
        else:
            # Fallback: if a dataset size column exists or we can infer from index length
            ds_summary.n_transitions = _safe_len(comparison_df)

    summary = {
        "timestamp": ts,
        "comparison": {
            "available": comparison_df is not None,
            "n_models": int(len(comparison_df)) if comparison_df is not None else 0,
            "columns": list(comparison_df.columns) if comparison_df is not None else [],
        },
        "best_model": asdict(best_model) if best_model is not None else None,
        "data_splits": asdict(ds_summary),
    }

    return summary


def render_text_summary(summary: Dict[str, Any]) -> str:
    """Render the structured summary into a human-readable multiline string."""
    lines = []
    lines.append("=== MODEL SUMMARY ===")
    lines.append(f"Generated: {summary['timestamp']}")

    # Comparison overview
    comp = summary["comparison"]
    if comp["available"]:
        lines.append("")
        lines.append(f"Models compared: {comp['n_models']}")
        if comp['columns']:
            lines.append("Comparison columns: " + ", ".join(comp['columns']))
    else:
        lines.append("")
        lines.append("No comparison DataFrame provided.")

    # Best model
    bm = summary.get("best_model")
    if bm:
        lines.append("")
        lines.append("=== BEST MODEL ===")
        lines.append(f"Name: {bm.get('model_name')}")
        if bm.get('model_index') is not None:
            lines.append(f"Index: {bm.get('model_index')}")
        metrics = bm.get('metrics') or {}
        if metrics:
            lines.append("Metrics:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    lines.append(f"  - {k}: {v:.4f}")
                else:
                    lines.append(f"  - {k}: {v}")
    else:
        lines.append("")
        lines.append("No best model information provided.")

    # Data splits
    ds = summary.get("data_splits", {})
    lines.append("")
    lines.append("=== DATA SPLITS ===")
    def _fmt_int(x):
        return f"{x:,}" if isinstance(x, int) and x is not None else str(x)
    lines.append(f"Train size: {_fmt_int(ds.get('n_train'))}")
    lines.append(f"Val size: {_fmt_int(ds.get('n_val'))}")
    lines.append(f"Test size: {_fmt_int(ds.get('n_test'))}")
    lines.append(f"Features: {_fmt_int(ds.get('n_features'))}")
    lines.append(f"Transitions: {_fmt_int(ds.get('n_transitions'))}")

    # Quit rates
    lines.append("")
    lines.append("=== QUIT RATES ===")
    def _pct(x):
        return f"{x*100:.1f}%" if isinstance(x, float) and x is not None else str(x)
    lines.append(f"Train: {_pct(ds.get('quit_rate_train'))}")
    lines.append(f"Validation: {_pct(ds.get('quit_rate_val'))}")
    lines.append(f"Test: {_pct(ds.get('quit_rate_test'))}")

    return "\n".join(lines)


def write_markdown_summary(summary: Dict[str, Any], path: str) -> None:
    """Write a Markdown version of the summary to disk."""
    text = render_text_summary(summary)
    md_lines = ["# Model Results Summary", "", "```", text, "```", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


__all__ = [
    "ModelSummary",
    "DataSplitSummary",
    "summarize_results",
    "render_text_summary",
    "write_markdown_summary",
]
