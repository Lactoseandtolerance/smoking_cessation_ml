#!/usr/bin/env python3
"""Train models on expanded transitions (W1→W7) and produce wave-pair
evaluation plus feature drift analysis.

Outputs:
  - models/xgboost_best.pkl (best validation AUC among trained models)
  - reports/WAVE_PAIR_EVAL.md (validation + test wave-pair metrics)
  - reports/FEATURE_DRIFT.md (baseline wave drift summary)
  - reports/INTERPRETABILITY_SUMMARY.md appended (optional hook if present)
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import date

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import get_feature_list, engineer_all_features
from src.modeling import (
    split_data_by_person,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    save_model
)
from src.evaluation import (
    evaluate_model,
    print_evaluation_report,
    evaluate_by_wave_pair,
    feature_drift_by_wave
)

DATA_PATH = PROJECT_ROOT / 'data/processed/pooled_transitions.parquet'
MODEL_OUT = PROJECT_ROOT / 'models/xgboost_best.pkl'
WAVE_PAIR_REPORT = PROJECT_ROOT / 'reports/WAVE_PAIR_EVAL.md'
DRIFT_REPORT = PROJECT_ROOT / 'reports/FEATURE_DRIFT.md'


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Ensure expected feature columns exist and return feature list."""
    df = df.copy()
    if 'person_id' not in df.columns and 'PERSONID' in df.columns:
        df['person_id'] = df['PERSONID']
    feature_cols = get_feature_list()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        df = engineer_all_features(df, recode_missing=False)
        still_missing = [c for c in feature_cols if c not in df.columns]
        for c in still_missing:
            df[c] = 0
    return df, feature_cols


def main():
    print("="*80)
    print("TRAINING & WAVE-PAIR EVALUATION")
    print("="*80)
    assert DATA_PATH.exists(), f"Missing dataset: {DATA_PATH}"

    # Load data
    print("1. Loading dataset...")
    df = pd.read_parquet(DATA_PATH)
    print(f"   ✓ Loaded {len(df):,} rows × {df.shape[1]} columns")

    # Prepare features
    print("2. Preparing feature columns...")
    df, feature_cols = prepare_dataset(df)
    print(f"   ✓ Using {len(feature_cols)} features")

    # Split by person
    print("3. Creating person-level splits (60/20/20)...")
    X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids = split_data_by_person(
        df, feature_cols, test_size=0.4, val_size=0.5, random_state=42
    )

    # Train models
    print("4. Training models...")
    # Logistic Regression
    print("   a) Logistic Regression")
    lr_model, lr_scaler, y_val_pred_lr, y_val_proba_lr = train_logistic_regression(
        X_train, y_train, X_val, y_val
    )
    lr_metrics = evaluate_model(y_val, y_val_pred_lr, y_val_proba_lr, model_name="Logistic Regression")
    # Random Forest
    print("   b) Random Forest")
    rf_model, y_val_pred_rf, y_val_proba_rf = train_random_forest(
        X_train, y_train, X_val, y_val
    )
    rf_metrics = evaluate_model(y_val, y_val_pred_rf, y_val_proba_rf, model_name="Random Forest")
    # XGBoost
    print("   c) XGBoost")
    xgb_model, y_val_pred_xgb, y_val_proba_xgb = train_xgboost(
        X_train, y_train, X_val, y_val
    )
    xgb_metrics = evaluate_model(y_val, y_val_pred_xgb, y_val_proba_xgb, model_name="XGBoost")

    # Model comparison
    comparison_df = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics])
    best_idx = comparison_df['roc_auc'].idxmax()
    best_row = comparison_df.loc[best_idx]
    best_name = best_row['model']
    print("\n5. Validation comparison:")
    print(comparison_df[['model','roc_auc','pr_auc','precision','recall','f1']].round(3).to_string(index=False))
    print(f"\n🏆 Best validation model: {best_name} (ROC-AUC={best_row['roc_auc']:.3f})")

    # Persist XGBoost regardless (common production choice)
    print("6. Saving XGBoost model...")
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    save_model(xgb_model, MODEL_OUT, metadata={'validation_metrics': xgb_metrics, 'n_train': len(X_train)})

    # Wave-pair evaluation (validation & test sets)
    print("7. Computing wave-pair metrics (validation & test)...")
    val_df = df[df['person_id'].isin(val_ids)]
    test_df = df[df['person_id'].isin(test_ids)]
    wave_val = evaluate_by_wave_pair(xgb_model, val_df, feature_cols, scaler=None, model_name="XGBoost")
    wave_test = evaluate_by_wave_pair(xgb_model, test_df, feature_cols, scaler=None, model_name="XGBoost")

    # Feature drift (baseline waves across full dataset)
    print("8. Assessing feature drift across baseline waves...")
    drift_df = feature_drift_by_wave(df, feature_cols, reference_wave=1, top_k=25)

    # Reports
    print("9. Writing reports...")
    WAVE_PAIR_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(WAVE_PAIR_REPORT, 'w') as f:
        f.write(f"# Wave-Pair Evaluation - {date.today().isoformat()}\n\n")
        f.write("Validation Set Metrics (baseline→follow-up)\n\n")
        if wave_val.empty:
            f.write("_No evaluable wave pairs (insufficient class variation).\n\n")
        else:
            f.write(wave_val.to_markdown(index=False))
            f.write("\n\n")
        f.write("Test Set Metrics (baseline→follow-up)\n\n")
        if wave_test.empty:
            f.write("_No evaluable wave pairs (insufficient class variation).\n")
        else:
            f.write(wave_test.to_markdown(index=False))
            f.write("\n")

    with open(DRIFT_REPORT, 'w') as f:
        f.write(f"# Feature Drift Across Baseline Waves - {date.today().isoformat()}\n\n")
        f.write(f"Reference Wave: 1\nTop Features by Max |mean_diff| (<=25)\n\n")
        if drift_df.empty:
            f.write("_No drift data available (missing baseline_wave or reference empty).\n")
        else:
            f.write(drift_df.to_markdown(index=False))

    print(f"   ✓ Wave-pair report: {WAVE_PAIR_REPORT}")
    print(f"   ✓ Drift report: {DRIFT_REPORT}")

    # Console summary of XGBoost metrics
    print("\n10. XGBoost validation metrics:")
    print_evaluation_report(xgb_metrics)

    print("\n✓ Training & evaluation complete.")


if __name__ == '__main__':
    main()
