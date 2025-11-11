"""Run held-out test evaluation for XGBoost model.
This script re-loads the processed dataset, re-creates the person-level splits
with the same random_state, loads the saved best XGBoost model, and computes
final metrics on the test set.
"""
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
from datetime import date

# Ensure project root is on sys.path for `src` imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports from project
from src.modeling import split_data_by_person, load_model
from src.evaluation import evaluate_model, print_evaluation_report
from src.feature_engineering import get_feature_list, engineer_all_features

DATA_PATH = Path('data/processed/pooled_transitions.parquet')
MODEL_PATH = Path('models/xgboost_best.pkl')
REPORT_PATH = Path('reports/TEST_SET_RESULTS.md')


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return dataframe with exact feature set used in training.

    We avoid dynamically including extra columns (e.g., categorical labels like
    age_cohort or race_ethnicity) that the model was never trained on. Instead,
    we pull the canonical list from feature_engineering.get_feature_list(). If
    any expected feature is missing (possible if the processed parquet was
    produced before a new feature was added), we run engineer_all_features to
    create it, preserving existing columns.
    """
    df = df.copy()
    if 'person_id' not in df.columns and 'PERSONID' in df.columns:
        df['person_id'] = df['PERSONID']

    feature_cols = get_feature_list()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        # Re-run feature engineering to populate missing expected columns.
        df = engineer_all_features(df, recode_missing=False)
        still_missing = [c for c in feature_cols if c not in df.columns]
        if still_missing:
            # Create any truly absent columns as zeros to avoid KeyErrors.
            for c in still_missing:
                df[c] = 0
    return df, feature_cols


def main():
    assert DATA_PATH.exists(), f"Missing dataset: {DATA_PATH}"
    assert MODEL_PATH.exists(), f"Missing model: {MODEL_PATH}"

    # Load data
    df = pd.read_parquet(DATA_PATH)
    df, feature_cols = prepare_features(df)

    # Create person-level splits (60/20/20)
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = split_data_by_person(
        df, feature_cols, test_size=0.4, val_size=0.5, random_state=42
    )

    # Load model
    model, metadata = load_model(str(MODEL_PATH))

    # Fill NaNs with training means to mimic training-time handling
    train_means = X_train.mean()
    X_test_filled = X_test.fillna(train_means)

    # Predict
    y_pred_proba = model.predict_proba(X_test_filled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, model_name='XGBoost (Test)')
    print_evaluation_report(metrics)

    # Save concise report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        f.write(f"# Test Set Results - {date.today().isoformat()}\n\n")
        f.write("Model: XGBoost (best from validation)\n")
        f.write(f"Samples: {len(X_test):,}\n\n")
        f.write(f"- ROC-AUC: {metrics['roc_auc']:.3f}\n")
        f.write(f"- PR-AUC: {metrics['pr_auc']:.3f}\n")
        f.write(f"- F1: {metrics['f1']:.3f}\n")
        f.write(f"- Precision: {metrics['precision']:.3f}\n")
        f.write(f"- Recall: {metrics['recall']:.3f}\n\n")
        f.write(f"Train/Val/Test sizes: {len(X_train):,}/{len(X_val):,}/{len(X_test):,}\n")

    print(f"\nSaved test metrics to: {REPORT_PATH.resolve()}")


if __name__ == '__main__':
    main()
