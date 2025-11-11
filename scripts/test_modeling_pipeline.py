#!/usr/bin/env python3
"""
Quick test of the modeling pipeline with the Phase 3 sample data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modeling import (
    split_data_by_person,
    train_logistic_regression,
    train_random_forest,
    train_xgboost
)
from src.evaluation import evaluate_model, print_evaluation_report

def main():
    print("="*70)
    print("MODELING PIPELINE TEST")
    print("="*70)
    
    # Load data
    data_path = Path(__file__).parent.parent / 'data/processed/engineered_phase3_sample.parquet'
    print(f"\n1. Loading data from {data_path.name}...")
    df = pd.read_parquet(data_path)
    print(f"   ✓ Loaded: {df.shape[0]} rows × {df.shape[1]} features")
    
    # Create synthetic target
    print("\n2. Creating synthetic target variable...")
    np.random.seed(42)
    base_prob = 0.3
    prob = np.full(len(df), base_prob)
    
    if 'high_dependence' in df.columns:
        prob = np.where(df['high_dependence'] == 0, prob + 0.15, prob - 0.1)
    if 'used_any_method' in df.columns:
        prob = np.where(df['used_any_method'] == 1, prob + 0.2, prob)
    
    prob = np.clip(prob, 0.05, 0.95)
    df['quit_success'] = np.random.binomial(1, prob)
    df['person_id'] = range(len(df))
    
    print(f"   ✓ Quit success rate: {df['quit_success'].mean():.3f}")
    
    # Prepare features
    print("\n3. Preparing feature list...")
    exclude_cols = ['quit_success', 'person_id', 'race_ethnicity']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"   ✓ Using {len(feature_cols)} features")
    
    # Split data
    print("\n4. Splitting data (60/20/20)...")
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = split_data_by_person(
        df, feature_cols, test_size=0.4, val_size=0.5, random_state=42
    )
    
    # Train models
    print("\n5. Training models...")
    print("   a) Logistic Regression...")
    lr_model, lr_scaler, y_val_pred_lr, y_val_proba_lr = train_logistic_regression(
        X_train, y_train, X_val, y_val
    )
    print("      ✓ Done")
    
    print("   b) Random Forest...")
    rf_model, y_val_pred_rf, y_val_proba_rf = train_random_forest(
        X_train, y_train, X_val, y_val
    )
    print("      ✓ Done")
    
    print("   c) XGBoost...")
    xgb_model, y_val_pred_xgb, y_val_proba_xgb = train_xgboost(
        X_train, y_train, X_val, y_val
    )
    print("      ✓ Done")
    
    # Evaluate
    print("\n6. Evaluating models...")
    lr_metrics = evaluate_model(y_val, y_val_pred_lr, y_val_proba_lr, "Logistic Regression")
    rf_metrics = evaluate_model(y_val, y_val_pred_rf, y_val_proba_rf, "Random Forest")
    xgb_metrics = evaluate_model(y_val, y_val_pred_xgb, y_val_proba_xgb, "XGBoost")
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print_evaluation_report(lr_metrics)
    print("\n")
    print_evaluation_report(rf_metrics)
    print("\n")
    print_evaluation_report(xgb_metrics)
    
    # Comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    comparison_df = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics])
    display_cols = ['model', 'roc_auc', 'pr_auc', 'precision', 'recall', 'f1']
    print(comparison_df[display_cols].round(3).to_string(index=False))
    
    best_idx = comparison_df['roc_auc'].idxmax()
    best_model = comparison_df.loc[best_idx, 'model']
    best_auc = comparison_df.loc[best_idx, 'roc_auc']
    print(f"\n🏆 Best model: {best_model} (ROC-AUC: {best_auc:.3f})")
    
    print("\n" + "="*70)
    print("✓ MODELING PIPELINE TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
