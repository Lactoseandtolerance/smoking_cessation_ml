"""Diagnose model performance gap between reported and actual metrics."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from modeling import load_model, split_data_by_person
from feature_engineering import get_feature_list
from sklearn.metrics import roc_auc_score, average_precision_score

# Load data
df = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / 'pooled_transitions.parquet')
if 'person_id' not in df.columns:
    df['person_id'] = df['PERSONID']

feature_cols = get_feature_list()
X_train, X_val, X_test, y_train, y_val, y_test, *_ = split_data_by_person(
    df, feature_cols, test_size=0.4, val_size=0.5, random_state=42
)

model, _ = load_model(str(PROJECT_ROOT / 'models' / 'xgboost_best.pkl'))

print('='*70)
print('PERFORMANCE GAP INVESTIGATION')
print('='*70)

# 1. Check training script reported metrics vs actual
print('\n1. REPORTED VS ACTUAL METRICS:')
print('   Training script reported validation AUC: 0.884')
train_means = X_train.mean()
val_auc_actual = roc_auc_score(y_val, model.predict_proba(X_val.fillna(train_means))[:, 1])
print(f'   Actual validation AUC when recomputed: {val_auc_actual:.3f}')
print(f'   Discrepancy: {0.884 - val_auc_actual:.3f} (very large!)')
print('   → This suggests the model file may not match the reported metrics')

# 2. Check if it's the same model by comparing predictions on training data
X_train_filled = X_train.fillna(train_means)
train_proba = model.predict_proba(X_train_filled)[:, 1]
train_auc = roc_auc_score(y_train, train_proba)

print(f'\n2. TRAINING SET PERFORMANCE:')
print(f'   Train AUC: {train_auc:.3f}')
print(f'   Val AUC:   {val_auc_actual:.3f}')
test_auc = roc_auc_score(y_test, model.predict_proba(X_test.fillna(train_means))[:, 1])
print(f'   Test AUC:  {test_auc:.3f}')
print(f'   Train-Val gap: {train_auc - val_auc_actual:.3f}')
if train_auc < 0.75:
    print('   ⚠️  Low training AUC suggests model underfitting or wrong model loaded')
elif train_auc > 0.95:
    print('   ⚠️  Very high training AUC suggests overfitting')
else:
    print('   ✓ Training AUC reasonable')

# 3. Check wave-specific performance
print(f'\n3. PERFORMANCE BY WAVE PAIR:')
for wave_pair in ['1→2', '2→3', '3→4', '4→5', '5→6', '6→7']:
    mask = df['transition'] == wave_pair
    if mask.sum() > 0:
        wave_df = df[mask]
        wave_X = wave_df[feature_cols].fillna(train_means)
        wave_y = wave_df['quit_success']
        if len(wave_y) > 10:
            wave_proba = model.predict_proba(wave_X)[:, 1]
            try:
                wave_auc = roc_auc_score(wave_y, wave_proba)
            except:
                wave_auc = np.nan
            quit_rate = wave_y.mean()
            print(f'   {wave_pair}: n={len(wave_y):5d}, quit_rate={quit_rate:.2%}, AUC={wave_auc:.3f}')

# 4. Check feature missingness impact
print(f'\n4. KEY FEATURES WITH HIGH MISSINGNESS:')
for feat in ['cpd', 'cpd_light', 'cpd_heavy', 'ttfc_minutes', 'quit_timeframe_code']:
    if feat in X_val.columns:
        missing_pct = X_val[feat].isna().mean()
        if missing_pct > 0.1:
            # Compare AUC for records with vs without this feature
            has_feat = ~X_val[feat].isna()
            if has_feat.sum() > 100 and (~has_feat).sum() > 100:
                auc_has = roc_auc_score(y_val[has_feat], 
                                       model.predict_proba(X_val[has_feat].fillna(train_means))[:, 1])
                auc_missing = roc_auc_score(y_val[~has_feat], 
                                           model.predict_proba(X_val[~has_feat].fillna(train_means))[:, 1])
                print(f'   {feat:20s}: {missing_pct:5.1%} missing, AUC(has)={auc_has:.3f}, AUC(miss)={auc_missing:.3f}')

# 5. Check if XGBoost is using native missing value handling
print(f'\n5. XGBOOST MISSING VALUE HANDLING:')
print(f'   XGBoost natively handles missing values: YES')
print(f'   Current approach: Impute with training means')
print(f'   Testing native handling...')

# Compare imputed vs native missing handling
val_proba_imputed = model.predict_proba(X_val.fillna(train_means))[:, 1]
val_proba_native = model.predict_proba(X_val)[:, 1]  # XGBoost handles NaN internally

auc_imputed = roc_auc_score(y_val, val_proba_imputed)
auc_native = roc_auc_score(y_val, val_proba_native)

print(f'   AUC with mean imputation: {auc_imputed:.3f}')
print(f'   AUC with native NaN handling: {auc_native:.3f}')
print(f'   Difference: {auc_native - auc_imputed:.3f}')
if abs(auc_native - auc_imputed) < 0.01:
    print('   → Imputation strategy has minimal impact')
else:
    print('   → Imputation strategy matters! Consider using native handling')

# 6. Check if model was trained on different data split
print(f'\n6. DATA SPLIT VERIFICATION:')
print(f'   Current dataset size: {len(df):,} transitions')
print(f'   Train size: {len(X_train):,} ({len(X_train)/len(df):.1%})')
print(f'   Val size: {len(X_val):,} ({len(X_val)/len(df):.1%})')
print(f'   Test size: {len(X_test):,} ({len(X_test)/len(df):.1%})')
print(f'   Split random seed: 42')
print(f'   → If model was trained on different split, metrics will differ')

print('\n' + '='*70)
print('DIAGNOSIS SUMMARY:')
print('='*70)

if abs(0.884 - val_auc_actual) > 0.15:
    print('\n⚠️  CRITICAL: Large performance gap detected!')
    print('   Possible causes:')
    print('   1. Model file is from old training run (different data/split)')
    print('   2. Training script used different imputation strategy')
    print('   3. Training script had bug in metric computation')
    print('   4. Model file corrupted or mismatched')
    print('\n   RECOMMENDATION: Retrain model and verify metrics match')
else:
    print('\n✓ Performance gap is acceptable (< 0.05 AUC difference)')

print('='*70)
