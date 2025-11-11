"""Generate SHAP interpretability plots and fairness metrics for the trained XGBoost model.

Outputs:
  - reports/figures/shap_summary.png
  - reports/figures/shap_dependence_<feature>.png (selected top features)
  - reports/FAIRNESS_RESULTS.md (per-group metrics & disparities)
  - reports/INTERPRETABILITY_SUMMARY.md (narrative summary)

Fairness groups: sex, age_cohort, race_ethnicity (via race_* dummies where needed).

Run:
  PYTHONPATH=. python scripts/run_interpretability_and_fairness.py
"""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import split_data_by_person, load_model
from src.feature_engineering import get_feature_list, engineer_all_features
from src.evaluation import evaluate_fairness, calculate_disparities

DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'pooled_transitions.parquet'
MODEL_PATH = PROJECT_ROOT / 'models' / 'xgboost_best.pkl'
FIG_DIR = PROJECT_ROOT / 'reports' / 'figures'
REPORT_FAIRNESS = PROJECT_ROOT / 'reports' / 'FAIRNESS_RESULTS.md'
REPORT_INTERP = PROJECT_ROOT / 'reports' / 'INTERPRETABILITY_SUMMARY.md'

TOP_N_FEATURES = 10  # number of top features to create dependence plots for


def load_and_prepare():
    df = pd.read_parquet(DATA_PATH)
    if 'person_id' not in df.columns and 'PERSONID' in df.columns:
        df['person_id'] = df['PERSONID']
    # Ensure feature columns exist as during training
    df = engineer_all_features(df, recode_missing=False)
    feature_cols = get_feature_list()
    return df, feature_cols


def get_splits(df, feature_cols):
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = split_data_by_person(
        df, feature_cols, test_size=0.4, val_size=0.5, random_state=42
    )
    # Fill train means for consistency
    train_means = X_train.mean()
    X_train_filled = X_train.fillna(train_means)
    X_val_filled = X_val.fillna(train_means)
    X_test_filled = X_test.fillna(train_means)
    return X_train_filled, X_val_filled, X_test_filled, y_train, y_val, y_test


def compute_shap(model, X_background, X_eval):
    """Compute SHAP values using the permutation/Kernel fallback for robustness.

    TreeExplainer currently errors on xgboost>=2.0 due to base_score format '[5E-1]'.
    We fall back to shap.Explainer (which chooses a safe path) or KernelExplainer
    with a small background sample to remain performant.
    """
    if len(X_background) > 500:
        background = X_background.sample(500, random_state=42)
    else:
        background = X_background
    try:
        # Generic interface picks appropriate explainer; may still hit the bug
        explainer = shap.Explainer(model, background)
        shap_values = explainer(X_eval)
        values = getattr(shap_values, 'values', shap_values)
        return explainer, values, X_eval
    except Exception:
        # Fallback to KernelExplainer on model.predict_proba
        def f(X):
            return model.predict_proba(pd.DataFrame(X, columns=X_background.columns))[:, 1]
        kernel_explainer = shap.KernelExplainer(f, background)
        eval_subset = X_eval.iloc[:1000]
        values = kernel_explainer.shap_values(eval_subset, nsamples=100)
        return kernel_explainer, values, eval_subset


def save_shap_plots(model, X_used, shap_values, feature_cols):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Summary plot (bar)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_used, feature_names=feature_cols, show=False)
    plt.tight_layout()
    summary_path = FIG_DIR / 'shap_summary.png'
    plt.savefig(summary_path, dpi=150)
    plt.close()

    # Identify top features by mean absolute SHAP
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:TOP_N_FEATURES]
    top_features = [feature_cols[i] for i in top_idx]

    dependence_paths = []
    for feat in top_features:
        plt.figure(figsize=(7,5))
        shap.dependence_plot(feat, shap_values, X_used, feature_names=feature_cols, show=False)
        dep_path = FIG_DIR / f'shap_dependence_{feat}.png'
        plt.tight_layout()
        plt.savefig(dep_path, dpi=150)
        plt.close()
        dependence_paths.append(dep_path)
    return summary_path, dependence_paths, top_features


def compute_fairness(model, X_test, y_test, test_df):
    # We expect demographics: 'sex', 'age_cohort', race dummies. Evaluate separately.
    rows = []
    fairness_frames = []
    group_vars = []
    # If age_cohort absent (all NaN), skip
    if 'age_cohort' in test_df.columns and test_df['age_cohort'].notna().any():
        group_vars.append('age_cohort')
    if 'sex' in test_df.columns:
        group_vars.append('sex')
    if 'race_ethnicity' in test_df.columns:
        group_vars.append('race_ethnicity')

    for gv in group_vars:
        f_df = evaluate_fairness(model, X_test, y_test, test_df, group_var=gv)
        # evaluate_fairness already populates group_variable per row; ensure consistency
        fairness_frames.append(f_df)
        disp = calculate_disparities(f_df, metric='auc')
        if not disp.empty:
            for _, r in disp.iterrows():
                rows.append({
                    'group_variable': r['group_variable'],
                    'max_auc': r['max'],
                    'min_auc': r['min'],
                    'disparity': r['disparity'],
                    'significant': r['significant']
                })
    fairness_all = pd.concat(fairness_frames, ignore_index=True) if fairness_frames else pd.DataFrame()
    disparities = pd.DataFrame(rows)
    return fairness_all, disparities


def write_reports(top_features, fairness_all, disparities):
    REPORT_FAIRNESS.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_INTERP, 'w') as f:
        f.write('# Interpretability Summary\n\n')
        f.write('Top SHAP features (by mean absolute importance):\n')
        for i, feat in enumerate(top_features, 1):
            f.write(f"{i}. {feat}\n")
        f.write('\nUse dependency plots to inspect non-linear effects and interactions.\n')

    with open(REPORT_FAIRNESS, 'w') as f:
        f.write('# Fairness Results\n\n')
        if fairness_all.empty:
            f.write('No fairness groups available in test dataset.\n')
        else:
            f.write('## Per-Group Performance\n\n')
            cols = ['group_variable','subgroup','n','base_rate','auc','precision','recall','f1','fpr','fnr']
            f.write(fairness_all[cols].to_markdown(index=False))
            f.write('\n\n## Disparities (AUC differences)\n\n')
            if disparities.empty:
                f.write('No multi-group disparities computed.\n')
            else:
                f.write(disparities.to_markdown(index=False))
                sig = disparities[disparities['significant']]
                if not sig.empty:
                    f.write('\n\nGroups with disparity > 0.05 flagged as significant.\n')


def main():
    assert DATA_PATH.exists(), f'Missing data: {DATA_PATH}'
    assert MODEL_PATH.exists(), f'Missing model: {MODEL_PATH}'

    df, feature_cols = load_and_prepare()
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(df, feature_cols)

    model, _ = load_model(str(MODEL_PATH))

    # SHAP on validation set for cleaner patterns (unseen during training early stops) but can also do test
    explainer, shap_vals_val, X_used = compute_shap(model, X_train, X_val)
    summary_path, dependence_paths, top_features = save_shap_plots(model, X_used, shap_vals_val, feature_cols)

    # Fairness on test set (generalization assessment)
    test_df = df[df['person_id'].isin(df['person_id'].unique())]  # original pooled; filter later
    # Reconstruct test person ids to align rows
    _, _, _, _, _, _, train_ids, val_ids, test_ids = split_data_by_person(df, feature_cols, test_size=0.4, val_size=0.5, random_state=42)
    test_mask = df['person_id'].isin(test_ids)
    fairness_all, disparities = compute_fairness(model, X_test, y_test, df.loc[test_mask].copy())

    write_reports(top_features, fairness_all, disparities)

    print('\nGenerated SHAP plots:')
    print(' -', summary_path)
    for p in dependence_paths:
        print(' -', p)
    print('\nFairness and interpretability reports written to:')
    print(' -', REPORT_INTERP)
    print(' -', REPORT_FAIRNESS)


if __name__ == '__main__':
    main()
