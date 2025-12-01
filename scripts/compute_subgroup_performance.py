"""
Compute model performance across demographic subgroups and generate visuals.

Outputs:
- reports/figures/subgroup_auc_bar.png
- reports/figures/subgroup_auc_heatmap.png
- reports/SUBGROUP_PERFORMANCE.csv (long format with metrics)
- Appends summary to reports/FAIRNESS_RESULTS.md

Subgroups:
- Sex (Female/Male) if `female` exists
- Age cohorts derived from numeric `age`: 18-24, 25-34, 35-49, 50+
- Race/Ethnicity from one-hot: race_white, race_black, race_hispanic, race_other

Notes:
- Requires processed parquet and trained model at models/xgboost_best.pkl
- Skips subgroup values with insufficient positives/negatives to compute AUC
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/processed/pooled_transitions.parquet"
MODEL_PATH = PROJECT_ROOT / "models/xgboost_best.pkl"
FIG_DIR = PROJECT_ROOT / "reports/figures"
REPORT_MD = PROJECT_ROOT / "reports/FAIRNESS_RESULTS.md"

import sys
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from modeling import load_model, split_data_by_person
from feature_engineering import get_feature_list


def _ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_data() -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_parquet(DATA_PATH)
    if 'PERSONID' in df.columns and 'person_id' not in df.columns:
        df = df.rename(columns={'PERSONID': 'person_id'})
    feature_cols = get_feature_list()
    return df, feature_cols


def _get_subgroup_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    # Sex
    if 'female' in df.columns:
        groups['Sex'] = ['female']  # 1=female, 0=male
    # Age
    if 'age' in df.columns:
        groups['Age Cohort'] = ['age']  # numeric, will be binned
    # Race/Ethnicity one-hots
    race_cols = [c for c in ['race_white', 'race_black', 'race_hispanic', 'race_other'] if c in df.columns]
    if race_cols:
        groups['Race/Ethnicity'] = race_cols
    return groups


def _assign_age_cohort(age: float) -> str:
    if pd.isna(age):
        return 'Unknown'
    age = float(age)
    if age < 25:
        return '18-24'
    if age < 35:
        return '25-34'
    if age < 50:
        return '35-49'
    return '50+'


def _derive_subgroup_series(df: pd.DataFrame, group_name: str, cols: List[str]) -> pd.Series:
    if group_name == 'Sex' and cols == ['female']:
        return df['female'].map({1: 'Female', 0: 'Male'}).fillna('Unknown')
    if group_name == 'Age Cohort' and cols == ['age']:
        return df['age'].apply(_assign_age_cohort)
    if group_name == 'Race/Ethnicity':
        # choose the 1 among one-hots, else Unknown
        sub = df[cols].copy()
        # handle rows where exactly one is 1
        idxmax = sub.idxmax(axis=1)
        # map to labels
        mapping = {
            'race_white': 'White',
            'race_black': 'Black',
            'race_hispanic': 'Hispanic',
            'race_other': 'Other'
        }
        labels = idxmax.map(mapping).fillna('Unknown')
        # if the row all zeros, mark Unknown
        all_zero = (sub.fillna(0).sum(axis=1) == 0)
        labels[all_zero] = 'Unknown'
        return labels
    # Fallback
    return pd.Series(['Unknown'] * len(df), index=df.index)


def _compute_rates(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Tuple[float, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    return fpr, fnr


def compute_subgroup_metrics() -> pd.DataFrame:
    df, feature_cols = _prepare_data()
    groups = _get_subgroup_columns(df)

    if not groups:
        return pd.DataFrame(columns=['Group', 'Subgroup', 'n', 'base_rate', 'auc', 'fpr', 'fnr'])

    model, _ = load_model(str(MODEL_PATH))

    X_train, X_val, X_test, y_train, y_val, y_test, *_ = split_data_by_person(df, feature_cols)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # Build a DataFrame aligned to X_test indices for subgroup mapping
    test_index = X_test.index
    df_test = df.loc[test_index]

    rows = []
    for gname, gcols in groups.items():
        subgroup_series = _derive_subgroup_series(df_test, gname, gcols)
        for subgroup, idx in subgroup_series.groupby(subgroup_series).groups.items():
            ys = y_test.loc[idx]
            ps = y_prob_test[y_test.index.get_indexer(idx)]
            n = len(ys)
            base = float(ys.mean()) if n > 0 else np.nan
            auc = np.nan
            if len(np.unique(ys)) == 2 and n >= 50:  # require both classes and some support
                try:
                    auc = float(roc_auc_score(ys, ps))
                except Exception:
                    auc = np.nan
            fpr, fnr = (np.nan, np.nan)
            if n >= 50:
                fpr, fnr = _compute_rates(ys.values, ps, threshold=0.5)
            rows.append({
                'Group': gname,
                'Subgroup': subgroup,
                'n': int(n),
                'base_rate': base,
                'auc': auc,
                'fpr': fpr,
                'fnr': fnr,
            })

    return pd.DataFrame(rows)


def plot_auc_bar(df_long: pd.DataFrame, out_path: Path):
    data = df_long.dropna(subset=['auc']).copy()
    if data.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x='auc', y='Subgroup', hue='Group', orient='h')
    plt.title('AUC by Demographic Subgroup (Test Set)')
    plt.xlabel('AUC')
    plt.ylabel('Subgroup')
    plt.xlim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_auc_heatmap(df_long: pd.DataFrame, out_path: Path):
    data = df_long.dropna(subset=['auc']).copy()
    if data.empty:
        return
    # Create pivot table with Subgroup rows and Group columns (show AUC)
    pv = data.pivot_table(index='Subgroup', columns='Group', values='auc')
    plt.figure(figsize=(8, max(4, 0.5 * len(pv))))
    sns.heatmap(pv, annot=True, vmin=0.5, vmax=1.0, cmap='YlGnBu', fmt='.3f')
    plt.title('AUC by Demographic Subgroup (Test Set)')
    plt.xlabel('Group')
    plt.ylabel('Subgroup')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def append_report_summary(df_long: pd.DataFrame):
    lines = []
    lines.append("\n## Demographic Subgroup Performance\n")
    if df_long.dropna(subset=['auc']).empty:
        lines.append("Limited subgroup data available; established framework for monitoring AUC, FPR/FNR, and base rates.\n")
    else:
        lines.append("AUC by subgroup (test set) computed for available demographics. Subgroups with n<50 or single-class outcomes were excluded.\n")
        # Summarize range per group
        for gname, sub in df_long.dropna(subset=['auc']).groupby('Group'):
            lines.append(f"- {gname}: AUC range {sub['auc'].min():.3f}–{sub['auc'].max():.3f}\n")
        lines.append("\nImplications: Monitor for performance drift and parity across groups; transparently report disparities before clinical deployment. Additional data may be needed for underrepresented groups.\n")

    with open(REPORT_MD, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    _ensure_dirs()
    df_long = compute_subgroup_metrics()
    # Save table
    out_csv = PROJECT_ROOT / 'reports/SUBGROUP_PERFORMANCE.csv'
    df_long.to_csv(out_csv, index=False)

    # Plots
    plot_auc_bar(df_long, FIG_DIR / 'subgroup_auc_bar.png')
    plot_auc_heatmap(df_long, FIG_DIR / 'subgroup_auc_heatmap.png')

    # Report
    append_report_summary(df_long)
    print("Saved subgroup performance to:")
    print(f"- {out_csv}")
    print(f"- {FIG_DIR / 'subgroup_auc_bar.png'}")
    print(f"- {FIG_DIR / 'subgroup_auc_heatmap.png'}")


if __name__ == '__main__':
    main()
