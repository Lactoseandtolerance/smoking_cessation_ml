"""Generate interpretability plots and fairness metrics for the trained XGBoost model.

SIMPLIFIED APPROACH (No SHAP dependency):
- Use XGBoost's built-in feature importance (gain, weight, cover)
- Generate partial dependence plots for top features
- Create individual prediction breakdowns
- Compare feature distributions across outcome groups
- Fairness analysis across demographic subgroups

Outputs:
  - reports/figures/feature_importance_gain.png (what drives splits)
  - reports/figures/feature_importance_weight.png (how often features are used)
  - reports/figures/partial_dependence_top5.png (feature effects)
  - reports/figures/prediction_breakdown_examples.png (individual cases)
  - reports/figures/feature_distributions_by_outcome.png
  - reports/FAIRNESS_RESULTS.md
  - reports/INTERPRETABILITY_SUMMARY.md (actionable recommendations)

Run:
    python scripts/run_interpretability_simple.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from modeling import load_model, split_data_by_person
from feature_engineering import engineer_all_features, get_feature_list
from evaluation import evaluate_fairness, calculate_disparities

DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'pooled_transitions.parquet'
MODEL_PATH = PROJECT_ROOT / 'models' / 'xgboost_best.pkl'
FIG_DIR = PROJECT_ROOT / 'reports' / 'figures'
REPORT_FAIRNESS = PROJECT_ROOT / 'reports' / 'FAIRNESS_RESULTS.md'
REPORT_INTERP = PROJECT_ROOT / 'reports' / 'INTERPRETABILITY_SUMMARY.md'

TOP_N_FEATURES = 10  # Top features to analyze in detail
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100


def load_and_prepare():
    df = pd.read_parquet(DATA_PATH)
    if 'person_id' not in df.columns and 'PERSONID' in df.columns:
        df['person_id'] = df['PERSONID']
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


def get_feature_importance(model, feature_names):
    """Extract XGBoost feature importance scores."""
    importance_dict = {}
    
    for imp_type in ['gain', 'weight', 'cover']:
        scores = model.get_booster().get_score(importance_type=imp_type)
        # Normalize to 0-100 scale
        total = sum(scores.values())
        importance_dict[imp_type] = {k: (v/total)*100 for k, v in scores.items()}
    
    # Create DataFrame with all importance types
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'gain': [importance_dict['gain'].get(f, 0) for f in feature_names],
        'weight': [importance_dict['weight'].get(f, 0) for f in feature_names],
        'cover': [importance_dict['cover'].get(f, 0) for f in feature_names]
    })
    
    importance_df = importance_df.sort_values('gain', ascending=False)
    return importance_df


def plot_feature_importance(importance_df, top_n=15):
    """Generate feature importance bar charts."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Gain (most interpretable - avg improvement in loss)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = importance_df.head(top_n)
    ax.barh(range(len(top_features)), top_features['gain'], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (% of total gain)', fontsize=11)
    ax.set_title('Top Features by Gain (Average Split Quality)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    gain_path = FIG_DIR / 'feature_importance_gain.png'
    plt.savefig(gain_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Weight (how often used)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_features)), top_features['weight'], color='coral')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (% of total splits)', fontsize=11)
    ax.set_title('Top Features by Weight (Split Frequency)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    weight_path = FIG_DIR / 'feature_importance_weight.png'
    plt.savefig(weight_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Feature importance plots saved:")
    print(f"  - {gain_path.name}")
    print(f"  - {weight_path.name}")
    
    return gain_path, weight_path


def plot_partial_dependence(model, X_data, feature_names, top_features, n_features=5):
    """Generate partial dependence plots for top features."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Select top N features by importance
    features_to_plot = [f for f in top_features[:n_features] if f in feature_names]
    feature_indices = [feature_names.index(f) for f in features_to_plot]
    
    # Create partial dependence plot
    fig, ax = plt.subplots(figsize=(14, 8))
    display = PartialDependenceDisplay.from_estimator(
        model,
        X_data,
        features=feature_indices,
        feature_names=feature_names,
        grid_resolution=50,
        ax=ax,
        n_cols=3,
        line_kw={'color': 'steelblue', 'linewidth': 2}
    )
    
    plt.suptitle('Partial Dependence: Effect of Top Features on Quit Probability', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    pd_path = FIG_DIR / 'partial_dependence_top5.png'
    plt.savefig(pd_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Partial dependence plot saved: {pd_path.name}")
    return pd_path


def plot_feature_distributions(X_data, y_data, top_features, n_features=6):
    """Compare feature distributions between quit/no-quit groups."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feat in enumerate(top_features[:n_features]):
        if feat not in X_data.columns:
            continue
            
        ax = axes[i]
        
        # Split by outcome
        quit_vals = X_data.loc[y_data == 1, feat].dropna()
        no_quit_vals = X_data.loc[y_data == 0, feat].dropna()
        
        # Determine if categorical (few unique values) or continuous
        n_unique = X_data[feat].nunique()
        
        if n_unique <= 5:  # Categorical
            # Bar chart
            quit_counts = quit_vals.value_counts(normalize=True).sort_index()
            no_quit_counts = no_quit_vals.value_counts(normalize=True).sort_index()
            
            x = np.arange(len(quit_counts))
            width = 0.35
            ax.bar(x - width/2, no_quit_counts, width, label='No Quit', color='lightcoral', alpha=0.8)
            ax.bar(x + width/2, quit_counts, width, label='Quit', color='seagreen', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(quit_counts.index)
            ax.set_ylabel('Proportion')
        else:  # Continuous
            # Overlapping histograms
            ax.hist(no_quit_vals, bins=30, alpha=0.6, label='No Quit', color='lightcoral', density=True)
            ax.hist(quit_vals, bins=30, alpha=0.6, label='Quit', color='seagreen', density=True)
            ax.set_ylabel('Density')
        
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Feature Distributions: Quit vs No Quit', fontsize=14, fontweight='bold')
    plt.tight_layout()
    dist_path = FIG_DIR / 'feature_distributions_by_outcome.png'
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Distribution comparison saved: {dist_path.name}")
    return dist_path


def analyze_individual_predictions(model, X_data, y_data, importance_df, n_examples=4):
    """Show prediction breakdowns for representative cases."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get predictions
    proba = model.predict_proba(X_data)[:, 1]
    
    # Select examples: high/low prediction × actual outcome
    high_quit_pred_correct = np.where((proba > 0.7) & (y_data == 1))[0]
    high_quit_pred_wrong = np.where((proba > 0.7) & (y_data == 0))[0]
    low_quit_pred_correct = np.where((proba < 0.3) & (y_data == 0))[0]
    low_quit_pred_wrong = np.where((proba < 0.3) & (y_data == 1))[0]
    
    examples = []
    if len(high_quit_pred_correct) > 0:
        examples.append(('High Pred, Quit ✓', high_quit_pred_correct[0]))
    if len(high_quit_pred_wrong) > 0:
        examples.append(('High Pred, No Quit ✗', high_quit_pred_wrong[0]))
    if len(low_quit_pred_correct) > 0:
        examples.append(('Low Pred, No Quit ✓', low_quit_pred_correct[0]))
    if len(low_quit_pred_wrong) > 0:
        examples.append(('Low Pred, Quit ✗', low_quit_pred_wrong[0]))
    
    # Plot top features for each example
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    top_feats = importance_df.head(10)['feature'].tolist()
    
    for i, (label, idx) in enumerate(examples[:4]):
        ax = axes[i]
        
        # Get feature values for this person
        feat_vals = X_data.iloc[idx][top_feats].values
        
        # Normalize for visualization (show deviation from mean)
        feat_means = X_data[top_feats].mean().values
        feat_stds = X_data[top_feats].std().values
        normalized = (feat_vals - feat_means) / (feat_stds + 1e-10)
        
        colors = ['green' if v > 0 else 'red' for v in normalized]
        ax.barh(range(len(top_feats)), normalized, color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels(top_feats, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Std Deviations from Mean', fontsize=10)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(f'{label}\nPred: {proba[idx]:.2f}, Actual: {y_data.iloc[idx]}', 
                     fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Individual Prediction Examples: Top Feature Values', fontsize=14, fontweight='bold')
    plt.tight_layout()
    examples_path = FIG_DIR / 'prediction_breakdown_examples.png'
    plt.savefig(examples_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Individual examples saved: {examples_path.name}")
    return examples_path


def compute_fairness(model, X_test, y_test, test_df):
    """Compute fairness metrics across demographic groups."""
    rows = []
    fairness_frames = []
    group_vars = []
    
    if 'age_cohort' in test_df.columns and test_df['age_cohort'].notna().any():
        group_vars.append('age_cohort')
    if 'sex' in test_df.columns:
        group_vars.append('sex')
    if 'race_ethnicity' in test_df.columns:
        group_vars.append('race_ethnicity')

    for gv in group_vars:
        f_df = evaluate_fairness(model, X_test, y_test, test_df, group_var=gv)
        if f_df is not None and not f_df.empty:
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


def write_reports(importance_df, fairness_all, disparities):
    """Generate actionable interpretability and fairness reports."""
    REPORT_FAIRNESS.parent.mkdir(parents=True, exist_ok=True)

    # Interpretability report
    with open(REPORT_INTERP, 'w') as f:
        f.write('# Model Interpretability Summary\n\n')
        f.write('## Top Predictive Features (XGBoost Feature Importance)\n\n')
        f.write('| Rank | Feature | Gain (%) | Weight (%) | Clinical Interpretation |\n')
        f.write('|------|---------|----------|------------|---------------------------|\n')
        
        interpretations = {
            'high_dependence': 'Higher nicotine dependence reduces quit success',
            'very_high_dependence': 'Very high dependence is a major barrier',
            'used_varenicline': 'Varenicline increases quit success (evidence-based)',
            'used_counseling': 'Counseling support improves outcomes',
            'used_nrt': 'NRT products aid cessation attempts',
            'cpd': 'Cigarettes per day indicates addiction severity',
            'cpd_light': 'Light smokers (<10/day) have higher quit rates',
            'cpd_heavy': 'Heavy smokers (20+/day) face more challenges',
            'ttfc_minutes': 'Time to first cigarette measures dependence',
            'age_young': 'Younger smokers may have different patterns',
            'longest_quit_duration': 'Prior quit success predicts future success',
            'plans_to_quit': 'Readiness/motivation is predictive',
            'smokefree_home': 'Supportive environment aids quitting',
            'used_any_medication': 'Pharmacotherapy improves quit rates',
            'quit_timeframe_code': 'Quit timing plans indicate motivation',
            'motivation_high': 'High motivation predicts success'
        }
        
        for i, row in importance_df.head(15).iterrows():
            feat = row['feature']
            interp = interpretations.get(feat, 'Feature impact on quit prediction')
            f.write(f"| {i+1} | `{feat}` | {row['gain']:.1f} | {row['weight']:.1f} | {interp} |\n")
        
        f.write('\n## Key Insights\n\n')
        f.write('### Methodology Note\n\n')
        f.write('**Importance Metrics:**\n')
        f.write('- **Gain**: Average improvement in model accuracy when this feature is used for splitting\n')
        f.write('- **Weight**: How frequently the feature is selected for splits across all trees\n')
        f.write('- **Cover**: Average number of samples affected by splits on this feature\n\n')
        
        f.write('### Actionable Recommendations\n\n')
        
        # Check what's in top features
        top_5 = importance_df.head(5)['feature'].tolist()
        
        if any(med in top_5 for med in ['used_varenicline', 'used_counseling', 'used_nrt', 'used_any_medication']):
            f.write('1. **Pharmacotherapy & Counseling**: Model shows strong importance of varenicline, NRT, and counseling. ')
            f.write('Prioritize offering these evidence-based interventions to smokers attempting to quit.\n\n')
        
        if any(dep in top_5 for dep in ['high_dependence', 'very_high_dependence', 'cpd', 'ttfc_minutes']):
            f.write('2. **Dependence Screening**: Nicotine dependence measures (TTFC, CPD, dependence score) are highly predictive. ')
            f.write('Screen for dependence level and tailor intervention intensity accordingly.\n\n')
        
        if any(env in top_5 for env in ['smokefree_home', 'household_smokers']):
            f.write('3. **Environmental Support**: Smokefree home rules and household composition matter. ')
            f.write('Address environmental triggers and encourage supportive home policies.\n\n')
        
        f.write('### Model Limitations\n\n')
        f.write('- Feature importance reflects associations, not causal effects.\n')
        f.write('- Missing data (especially CPD: 77% missing) handled via XGBoost native splitting.\n')
        f.write('- Interactions between features (e.g., medication × dependence) captured in model but not shown separately.\n\n')
        
        f.write('## Visualization Guide\n\n')
        f.write('- **Feature Importance**: Shows which features drive model decisions\n')
        f.write('- **Partial Dependence**: Shows how changing one feature affects predicted quit probability\n')
        f.write('- **Distribution Comparison**: Shows how feature values differ between quit/no-quit groups\n')
        f.write('- **Individual Examples**: Shows how features combine to produce predictions for specific cases\n\n')

    # Fairness report
    with open(REPORT_FAIRNESS, 'w') as f:
        f.write('# Fairness Analysis Results\n\n')
        if fairness_all.empty:
            f.write('No fairness groups available in test dataset.\n')
        else:
            f.write('## Model Performance Across Demographic Groups\n\n')
            f.write('### Per-Group Metrics\n\n')
            cols = ['group_variable','subgroup','n','base_rate','auc','precision','recall','f1','fpr','fnr']
            available_cols = [c for c in cols if c in fairness_all.columns]
            f.write(fairness_all[available_cols].to_markdown(index=False))
            
            f.write('\n\n### Disparity Analysis (AUC Differences)\n\n')
            if disparities.empty:
                f.write('No multi-group disparities detected.\n')
            else:
                f.write(disparities.to_markdown(index=False))
                f.write('\n\n')
                sig = disparities[disparities['significant']]
                if not sig.empty:
                    f.write('⚠️  **Significant disparities** (>0.05 AUC difference) detected in:\n')
                    for _, row in sig.iterrows():
                        f.write(f"- {row['group_variable']}: {row['disparity']:.3f} difference\n")
                    f.write('\nConsider calibration or reweighting strategies to address fairness concerns.\n')
                else:
                    f.write('✓ No significant disparities (>0.05) detected across groups.\n')
    
    print(f"\n✓ Reports written:")
    print(f"  - {REPORT_INTERP.name}")
    print(f"  - {REPORT_FAIRNESS.name}")


def main():
    assert DATA_PATH.exists(), f'Missing data: {DATA_PATH}'
    assert MODEL_PATH.exists(), f'Missing model: {MODEL_PATH}'

    print("Loading data and model...")
    df, feature_cols = load_and_prepare()
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(df, feature_cols)
    model, _ = load_model(str(MODEL_PATH))
    
    print(f"\nData splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # 1. Feature importance
    print("\n1. Computing feature importance...")
    importance_df = get_feature_importance(model, feature_cols)
    print(f"   Top 5 features: {', '.join(importance_df.head(5)['feature'].tolist())}")
    
    # 2. Plot importance
    print("\n2. Generating feature importance plots...")
    plot_feature_importance(importance_df, top_n=15)
    
    # 3. Partial dependence
    print("\n3. Generating partial dependence plots...")
    top_features = importance_df.head(TOP_N_FEATURES)['feature'].tolist()
    plot_partial_dependence(model, X_val, feature_cols, top_features, n_features=6)
    
    # 4. Distribution comparison
    print("\n4. Comparing feature distributions...")
    plot_feature_distributions(X_val, y_val, top_features, n_features=6)
    
    # 5. Individual examples
    print("\n5. Analyzing individual predictions...")
    analyze_individual_predictions(model, X_val, y_val, importance_df, n_examples=4)
    
    # 6. Fairness analysis
    print("\n6. Computing fairness metrics...")
    _, _, _, _, _, _, train_ids, val_ids, test_ids = split_data_by_person(df, feature_cols, test_size=0.4, val_size=0.5, random_state=42)
    test_mask = df['person_id'].isin(test_ids)
    fairness_all, disparities = compute_fairness(model, X_test, y_test, df.loc[test_mask].copy())
    
    # 7. Write reports
    print("\n7. Writing reports...")
    write_reports(importance_df, fairness_all, disparities)
    
    print("\n" + "="*60)
    print("✓ INTERPRETABILITY ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey Takeaways:")
    print(f"  • Top feature: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['gain']:.1f}% gain)")
    print(f"  • {len(importance_df[importance_df['gain'] > 1])} features contribute >1% each")
    print(f"  • Visualizations saved in {FIG_DIR}/")


if __name__ == '__main__':
    main()
