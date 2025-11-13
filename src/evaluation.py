"""
Evaluation and metrics utilities for smoking cessation prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, 
    precision_recall_curve, auc,
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)


def evaluate_model(y_true, y_pred, y_pred_proba, model_name="Model"):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        model_name: Name of model for display
        
    Returns:
        dict: Dictionary of metrics
    """
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Classification metrics
    precision_val = precision_score(y_true, y_pred, zero_division=0)
    recall_val = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'model': model_name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'precision': precision_val,
        'recall': recall_val,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }
    
    return metrics


def print_evaluation_report(metrics):
    """
    Print formatted evaluation report.
    
    Args:
        metrics: Dictionary of metrics from evaluate_model()
    """
    print("=" * 50)
    print(f"{metrics['model'].upper()} RESULTS")
    print("=" * 50)
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"PR-AUC: {metrics['pr_auc']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1']:.3f}")
    print("\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['tp']}")
    print(f"  True Negatives:  {metrics['tn']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")


def plot_roc_curve(y_true, y_pred_proba, model_name="Model", ax=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name for legend
        ax: Matplotlib axis (optional)
        
    Returns:
        Matplotlib axis
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(alpha=0.3)
    
    return ax


def plot_precision_recall_curve(y_true, y_pred_proba, model_name="Model", ax=None):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name for legend
        ax: Matplotlib axis (optional)
        
    Returns:
        Matplotlib axis
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(alpha=0.3)
    
    return ax


def plot_confusion_matrix(y_true, y_pred, model_name="Model", ax=None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name for title
        ax: Matplotlib axis (optional)
        
    Returns:
        Matplotlib axis
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_xticklabels(['Failed', 'Success'])
    ax.set_yticklabels(['Failed', 'Success'])
    
    return ax


def evaluate_fairness(model, X_test, y_test, test_data, group_var, scaler=None):
    """
    Evaluate model performance across demographic subgroups.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        test_data: Full test dataset with demographic variables
        group_var: Name of demographic variable (e.g., 'gender', 'age_cohort')
        scaler: Optional scaler for linear models
        
    Returns:
        pd.DataFrame: Performance metrics by subgroup
    """
    results = []
    
    for group_value in test_data[group_var].unique():
        mask = test_data[group_var] == group_value
        
        if mask.sum() == 0:
            continue
        
        X_sub = X_test[mask]
        y_sub = y_test[mask]
        
        # Prepare data
        if scaler is not None:
            X_sub_prepared = scaler.transform(X_sub)
        else:
            X_sub_prepared = X_sub
        
        # Predict
        y_pred_proba = model.predict_proba(X_sub_prepared)[:, 1]
        y_pred = model.predict(X_sub_prepared)
        
        # Calculate metrics
        try:
            roc_auc = roc_auc_score(y_sub, y_pred_proba)
        except:
            roc_auc = np.nan
        
        precision_val = precision_score(y_sub, y_pred, zero_division=0)
        recall_val = recall_score(y_sub, y_pred, zero_division=0)
        f1 = f1_score(y_sub, y_pred, zero_division=0)
        
        # False positive/negative rates
        tn, fp, fn, tp = confusion_matrix(y_sub, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        
        results.append({
            'subgroup': f"{group_var}={group_value}",
            'group_variable': group_var,
            'n': len(y_sub),
            'base_rate': y_sub.mean(),
            'auc': roc_auc,
            'precision': precision_val,
            'recall': recall_val,
            'f1': f1,
            'fpr': fpr,
            'fnr': fnr
        })
    
    return pd.DataFrame(results)


def calculate_disparities(fairness_df, metric='auc'):
    """
    Calculate disparity metrics across demographic groups.
    
    Args:
        fairness_df: DataFrame from evaluate_fairness()
        metric: Metric to analyze ('auc', 'precision', etc.)
        
    Returns:
        pd.DataFrame: Disparity statistics by group variable
    """
    disparities = []

    # Guard empty input
    if fairness_df is None or len(fairness_df) == 0 or 'group_variable' not in fairness_df.columns:
        return pd.DataFrame(columns=['group_variable', 'max', 'min', 'disparity', 'significant'])

    for group_var in fairness_df['group_variable'].unique():
        group_data = fairness_df[fairness_df['group_variable'] == group_var]
        
        if len(group_data) > 1:
            max_val = group_data[metric].max()
            min_val = group_data[metric].min()
            disparity = max_val - min_val
            
            disparities.append({
                'group_variable': group_var,
                'max': max_val,
                'min': min_val,
                'disparity': disparity,
                'significant': disparity > 0.05  # Flag if > 0.05 difference
            })
    
    if len(disparities) == 0:
        return pd.DataFrame(columns=['group_variable', 'max', 'min', 'disparity', 'significant'])
    return pd.DataFrame(disparities).sort_values('disparity', ascending=False)
