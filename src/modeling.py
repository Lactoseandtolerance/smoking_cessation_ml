"""
Modeling utilities for smoking cessation prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from pathlib import Path


def split_data_by_person(pooled_data, feature_cols, test_size=0.4, val_size=0.5, random_state=42):
    """
    Split data by person_id to prevent data leakage.
    
    Same person should not appear in both train and test sets.
    
    Args:
        pooled_data: Person-period dataset
        feature_cols: List of feature column names
        test_size: Proportion for val+test combined
        val_size: Proportion of test_size for validation
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids)
    """
    # Get unique person IDs
    unique_persons = pooled_data['person_id'].unique()
    
    # Split persons into train (60%), temp (40%)
    train_ids, temp_ids = train_test_split(
        unique_persons, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Split temp into validation (20%) and test (20%)
    val_ids, test_ids = train_test_split(
        temp_ids, 
        test_size=val_size, 
        random_state=random_state
    )
    
    # Create data splits
    train_data = pooled_data[pooled_data['person_id'].isin(train_ids)].copy()
    val_data = pooled_data[pooled_data['person_id'].isin(val_ids)].copy()
    test_data = pooled_data[pooled_data['person_id'].isin(test_ids)].copy()
    
    # Extract features and outcomes
    X_train = train_data[feature_cols].copy()
    y_train = train_data['quit_success'].copy()
    
    X_val = val_data[feature_cols].copy()
    y_val = val_data['quit_success'].copy()
    
    X_test = test_data[feature_cols].copy()
    y_test = test_data['quit_success'].copy()
    
    print(f"Train size: {len(X_train)} ({len(train_ids)} persons)")
    print(f"Val size: {len(X_val)} ({len(val_ids)} persons)")
    print(f"Test size: {len(X_test)} ({len(test_ids)} persons)")
    print(f"\nTrain cessation rate: {y_train.mean():.3f}")
    print(f"Val cessation rate: {y_val.mean():.3f}")
    print(f"Test cessation rate: {y_test.mean():.3f}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids


def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train logistic regression with class weighting.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        Tuple of (model, scaler, predictions, probabilities)
    """
    # Fill NaN values with column means
    X_train_filled = X_train.fillna(X_train.mean())
    X_val_filled = X_val.fillna(X_train.mean())  # Use training means
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filled)
    X_val_scaled = scaler.transform(X_val_filled)
    
    # Train model with class weighting
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    
    lr.fit(X_train_scaled, y_train)
    
    # Predict on validation set
    y_val_pred_proba = lr.predict_proba(X_val_scaled)[:, 1]
    y_val_pred = lr.predict(X_val_scaled)
    
    return lr, scaler, y_val_pred, y_val_pred_proba


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train random forest with class weighting.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        Tuple of (model, predictions, probabilities)
    """
    # Fill NaN values with column means
    X_train_filled = X_train.fillna(X_train.mean())
    X_val_filled = X_val.fillna(X_train.mean())
    
    # Train model with class weighting
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train_filled, y_train)
    
    # Predict on validation set
    y_val_pred_proba = rf.predict_proba(X_val_filled)[:, 1]
    y_val_pred = rf.predict(X_val_filled)
    
    return rf, y_val_pred, y_val_pred_proba


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost with native NaN handling.
    
    XGBoost can handle missing values (NaN) natively by learning the optimal
    direction for missing values at each split. This is superior to simple
    imputation as it allows the model to learn patterns in missingness.
    
    Args:
        X_train, y_train: Training data (can contain NaN)
        X_val, y_val: Validation data (can contain NaN)
        
    Returns:
        Tuple of (model, predictions, probabilities)
    """
    # NO imputation needed - XGBoost handles NaN natively!
    # Just ensure data is numeric (already done in feature engineering)
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # Check for NaN in training data
    nan_cols = X_train.columns[X_train.isna().any()].tolist()
    if nan_cols:
        print(f"Training with {len(nan_cols)} features containing NaN (XGBoost handles natively)")
        print(f"  Top features with missing data:")
        missing_pct = (X_train.isna().sum() / len(X_train) * 100).sort_values(ascending=False)
        for col in missing_pct.head(5).index:
            print(f"    • {col}: {missing_pct[col]:.1f}% missing")
    
    # Train XGBoost with native missing value handling
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        missing=np.nan,  # Explicitly tell XGBoost that NaN means missing
        random_state=42,
        eval_metric='auc',
        early_stopping_rounds=10
    )
    
    # Fit with early stopping - use raw data with NaN
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predict on validation set - XGBoost handles NaN in predictions too
    y_val_pred_proba = xgb_model.predict_proba(X_val)[:, 1]
    y_val_pred = xgb_model.predict(X_val)
    
    return xgb_model, y_val_pred, y_val_pred_proba


def save_model(model, filepath, metadata=None):
    """
    Save trained model and metadata.
    
    Args:
        model: Trained model object
        filepath: Path to save model
        metadata: Optional dictionary of metadata
    """
    # Ensure path-like
    path = Path(filepath)
    joblib.dump(model, path)
    
    if metadata:
        # Build a sibling metadata file name safely using pathlib
        metadata_path = path.with_name(path.stem + '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
    
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load trained model.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Loaded model object
    """
    model = joblib.load(filepath)
    
    # Try to load metadata
    try:
        metadata_path = filepath.replace('.pkl', '_metadata.pkl')
        metadata = joblib.load(metadata_path)
        return model, metadata
    except:
        return model, None
