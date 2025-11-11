"""
Data preprocessing utilities for PATH Study smoking cessation analysis.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def load_wave_data(wave_num, data_dir='../data/raw', file_format='dta'):
    """
    Load PATH Study wave data from STATA (.dta) or SPSS (.sav) format.
    
    PATH Study provides data in STATA or SPSS format, not CSV.
    
    Args:
        wave_num (int): Wave number (1-5)
        data_dir (str): Directory containing data files
        file_format (str): 'dta' for STATA or 'sav' for SPSS
        
    Returns:
        pd.DataFrame: Wave data
    """
    # Try to find the file with various naming conventions
    possible_names = [
        f"PATH_W{wave_num}_Adult_Public.{file_format}",
        f"PATH_Wave{wave_num}_Adult.{file_format}",
        f"wave{wave_num}_adult.{file_format}",
        f"W{wave_num}_adult.{file_format}"
    ]
    
    for filename in possible_names:
        filepath = f"{data_dir}/{filename}"
        try:
            if file_format == 'dta':
                df = pd.read_stata(filepath)
                print(f"Loaded Wave {wave_num}: {len(df)} observations (STATA format)")
                return df
            elif file_format == 'sav':
                df = pd.read_spss(filepath)
                print(f"Loaded Wave {wave_num}: {len(df)} observations (SPSS format)")
                return df
        except FileNotFoundError:
            continue
    
    # If no file found, raise error with helpful message
    raise FileNotFoundError(
        f"Could not find Wave {wave_num} data file in {data_dir}.\n"
        f"Tried: {', '.join(possible_names)}\n"
        f"Please check the actual filename and update the path accordingly."
    )


def create_transition(wave_t, wave_t1, transition_name):
    """
    Create person-period observations for one wave transition.
    
    Args:
        wave_t: Baseline wave (e.g., Wave 1)
        wave_t1: Follow-up wave (e.g., Wave 2)
        transition_name: Label (e.g., 'W1_W2')
    
    Returns:
        DataFrame with person-period observations
    """
    # Merge waves on person ID
    merged = wave_t.merge(wave_t1, on='person_id', suffixes=('_t', '_t1'))
    
    # Filter to smokers at baseline who attempted to quit
    quit_attempters = merged[
        (merged['current_smoker_t'] == 1) & 
        (merged['quit_attempt_t1'] == 1)
    ].copy()
    
    # Define outcome: abstinent at follow-up
    quit_attempters['quit_success'] = (
        quit_attempters['current_smoker_t1'] == 0
    ).astype(int)
    
    # Add transition identifier
    quit_attempters['transition'] = transition_name
    
    return quit_attempters


def pool_transitions(waves):
    """
    Pool multiple wave transitions into person-period dataset.
    
    Args:
        waves (list): List of wave DataFrames
        
    Returns:
        pd.DataFrame: Pooled person-period dataset
    """
    transitions = []
    
    for i in range(len(waves) - 1):
        transition_name = f'W{i+1}_W{i+2}'
        transition_data = create_transition(waves[i], waves[i+1], transition_name)
        transitions.append(transition_data)
        print(f"Created transition {transition_name}: {len(transition_data)} observations")
    
    pooled_data = pd.concat(transitions, ignore_index=True)
    
    print(f"\nTotal person-periods: {len(pooled_data)}")
    print(f"Unique individuals: {pooled_data['person_id'].nunique()}")
    
    return pooled_data


def handle_missing_codes(df):
    """
    Replace PATH Study missing value codes with NaN.
    
    PATH uses negative values for missing:
    -9 (refused), -1 (inapplicable), -7 (don't know), etc.
    
    Args:
        df (pd.DataFrame): Dataset with PATH missing codes
        
    Returns:
        pd.DataFrame: Dataset with NaN for missing values
    """
    missing_codes = [-9, -8, -7, -4, -1]
    df = df.replace(missing_codes, np.nan)
    return df


def impute_missing_data(X_train, X_val, X_test):
    """
    Impute missing values in train/val/test sets.
    
    Strategy: Median for numeric, most frequent for categorical.
    Fit on train, transform all sets.
    
    Args:
        X_train, X_val, X_test: Feature matrices
        
    Returns:
        Tuple of (X_train_imputed, X_val_imputed, X_test_imputed, imputers)
    """
    numeric_features = X_train.select_dtypes(include=[np.number]).columns
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns
    
    # Numeric imputation
    numeric_imputer = SimpleImputer(strategy='median')
    X_train[numeric_features] = numeric_imputer.fit_transform(X_train[numeric_features])
    X_val[numeric_features] = numeric_imputer.transform(X_val[numeric_features])
    X_test[numeric_features] = numeric_imputer.transform(X_test[numeric_features])
    
    # Categorical imputation
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    if len(categorical_features) > 0:
        X_train[categorical_features] = categorical_imputer.fit_transform(X_train[categorical_features])
        X_val[categorical_features] = categorical_imputer.transform(X_val[categorical_features])
        X_test[categorical_features] = categorical_imputer.transform(X_test[categorical_features])
    
    imputers = {
        'numeric': numeric_imputer,
        'categorical': categorical_imputer
    }
    
    return X_train, X_val, X_test, imputers


def calculate_cessation_rates(pooled_data):
    """
    Calculate cessation rates overall and by subgroups.
    
    Args:
        pooled_data (pd.DataFrame): Person-period dataset
        
    Returns:
        dict: Cessation rate statistics
    """
    overall_rate = pooled_data['quit_success'].mean()
    
    rates_by_wave = pooled_data.groupby('transition')['quit_success'].agg(['mean', 'count'])
    
    results = {
        'overall': overall_rate,
        'by_wave': rates_by_wave,
        'sample_size': len(pooled_data),
        'n_individuals': pooled_data['person_id'].nunique()
    }
    
    return results
