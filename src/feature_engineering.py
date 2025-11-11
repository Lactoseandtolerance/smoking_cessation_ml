"""
Feature engineering utilities for smoking cessation prediction.
"""

import pandas as pd
import numpy as np


def engineer_dependence_features(df):
    """
    Create nicotine dependence features.
    
    Args:
        df (pd.DataFrame): Dataset with smoking variables
        
    Returns:
        pd.DataFrame: Dataset with dependence features added
    """
    # Time to first cigarette (TTFC) - strongest dependence predictor
    df['high_dependence'] = (df['ttfc_minutes'] < 30).astype(int)
    df['very_high_dependence'] = (df['ttfc_minutes'] < 5).astype(int)
    
    # Cigarettes per day
    df['cpd_heavy'] = (df['cpd'] >= 20).astype(int)
    df['cpd_light'] = (df['cpd'] <= 10).astype(int)
    
    # Composite dependence score
    df['dependence_score'] = (
        df['high_dependence'] + 
        df['cpd_heavy']
    )
    
    return df


def engineer_demographic_features(df):
    """
    Create demographic features.
    
    Args:
        df (pd.DataFrame): Dataset with demographic variables
        
    Returns:
        pd.DataFrame: Dataset with demographic features added
    """
    # Age cohorts
    df['age_cohort'] = pd.cut(
        df['age'], 
        bins=[18, 25, 35, 45, 55, 65, 100],
        labels=['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    )
    df['age_young'] = (df['age'] < 35).astype(int)
    
    # Gender
    df['female'] = (df['sex'] == 'Female').astype(int)
    
    # Education
    df['college_degree'] = (df['education_cat'] == 'College+').astype(int)
    
    # Income
    income_median = df['income'].median()
    df['high_income'] = (df['income'] > income_median).astype(int)
    
    return df


def engineer_cessation_method_features(df):
    """
    Create cessation method features.
    
    Planned-but-not-used methods are coded as 0 (not used).
    
    Args:
        df (pd.DataFrame): Dataset with cessation method variables
        
    Returns:
        pd.DataFrame: Dataset with method features added
    """
    # NRT products
    df['used_nrt'] = df['nrt_any'].fillna(0).astype(int)
    df['used_patch'] = df['nrt_patch'].fillna(0).astype(int)
    df['used_gum'] = df['nrt_gum'].fillna(0).astype(int)
    df['used_lozenge'] = df['nrt_lozenge'].fillna(0).astype(int)
    
    # Prescription medications
    df['used_varenicline'] = df['varenicline'].fillna(0).astype(int)
    df['used_bupropion'] = df['bupropion'].fillna(0).astype(int)
    df['used_any_medication'] = (
        (df['used_varenicline'] == 1) | (df['used_bupropion'] == 1)
    ).astype(int)
    
    # Behavioral support
    df['used_counseling'] = df['counseling'].fillna(0).astype(int)
    df['used_quitline'] = df['quitline'].fillna(0).astype(int)
    
    # Cold turkey (no methods)
    df['cold_turkey'] = (
        (df['used_nrt'] == 0) & 
        (df['used_varenicline'] == 0) & 
        (df['used_bupropion'] == 0) & 
        (df['used_counseling'] == 0)
    ).astype(int)
    
    return df


def engineer_quit_history_features(df):
    """
    Create quit history features.
    
    Args:
        df (pd.DataFrame): Dataset with quit history variables
        
    Returns:
        pd.DataFrame: Dataset with quit history features added
    """
    df['num_previous_quits'] = df['lifetime_quit_attempts'].fillna(0)
    df['previous_quit_success'] = (df['num_previous_quits'] > 0).astype(int)
    df['longest_quit_duration'] = df['longest_abstinence_days'].fillna(0)
    
    return df


def engineer_motivation_features(df):
    """
    Create motivation and intention features.
    
    Args:
        df (pd.DataFrame): Dataset with motivation variables
        
    Returns:
        pd.DataFrame: Dataset with motivation features added
    """
    # Readiness to quit (assuming 1-10 scale)
    df['motivation_high'] = (df['readiness_to_quit'] >= 7).astype(int)
    df['plans_to_quit'] = df['plans_quit_next_month'].fillna(0).astype(int)
    
    return df


def engineer_environmental_features(df):
    """
    Create environmental features.
    
    Args:
        df (pd.DataFrame): Dataset with environmental variables
        
    Returns:
        pd.DataFrame: Dataset with environmental features added
    """
    df['household_smokers'] = (df['num_household_smokers'] > 0).astype(int)
    df['smokefree_home'] = df['home_smoking_rules'].fillna(0).astype(int)
    df['workplace_smokefree'] = df['workplace_policy'].fillna(0).astype(int)
    
    return df


def engineer_interaction_features(df):
    """
    Create interaction features (advanced).
    
    Args:
        df (pd.DataFrame): Dataset with base features
        
    Returns:
        pd.DataFrame: Dataset with interaction features added
    """
    # Method combinations
    df['med_plus_counseling'] = (
        (df['used_any_medication'] == 1) & (df['used_counseling'] == 1)
    ).astype(int)
    
    df['nrt_plus_med'] = (
        (df['used_nrt'] == 1) & (df['used_any_medication'] == 1)
    ).astype(int)
    
    # High dependence × medication interactions
    df['highdep_x_varenicline'] = df['high_dependence'] * df['used_varenicline']
    df['highdep_x_nrt'] = df['high_dependence'] * df['used_nrt']
    
    # Age × method interactions
    df['young_x_counseling'] = df['age_young'] * df['used_counseling']
    
    return df


def engineer_all_features(df):
    """
    Apply all feature engineering functions.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Dataset with all engineered features
    """
    df = engineer_dependence_features(df)
    df = engineer_demographic_features(df)
    df = engineer_cessation_method_features(df)
    df = engineer_quit_history_features(df)
    df = engineer_motivation_features(df)
    df = engineer_environmental_features(df)
    df = engineer_interaction_features(df)
    
    return df


def get_feature_list():
    """
    Return list of feature columns for modeling.
    
    Returns:
        list: Feature column names
    """
    features = [
        # Nicotine dependence
        'high_dependence', 'very_high_dependence', 'cpd', 'cpd_heavy', 'cpd_light',
        'dependence_score', 'ttfc_minutes',
        
        # Demographics
        'age', 'age_young', 'female', 'college_degree', 'high_income',
        
        # Cessation methods
        'used_nrt', 'used_patch', 'used_gum', 'used_lozenge',
        'used_varenicline', 'used_bupropion', 'used_any_medication',
        'used_counseling', 'used_quitline', 'cold_turkey',
        
        # Method combinations
        'med_plus_counseling', 'nrt_plus_med',
        
        # Quit history
        'num_previous_quits', 'previous_quit_success', 'longest_quit_duration',
        
        # Motivation
        'motivation_high', 'plans_to_quit',
        
        # Environmental
        'household_smokers', 'smokefree_home',
        
        # Interactions
        'highdep_x_varenicline', 'highdep_x_nrt', 'young_x_counseling'
    ]
    
    return features
