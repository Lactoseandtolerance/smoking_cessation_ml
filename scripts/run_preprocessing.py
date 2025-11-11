#!/usr/bin/env python3
"""
Run the full data preprocessing pipeline to create pooled_transitions.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import engineer_all_features, _extract_numeric_code

# Configuration
DATA_DIR = Path(__file__).parent.parent / 'data/raw'
OUTPUT_DIR = Path(__file__).parent.parent / 'data/processed'
OUTPUT_DIR.mkdir(exist_ok=True)

WAVE_FILES = {
    1: DATA_DIR / 'PATH_W1_Adult_Public.dta',
    2: DATA_DIR / 'PATH_W2_Adult_Public.dta',
    3: DATA_DIR / 'PATH_W3_Adult_Public.dta',
    4: DATA_DIR / 'PATH_W4_Adult_Public.dta',
    5: DATA_DIR / 'PATH_W5_Adult_Public.dta',
}

# Sample size (None for full data, or number for testing)
SAMPLE_SIZE = None  # Change to 1000 for quick testing

def load_wave(wave_num, nrows=None):
    """Load a single wave of PATH data."""
    path = WAVE_FILES[wave_num]
    
    if not path.exists():
        print(f"⚠️  Wave {wave_num} file not found: {path}")
        return None
    
    print(f"Loading Wave {wave_num}...", end=' ', flush=True)
    
    # Load data - disable convert_categoricals to avoid duplicate label errors
    reader = pd.read_stata(path, iterator=True, convert_categoricals=False)
    df = reader.read(nrows=nrows)
    
    # Add wave identifier
    df['wave'] = wave_num
    
    print(f"✓ {len(df):,} rows, {len(df.columns):,} columns")
    return df

def identify_current_smokers(df, wave_num):
    """Identify current smokers in a given wave."""
    smoked_30d = f'R0{wave_num}_AC1002'
    freq_var = f'R0{wave_num}_AC1003'
    
    is_smoker = pd.Series(False, index=df.index)
    
    if smoked_30d in df.columns:
        smoked_code = _extract_numeric_code(df[smoked_30d])
        is_smoker |= (smoked_code == 1)
    
    if freq_var in df.columns:
        freq_code = _extract_numeric_code(df[freq_var])
        is_smoker |= (freq_code.isin([1, 2]))
    
    return is_smoker

def create_transitions(wave_t_data, wave_t1_data, wave_t, wave_t1):
    """Create transition records from wave t to wave t+1."""
    print(f"\nCreating transitions: Wave {wave_t} → Wave {wave_t1}")
    
    # Get smokers at baseline
    smokers_t = wave_t_data[wave_t_data['is_current_smoker']].copy()
    print(f"  Baseline smokers: {len(smokers_t):,}")
    
    # Merge with follow-up data
    # Select key baseline demographic and dependence inputs for wave-aware feature engineering
    baseline_cols = [
        'PERSONID',
        # Demographics
        f'R0{wave_t}R_A_AGE', f'R0{wave_t}R_A_AGECAT7', f'R0{wave_t}R_A_AGECAT6', f'R0{wave_t}R_A_SEX',
        f'R0{wave_t}R_POVCAT3', f'R0{wave_t}R_A_INCOME', f'R0{wave_t}R_A_RACECAT3', f'R0{wave_t}R_A_RACE', f'R0{wave_t}R_A_HISP',
        # Core smoking behavior
        f'R0{wave_t}R_A_PERDAY_P30D_CIGS', f'R0{wave_t}R_A_MINFIRST_CIGS'
    ]
    existing_baseline_cols = [c for c in baseline_cols if c in smokers_t.columns]

    followup_cols = [
        'PERSONID', 'is_current_smoker'
    ]
    # Avoid duplicate PERSONID column selection
    left_cols = list(dict.fromkeys(existing_baseline_cols + ['PERSONID', 'is_current_smoker']))
    transitions = smokers_t[left_cols].merge(
        wave_t1_data[followup_cols],
        on='PERSONID',
        how='inner',
        suffixes=('', '_t1')
    )
    
    print(f"  With follow-up data: {len(transitions):,}")
    
    # Define quit success
    transitions['quit_success'] = (~transitions['is_current_smoker_t1']).astype(int)
    
    # Add transition info
    transitions['baseline_wave'] = wave_t
    transitions['followup_wave'] = wave_t1
    transitions['transition'] = f'W{wave_t}→W{wave_t1}'
    
    quit_rate = 100 * transitions['quit_success'].mean()
    print(f"  Quit rate: {quit_rate:.1f}%")
    
    return transitions

def main():
    print("="*70)
    print("PATH DATA PREPROCESSING PIPELINE")
    print("="*70)
    print(f"Sample size: {SAMPLE_SIZE if SAMPLE_SIZE else 'FULL DATA'}\n")
    
    # 1. Load all waves
    print("STEP 1: Loading waves...")
    print("-"*70)
    
    waves_data = {}
    for wave_num in range(1, 6):
        df = load_wave(wave_num, nrows=SAMPLE_SIZE)
        if df is not None:
            waves_data[wave_num] = df
    
    print(f"\n✓ Loaded {len(waves_data)} waves\n")
    
    # 2. Identify smokers
    print("STEP 2: Identifying current smokers...")
    print("-"*70)
    
    for wave_num, df in waves_data.items():
        is_smoker = identify_current_smokers(df, wave_num)
        n_smokers = is_smoker.sum()
        pct_smokers = 100 * n_smokers / len(df)
        
        waves_data[wave_num]['is_current_smoker'] = is_smoker
        print(f"Wave {wave_num}: {n_smokers:>6,} / {len(df):>6,} ({pct_smokers:>5.1f}%)")
    
    print()
    
    # 3. Create transitions
    print("STEP 3: Creating person-period transitions...")
    print("-"*70)
    
    all_transitions = []
    
    for wave_t in range(1, 5):
        wave_t1 = wave_t + 1
        
        if wave_t in waves_data and wave_t1 in waves_data:
            transitions = create_transitions(
                waves_data[wave_t],
                waves_data[wave_t1],
                wave_t,
                wave_t1
            )
            all_transitions.append(transitions)
    
    if all_transitions:
        pooled = pd.concat(all_transitions, ignore_index=True)
        print("\n" + "-"*70)
        print(f"✓ Total transitions: {len(pooled):,}")
        print(f"✓ Overall quit rate: {100 * pooled['quit_success'].mean():.1f}%")
        print(f"✓ Unique persons: {pooled['PERSONID'].nunique():,}")
    else:
        print("⚠️  No transitions created")
        return
    
    # 4. Apply feature engineering
    print("\n" + "="*70)
    print("STEP 4: Applying feature engineering...")
    print("-"*70)
    
    # Codebook overrides: keep only general mapping hints; wave-aware mapping handled in feature engineering
    codebook_overrides = {
        'race_map': {1: 'White', 2: 'Black', 3: 'Other'},
        'hisp_yes_values': (1,),
        'race_collapse_to_other': (),
    }
    
    print("Running feature engineering on pooled transitions...")
    engineered = engineer_all_features(
        pooled.copy(),
        codebook_overrides=codebook_overrides,
        recode_missing=True
    )
    
    print(f"✓ Features created: {engineered.shape[1]} columns")
    print(f"✓ Records: {len(engineered):,}")
    
    # 5. Validate and save
    print("\n" + "="*70)
    print("STEP 5: Validating and saving...")
    print("-"*70)
    
    print(f"\nOutcome distribution:")
    print(engineered['quit_success'].value_counts())
    print(f"Quit rate: {100 * engineered['quit_success'].mean():.1f}%")
    
    # Select features for modeling + preserve raw wave-specific columns for downstream re-engineering
    from src.feature_engineering import get_feature_list
    feature_cols = get_feature_list()
    available_features = [f for f in feature_cols if f in engineered.columns]
    
    print(f"\nFeature availability: {len(available_features)}/{len(feature_cols)}")
    
    # Preserve raw wave-specific demographic/dependence columns alongside engineered features
    raw_wave_cols = [c for c in engineered.columns if c.startswith('R0') and ('_A_' in c or 'POVCAT' in c)]
    
    # Create final dataset with modeling columns + raw inputs
    modeling_cols = ['PERSONID', 'baseline_wave', 'followup_wave', 'transition', 'quit_success'] + available_features + raw_wave_cols
    existing_cols = list(dict.fromkeys([c for c in modeling_cols if c in engineered.columns]))  # dedupe
    modeling_data = engineered[existing_cols].copy()
    
    # Save
    csv_path = OUTPUT_DIR / 'pooled_transitions.csv'
    parquet_path = OUTPUT_DIR / 'pooled_transitions.parquet'
    
    modeling_data.to_csv(csv_path, index=False)
    modeling_data.to_parquet(parquet_path, index=False)
    
    print(f"\n✓ Saved: {csv_path}")
    print(f"✓ Saved: {parquet_path}")
    print(f"\nFinal dataset: {len(modeling_data):,} rows × {len(modeling_data.columns):,} columns")
    
    print("\n" + "="*70)
    print("✓ PREPROCESSING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
