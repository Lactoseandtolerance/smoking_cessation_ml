"""
Feature engineering utilities for smoking cessation prediction.
"""

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Phase 3 scaffolding: mapping PATH codebook variables to canonical columns
# -----------------------------------------------------------------------------

# Unified list of PATH missing codes observed in documentation
PATH_MISSING_CODES = [
    -9, -8, -7, -4, -1,           # General public-use missing codes (short list)
    -99988, -99977, -99955, -99911, -97777  # Extended codes seen in user guide
]


# Centralized candidate variable names to ease maintenance and notebook wiring.
# Replace or extend these in the Phase 3 notebook as you confirm codebook names.
VARIABLE_CANDIDATES = {
    # Demographics
    'age': ['age', 'R01R_A_AGE', 'R02R_A_AGE', 'R03R_A_AGE', 'R04R_A_AGE', 'R05R_A_AGE', 'baseline_age'],
    'sex': ['sex', 'SEX', 'R01R_A_SEX', 'R02R_A_SEX', 'R03R_A_SEX', 'R04R_A_SEX', 'R05R_A_SEX', 'baseline_sex'],
    'education_cat': ['education_cat'],
    'education_code': ['education_code', 'edu_code', 'R01R_A_EDUC', 'R02R_A_EDUC', 'R03R_A_EDUC', 'R04R_A_EDUC', 'R05R_A_EDUC', 'baseline_education_code'],
    'income': ['income', 'household_income', 'R01R_A_INCOME', 'R02R_A_INCOME', 'R03R_A_INCOME', 'R04R_A_INCOME', 'R05R_A_INCOME', 'baseline_income'],
    # Race/Ethnicity
    'race': ['race', 'R01R_A_RACE', 'R02R_A_RACE', 'R03R_A_RACE', 'R04R_A_RACE', 'R05R_A_RACE', 'R01R_A_RACECAT', 'R02R_A_RACECAT', 'R03R_A_RACECAT', 'R04R_A_RACECAT', 'R05R_A_RACECAT'],
    'hispanic': ['hispanic', 'HISPANIC', 'R01R_A_HISP', 'R02R_A_HISP', 'R03R_A_HISP', 'R04R_A_HISP', 'R05R_A_HISP'],
    # Smoking behavior
    'cpd': ['cpd', 'baseline_cpd', 'R01R_A_PERDAY_P30D_CIGS', 'R02R_A_PERDAY_P30D_CIGS', 'R03R_A_PERDAY_P30D_CIGS', 'R04R_A_PERDAY_P30D_CIGS', 'R05R_A_PERDAY_P30D_CIGS'],
    'ttfc_minutes': ['ttfc_minutes', 'baseline_ttfc', 'R01R_A_MINFIRST_CIGS', 'R02R_A_MINFIRST_CIGS', 'R03R_A_MINFIRST_CIGS', 'R04R_A_MINFIRST_CIGS', 'R05R_A_MINFIRST_CIGS'],
    # Cessation methods (to be confirmed in codebook)
    'nrt_any': ['nrt_any', 'used_nrt'],
    'nrt_patch': ['nrt_patch', 'used_patch'],
    'nrt_gum': ['nrt_gum', 'used_gum'],
    'nrt_lozenge': ['nrt_lozenge', 'used_lozenge'],
    'varenicline': ['varenicline', 'used_varenicline'],
    'bupropion': ['bupropion', 'used_bupropion'],
    'counseling': ['counseling', 'used_counseling'],
    'quitline': ['quitline', 'used_quitline'],
}


def _replace_path_missing(series):
    """Replace PATH negative missing codes with np.nan on a Series.

    Args:
        series (pd.Series): Input series with potential PATH missing codes

    Returns:
        pd.Series: Series with PATH missing codes replaced by NaN
    """
    # Ensure we have a Series, not an array
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    # First extract numeric codes if it's categorical with labels
    if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
        # Extract number at start of string (e.g., "1 = Male" -> 1)
        s = series.astype(str).str.extract(r'^(-?\d+)', expand=False).astype(float)
    else:
        # Already numeric
        s = pd.to_numeric(series, errors='coerce')
        if not isinstance(s, pd.Series):
            s = pd.Series(s, index=series.index)
    
    # Now replace PATH missing codes
    return s.replace(PATH_MISSING_CODES, np.nan)


def _first_present_column(df, candidates):
    """Return the first column name present in df from candidates, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _series_or_default(df: pd.DataFrame, col: str, default=0) -> pd.Series:
    """Return df[col] if present as a Series; otherwise a Series filled with default.

    Ensures downstream .fillna/.astype calls are valid.
    """
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def _extract_numeric_code(series):
    """Extract numeric code from PATH categorical strings like '1 = Male'.
    
    Args:
        series (pd.Series): Series potentially containing categorical strings with codes
        
    Returns:
        pd.Series: Numeric codes extracted from strings, or original values if already numeric
    """
    if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
        # Extract number at start of string (e.g., "1 = Male" -> 1)
        return series.astype(str).str.extract(r'^(-?\d+)', expand=False).astype(float)
    else:
        # Already numeric
        return pd.to_numeric(series, errors='coerce')


def _wave_aware_pick(df: pd.DataFrame, baseline_wave_col: str, candidates_per_wave: dict[int, list[str]]):
    """Select values row-wise from wave-specific columns based on baseline_wave.

    Args:
        df: DataFrame containing pooled transitions and raw wave columns
        baseline_wave_col: column indicating the baseline wave (int 1..5)
        candidates_per_wave: mapping wave -> ordered list of candidate column names

    Returns:
        pd.Series of values assembled across waves (dtype object, to be coerced later)
    """
    s = pd.Series(np.nan, index=df.index)
    if baseline_wave_col not in df.columns:
        return s
    for w, cands in candidates_per_wave.items():
        mask = df[baseline_wave_col] == w
        if not mask.any():
            continue
        for col in cands:
            if col in df.columns:
                s.loc[mask] = df.loc[mask, col]
                break
    return s


def _normalize_race_ethnicity(df, race_col=None, hisp_col=None, *, race_map=None, hisp_yes_values=(1,), collapse_to_other=('Asian',)):
    """Derive a 4-level race/ethnicity label and one-hot dummies.

    If a Hispanic indicator is available, values in hisp_yes_values are treated
    as Hispanic, overriding the race mapping. Provide race_map to align with
    PATH's exact coding (e.g., {1:'White', 2:'Black', 3:'Asian', 4:'Other', ...}).

    Returns: (series_label, dict_of_dummies)
    """
    race_s = df[race_col] if race_col is not None else pd.Series(index=df.index, dtype='float')
    race_s = _replace_path_missing(race_s)
    race_s = _extract_numeric_code(race_s)

    # Use provided mapping or a conservative default
    base_map = race_map or {
        1: 'White',
        2: 'Black',
        3: 'Asian',  # often collapsed depending on sample size
        4: 'Other',  # AIAN/Other
        5: 'Other'
    }
    race_label = race_s.map(base_map).fillna('Other')

    # If we have a hispanic indicator, set those to 'Hispanic'
    if hisp_col is not None and hisp_col in df.columns:
        h = _replace_path_missing(df[hisp_col])
        h = _extract_numeric_code(h)
        hisp_mask = h.isin(list(hisp_yes_values))
        race_label = race_label.where(~hisp_mask, 'Hispanic')

    # Collapse specified labels into Other for a 4-level scheme if requested
    for lbl in collapse_to_other or ():
        race_label = race_label.replace({lbl: 'Other'})

    dummies = {
        'race_white': (race_label == 'White').astype(int),
        'race_black': (race_label == 'Black').astype(int),
        'race_hispanic': (race_label == 'Hispanic').astype(int),
        'race_other': (~race_label.isin(['White', 'Black', 'Hispanic'])).astype(int),
    }
    return race_label, dummies


def map_from_codebook(
    df,
    codebook_overrides=None,
    recode_missing=True,
):
    """Map PATH Study codebook variables into canonical columns used downstream.

    This function is designed to be robust to partially-available inputs and
    wave-specific variable names. You can pass explicit mappings via
    codebook_overrides; otherwise, the function will try common candidates.

    Canonical outputs created when possible:
      - age (numeric years)
      - sex (string: 'Male'/'Female')
      - education_cat (string: '<HS'|'HS'|'Some College'|'College+')
      - income (numeric proxy; uses raw numeric if available, otherwise ordinal)
      - cpd (cigarettes per day, numeric)
      - ttfc_minutes (time-to-first-cigarette in minutes, numeric)
      - nrt_any, nrt_patch, nrt_gum, nrt_lozenge (0/1)
      - varenicline, bupropion (0/1)
      - counseling, quitline (0/1)

    Args:
        df (pd.DataFrame): Input data (raw PATH or pooled dataset)
        codebook_overrides (dict|None): Optional explicit mapping
            Example keys: 'age', 'sex', 'education_code', 'education_cat', 'income',
            'cpd', 'ttfc_minutes', 'nrt_any', 'nrt_patch', 'nrt_gum', 'nrt_lozenge',
            'varenicline', 'bupropion', 'counseling', 'quitline'
        recode_missing (bool): If True, replace PATH negative codes with NaN

    Returns:
        pd.DataFrame: df with canonical columns added if derivable
    """
    codebook_overrides = codebook_overrides or {}

    # Optionally normalize PATH missing codes across numeric fields we touch
    def clean(s):
        return _replace_path_missing(s) if recode_missing else s

    # ------------------------- Demographics -------------------------
    # Wave-aware mapping when pooled data contains baseline_wave
    if 'baseline_wave' in df.columns:
        # AGE: prefer numeric age if available; else map AGECAT7 to midpoints
        age_num = _wave_aware_pick(
            df, 'baseline_wave',
            {w: [f'R0{w}R_A_AGE'] for w in range(1, 6)}
        )
        age_cat = _wave_aware_pick(
            df, 'baseline_wave',
            {w: [f'R0{w}R_A_AGECAT7'] for w in range(1, 6)}
        )
        age_cat6 = _wave_aware_pick(
            df, 'baseline_wave',
            {w: [f'R0{w}R_A_AGECAT6'] for w in range(1, 6)}
        )
        age_num = _extract_numeric_code(clean(age_num))
        age_cat_codes = _extract_numeric_code(clean(age_cat))
        age_cat6_codes = _extract_numeric_code(clean(age_cat6))
        age_cat_map = {
            1: 21,   # 18-24
            2: 29.5, # 25-34
            3: 39.5, # 35-44
            4: 49.5, # 45-54
            5: 59.5, # 55-64
            6: 69.5, # 65-74
            7: 80,   # 75+
        }
        age_from_cat = age_cat_codes.map(age_cat_map)
        age_cat6_map = {
            1: 21,   # 18-24
            2: 29.5, # 25-34
            3: 39.5, # 35-44
            4: 49.5, # 45-54
            5: 59.5, # 55-64
            6: 69.5, # 65+
        }
        age_from_cat6 = age_cat6_codes.map(age_cat6_map)
        df['age'] = age_num.where(age_num.notna(), age_from_cat)
        df['age'] = df['age'].where(df['age'].notna(), age_from_cat6)

        # SEX: 1=Male, 2=Female
        sex_wave = _wave_aware_pick(
            df, 'baseline_wave',
            {w: [f'R0{w}R_A_SEX'] for w in range(1, 6)}
        )
        sex_codes = _extract_numeric_code(clean(sex_wave))
        df['sex'] = sex_codes.map({1: 'Male', 2: 'Female'})

        # INCOME: try both POVCAT3 and A_INCOME
        income_wave = _wave_aware_pick(
            df, 'baseline_wave',
            {w: [f'R0{w}R_POVCAT3', f'R0{w}R_A_INCOME'] for w in range(1, 6)}
        )
        df['income'] = _extract_numeric_code(clean(income_wave))

        # RACE / HISPANIC
        race_wave = _wave_aware_pick(
            df, 'baseline_wave',
            {w: [f'R0{w}R_A_RACECAT3', f'R0{w}R_A_RACE'] for w in range(1, 6)}
        )
        hisp_wave = _wave_aware_pick(
            df, 'baseline_wave',
            {w: [f'R0{w}R_A_HISP'] for w in range(1, 6)}
        )
        # Stash temporarily to reuse existing normalization function
        df['__race_code_waveaware'] = race_wave
        df['__hisp_code_waveaware'] = hisp_wave
        race_label, dummies = _normalize_race_ethnicity(
            df,
            race_col='__race_code_waveaware',
            hisp_col='__hisp_code_waveaware',
            race_map=codebook_overrides.get('race_map') if codebook_overrides else None,
            hisp_yes_values=tuple(codebook_overrides.get('hisp_yes_values', (1,))) if codebook_overrides else (1,),
            collapse_to_other=tuple(codebook_overrides.get('race_collapse_to_other', ('Asian',))) if codebook_overrides else ('Asian',)
        )
        df['race_ethnicity'] = race_label
        for k, v in dummies.items():
            df[k] = v

        # CPD and TTFC (numeric)
        cpd_wave = _wave_aware_pick(
            df, 'baseline_wave',
            {w: [f'R0{w}R_A_PERDAY_P30D_CIGS'] for w in range(1, 6)}
        )
        df['cpd'] = pd.to_numeric(clean(cpd_wave), errors='coerce')

        ttfc_wave = _wave_aware_pick(
            df, 'baseline_wave',
            {w: [f'R0{w}R_A_MINFIRST_CIGS'] for w in range(1, 6)}
        )
        df['ttfc_minutes'] = pd.to_numeric(clean(ttfc_wave), errors='coerce')
    else:
        # age (handle both numeric and categorical) for single-wave frames
        if 'age' not in df.columns:
            age_candidates = codebook_overrides.get('age_candidates') or VARIABLE_CANDIDATES['age']
            col = codebook_overrides.get('age') or _first_present_column(df, age_candidates)
            if col is not None:
                age_series = clean(df[col])
                # Extract numeric code from categorical strings
                age_numeric = _extract_numeric_code(age_series)
                # Check if it's categorical (like PATH's AGECAT7)
                if 'AGECAT' in str(col).upper():
                    # Map age categories to midpoints
                    age_mapping = {
                        1: 21,   # 18-24
                        2: 29.5, # 25-34
                        3: 39.5, # 35-44
                        4: 49.5, # 45-54
                        5: 59.5, # 55-64
                        6: 69.5, # 65-74
                        7: 80,   # 75+
                    }
                    df['age'] = age_numeric.map(age_mapping)
                else:
                    # Direct numeric age
                    df['age'] = age_numeric

        # sex (normalize to Male/Female strings if coded 1/2)
        if 'sex' not in df.columns:
            sex_candidates = codebook_overrides.get('sex_candidates') or VARIABLE_CANDIDATES['sex']
            col = codebook_overrides.get('sex') or _first_present_column(df, sex_candidates)
            if col is not None:
                s = clean(df[col])
                # Extract numeric code and map 1=Male, 2=Female
                sex_numeric = _extract_numeric_code(s)
                df['sex'] = sex_numeric.map({1: 'Male', 2: 'Female'})

    # education_cat (derive from either pre-made category or numeric education code)
    if 'education_cat' not in df.columns:
        edu_cat_col = codebook_overrides.get('education_cat')
        edu_code_col = codebook_overrides.get('education_code')
        if edu_cat_col and edu_cat_col in df.columns:
            df['education_cat'] = df[edu_cat_col]
        else:
            edu_candidates = VARIABLE_CANDIDATES['education_code']
            edu_code_col = edu_code_col or _first_present_column(df, edu_candidates)
            if edu_code_col is not None:
                edu_code = pd.to_numeric(clean(df[edu_code_col]), errors='coerce')
                # Map common survey-style education codes to 4 bins used downstream
                # This mapping is a placeholder; confirm with PATH codebook
                edu_map = {
                    1: '<HS', 2: '<HS',               # < High school
                    3: 'HS', 4: 'HS',                 # High school/GED
                    5: 'Some College', 6: 'Some College',  # Some college/Associate
                    7: 'College+', 8: 'College+', 9: 'College+'  # Bachelor+
                }
                df['education_cat'] = edu_code.map(edu_map)

    # income (prefer raw numeric if available; else ordinal proxy)
        if 'income' not in df.columns:
            income_candidates = codebook_overrides.get('income_candidates') or VARIABLE_CANDIDATES['income']
            col = codebook_overrides.get('income') or _first_present_column(df, income_candidates)
            if col is not None:
                s = clean(df[col])
                # Extract numeric code from categorical strings
                income_numeric = _extract_numeric_code(s)
                df['income'] = income_numeric

    # Race/Ethnicity: create race_ethnicity label and one-hot dummies
    if 'baseline_wave' not in df.columns:
        race_col = codebook_overrides.get('race') if codebook_overrides else None
        if race_col is None:
            race_col = _first_present_column(df, VARIABLE_CANDIDATES['race'])
        hisp_col = codebook_overrides.get('hispanic') if codebook_overrides else None
        if hisp_col is None:
            hisp_col = _first_present_column(df, VARIABLE_CANDIDATES['hispanic'])
        if (race_col is not None or hisp_col is not None) and 'race_ethnicity' not in df.columns:
            race_label, dummies = _normalize_race_ethnicity(
                df, race_col, hisp_col,
                race_map=codebook_overrides.get('race_map') if codebook_overrides else None,
                hisp_yes_values=tuple(codebook_overrides.get('hisp_yes_values', (1,))) if codebook_overrides else (1,),
                collapse_to_other=tuple(codebook_overrides.get('race_collapse_to_other', ('Asian',))) if codebook_overrides else ('Asian',)
            )
            df['race_ethnicity'] = race_label
            for k, v in dummies.items():
                df[k] = v

    # ------------------ Core smoking inputs (if needed) ------------------
    if 'baseline_wave' not in df.columns:
        # cpd
        if 'cpd' not in df.columns:
            cpd_candidates = codebook_overrides.get('cpd_candidates') or VARIABLE_CANDIDATES['cpd']
            col = codebook_overrides.get('cpd') or _first_present_column(df, cpd_candidates)
            if col is not None:
                df['cpd'] = pd.to_numeric(clean(df[col]), errors='coerce')

        # ttfc_minutes
        if 'ttfc_minutes' not in df.columns:
            ttfc_candidates = codebook_overrides.get('ttfc_minutes_candidates') or VARIABLE_CANDIDATES['ttfc_minutes']
            col = codebook_overrides.get('ttfc_minutes') or _first_present_column(df, ttfc_candidates)
            if col is not None:
                df['ttfc_minutes'] = pd.to_numeric(clean(df[col]), errors='coerce')

    # --------------------- Cessation methods (binary) ---------------------
    # Helper to coerce a method column into 0/1 where 1 means used
    def method_binary(colname):
        s = clean(df[colname])
        # Extract numeric code from categorical strings
        s_num = _extract_numeric_code(s)
        # Common PATH coding: 1=Yes, 2=No
        used = (s_num == 1).astype(float)
        # Where explicit No is coded as 2, set to 0
        used = used.where(~(s_num == 2), 0)
        return used.fillna(0).astype(int)

    method_map = {}
    for key in ['nrt_any','nrt_patch','nrt_gum','nrt_lozenge','varenicline','bupropion','counseling','quitline']:
        explicit = codebook_overrides.get(key) if codebook_overrides else None
        method_map[key] = explicit or _first_present_column(df, VARIABLE_CANDIDATES[key])

    for canonical, raw_col in method_map.items():
        if raw_col is not None and canonical not in df.columns:
            try:
                df[canonical] = method_binary(raw_col)
            except Exception:
                # Fallback: simple 0/1 coercion
                df[canonical] = pd.to_numeric(clean(df[raw_col]), errors='coerce').fillna(0).astype(int)

    return df


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
    # Age cohorts (guard when age not available)
    if 'age' in df.columns:
        df['age_cohort'] = pd.cut(
            df['age'], 
            bins=[18, 25, 35, 45, 55, 65, 100],
            labels=['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        )
        df['age_young'] = (df['age'] < 35).astype(int)
    else:
        df['age_cohort'] = pd.Series(pd.Categorical([np.nan] * len(df), categories=['18-24','25-34','35-44','45-54','55-64','65+']))
        df['age_young'] = 0
    
    # Gender (robust to 'Female' string or numeric 2)
    sex_series = df['sex'] if 'sex' in df.columns else pd.Series(np.nan, index=df.index)
    female_mask = (
        (sex_series.astype(str).str.lower() == 'female') |
        (pd.to_numeric(sex_series, errors='coerce') == 2)
    )
    df['female'] = female_mask.fillna(False).astype(int)
    
    # Education (handle missing education_cat column)
    if 'education_cat' in df.columns:
        df['college_degree'] = (df['education_cat'] == 'College+').astype(int)
    else:
        df['college_degree'] = 0  # Default to 0 when education not available
    
    # Income (guard missing)
    raw_income = df['income'] if 'income' in df.columns else pd.Series(np.nan, index=df.index)
    income_series = pd.to_numeric(raw_income, errors='coerce')
    if income_series.notna().sum() > 0:
        income_median = income_series.median()
        df['high_income'] = (income_series > income_median).astype(int)
    else:
        df['high_income'] = 0  # Default when no income data
    
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
    df['used_nrt'] = _series_or_default(df, 'nrt_any', 0).fillna(0).astype(int)
    df['used_patch'] = _series_or_default(df, 'nrt_patch', 0).fillna(0).astype(int)
    df['used_gum'] = _series_or_default(df, 'nrt_gum', 0).fillna(0).astype(int)
    df['used_lozenge'] = _series_or_default(df, 'nrt_lozenge', 0).fillna(0).astype(int)
    
    # Prescription medications
    df['used_varenicline'] = _series_or_default(df, 'varenicline', 0).fillna(0).astype(int)
    df['used_bupropion'] = _series_or_default(df, 'bupropion', 0).fillna(0).astype(int)
    df['used_any_medication'] = (
        (df['used_varenicline'] == 1) | (df['used_bupropion'] == 1)
    ).astype(int)
    
    # Behavioral support
    df['used_counseling'] = _series_or_default(df, 'counseling', 0).fillna(0).astype(int)
    df['used_quitline'] = _series_or_default(df, 'quitline', 0).fillna(0).astype(int)
    df['used_any_behavioral'] = (
        (df['used_counseling'] == 1) | (df['used_quitline'] == 1)
    ).astype(int)
    
    # Cold turkey (no methods)
    df['cold_turkey'] = (
        (df['used_nrt'] == 0) & 
        (df['used_varenicline'] == 0) & 
        (df['used_bupropion'] == 0) & 
        (df['used_counseling'] == 0)
    ).astype(int)

    # Any method used at all
    df['used_any_method'] = (
        (df['used_nrt'] == 1) |
        (df['used_varenicline'] == 1) |
        (df['used_bupropion'] == 1) |
        (df['used_counseling'] == 1) |
        (df['used_quitline'] == 1)
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
    df['num_previous_quits'] = _series_or_default(df, 'lifetime_quit_attempts', 0).fillna(0)
    df['previous_quit_success'] = (df['num_previous_quits'] > 0).astype(int)
    df['longest_quit_duration'] = _series_or_default(df, 'longest_abstinence_days', 0).fillna(0)
    
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
    df['motivation_high'] = (pd.to_numeric(_series_or_default(df, 'readiness_to_quit', 0), errors='coerce') >= 7).astype(int)
    df['plans_to_quit'] = _series_or_default(df, 'plans_quit_next_month', 0).fillna(0).astype(int)
    
    return df


def engineer_environmental_features(df):
    """
    Create environmental features.
    
    Args:
        df (pd.DataFrame): Dataset with environmental variables
        
    Returns:
        pd.DataFrame: Dataset with environmental features added
    """
    df['household_smokers'] = (pd.to_numeric(_series_or_default(df, 'num_household_smokers', 0), errors='coerce') > 0).astype(int)
    df['smokefree_home'] = _series_or_default(df, 'home_smoking_rules', 0).fillna(0).astype(int)
    df['workplace_smokefree'] = _series_or_default(df, 'workplace_policy', 0).fillna(0).astype(int)
    
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

    # Additional combinations for Phase 3 exploration
    df['nrt_plus_counseling'] = (
        (df['used_nrt'] == 1) & (df['used_counseling'] == 1)
    ).astype(int)
    df['nrt_plus_quitline'] = (
        (df['used_nrt'] == 1) & (df['used_quitline'] == 1)
    ).astype(int)
    df['med_plus_quitline'] = (
        (df['used_any_medication'] == 1) & (df['used_quitline'] == 1)
    ).astype(int)
    
    # High dependence × medication interactions
    df['highdep_x_varenicline'] = df['high_dependence'] * df['used_varenicline']
    df['highdep_x_nrt'] = df['high_dependence'] * df['used_nrt']
    
    # Age × method interactions
    df['young_x_counseling'] = df['age_young'] * df['used_counseling']
    
    return df


def engineer_all_features(df, codebook_overrides=None, recode_missing=True):
    """
    Apply all feature engineering functions.
    
    Args:
        df (pd.DataFrame): Raw dataset
        codebook_overrides (dict|None): Optional explicit mapping from PATH
            codebook variable names to canonical columns. See map_from_codebook.
        recode_missing (bool): Replace PATH negative missing codes with NaN
        
    Returns:
        pd.DataFrame: Dataset with all engineered features
    """
    # Phase 3: Map codebook variables to canonical columns before feature creation
    df = map_from_codebook(df, codebook_overrides=codebook_overrides, recode_missing=recode_missing)

    df = engineer_dependence_features(df)
    df = engineer_demographic_features(df)
    df = engineer_cessation_method_features(df)
    df = engineer_quit_history_features(df)
    df = engineer_motivation_features(df)
    df = engineer_environmental_features(df)
    df = engineer_interaction_features(df)

    # Ensure all expected features exist to prevent downstream KeyErrors
    for col in get_feature_list():
        if col not in df.columns:
            # default to 0; downstream imputers/modeling can handle
            df[col] = 0

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
        'race_white', 'race_black', 'race_hispanic', 'race_other',
        
        # Cessation methods
        'used_nrt', 'used_patch', 'used_gum', 'used_lozenge',
        'used_varenicline', 'used_bupropion', 'used_any_medication',
        'used_counseling', 'used_quitline', 'used_any_behavioral', 'used_any_method', 'cold_turkey',
        
        # Method combinations
    'med_plus_counseling', 'nrt_plus_med', 'nrt_plus_counseling', 'nrt_plus_quitline', 'med_plus_quitline',
        
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
