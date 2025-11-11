# PATH Missing Codes: Impact on Models and Resolution

## Your Question
> "will the -99911 effect models or will the model understand that it means 0 or missing data"

## Short Answer
**Before fix**: YES, PATH missing codes like -99911 and -99999 would have severely damaged models by introducing large negative values that models interpret as legitimate data.

**After fix**: NO, all PATH missing codes are now properly converted to NaN (missing values) before modeling, so models handle them correctly through imputation or row dropping.

---

## The Problem

### What are PATH Missing Codes?

PATH Study uses negative values to indicate different types of missing or invalid data:

| Code | Meaning | Example |
|------|---------|---------|
| `-99999` | Inconsistent/missing value | Data quality issue |
| `-99988` | Don't know | Respondent answered "don't know" |
| `-99977` | Refused | Respondent refused to answer |
| `-99955` | Improbable/inconsistent | Value flagged as implausible |
| `-99911` | Skip pattern | Question legitimately skipped (e.g., "used NRT?" skipped if didn't try to quit) |
| `-97777` | Other missing | Other type of missing data |

### Why This Matters for Machine Learning

**Without proper handling, models would interpret these as real values:**

```python
# Example: Cigarettes per day (cpd)
# Raw PATH data contains:
cpd_values = [5, 10, 20, -99911, 15, -99988, 8]

# Model would see:
Mean cpd = (5 + 10 + 20 + (-99911) + 15 + (-99988) + 8) / 7 = -28,548 cpd 😱
```

**Catastrophic impacts on models:**

1. **Feature scaling breaks**: StandardScaler would compute mean = -28,548, making all real values appear identical
2. **Coefficients become meaningless**: Model learns "higher cpd = more likely to quit" because missing data (-99911) gets confused with heavy smoking
3. **Predictions fail**: New data with real cpd=20 gets scaled as if it's much higher than -99911
4. **Feature importance skews**: Features with more missing codes appear more predictive

### Real Example from Your Data

**Before fix:**
```
cpd column contained:
  Mean: -466.46 cpd  ← WRONG! Contaminated by -99999
  Min: -99999.00     ← Missing code treated as data
  Max: 2160.00

ttfc_minutes:
  Mean: 0.05 minutes ← WRONG! Should be ~30 minutes
  Min: -99999.00     ← Missing code treated as data
```

This would have caused models to:
- Think "low cpd" predicts quitting (because -99999 is "low")
- Assign huge weight to cpd feature (because of extreme range)
- Make terrible predictions on real-world data

---

## The Solution

### What We Fixed

**Updated `PATH_MISSING_CODES` list** in `src/feature_engineering.py`:

```python
PATH_MISSING_CODES = [
    -9, -8, -7, -4, -1,           # General public-use missing codes
    -99999,                        # ← ADDED: Inconsistent/missing value
    -99988,                        # Don't know
    -99977,                        # Refused
    -99955,                        # Improbable/inconsistent value
    -99911,                        # Skip pattern (legitimate skip)
    -97777                         # Other missing
]
```

### How It Works

The `_replace_path_missing()` function now converts ALL these codes to NaN:

```python
def _replace_path_missing(series):
    """Replace PATH negative missing codes with np.nan."""
    s = pd.to_numeric(series, errors='coerce')
    return s.replace(PATH_MISSING_CODES, np.nan)
```

**Before → After:**
```
Raw PATH data:
cpd: [5, 10, 20, -99911, 15, -99988, 8, -99999]

After _replace_path_missing():
cpd: [5, 10, 20, NaN, 15, NaN, 8, NaN]
```

---

## Validation Results

**After fix - all features clean:**

```
✅ age: No negative values (4 NaN, 0.0%)
✅ cpd: No negative values (37,190 NaN, 77.7%)
✅ ttfc_minutes: No negative values (12,647 NaN, 26.4%)
✅ longest_quit_duration: No negative values
✅ used_nrt: No negative values
✅ used_varenicline: No negative values
✅ household_smokers: No negative values
✅ college_degree: No negative values
```

**Example: cpd now has correct statistics:**
```
Mean: 8.31 cpd     ← Realistic average
Median: 3.00 cpd   ← Makes sense
Range: 0 to 2,160  ← All positive (2,160 may be data entry error for "216")
```

---

## How Models Handle Missing Data (NaN)

### Scikit-learn Models

**Most scikit-learn models cannot handle NaN directly and will raise errors:**

```python
from sklearn.ensemble import RandomForestClassifier

# This would FAIL:
model.fit(X_with_nan, y)  # ValueError: Input contains NaN
```

**Your pipeline must handle NaN before modeling:**

#### Option 1: Imputation (Recommended)
```python
from sklearn.impute import SimpleImputer

# Mean imputation for continuous features
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Or median (more robust to outliers)
imputer = SimpleImputer(strategy='median')

# Or most frequent (for categorical)
imputer = SimpleImputer(strategy='most_frequent')
```

#### Option 2: Drop rows with missing data
```python
# Remove any row with NaN
df_clean = df.dropna()

# More selective: drop only if key features are missing
df_clean = df.dropna(subset=['cpd', 'ttfc_minutes'])
```

#### Option 3: Indicator columns
```python
from sklearn.impute import SimpleImputer

# Add binary "was missing" columns
imputer = SimpleImputer(strategy='mean', add_indicator=True)
# Creates: cpd_imputed, cpd_missing_indicator
```

### XGBoost (Handles NaN Natively)

**XGBoost has built-in missing value handling:**

```python
import xgboost as xgb

# XGBoost automatically learns optimal direction for missing values
model = xgb.XGBClassifier()
model.fit(X_with_nan, y)  # Works fine! No imputation needed
```

XGBoost treats missing values as a separate category and learns during training whether to send missing values left or right at each split.

---

## Impact on Your Specific Dataset

### Missing Data Patterns

**High missingness in smoking behavior variables:**

| Feature | Missing % | Reason |
|---------|-----------|--------|
| `cpd` | 77.7% | Skip pattern (-99911): Asked only to current smokers in follow-up |
| `ttfc_minutes` | 26.4% | Skip pattern: Asked only to daily smokers |
| `age` | 0.0% | Core demographic - always collected |
| `longest_quit_duration` | 90.7% | Skip pattern: Only for those who attempted to quit in past 12 months |

**Why cpd has 77.7% missing:**
- Dataset contains transitions from smoker → quit
- Follow-up wave asks "How many cigarettes per day?" 
- If person quit, they get skip pattern (-99911) because they're no longer smoking
- This is **informative missingness**: Missing cpd in follow-up often means they quit!

### Modeling Implications

**Strategy 1: Drop rows with missing cpd**
```python
df_complete = df.dropna(subset=['cpd'])
# Keeps only 10,692 of 47,882 rows (22.3%)
# May lose important quit patterns
```

**Strategy 2: Impute cpd with baseline cpd**
```python
# Use baseline smoking rate as imputed value
df['cpd_imputed'] = df['cpd'].fillna(df['baseline_cpd'])
# Preserves all rows, assumes quit attempts use baseline intensity
```

**Strategy 3: Create "missing cpd" indicator**
```python
df['cpd_missing'] = df['cpd'].isna().astype(int)
# Model can learn that missing cpd predicts quitting
```

**Recommended approach for your project:**
```python
# Combine imputation + indicator
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median', add_indicator=True)
# Creates: cpd (imputed with median) + cpd_missing (0/1 indicator)
# Models can use both: imputed value AND knowledge that it was missing
```

---

## Summary

### Before Fix: ❌ Models Would Fail

- PATH missing codes (-99911, -99999) treated as extreme negative values
- Feature means/scales completely wrong
- Model coefficients meaningless
- Predictions unreliable

### After Fix: ✅ Models Will Work Correctly

- All PATH missing codes converted to NaN
- Models see clean numeric data or proper missing values
- Feature scaling works correctly (StandardScaler computes mean from real values only)
- XGBoost handles NaN natively
- Scikit-learn requires imputation (which you can control)

### Action Required in Your Modeling Pipeline

**✅ IMPLEMENTED: Option 2 - XGBoost with Native NaN Handling**

Updated `src/modeling.py` to use XGBoost's native missing value support:

```python
def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with native NaN handling."""
    # NO imputation needed - XGBoost handles NaN natively!
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        missing=np.nan,  # Native NaN handling
        # ... other params
    )
    
    # Train with raw data containing NaN
    xgb_model.fit(X_train, y_train)  # Works with NaN!
```

**Why This is Critical for Your Dataset:**

Missing data is **highly informative** in smoking cessation:

| Feature | Missing → Quit Rate | Present → Quit Rate | Difference |
|---------|-------------------|-------------------|-----------|
| `ttfc_minutes` | 77.1% | 12.7% | **+64.4 pp** 😱 |
| `cpd` | 30.8% | 26.0% | **+4.8 pp** |

**What this means:**
- Missing `ttfc_minutes` (time to first cigarette) is **strongly predictive** of quitting (77% vs 13%)
- This makes sense: If person quit, they don't have a "time to first cigarette" → skip pattern
- **Mean imputation would destroy this signal** by replacing NaN with ~30 minutes
- **XGBoost learns**: "When ttfc is missing → higher chance of quit success"

**Other model options:**
**Other model options:**

**Logistic Regression & Random Forest** (still use mean imputation):
```python
# These models require complete data
X_train_filled = X_train.fillna(X_train.mean())
X_val_filled = X_val.fillna(X_train.mean())
```

**Why keep imputation for LR/RF:**
- They cannot handle NaN natively (will raise ValueError)
- Serve as baseline comparisons to XGBoost
- Expected: XGBoost will outperform due to preserved missing data signal

---

## Files Modified

1. **`src/feature_engineering.py`** (line 12-19):
   - Added `-99999` to `PATH_MISSING_CODES` list
   - Now catches all missing codes from PATH data

## Testing

Validated on full dataset (47,882 transitions):
- ✅ Zero negative values in any feature
- ✅ NaN properly represents missing data
- ✅ Feature statistics now realistic (e.g., mean cpd = 8.31, not -466)
- ✅ Ready for imputation or XGBoost
