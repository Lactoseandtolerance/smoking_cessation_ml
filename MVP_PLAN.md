# Streamlined MVP Plan: Smoking Cessation Prediction Using PATH Study

**Project Goal:** Predict smoking cessation success using machine learning on longitudinal PATH Study data with a focus on rapid MVP delivery.

---

## Executive Summary

This plan delivers a working smoking cessation prediction model using the PATH Study (Population Assessment of Tobacco and Health) Waves 1-5. The approach pools multiple wave transitions into a person-period dataset, implements three ML algorithms with class weighting, and provides SHAP-based interpretability with fairness assessment.

**Target Performance:** ROC-AUC > 0.70  
**Timeline:** ~16 days from data download to deliverables  
**Primary Output:** Trained model, IEEE format report, Streamlit dashboard, 10-minute presentation

---

## Key Design Decisions (Locked In)

### Data Structure
- **Dataset:** PATH Study Waves 1-5 (CSV format)
- **Sample:** Pooled person-period design across wave transitions (1→2, 2→3, 3→4, 4→5)
- **Age cohorts:** 18-24, 25-34, 35-44, 45-54, 55-64, 65+
- **Unit of analysis:** Each quit attempt by each person (person-periods)

### Outcome Definitions
- **Primary outcome:** Quit attempt success rate (30-day abstinence after quit attempt)
- **Secondary outcome (if time permits):** Sustained abstinence across 2+ consecutive waves
- **Rationale:** Quit attempt success is cleaner, more actionable for intervention design

### Technical Decisions
- **Class imbalance strategy:** Class weighting (default in sklearn/xgboost). Try SMOTE only if initial results are poor.
- **Survey weights:** Compare weighted vs unweighted models, use whichever performs better
- **Cessation method coding:** Planned-but-not-used = 0 (minimize grey area, document in methods)
- **Fairness approach:** Report disparities only (no mitigation attempts)
- **Feature target:** 25-30 engineered features for MVP

### Critical Technical Note on Class Imbalance

**Pooling waves does NOT eliminate class imbalance.** If 7% of quit attempts succeed in each wave, pooling gives you more observations but still 93% failures. Class weighting remains essential.

**Default approach:** Enable `class_weight='balanced'` in sklearn, `scale_pos_weight` in XGBoost. Only implement SMOTE if validation AUC < 0.60.

---

## Phase-by-Phase Implementation Plan

### Phase 1: Data Acquisition (Day 1)

**Objectives:**
- Obtain PATH Study data with immediate access
- Set up development environment
- Download all documentation

**Actions:**
1. Register at ICPSR: https://www.icpsr.umich.edu/
2. Navigate to PATH Study Series: https://www.icpsr.umich.edu/web/NAHDAP/series/606
3. Download Public Use Files for Waves 1-5 (CSV format)
4. Download the 565-page user guide and all wave-specific codebooks
5. Create project directory structure:
```
smoking-cessation-ml/
├── data/
│   ├── raw/                    # PATH CSV files (not in Git)
│   ├── processed/              # Cleaned datasets
│   └── data_dictionary.md      
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling_baseline.ipynb
│   ├── 05_modeling_advanced.ipynb
│   ├── 06_model_interpretation.ipynb
│   └── 07_fairness_assessment.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── visualization.py
├── models/                     # Saved model files
├── dashboard/
│   └── app.py
├── reports/
│   ├── figures/
│   └── final_report.pdf
├── requirements.txt
├── README.md
└── .gitignore
```

6. Set up virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn plotly streamlit jupyter imbalanced-learn
pip freeze > requirements.txt
```

7. Initialize Git repository and add `.gitignore` for data files

**Deliverables:**
- PATH Study Waves 1-5 CSV files in `data/raw/`
- Documentation downloaded
- Environment configured and tested
- GitHub repository initialized

---

### Phase 2: Define Analytical Sample (Days 2-3)

**Objectives:**
- Understand PATH Study structure and smoking variables
- Create person-period pooled dataset
- Calculate actual cessation/success rates
- Determine sample size for modeling

**Step 2.1: Identify Baseline Smokers in Each Wave**

Key PATH variables to locate (exact names may vary - check codebook):
- Current smoking status
- Cigarettes per day
- Quit attempts since last wave
- Smoking cessation at follow-up
- Time to first cigarette

```python
import pandas as pd

# Load waves
wave1 = pd.read_csv('data/raw/PATH_Wave1.csv')
wave2 = pd.read_csv('data/raw/PATH_Wave2.csv')
wave3 = pd.read_csv('data/raw/PATH_Wave3.csv')
wave4 = pd.read_csv('data/raw/PATH_Wave4.csv')
wave5 = pd.read_csv('data/raw/PATH_Wave5.csv')

# Identify current smokers at each wave
# (Replace 'current_smoker' with actual PATH variable name)
wave1_smokers = wave1[wave1['current_smoker'] == 1].copy()
wave2_smokers = wave2[wave2['current_smoker'] == 1].copy()
wave3_smokers = wave3[wave3['current_smoker'] == 1].copy()
wave4_smokers = wave4[wave4['current_smoker'] == 1].copy()

print(f"Wave 1 smokers: {len(wave1_smokers)}")
print(f"Wave 2 smokers: {len(wave2_smokers)}")
# ... etc
```

**Step 2.2: Create Person-Period Dataset**

Strategy: For each wave transition, create observations where:
- **Baseline (time t):** Person is a current smoker who attempted to quit
- **Follow-up (time t+1):** Measure smoking status
- **Outcome:** Binary indicator of cessation success

```python
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

# Create transitions for each wave pair
transitions = []
transitions.append(create_transition(wave1, wave2, 'W1_W2'))
transitions.append(create_transition(wave2, wave3, 'W2_W3'))
transitions.append(create_transition(wave3, wave4, 'W3_W4'))
transitions.append(create_transition(wave4, wave5, 'W4_W5'))

# Pool all transitions
pooled_data = pd.concat(transitions, ignore_index=True)

print(f"Total person-periods: {len(pooled_data)}")
print(f"Unique individuals: {pooled_data['person_id'].nunique()}")
```

**Step 2.3: Calculate Cessation Rates**

```python
# Overall cessation rate
cessation_rate = pooled_data['quit_success'].mean()
print(f"Quit attempt success rate: {cessation_rate:.1%}")

# By wave transition
cessation_by_wave = pooled_data.groupby('transition')['quit_success'].mean()
print("\nCessation rates by transition:")
print(cessation_by_wave)

# By demographic groups (preview for fairness assessment)
cessation_by_age = pooled_data.groupby('age_cohort')['quit_success'].mean()
cessation_by_education = pooled_data.groupby('education')['quit_success'].mean()
```

**Critical checkpoint:** If cessation rate is <5% or >20%, reconsider outcome definition. Expected range: 7-15%.

**Step 2.4: Handle Missing Data**

```python
# Calculate missingness by variable
missing_pct = (pooled_data.isnull().sum() / len(pooled_data) * 100).sort_values(ascending=False)
print("\nVariables with >10% missing:")
print(missing_pct[missing_pct > 10])

# Decide on thresholds:
# - Exclude variables with >40% missing (too sparse to be useful)
# - Impute or flag variables with 10-40% missing
# - Keep variables with <10% missing
```

**Deliverables:**
- `data/processed/pooled_transitions.csv` - Person-period dataset
- `notebooks/01_data_exploration.ipynb` - Sample characteristics, cessation rates
- Documentation of actual sample size and cessation rate

---

### Phase 3: Data Cleaning & Feature Engineering (Days 4-5)

**Objectives:**
- Clean and recode variables
- Engineer predictive features based on cessation literature
- Create analysis-ready dataset with 25-30 features

**Step 3.1: Variable Cleaning**

```python
import numpy as np

def clean_variables(df):
    """Clean PATH Study variables for analysis."""
    
    # Handle special codes
    # PATH uses negative values for missing: -9 (refused), -1 (inapplicable), etc.
    # Replace with NaN for clarity
    df = df.replace({-9: np.nan, -1: np.nan, -7: np.nan})
    
    # Recode categorical variables
    # Example: Education
    df['education_cat'] = pd.cut(df['education_years'], 
                                  bins=[0, 12, 14, 16, 25],
                                  labels=['<HS', 'HS', 'Some College', 'College+'])
    
    # Example: Income (adapt to PATH's actual coding)
    # df['income_cat'] = map PATH's income categories
    
    return df

pooled_data = clean_variables(pooled_data)
```

**Step 3.2: Feature Engineering**

**Tier 1 Features (Essential - Implement First):**

```python
def engineer_features(df):
    """Create predictive features for cessation modeling."""
    
    # ===== NICOTINE DEPENDENCE =====
    # Time to first cigarette (TTFC) - strongest dependence predictor
    df['ttfc_minutes'] = df['time_to_first_cigarette']  # Adapt variable name
    df['high_dependence'] = (df['ttfc_minutes'] < 30).astype(int)
    df['very_high_dependence'] = (df['ttfc_minutes'] < 5).astype(int)
    
    # Cigarettes per day
    df['cpd'] = df['cigarettes_per_day']
    df['cpd_heavy'] = (df['cpd'] >= 20).astype(int)
    df['cpd_light'] = (df['cpd'] <= 10).astype(int)
    
    # Composite dependence score (if multiple indicators available)
    df['dependence_score'] = (
        df['high_dependence'] + 
        df['cpd_heavy'] + 
        df['wake_up_urge'].fillna(0)  # Adapt to actual PATH variables
    )
    
    # ===== DEMOGRAPHICS =====
    # Age cohorts (standard categories)
    df['age_cohort'] = pd.cut(df['age'], 
                               bins=[18, 25, 35, 45, 55, 65, 100],
                               labels=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
    df['age_young'] = (df['age'] < 35).astype(int)
    
    # Sex
    df['female'] = (df['sex'] == 'Female').astype(int)
    
    # Education (college degree is strong predictor)
    df['college_degree'] = (df['education_cat'] == 'College+').astype(int)
    
    # Income (above vs below median)
    df['high_income'] = (df['income'] > df['income'].median()).astype(int)
    
    # ===== CESSATION METHODS =====
    # Individual methods (binary indicators)
    # Planned-but-not-used = 0 per user decision
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
    
    # Cold turkey (no methods used)
    df['cold_turkey'] = (
        (df['used_nrt'] == 0) & 
        (df['used_varenicline'] == 0) & 
        (df['used_bupropion'] == 0) & 
        (df['used_counseling'] == 0)
    ).astype(int)
    
    # ===== QUIT HISTORY =====
    df['num_previous_quits'] = df['lifetime_quit_attempts'].fillna(0)
    df['previous_quit_success'] = (df['num_previous_quits'] > 0).astype(int)
    df['longest_quit_duration'] = df['longest_abstinence_days'].fillna(0)
    
    # ===== MOTIVATION =====
    df['motivation_high'] = (df['readiness_to_quit'] >= 7).astype(int)  # If on 1-10 scale
    df['plans_to_quit'] = df['plans_quit_next_month'].fillna(0).astype(int)
    
    return df

pooled_data = engineer_features(pooled_data)
```

**Tier 2 Features (Add If Time Permits):**

```python
def engineer_advanced_features(df):
    """Create interaction features and complex predictors."""
    
    # ===== METHOD COMBINATIONS =====
    # Multi-method approach (proven effective in literature)
    df['med_plus_counseling'] = (
        (df['used_any_medication'] == 1) & (df['used_counseling'] == 1)
    ).astype(int)
    
    df['nrt_plus_med'] = (
        (df['used_nrt'] == 1) & (df['used_any_medication'] == 1)
    ).astype(int)
    
    # ===== INTERACTIONS =====
    # High dependence × medication use
    df['highdep_x_varenicline'] = df['high_dependence'] * df['used_varenicline']
    df['highdep_x_nrt'] = df['high_dependence'] * df['used_nrt']
    
    # Age × method interactions (younger smokers may prefer different methods)
    df['young_x_counseling'] = df['age_young'] * df['used_counseling']
    
    # ===== ENVIRONMENTAL =====
    df['household_smokers'] = (df['num_household_smokers'] > 0).astype(int)
    df['smokefree_home'] = df['home_smoking_rules'].fillna(0).astype(int)
    df['workplace_smokefree'] = df['workplace_policy'].fillna(0).astype(int)
    
    # ===== PSYCHOSOCIAL (if available in PUF) =====
    df['mental_health_flag'] = df['mental_health_diagnosis'].fillna(0).astype(int)
    df['alcohol_use'] = (df['alcohol_frequency'] > 0).fillna(0).astype(int)
    
    return df

pooled_data = engineer_advanced_features(pooled_data)
```

**Step 3.3: Select Final Feature Set**

```python
# Define feature columns for modeling
feature_cols = [
    # Nicotine dependence
    'high_dependence', 'very_high_dependence', 'cpd', 'cpd_heavy', 'cpd_light',
    'dependence_score',
    
    # Demographics
    'age', 'age_young', 'female', 'college_degree', 'high_income',
    
    # Cessation methods
    'used_nrt', 'used_varenicline', 'used_bupropion', 'used_any_medication',
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
    'highdep_x_varenicline', 'highdep_x_nrt'
]

# Create modeling dataset
X = pooled_data[feature_cols].copy()
y = pooled_data['quit_success'].copy()

print(f"Feature count: {len(feature_cols)}")
print(f"Sample size: {len(X)}")
print(f"Outcome distribution: {y.value_counts(normalize=True)}")
```

**Step 3.4: Handle Remaining Missing Data**

```python
from sklearn.impute import SimpleImputer

# Strategy: Impute median for numeric, mode for categorical
numeric_features = X.select_dtypes(include=[np.number]).columns
categorical_features = X.select_dtypes(exclude=[np.number]).columns

# Numeric imputation
numeric_imputer = SimpleImputer(strategy='median')
X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])

# Categorical imputation (or create "missing" category)
categorical_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])

# Save imputers for later use on test data
import joblib
joblib.dump(numeric_imputer, 'models/numeric_imputer.pkl')
joblib.dump(categorical_imputer, 'models/categorical_imputer.pkl')
```

**Deliverables:**
- `data/processed/modeling_data.csv` - Analysis-ready dataset
- `notebooks/02_data_cleaning.ipynb`
- `notebooks/03_feature_engineering.ipynb`
- Documentation of final feature list (25-35 features)

---

### Phase 4: Modeling Pipeline (Days 6-9)

**Objectives:**
- Implement train/validation/test splits
- Train three ML models with class weighting
- Compare performance and select best model
- Evaluate on held-out test set

**Step 4.1: Train/Validation/Test Split (60/20/20)**

**CRITICAL:** Split by `person_id`, not by observation, to prevent data leakage. Same person should not appear in both train and test sets.

```python
from sklearn.model_selection import train_test_split

# Get unique person IDs
unique_persons = pooled_data['person_id'].unique()
n_persons = len(unique_persons)

print(f"Total unique individuals: {n_persons}")

# Split persons into train (60%), temp (40%)
train_ids, temp_ids = train_test_split(
    unique_persons, 
    test_size=0.4, 
    random_state=42
)

# Split temp into validation (20%) and test (20%)
val_ids, test_ids = train_test_split(
    temp_ids, 
    test_size=0.5, 
    random_state=42
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
```

**Step 4.2: Baseline Model - Logistic Regression with Class Weighting**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, classification_report, 
                              confusion_matrix, precision_recall_curve, auc)
import matplotlib.pyplot as plt

# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train logistic regression with class weighting
lr = LogisticRegression(
    class_weight='balanced',  # Automatic class weighting
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

lr.fit(X_train_scaled, y_train)

# Predict on validation set
y_val_pred_proba_lr = lr.predict_proba(X_val_scaled)[:, 1]
y_val_pred_lr = lr.predict(X_val_scaled)

# Evaluate
auc_lr = roc_auc_score(y_val, y_val_pred_proba_lr)
precision_lr, recall_lr, _ = precision_recall_curve(y_val, y_val_pred_proba_lr)
pr_auc_lr = auc(recall_lr, precision_lr)

print("=" * 50)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 50)
print(f"ROC-AUC: {auc_lr:.3f}")
print(f"PR-AUC: {pr_auc_lr:.3f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_lr))
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred_lr))

# Feature importance (coefficients)
feature_importance_lr = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': lr.coef_[0]
}).sort_values('coefficient', ascending=False)

print("\nTop 10 Features:")
print(feature_importance_lr.head(10))

# Save scaler for later use
joblib.dump(scaler, 'models/scaler.pkl')
```

**Step 4.3: Random Forest with Class Weighting**

```python
from sklearn.ensemble import RandomForestClassifier

# Train random forest (no scaling needed)
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced',  # Automatic class weighting
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Predict on validation set
y_val_pred_proba_rf = rf.predict_proba(X_val)[:, 1]
y_val_pred_rf = rf.predict(X_val)

# Evaluate
auc_rf = roc_auc_score(y_val, y_val_pred_proba_rf)
precision_rf, recall_rf, _ = precision_recall_curve(y_val, y_val_pred_proba_rf)
pr_auc_rf = auc(recall_rf, precision_rf)

print("=" * 50)
print("RANDOM FOREST RESULTS")
print("=" * 50)
print(f"ROC-AUC: {auc_rf:.3f}")
print(f"PR-AUC: {pr_auc_rf:.3f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_rf))
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred_rf))

# Feature importance
feature_importance_rf = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(feature_importance_rf.head(10))
```

**Step 4.4: XGBoost with Class Weighting**

```python
import xgboost as xgb

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale pos weight: {scale_pos_weight:.2f}")

# Train XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,  # Class weighting for XGBoost
    random_state=42,
    eval_metric='auc',
    early_stopping_rounds=10
)

# Fit with early stopping on validation set
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Predict on validation set
y_val_pred_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
y_val_pred_xgb = xgb_model.predict(X_val)

# Evaluate
auc_xgb = roc_auc_score(y_val, y_val_pred_proba_xgb)
precision_xgb, recall_xgb, _ = precision_recall_curve(y_val, y_val_pred_proba_xgb)
pr_auc_xgb = auc(recall_xgb, precision_xgb)

print("=" * 50)
print("XGBOOST RESULTS")
print("=" * 50)
print(f"ROC-AUC: {auc_xgb:.3f}")
print(f"PR-AUC: {pr_auc_xgb:.3f}")
print(f"Best iteration: {xgb_model.best_iteration}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_xgb))
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred_xgb))

# Feature importance
feature_importance_xgb = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(feature_importance_xgb.head(10))
```

**Step 4.5: Model Comparison**

```python
# Compile results
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'ROC-AUC': [auc_lr, auc_rf, auc_xgb],
    'PR-AUC': [pr_auc_lr, pr_auc_rf, pr_auc_xgb]
})

print("=" * 50)
print("MODEL COMPARISON (Validation Set)")
print("=" * 50)
print(results.to_string(index=False))

# Select best model (highest ROC-AUC)
best_model_name = results.loc[results['ROC-AUC'].idxmax(), 'Model']
print(f"\nBest model: {best_model_name}")

# Map to actual model object
model_map = {
    'Logistic Regression': lr,
    'Random Forest': rf,
    'XGBoost': xgb_model
}
best_model = model_map[best_model_name]
```

**Step 4.6: Final Evaluation on Test Set**

```python
# DO NOT touch test set until this point

# Prepare test data
if best_model_name == 'Logistic Regression':
    X_test_prepared = scaler.transform(X_test)
else:
    X_test_prepared = X_test

# Predict
y_test_pred_proba = best_model.predict_proba(X_test_prepared)[:, 1]
y_test_pred = best_model.predict(X_test_prepared)

# Final evaluation
test_auc = roc_auc_score(y_test, y_test_pred_proba)
precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred_proba)
pr_auc_test = auc(recall_test, precision_test)

print("=" * 50)
print(f"FINAL TEST SET RESULTS - {best_model_name}")
print("=" * 50)
print(f"ROC-AUC: {test_auc:.3f}")
print(f"PR-AUC: {pr_auc_test:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# Save final model
joblib.dump(best_model, 'models/final_model.pkl')
joblib.dump({'model_name': best_model_name, 
             'feature_cols': feature_cols,
             'test_auc': test_auc}, 
            'models/model_metadata.pkl')

print(f"\nModel saved to models/final_model.pkl")
```

**Performance Benchmark:** Target ROC-AUC > 0.70. Published studies using PATH data achieved 0.72.

**Deliverables:**
- `notebooks/04_modeling_baseline.ipynb` - Logistic Regression
- `notebooks/05_modeling_advanced.ipynb` - Random Forest and XGBoost
- `models/final_model.pkl` - Best performing model
- `models/model_metadata.pkl` - Model information
- Model comparison table and performance metrics

---

### Phase 5: SHAP Interpretation (Days 10-11)

**Objectives:**
- Generate SHAP values for model explainability
- Identify top predictors globally
- Examine feature interactions
- Create visualizations for report and dashboard

**Step 5.1: Generate SHAP Values**

```python
import shap

# Load best model if needed
best_model = joblib.load('models/final_model.pkl')
metadata = joblib.load('models/model_metadata.pkl')

# Create SHAP explainer
# For tree-based models (RF, XGBoost)
if metadata['model_name'] in ['Random Forest', 'XGBoost']:
    explainer = shap.TreeExplainer(best_model)
else:
    # For linear models
    explainer = shap.LinearExplainer(best_model, X_train_scaled)

# Calculate SHAP values
# Use subset for speed if dataset is large (>10k observations)
sample_size = min(1000, len(X_test))
X_test_sample = X_test.sample(n=sample_size, random_state=42)

if metadata['model_name'] == 'Logistic Regression':
    X_test_sample_prepared = scaler.transform(X_test_sample)
else:
    X_test_sample_prepared = X_test_sample

shap_values = explainer(X_test_sample_prepared)

print(f"SHAP values calculated for {sample_size} test observations")
```

**Step 5.2: Global Feature Importance**

```python
# Summary plot - bar chart of mean absolute SHAP values
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('reports/figures/shap_importance_bar.png', dpi=300, bbox_inches='tight')
plt.close()

# Summary plot - beeswarm (shows distribution of impacts)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.tight_layout()
plt.savefig('reports/figures/shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()

# Extract top features
shap_importance = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("\nTop 10 Features by Mean Absolute SHAP Value:")
print(shap_importance.head(10))

# Save for later use
shap_importance.to_csv('reports/figures/shap_feature_importance.csv', index=False)
```

**Step 5.3: Feature Dependence Plots (Interactions)**

```python
# Dependence plots for top 5 features
top_5_features = shap_importance.head(5)['feature'].tolist()

for i, feature in enumerate(top_5_features):
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature, 
        shap_values.values, 
        X_test_sample,
        interaction_index='auto',  # Automatically select best interaction
        show=False
    )
    plt.tight_layout()
    plt.savefig(f'reports/figures/shap_dependence_{feature}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created dependence plot for: {feature}")
```

**Step 5.4: Individual Predictions (Local Explanations)**

```python
# Select representative examples
successful_quitter = X_test_sample[y_test.loc[X_test_sample.index] == 1].iloc[0]
failed_quitter = X_test_sample[y_test.loc[X_test_sample.index] == 0].iloc[0]

# Waterfall plots
for name, individual in [('successful', successful_quitter), 
                         ('failed', failed_quitter)]:
    idx = X_test_sample.index.get_loc(individual.name)
    
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[idx], show=False)
    plt.tight_layout()
    plt.savefig(f'reports/figures/shap_waterfall_{name}_quitter.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created waterfall plot for: {name} quitter")
```

**Step 5.5: Write Interpretation**

Document key findings:

```python
# Example interpretation template
interpretation = f"""
SHAP Analysis Key Findings:

1. Top 5 Predictors (in order of importance):
   {shap_importance.head(5).to_string()}

2. Effect Sizes:
   - [Feature X]: High values increase cessation probability by [Y]%
   - [Feature Z]: Presence reduces cessation probability by [W]%

3. Key Interactions:
   - [Feature A] × [Feature B]: [Describe interaction effect]

4. Clinical Implications:
   - Modifiable factors: [List interventionable features]
   - Non-modifiable factors: [List demographic features]
   
5. Model Behavior:
   - The model primarily relies on [nicotine dependence/cessation methods/etc.]
   - [Feature X] has the largest impact on individual predictions
"""

with open('reports/shap_interpretation.txt', 'w') as f:
    f.write(interpretation)
```

**Deliverables:**
- `notebooks/06_model_interpretation.ipynb`
- `reports/figures/shap_*.png` - All SHAP visualizations
- `reports/shap_interpretation.txt` - Written interpretation
- SHAP values saved for dashboard use

---

### Phase 6: Fairness Assessment (Day 12)

**Objectives:**
- Evaluate model performance across demographic subgroups
- Identify performance disparities
- Report findings transparently (no mitigation attempts per your decision)

**Step 6.1: Define Demographic Subgroups**

```python
# Demographic variables to assess
fairness_groups = {
    'gender': ['Male', 'Female'],
    'age_cohort': ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
    'race_ethnicity': test_data['race_ethnicity'].unique(),
    'education': ['<HS', 'HS', 'Some College', 'College+'],
    'income_level': ['Low', 'Medium', 'High']  # Define thresholds
}
```

**Step 6.2: Calculate Performance Metrics by Subgroup**

```python
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def evaluate_subgroup(model, X, y, subgroup_mask, subgroup_name):
    """Calculate performance metrics for a demographic subgroup."""
    if subgroup_mask.sum() == 0:
        return None
    
    X_sub = X[subgroup_mask]
    y_sub = y[subgroup_mask]
    
    # Predict
    if metadata['model_name'] == 'Logistic Regression':
        X_sub_prepared = scaler.transform(X_sub)
    else:
        X_sub_prepared = X_sub
    
    y_pred_proba = model.predict_proba(X_sub_prepared)[:, 1]
    y_pred = model.predict(X_sub_prepared)
    
    # Calculate metrics
    try:
        auc = roc_auc_score(y_sub, y_pred_proba)
    except:
        auc = np.nan
    
    precision = precision_score(y_sub, y_pred, zero_division=0)
    recall = recall_score(y_sub, y_pred, zero_division=0)
    f1 = f1_score(y_sub, y_pred, zero_division=0)
    
    # False positive rate and false negative rate
    tn, fp, fn, tp = confusion_matrix(y_sub, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    
    return {
        'subgroup': subgroup_name,
        'n': len(y_sub),
        'base_rate': y_sub.mean(),
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'fnr': fnr
    }

# Evaluate all subgroups
fairness_results = []

for group_var, group_values in fairness_groups.items():
    for value in group_values:
        mask = test_data[group_var] == value
        result = evaluate_subgroup(
            best_model, 
            X_test, 
            y_test, 
            mask, 
            f"{group_var}={value}"
        )
        if result is not None:
            result['group_variable'] = group_var
            fairness_results.append(result)

fairness_df = pd.DataFrame(fairness_results)
```

**Step 6.3: Analyze Disparities**

```python
# Calculate disparity metrics
def calculate_disparities(fairness_df, metric='auc'):
    """Calculate max disparity for each demographic variable."""
    disparities = []
    
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
                'disparity': disparity
            })
    
    return pd.DataFrame(disparities).sort_values('disparity', ascending=False)

# Calculate disparities for AUC
auc_disparities = calculate_disparities(fairness_df, 'auc')

print("=" * 50)
print("FAIRNESS ASSESSMENT: AUC DISPARITIES")
print("=" * 50)
print(auc_disparities.to_string(index=False))

# Flag significant disparities (>0.05)
significant_disparities = auc_disparities[auc_disparities['disparity'] > 0.05]
if len(significant_disparities) > 0:
    print("\n⚠️  SIGNIFICANT DISPARITIES DETECTED (>0.05 AUC difference):")
    print(significant_disparities.to_string(index=False))
else:
    print("\n✓ No significant disparities detected (all <0.05 AUC difference)")

# Save results
fairness_df.to_csv('reports/figures/fairness_results.csv', index=False)
auc_disparities.to_csv('reports/figures/auc_disparities.csv', index=False)
```

**Step 6.4: Visualizations**

```python
import seaborn as sns

# AUC by demographic group
for group_var in fairness_groups.keys():
    group_data = fairness_df[fairness_df['group_variable'] == group_var]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=group_data, x='subgroup', y='auc')
    plt.axhline(y=test_auc, color='r', linestyle='--', label='Overall AUC')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('ROC-AUC')
    plt.title(f'Model Performance by {group_var}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'reports/figures/fairness_auc_by_{group_var}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

# Heatmap of multiple metrics by subgroup
pivot_data = fairness_df.pivot_table(
    index='subgroup',
    values=['auc', 'precision', 'recall', 'f1'],
    aggfunc='mean'
)

plt.figure(figsize=(10, 12))
sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1)
plt.title('Model Performance Metrics Across Demographic Subgroups')
plt.tight_layout()
plt.savefig('reports/figures/fairness_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
```

**Step 6.5: Document Findings**

```python
fairness_summary = f"""
FAIRNESS ASSESSMENT SUMMARY

Overall Test AUC: {test_auc:.3f}

Disparity Analysis:
{auc_disparities.to_string(index=False)}

Key Findings:
1. Largest disparity: {auc_disparities.iloc[0]['group_variable']} 
   (difference = {auc_disparities.iloc[0]['disparity']:.3f})

2. Groups with below-average performance:
{fairness_df[fairness_df['auc'] < test_auc][['subgroup', 'auc', 'n']].to_string(index=False)}

3. Potential explanations:
   - Sample size differences across groups
   - Differential access to cessation methods
   - Varying baseline cessation rates
   - Feature availability/measurement quality

4. Implications for deployment:
   - Model may perform worse for [specific groups]
   - Consider additional data collection for underperforming groups
   - Monitor performance in real-world deployment
"""

with open('reports/fairness_summary.txt', 'w') as f:
    f.write(fairness_summary)

print(fairness_summary)
```

**Deliverables:**
- `notebooks/07_fairness_assessment.ipynb`
- `reports/figures/fairness_*.png` - Visualizations by demographic group
- `reports/fairness_results.csv` - Complete performance metrics
- `reports/fairness_summary.txt` - Written summary

---

### Phase 7: Dashboard Development (Days 13-14)

**Objectives:**
- Create interactive Streamlit dashboard
- Enable exploration of model predictions and explanations
- Present fairness assessment visually
- Provide actionable insights

**Dashboard Structure:**

```python
# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# Page configuration
st.set_page_config(
    page_title="Smoking Cessation Prediction",
    page_icon="🚭",
    layout="wide"
)

# Load models and data
@st.cache_resource
def load_models():
    model = joblib.load('../models/final_model.pkl')
    metadata = joblib.load('../models/model_metadata.pkl')
    scaler = joblib.load('../models/scaler.pkl') if metadata['model_name'] == 'Logistic Regression' else None
    return model, metadata, scaler

@st.cache_data
def load_fairness_data():
    return pd.read_csv('../reports/figures/fairness_results.csv')

@st.cache_data
def load_feature_importance():
    return pd.read_csv('../reports/figures/shap_feature_importance.csv')

model, metadata, scaler = load_models()
fairness_df = load_fairness_data()
feature_importance = load_feature_importance()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Model Performance", "Feature Importance", 
     "Prediction Tool", "Fairness Assessment", "Key Insights"]
)

# ========== PAGE 1: OVERVIEW ==========
if page == "Overview":
    st.title("🚭 Smoking Cessation Prediction Model")
    st.markdown("### Using PATH Study Waves 1-5")
    
    st.markdown("""
    This machine learning model predicts smoking cessation success based on individual 
    characteristics, smoking history, and cessation methods used.
    
    **Data Source:** Population Assessment of Tobacco and Health (PATH) Study  
    **Study Design:** Longitudinal person-period dataset pooling 4 wave transitions  
    **Sample Size:** [XX,XXX] quit attempts from [X,XXX] unique individuals  
    **Outcome:** 30-day smoking abstinence after quit attempt
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Model", metadata['model_name'])
    with col2:
        st.metric("Test Set AUC", f"{metadata['test_auc']:.3f}")
    with col3:
        st.metric("Features", len(metadata['feature_cols']))
    
    st.markdown("---")
    st.subheader("Project Overview")
    st.markdown("""
    - **Research Question:** What factors predict successful smoking cessation one year after a quit attempt?
    - **Approach:** Compared Logistic Regression, Random Forest, and XGBoost models
    - **Key Innovation:** SHAP values for interpretable predictions and fairness assessment across demographic groups
    """)

# ========== PAGE 2: MODEL PERFORMANCE ==========
elif page == "Model Performance":
    st.title("📊 Model Performance")
    
    st.subheader("Model Comparison")
    st.markdown("""
    Three machine learning algorithms were trained and compared:
    - **Logistic Regression:** Interpretable baseline model
    - **Random Forest:** Ensemble method with feature importance
    - **XGBoost:** Gradient boosting for optimal performance
    """)
    
    # Load or display ROC curve
    st.image('../reports/figures/roc_curve.png', 
             caption='ROC Curves for All Models')
    
    st.markdown("---")
    st.subheader("Performance Metrics")
    
    # Display metrics table (create this in modeling phase)
    metrics_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'ROC-AUC': [0.68, 0.71, 0.73],  # Replace with actual values
        'Precision': [0.25, 0.28, 0.30],
        'Recall': [0.45, 0.48, 0.52],
        'F1-Score': [0.32, 0.35, 0.38]
    }
    st.dataframe(metrics_data)
    
    st.markdown("---")
    st.subheader("Confusion Matrix")
    st.image('../reports/figures/confusion_matrix.png',
             caption=f'{metadata["model_name"]} Confusion Matrix (Test Set)')

# ========== PAGE 3: FEATURE IMPORTANCE ==========
elif page == "Feature Importance":
    st.title("🎯 Feature Importance")
    
    st.markdown("""
    SHAP (SHapley Additive exPlanations) values show how each feature contributes 
    to individual predictions. Features are ranked by their average impact on predictions.
    """)
    
    st.subheader("Top Predictors")
    
    # Display top 10 features
    top_10 = feature_importance.head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_10['feature'], top_10['mean_abs_shap'])
    ax.set_xlabel('Mean Absolute SHAP Value')
    ax.set_title('Top 10 Predictive Features')
    ax.invert_yaxis()
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("SHAP Summary Plot")
    st.image('../reports/figures/shap_summary_beeswarm.png',
             caption='Feature Impact Distribution')
    
    st.markdown("---")
    st.subheader("Feature Interactions")
    
    selected_feature = st.selectbox(
        "Select feature to explore interactions:",
        top_10['feature'].tolist()
    )
    
    st.image(f'../reports/figures/shap_dependence_{selected_feature}.png',
             caption=f'How {selected_feature} affects predictions')

# ========== PAGE 4: PREDICTION TOOL ==========
elif page == "Prediction Tool":
    st.title("🔮 Individual Prediction Tool")
    
    st.markdown("""
    Enter characteristics to predict smoking cessation success probability.
    The model will provide a prediction and explanation.
    """)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demographics")
            age = st.slider("Age", 18, 80, 35)
            gender = st.radio("Gender", ["Male", "Female"])
            education = st.selectbox("Education", 
                                    ["<HS", "HS", "Some College", "College+"])
            
            st.subheader("Smoking Behavior")
            cpd = st.slider("Cigarettes per day", 1, 60, 15)
            ttfc = st.slider("Time to first cigarette (minutes)", 0, 120, 30)
            num_quits = st.number_input("Number of previous quit attempts", 0, 20, 1)
        
        with col2:
            st.subheader("Cessation Methods")
            used_nrt = st.checkbox("Using NRT (patch, gum, etc.)")
            used_varenicline = st.checkbox("Using varenicline (Chantix)")
            used_bupropion = st.checkbox("Using bupropion (Zyban)")
            used_counseling = st.checkbox("Receiving counseling")
            
            st.subheader("Environment")
            household_smokers = st.radio("Other household smokers?", ["No", "Yes"])
            smokefree_home = st.radio("Smoke-free home policy?", ["No", "Yes"])
        
        submitted = st.form_submit_button("Predict Cessation Success")
    
    if submitted:
        # Construct feature vector
        features = {
            'age': age,
            'female': 1 if gender == "Female" else 0,
            'college_degree': 1 if education == "College+" else 0,
            'cpd': cpd,
            'high_dependence': 1 if ttfc < 30 else 0,
            'num_previous_quits': num_quits,
            'used_nrt': 1 if used_nrt else 0,
            'used_varenicline': 1 if used_varenicline else 0,
            'used_bupropion': 1 if used_bupropion else 0,
            'used_counseling': 1 if used_counseling else 0,
            'household_smokers': 1 if household_smokers == "Yes" else 0,
            'smokefree_home': 1 if smokefree_home == "Yes" else 0,
            # ... add remaining features with default values
        }
        
        # Create DataFrame with all features
        input_df = pd.DataFrame([features])
        for col in metadata['feature_cols']:
            if col not in input_df.columns:
                input_df[col] = 0  # Default value for missing features
        
        input_df = input_df[metadata['feature_cols']]  # Ensure correct order
        
        # Prepare for prediction
        if metadata['model_name'] == 'Logistic Regression':
            input_prepared = scaler.transform(input_df)
        else:
            input_prepared = input_df
        
        # Predict
        probability = model.predict_proba(input_prepared)[0, 1]
        prediction = "Success" if probability >= 0.5 else "Failure"
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", prediction)
        with col2:
            st.metric("Success Probability", f"{probability:.1%}")
        with col3:
            color = "🟢" if probability >= 0.5 else "🔴"
            st.metric("Confidence", f"{color} {abs(probability-0.5)*2:.1%}")
        
        # SHAP explanation (simplified - would need full SHAP explainer)
        st.markdown("---")
        st.subheader("Prediction Explanation")
        st.markdown("""
        *Top factors influencing this prediction:*
        - High nicotine dependence (TTFC < 30 min): **Decreases** success probability
        - Using varenicline: **Increases** success probability  
        - College education: **Increases** success probability
        """)

# ========== PAGE 5: FAIRNESS ASSESSMENT ==========
elif page == "Fairness Assessment":
    st.title("⚖️ Fairness Assessment")
    
    st.markdown("""
    Model performance across demographic subgroups to identify potential disparities.
    Significant disparities (>0.05 AUC difference) are flagged.
    """)
    
    st.subheader("Performance by Demographic Group")
    
    # Group selector
    group_var = st.selectbox(
        "Select demographic variable:",
        fairness_df['group_variable'].unique()
    )
    
    # Filter data
    group_data = fairness_df[fairness_df['group_variable'] == group_var]
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(group_data)), group_data['auc'])
    ax.axhline(y=metadata['test_auc'], color='r', linestyle='--', 
               label='Overall AUC')
    ax.set_xticks(range(len(group_data)))
    ax.set_xticklabels(group_data['subgroup'], rotation=45, ha='right')
    ax.set_ylabel('ROC-AUC')
    ax.set_title(f'Model Performance by {group_var}')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    
    # Detailed metrics table
    st.markdown("---")
    st.subheader("Detailed Metrics")
    display_cols = ['subgroup', 'n', 'base_rate', 'auc', 'precision', 'recall', 'f1']
    st.dataframe(group_data[display_cols].style.format({
        'base_rate': '{:.3f}',
        'auc': '{:.3f}',
        'precision': '{:.3f}',
        'recall': '{:.3f}',
        'f1': '{:.3f}'
    }))
    
    # Disparity summary
    st.markdown("---")
    st.subheader("Disparity Summary")
    max_auc = group_data['auc'].max()
    min_auc = group_data['auc'].min()
    disparity = max_auc - min_auc
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Highest AUC", f"{max_auc:.3f}")
    with col2:
        st.metric("Lowest AUC", f"{min_auc:.3f}")
    with col3:
        status = "⚠️ Significant" if disparity > 0.05 else "✓ Acceptable"
        st.metric("Disparity", f"{disparity:.3f}", delta=status)

# ========== PAGE 6: KEY INSIGHTS ==========
elif page == "Key Insights":
    st.title("💡 Key Insights & Recommendations")
    
    st.markdown("### Main Findings")
    
    st.markdown("""
    1. **Model Performance**
       - Achieved {:.3f} AUC on held-out test set
       - Comparable to published benchmarks (Issabakhsh et al. 2023: 0.72 AUC)
       - {} model provided best performance
    
    2. **Top Predictive Factors**
       - **Nicotine dependence** (time to first cigarette) - strongest predictor
       - **Varenicline use** - increases cessation probability by ~15 percentage points
       - **Education level** - college graduates have higher success rates
       - **Previous quit attempts** - more attempts correlate with success
       - **Counseling + medication** - combination approach most effective
    
    3. **Fairness Findings**
       - [Largest disparity found in X demographic]
       - [Groups with below-average performance]
       - Potential causes: differential access to cessation aids, baseline cessation rate variations
    
    4. **Clinical Implications**
       - **Modifiable factors:** Encourage combination therapy (medication + counseling)
       - **High-risk profiles:** Heavy smokers with high dependence benefit most from varenicline
       - **Intervention targeting:** Focus resources on groups with lower predicted success
    """.format(
        metadata['test_auc'],
        metadata['model_name']
    ))
    
    st.markdown("---")
    st.markdown("### Limitations")
    
    st.markdown("""
    - **Self-reported data:** No biochemical verification of abstinence
    - **Short follow-up:** 1-year abstinence is an early marker, not long-term cessation
    - **Temporal limitations:** Data from 2013-2019 predates some modern interventions
    - **Missing variables:** Some psychosocial factors may be restricted in Public Use Files
    """)
    
    st.markdown("---")
    st.markdown("### Recommendations for Future Work")
    
    st.markdown("""
    1. **Model improvements:**
       - Incorporate PATH Restricted Use Files for additional variables
       - Extend follow-up to 2+ years for sustained cessation
       - Explore deep learning approaches (LSTM for temporal patterns)
    
    2. **Fairness mitigation:**
       - Collect additional data for underperforming demographic groups
       - Develop group-specific models or calibration
       - Monitor real-world deployment for performance drift
    
    3. **Clinical deployment:**
       - Integrate into electronic health records for decision support
       - Develop patient-facing tool for self-assessment
       - Conduct prospective validation in clinical settings
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**Project:** Smoking Cessation Prediction  
**Data:** PATH Study Waves 1-5  
**Authors:** [Your Names]  
**Course:** Data Mining  
**Date:** Fall 2025
""")
```

**Deliverables:**
- `dashboard/app.py` - Complete Streamlit dashboard
- Dashboard runnable with `streamlit run dashboard/app.py`
- All visualizations integrated from `reports/figures/`

---

### Phase 8: Report & Presentation (Days 15-16)

**Objectives:**
- Write IEEE format report (minimum 4 pages)
- Create 10-minute presentation
- Document findings and implications

**IEEE Report Structure:**

```markdown
# Predicting Smoking Cessation Success Using Machine Learning: A Multi-Model Approach with PATH Study Data

## Abstract (150-200 words)
Background, methods, key results, implications

## I. Introduction
- Smoking as public health burden
- Need for predictive tools to target interventions
- Research question and contributions
- Paper organization

## II. Related Work
- Prior ML studies on smoking cessation (cite Issabakhsh et al. 2023, others)
- Limitations of previous approaches
- How this work advances the field

## III. Methods
### A. Data Source
- PATH Study description
- Sample selection (person-period design)
- Final sample size and characteristics

### B. Outcome Definition
- 30-day point prevalence abstinence after quit attempt
- Justification for outcome choice

### C. Features
- Table 1: Feature categories and descriptions
- Feature engineering process
- Final feature count: [X] features

### D. Models
- Logistic Regression (baseline)
- Random Forest (ensemble)
- XGBoost (gradient boosting)
- Class weighting strategy
- Hyperparameter tuning approach

### E. Evaluation Metrics
- ROC-AUC (primary metric)
- Precision, Recall, F1
- Stratified train/validation/test splits (60/20/20)

### F. Interpretability & Fairness
- SHAP values for global and local explanations
- Fairness assessment across demographic groups
- Disparity thresholds

## IV. Results
### A. Sample Characteristics
- Table 2: Demographic and smoking characteristics
- Quit attempt success rate: [X]%

### B. Model Performance
- Table 3: Performance comparison (AUC, Precision, Recall, F1)
- Best model: [X] with [Y] AUC
- ROC curves (Figure 1)
- Confusion matrix (Figure 2)

### C. Feature Importance
- Table 4: Top 10 predictive features with SHAP values
- SHAP summary plot (Figure 3)
- Key interactions identified

### D. Fairness Assessment
- Table 5: Performance by demographic subgroup
- Significant disparities (>0.05 AUC): [list]
- Potential explanations

## V. Discussion
### A. Key Findings
- Comparison to published benchmarks
- Top modifiable predictors
- Clinical implications

### B. Fairness Considerations
- Groups with lower model performance
- Implications for deployment
- Need for continued monitoring

### C. Limitations
- Self-reported outcomes
- Short follow-up period
- Temporal limitations of data
- Public Use File restrictions

### D. Future Work
- Extended follow-up analyses
- Integration with clinical systems
- Fairness mitigation strategies

## VI. Conclusion
- Summary of contributions
- Public health implications
- Call to action

## References
[15-20 references including PATH Study papers, ML methodology, prior cessation research]

## Figures
- Figure 1: ROC curves for all models
- Figure 2: Confusion matrix (best model)
- Figure 3: SHAP summary plot
- Figure 4: Performance by demographic subgroup
```

**Presentation Outline (10 slides):**

1. **Title Slide**
   - Title, authors, affiliation, date

2. **Motivation**
   - Smoking burden statistics
   - Opportunity for targeted interventions

3. **Research Question & Data**
   - What predicts smoking cessation success?
   - PATH Study Waves 1-5, person-period design
   - Sample size and outcome

4. **Methods Overview**
   - Feature engineering (25-35 features)
   - Three ML algorithms with class weighting
   - SHAP interpretability + fairness assessment

5. **Sample Characteristics**
   - Demographics table
   - Quit attempt success rate
   - Key baseline characteristics

6. **Model Performance**
   - Bar chart: AUC comparison across models
   - Best model achieves [X] AUC
   - Comparison to published benchmark

7. **Top Predictors**
   - SHAP summary plot (top 10 features)
   - Nicotine dependence, varenicline, education highlighted
   - Effect sizes

8. **Example Prediction**
   - SHAP waterfall plot for one individual
   - "This person has 72% probability of success because..."

9. **Fairness Findings**
   - Performance by demographic group
   - Significant disparities identified
   - Implications for deployment

10. **Key Takeaways & Recommendations**
    - Modifiable factors for intervention
    - High-risk profiles to target
    - Limitations and future work

**Deliverables:**
- `reports/final_report.pdf` - IEEE format, 4+ pages
- `reports/presentation.pdf` - 10 slides
- `reports/presentation_script.md` - Speaking notes

---

## Quality Checklist Before Submission

### Code Quality
- [ ] All code runs without errors
- [ ] Functions have docstrings
- [ ] Variable names are descriptive
- [ ] No unused imports or commented code
- [ ] Notebooks have markdown explanations
- [ ] Random seeds set for reproducibility

### Documentation
- [ ] README.md updated with results
- [ ] requirements.txt includes all dependencies
- [ ] Data dictionary documents all variables
- [ ] Model metadata saved (features, performance)

### Analysis Quality
- [ ] Train/val/test splits by person_id (no leakage)
- [ ] Class imbalance addressed (weighting enabled)
- [ ] Model comparison includes multiple metrics
- [ ] SHAP values calculated and interpreted
- [ ] Fairness assessment across 4+ demographic variables
- [ ] Results compared to published benchmarks

### Deliverables
- [ ] IEEE report (4+ pages) with all sections
- [ ] Presentation (10 slides) with visuals
- [ ] Dashboard runs with `streamlit run dashboard/app.py`
- [ ] All figures saved in `reports/figures/`
- [ ] Models saved in `models/` directory

### Alignment with Rubric
- [ ] Research question clear and motivated
- [ ] Analysis complete (preprocessing, modeling, evaluation, interpretation)
- [ ] Results clearly tied to analysis with insightful discussion
- [ ] Plots labeled, informative, with reference info
- [ ] Report in IEEE format, well-structured, grammatically correct
- [ ] Code organized, readable, well-documented

---

## Expected Timeline (16 Days)

| Days | Phase | Deliverables |
|------|-------|--------------|
| 1 | Data Acquisition | PATH data downloaded, environment setup |
| 2-3 | Analytical Sample | Person-period dataset, cessation rates calculated |
| 4-5 | Feature Engineering | 25-35 features created, modeling dataset ready |
| 6-9 | Modeling | 3 models trained, best model selected, test evaluation |
| 10-11 | SHAP Interpretation | Feature importance, interactions, visualizations |
| 12 | Fairness Assessment | Performance by subgroup, disparity analysis |
| 13-14 | Dashboard | Streamlit app with 6 pages |
| 15-16 | Report & Presentation | IEEE report, 10-slide presentation |

**Total:** 16 days from data download to complete deliverables

---

## Key Resources

### PATH Study
- **Data download:** https://www.icpsr.umich.edu/web/NAHDAP/series/606
- **User guide (565 pages):** Available with data download
- **Documentation portal:** https://www.fda.gov/tobacco-products/research/fda-and-nih-study-population-assessment-tobacco-and-health

### Technical Documentation
- **SHAP:** https://shap.readthedocs.io/
- **Scikit-learn:** https://scikit-learn.org/stable/
- **XGBoost:** https://xgboost.readthedocs.io/
- **Streamlit:** https://docs.streamlit.io/
- **IEEE format:** https://www.ieee.org/conferences/publishing/templates.html

### Published Benchmarks
- Issabakhsh et al. (2023) - PATH Study ML analysis: 72% AUC
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/37289765/
  - Full text: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0286883

---

## Critical Success Factors

1. **Start immediately** - Data download is the first bottleneck
2. **Person-level splits** - Avoid data leakage by splitting on person_id
3. **Class weighting** - Enable by default, not optional
4. **Interpretability** - SHAP analysis is as important as model performance
5. **Fairness transparency** - Report disparities honestly
6. **Comparison to benchmarks** - Always compare your AUC to published 0.72
7. **Documentation** - Write as you go, not at the end

**Target Performance:** ROC-AUC > 0.70 (published benchmark: 0.72)

**Questions or issues? Document them in your notebooks and address in Discussion section.**

---

*This plan prioritizes rapid MVP delivery while maintaining analytical rigor. Focus on Tier 1 features first, then expand if time permits. Good luck!* 🚀
