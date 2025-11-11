# Quick Start: What to Do Next

## You Just Completed Phase 5 ✅

Your smoking cessation prediction model is trained and working! Here's what you accomplished:

- ✅ Loaded 47,882 real transitions from PATH Study
- ✅ Trained 3 models (Logistic Regression, Random Forest, XGBoost)  
- ✅ XGBoost achieved **0.830 ROC-AUC** and **0.732 F1-score**
- ✅ Identified top predictors: race_other (69%), high_income (14%), ttfc_minutes (9%)
- ✅ Saved best model to `models/xgboost_best.pkl`

## What to Do Now (Choose One)

### 🎯 Option A: Test Set Evaluation (30 minutes) ← START HERE

**Why**: Get final unbiased performance metrics on data the model has never seen.

**How**:
```python
# Open notebooks/04_modeling.ipynb
# Add a new cell at the end and run:

from src.evaluation import evaluate_model

print("=== FINAL TEST SET EVALUATION ===\n")
test_metrics = evaluate_model(xgb_model, X_test, y_test)

print("\nTest Set Results:")
print(f"ROC-AUC: {test_metrics['roc_auc']:.3f}")
print(f"F1-Score: {test_metrics['f1']:.3f}")
print(f"Precision: {test_metrics['precision']:.3f}")
print(f"Recall: {test_metrics['recall']:.3f}")
```

**Expected outcome**: Similar to validation results (0.82-0.84 ROC-AUC)

---

### 🔍 Option B: SHAP Interpretation (2 hours)

**Why**: Understand WHY the model makes predictions for individual smokers.

**How**:
```python
# Install SHAP if needed
!pip install shap

# In notebook:
import shap

# Create explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test[:100])  # First 100 test cases

# Summary plot
shap.summary_plot(shap_values, X_test[:100])

# Individual prediction explanation
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=X_test.columns.tolist()
))
```

**Expected outcome**: Visualizations showing how features contribute to predictions

---

### 🎛️ Option C: Hyperparameter Tuning (3-4 hours)

**Why**: Potentially improve XGBoost performance beyond 0.830 ROC-AUC.

**How**:
```python
from sklearn.model_selection import RandomizedSearchCV

# Define parameter space
param_dist = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Run search
search = RandomizedSearchCV(
    xgb.XGBClassifier(scale_pos_weight=2.34, random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    n_jobs=-1
)

search.fit(X_train, y_train)
print(f"Best ROC-AUC: {search.best_score_:.3f}")
print(f"Best params: {search.best_params_}")
```

**Expected outcome**: May gain 1-3% improvement in ROC-AUC

---

### ⚖️ Option D: Fairness Analysis (1-2 hours)

**Why**: Check if model performs differently for different demographic groups.

**How**:
```python
# Evaluate by race
for race in df['race_ethnicity'].unique():
    mask = df[df.index.isin(X_test.index)]['race_ethnicity'] == race
    X_subset = X_test[mask]
    y_subset = y_test[mask]
    
    if len(X_subset) > 50:  # Enough samples
        metrics = evaluate_model(xgb_model, X_subset, y_subset)
        print(f"\n{race}: ROC-AUC = {metrics['roc_auc']:.3f}")

# Similar analysis for gender, income, age groups
```

**Expected outcome**: Performance metrics broken down by demographic subgroups

---

### 📊 Option E: Dashboard Development (4-6 hours)

**Why**: Create interactive visualization for stakeholders.

**Steps**:
1. Create `dashboard/app.py` with Streamlit
2. Add model performance visualizations
3. Add prediction interface (input features → quit probability)
4. Run locally: `streamlit run dashboard/app.py`

**Expected outcome**: Interactive web app for model exploration

---

## Quick Commands

### View results:
```bash
cat reports/PHASE5_RESULTS.md          # Detailed results
cat PHASE5_SUMMARY.md                   # Quick summary
jupyter notebook notebooks/04_modeling.ipynb  # Executed notebook
```

### Check files:
```bash
ls -lh models/                          # See saved models
wc -l data/processed/pooled_transitions.csv  # 47,883 lines
```

### Next notebook:
```bash
jupyter notebook notebooks/04_modeling.ipynb
# Add cells at the end for Option A, B, C, or D above
```

---

## Need Help?

- **Stuck?** Check `reports/PHASE5_RESULTS.md` for full details
- **Want overview?** Read `PHASE5_SUMMARY.md`
- **Need code?** See `notebooks/04_modeling.ipynb` (all cells executed)
- **Want structure?** Check `PROJECT_STATUS.md` (updated with Phase 5)

---

## My Recommendation

**Start with Option A (Test Set Evaluation)** - it's quick (30 min) and gives you confidence that your model generalizes to new data. Then move to Option B (SHAP) if you want to understand predictions, or Option E (Dashboard) if you want to show results to others.

Good luck! 🎉
