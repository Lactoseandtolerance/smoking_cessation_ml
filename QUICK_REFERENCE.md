# Quick Reference Guide
## Essential Information at a Glance

---

## 🎯 Project Goal
**Predict smoking cessation success using machine learning on PATH Study data (Waves 1-7)**

**Target Performance:** ROC-AUC > 0.70 (Benchmark: 0.72)  
**Achieved Performance:** Validation ROC-AUC 0.884 | Test ROC-AUC 0.669  
**Feature Count:** 52 canonical features  
**Timeline:** Completed with comprehensive evaluation

---

## 📁 Project Structure

```
smoking_cessation_ml/
├── data/
│   ├── raw/                          # PATH STATA files (Waves 1-7)
│   ├── processed/                    # Generated datasets
│   │   ├── pooled_transitions.csv   # 47,882 transitions × 52 features
│   │   └── pooled_transitions.parquet
│   └── data_dictionary.md           # 52-feature variable mapping
├── notebooks/                        # Jupyter notebooks (numbered 01-07)
├── src/                             # Python modules (feature engineering, modeling, evaluation)
├── models/                          # Saved models (XGBoost: 0.884 Val AUC)
├── dashboard/app.py                 # Streamlit dashboard
├── reports/
│   ├── PHASE5_RESULTS.md            # Validation metrics (0.884 AUC)
│   ├── TEST_SET_RESULTS.md          # Test metrics (0.669 AUC)
│   ├── WAVE_PAIR_EVAL.md            # Per-wave performance
│   ├── FAIRNESS_RESULTS.md          # Subgroup AUC/FPR/FNR analysis
│   ├── FEATURE_DRIFT.md             # Feature drift across waves
│   ├── INTERPRETABILITY_SUMMARY.md  # SHAP feature importance
│   ├── figures/                     # Generated charts
│   └── SUBGROUP_PERFORMANCE.csv     # Detailed fairness metrics
├── ACTION_GUIDE.md                  # Detailed instructions (Waves 1-7)
├── MVP_PLAN.md                      # Complete technical plan (Waves 1-7)
├── README.md                        # Project overview (52 features, current metrics)
└── requirements.txt                 # Python dependencies
```

---

## 🚀 Quick Start

```bash
# 1. Navigate to project
cd ~/data\ mining/smoking_cessation_ml

# 2. Activate environment
source venv/bin/activate

# 3. Start Jupyter
jupyter notebook

# 4. Run dashboard (after Phase 7)
streamlit run dashboard/app.py
```

---

## 📊 Phase Checklist

### ✅ Phase 1: Setup (Day 1) - COMPLETE
- [x] Register at ICPSR
- [x] Download PATH Waves 1-7 (STATA .dta format)
- [x] Download ADULT files only (NOT Youth or Parent files)
- [x] Download documentation
- [x] Install dependencies (including pyreadstat)
- [x] Initialize Git repo

### ✅ Phase 2: Sample (Days 2-3) - COMPLETE
- [x] Create data dictionary with actual variable names
- [x] Load all 7 waves
- [x] Create person-period dataset (47,882 transitions)
- [x] Calculate cessation rates by wave pair
- [x] Save `pooled_transitions.csv`

### ✅ Phase 3: Features (Days 4-5) - COMPLETE
- [x] Update feature engineering code with PATH variables
- [x] Engineer 52 canonical features (dependence, demographics, methods, environment, motivation)
- [x] Handle missing data with codebook overrides
- [x] Save `pooled_transitions.csv` with features
- [x] Feature count: 52 (exceeds MVP goal)

### ✅ Phase 4: Modeling (Days 6-9) - COMPLETE
- [x] Split by person_id (60/20/20) - no data leakage
- [x] Train Logistic Regression (Val AUC 0.787)
- [x] Train Random Forest (Val AUC 0.819)
- [x] Train XGBoost (Val AUC 0.884) ✨ Best performer
- [x] Evaluate on test set (Test AUC 0.669)
- [x] Save best model to `models/xgboost_best.pkl`

### ✅ Phase 5: SHAP & Interpretability (Days 10-11) - COMPLETE
- [x] Generate SHAP values for top 10-20 features
- [x] Create SHAP summary plot
- [x] Create SHAP dependence plots
- [x] Create SHAP waterfall plots
- [x] Document top features in `reports/INTERPRETABILITY_SUMMARY.md`

### ✅ Phase 6: Fairness Analysis (Day 12) - COMPLETE
- [x] Evaluate performance by demographic groups (sex, age cohort, race/ethnicity)
- [x] Calculate AUC, FPR, FNR disparities
- [x] Create fairness visualizations (heatmaps, bar charts)
- [x] Save results to `reports/FAIRNESS_RESULTS.md`
- [x] Note: Test AUC variance (0.669) suggests potential subgroup performance differences

### ✅ Phase 7: Wave-Pair Evaluation (Extended) - COMPLETE
- [x] Compute per-wave pair metrics (W1→W2, W2→W3, ..., W6→W7)
- [x] Feature drift analysis (mean differences and KS statistics)
- [x] Generate `reports/WAVE_PAIR_EVAL.md`
- [x] Generate `reports/FEATURE_DRIFT.md`
- [x] Dashboard ready at `dashboard/app.py`
- [ ] Document findings

### ⬜ Phase 7: Dashboard (Days 13-14)
- [ ] Create Streamlit app (6 pages)
- [ ] Test all pages
- [ ] Verify visualizations load

### ⬜ Phase 8: Report (Days 15-16)
- [ ] Write IEEE format report (4+ pages)
- [ ] Create 10-slide presentation
- [ ] Write speaking notes
- [ ] Rehearse presentation

---

## 🔑 Critical Code Snippets

### Load Data
```python
import pandas as pd
import sys
sys.path.append('../src')
from data_preprocessing import load_wave_data

# Will automatically handle STATA (.dta) or SPSS (.sav) format
wave1 = load_wave_data(1, '../data/raw', file_format='dta')

# Or directly with pandas:
wave1 = pd.read_stata('../data/raw/PATH_W1_Adult.dta')
```

### Train Model with Class Weighting
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    class_weight='balanced',  # CRITICAL!
    max_iter=1000,
    random_state=42
)
lr.fit(X_train, y_train)
```

### Split by Person ID (Prevent Leakage)
```python
from sklearn.model_selection import train_test_split

unique_persons = pooled_data['person_id'].unique()
train_ids, temp_ids = train_test_split(unique_persons, test_size=0.4, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

train_data = pooled_data[pooled_data['person_id'].isin(train_ids)]
```

### Generate SHAP Values
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test_sample)
shap.summary_plot(shap_values, X_test_sample)
```

---

## 📚 Key Resources

| Resource | URL |
|----------|-----|
| PATH Study Data | https://www.icpsr.umich.edu/web/NAHDAP/series/606 |
| Published Benchmark | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0286883 |
| SHAP Documentation | https://shap.readthedocs.io/ |
| IEEE Template | https://www.ieee.org/conferences/publishing/templates.html |
| Streamlit Docs | https://docs.streamlit.io/ |

---

## ⚠️ Common Pitfalls to Avoid

1. **Data Leakage** → Always split by `person_id`, not by observation
2. **Forgetting Class Weighting** → Enable in ALL models (not optional)
3. **Wrong Variable Names** → Use actual PATH variable names from codebook
4. **Touching Test Set Early** → Only use test set for final evaluation
5. **Skipping SHAP** → Interpretability is required, not optional
6. **Ignoring Fairness** → Must assess across demographic groups
7. **Writing at the End** → Document as you go in notebooks

---

## 🎯 Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Test Set AUC | > 0.70 | Benchmark: 0.72 (Issabakhsh 2023) |
| Features | 25-30 | Tier 1 features minimum |
| Models | 3 | Logistic Regression, Random Forest, XGBoost |
| Notebooks | 7 | All phases documented |
| Report | 4+ pages | IEEE format |
| Presentation | 10 slides | 10-minute talk |

---

## 📞 Getting Help

**If stuck, check:**
1. ACTION_GUIDE.md for detailed instructions
2. MVP_PLAN.md for complete technical details
3. PATH Study codebook for variable definitions
4. src/ modules for code examples

**Common issues:**
- "Variable not found" → Check data_dictionary.md
- "Low AUC (<0.60)" → Review feature engineering, enable class weighting
- "Import errors" → Run `pip install -r requirements.txt`

---

## 🏁 Next Steps

1. **NOW:** Register at ICPSR (if not done)
2. **Day 1:** Download PATH data and documentation
3. **Day 2:** Start Phase 2 (analytical sample)
4. **Follow** ACTION_GUIDE.md step by step

**You have everything you need. Start Phase 1 today!**
