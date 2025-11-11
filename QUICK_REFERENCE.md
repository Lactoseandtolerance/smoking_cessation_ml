# Quick Reference Guide
## Essential Information at a Glance

---

## 🎯 Project Goal
**Predict smoking cessation success using machine learning on PATH Study data**

**Target Performance:** ROC-AUC > 0.70 (Benchmark: 0.72)  
**Timeline:** 16 days from data acquisition to deliverables

---

## 📁 Project Structure

```
smoking_cessation_ml/
├── data/
│   ├── raw/                          # PATH CSV files (YOU MUST DOWNLOAD)
│   ├── processed/                    # Generated datasets
│   │   ├── pooled_transitions.csv   # Phase 2 output
│   │   └── modeling_data.csv        # Phase 3 output
│   └── data_dictionary.md           # Variable mapping
├── notebooks/                        # Jupyter notebooks (numbered 01-07)
├── src/                             # Python modules (already created)
├── models/                          # Saved models
├── dashboard/app.py                 # Streamlit dashboard
├── reports/
│   ├── figures/                     # All plots and visualizations
│   ├── final_report.pdf            # IEEE format report
│   └── presentation.pdf            # 10-slide presentation
├── ACTION_GUIDE.md                  # Detailed instructions (READ THIS)
├── MVP_PLAN.md                      # Complete technical plan
├── README.md                        # Project overview
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

### ✅ Phase 1: Setup (Day 1)
- [ ] Register at ICPSR
- [ ] Download PATH Waves 1-5 (STATA .dta or SPSS .sav format)
- [ ] Download ADULT files only (NOT Youth or Parent files)
- [ ] Download documentation
- [ ] Install dependencies (including pyreadstat)
- [ ] Initialize Git repo

### ⬜ Phase 2: Sample (Days 2-3)
- [ ] Create data dictionary with actual variable names
- [ ] Load all 5 waves
- [ ] Create person-period dataset
- [ ] Calculate cessation rates
- [ ] Save `pooled_transitions.csv`

### ⬜ Phase 3: Features (Days 4-5)
- [ ] Update feature engineering code with PATH variables
- [ ] Engineer 25-30 features
- [ ] Handle missing data
- [ ] Save `modeling_data.csv`

### ⬜ Phase 4: Modeling (Days 6-9)
- [ ] Split by person_id (60/20/20)
- [ ] Train Logistic Regression with class_weight='balanced'
- [ ] Train Random Forest with class_weight='balanced'
- [ ] Train XGBoost with scale_pos_weight
- [ ] Select best model
- [ ] Evaluate on test set (AUC > 0.70)
- [ ] Save models

### ⬜ Phase 5: SHAP (Days 10-11)
- [ ] Generate SHAP values
- [ ] Create summary plots
- [ ] Create dependence plots
- [ ] Create waterfall plots
- [ ] Document top features

### ⬜ Phase 6: Fairness (Day 12)
- [ ] Evaluate performance by demographic groups
- [ ] Calculate disparities
- [ ] Create visualizations
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
