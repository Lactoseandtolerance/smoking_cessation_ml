# IMPLEMENTATION ACTION GUIDE
## Smoking Cessation ML Project - Your Step-by-Step Instructions

---

## 🎯 PHASE 1: DATA ACQUISITION & SETUP (Day 1)

### ACTIONS YOU MUST TAKE:

1. **Register for PATH Study Data**
   - Go to: https://www.icpsr.umich.edu/
   - Click "Create Account" and complete registration
   - Verification may take 1-2 business days

2. **Download PATH Study Data**
   - Navigate to: https://www.icpsr.umich.edu/web/NAHDAP/series/606
   - **Data Format:** PATH Study provides data in **STATA (.dta)** or **SPSS (.sav)** format
   - **Which Files to Download:**
     * **ADULT data only** (ages 18+) - this is what you need for smoking cessation
     * Download for **Waves 1-7**: Adult questionnaire data files (2013–2020 panel)
     * ⚠️ **Do NOT download Youth or Parent files** - not relevant for adult smoking cessation
   
   **Specific files needed:**
   - Wave 1: Adult Public Use Files (.dta format)
   - Wave 2: Adult Public Use Files (.dta format)
   - Wave 3: Adult Public Use Files (.dta format)
   - Wave 4: Adult Public Use Files (.dta format)
   - Wave 5: Adult Public Use Files (.dta format)
   - Wave 6: Adult Public Use Files (.dta format)
   - Wave 7: Adult Public Use Files (.dta format)
   
   **Documentation to download:**
   - User Guide (565 pages PDF)
   - Adult questionnaire codebooks for each wave
   - Variable codebooks
   
3. **Place Downloaded Files**
   ```bash
   # Move .dta or .sav files to:
   ~/data mining/smoking_cessation_ml/data/raw/
   
   # Files might be named like:
   # PATH_W1_Adult_Public.dta
   # PATH_W2_Adult_Public.dta
   # ... etc
   # (exact names may vary - keep original names)
   ```

4. **Set Up Python Environment**
   ```bash
   cd ~/data\ mining/smoking_cessation_ml
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Initialize Git Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial project structure"
   # Create GitHub repo and push (optional)
   ```

### ✅ COMPLETION CHECKLIST:
- [ ] PATH Study account created and verified
- [ ] All 7 waves downloaded (STATA .dta format)
- [ ] Documentation downloaded
- [ ] Files placed in `data/raw/`
- [ ] Virtual environment created and activated
- [ ] All dependencies installed without errors
- [ ] Git repository initialized

**STOP HERE until data is downloaded. This is the CRITICAL PATH.**

---

## 🎯 PHASE 2: DEFINE ANALYTICAL SAMPLE (Days 2-3)

### ACTIONS YOU MUST TAKE:

1. **Examine the PATH Study Codebook**
   - Open the User Guide PDF
   - Locate these key variables (names may vary):
     * Current smoking status indicator
     * Quit attempt indicator
     * Cigarettes per day
     * Time to first cigarette
     * NRT usage variables
     * Varenicline/bupropion usage
     * Demographics (age, sex, education, income)
   
2. **Create Data Dictionary**
   - Create file: `data/data_dictionary.md`
   - Document actual variable names from PATH Study
   - Map them to the features needed in the plan

3. **Run Data Exploration Notebook**
   ```bash
   jupyter notebook
   # Create and run: notebooks/01_data_exploration.ipynb
   ```
   
   In the notebook:
   ```python
   import pandas as pd
   import sys
   sys.path.append('../src')
   
   from data_preprocessing import load_wave_data, pool_transitions, calculate_cessation_rates
   
   # Load all waves
   wave1 = load_wave_data(1)
   wave2 = load_wave_data(2)
   wave3 = load_wave_data(3)
   wave4 = load_wave_data(4)
   wave5 = load_wave_data(5)
   
   # Explore structure
   print(wave1.columns)
   print(wave1.head())
   
   # Identify smokers
   # ADJUST variable names based on actual PATH codebook
   wave1_smokers = wave1[wave1['R01_CURR_SMOKER'] == 1]
   print(f"Wave 1 smokers: {len(wave1_smokers)}")
   ```

4. **Create Person-Period Dataset**
   - Modify the `create_transition()` function in `src/data_preprocessing.py` to use ACTUAL PATH variable names
   - Pool transitions across waves
   - Calculate sample size and cessation rates

5. **Save Pooled Dataset**
   ```python
   # In notebook:
   pooled_data.to_csv('../data/processed/pooled_transitions.csv', index=False)
   ```

### ✅ COMPLETION CHECKLIST:
- [ ] Data dictionary created with actual PATH variable names
- [ ] `01_data_exploration.ipynb` completed
- [ ] Identified current smokers in each wave
- [ ] Created person-period pooled dataset
- [ ] Calculated overall quit attempt success rate (7-15% expected)
- [ ] Saved `data/processed/pooled_transitions.csv`
- [ ] Documented sample size: _____ quit attempts from _____ individuals

**CRITICAL CHECKPOINT:** If cessation rate is <5% or >20%, review outcome definition.

---

## 🎯 PHASE 3: FEATURE ENGINEERING (Days 4-5)

### ACTIONS YOU MUST TAKE:

1. **Update Feature Engineering Code**
   - Open `src/feature_engineering.py`
   - Replace placeholder variable names with actual PATH variable names
   - Example:
     ```python
     # Change this:
     df['ttfc_minutes'] = df['time_to_first_cigarette']
     
     # To this (using actual PATH variable):
     df['ttfc_minutes'] = df['R01_TTFC_MINUTES']
     ```

2. **Create Feature Engineering Notebook**
   ```bash
   # Create: notebooks/03_feature_engineering.ipynb
   ```
   
   In the notebook:
   ```python
   import pandas as pd
   import sys
   sys.path.append('../src')
   
   from feature_engineering import engineer_all_features, get_feature_list
   from data_preprocessing import handle_missing_codes
   
   # Load pooled data
   pooled_data = pd.read_csv('../data/processed/pooled_transitions.csv')
   
   # Handle missing codes
   pooled_data = handle_missing_codes(pooled_data)
   
   # Engineer features
   pooled_data = engineer_all_features(pooled_data)
   
   # Get feature list
   feature_cols = get_feature_list()
   
   # Check for missing features
   missing_features = [f for f in feature_cols if f not in pooled_data.columns]
   print(f"Missing features: {missing_features}")
   
   # Save modeling dataset
   pooled_data.to_csv('../data/processed/modeling_data.csv', index=False)
   ```

3. **Handle Missing Data**
   - Calculate missingness percentages
   - Decide on exclusion thresholds
   - Document decisions in notebook

### ✅ COMPLETION CHECKLIST:
- [ ] Updated `src/feature_engineering.py` with actual variable names
- [ ] Created `03_feature_engineering.ipynb`
- [ ] Engineered all Tier 1 features (25-30 features minimum)
- [ ] Handled missing data appropriately
- [ ] Saved `data/processed/modeling_data.csv`
- [ ] Documented final feature count: _____ features

---

## 🎯 PHASE 4: MODELING (Days 6-9)

### ACTIONS YOU MUST TAKE:

1. **Create Baseline Model Notebook**
   ```bash
   # Create: notebooks/04_modeling_baseline.ipynb
   ```
   
   Follow the code from MVP_PLAN.md Phase 4 to:
   - Split data by `person_id` (60/20/20)
   - Train Logistic Regression with `class_weight='balanced'`
   - Evaluate on validation set
   - Save ROC-AUC score

2. **Create Advanced Models Notebook**
   ```bash
   # Create: notebooks/05_modeling_advanced.ipynb
   ```
   
   - Train Random Forest with `class_weight='balanced'`
   - Train XGBoost with `scale_pos_weight`
   - Compare all 3 models
   - Select best model

3. **Final Test Set Evaluation**
   - Evaluate best model on held-out test set
   - Save final model:
     ```python
     import joblib
     joblib.dump(best_model, '../models/final_model.pkl')
     joblib.dump({'model_name': best_model_name,
                  'feature_cols': feature_cols,
                  'test_auc': test_auc}, 
                 '../models/model_metadata.pkl')
     ```

### ✅ COMPLETION CHECKLIST:
- [ ] `04_modeling_baseline.ipynb` completed
- [ ] `05_modeling_advanced.ipynb` completed
- [ ] All 3 models trained with class weighting
- [ ] Model comparison table created
- [ ] Best model selected based on validation AUC
- [ ] Final test set AUC: _____ (target: >0.70)
- [ ] Models saved in `models/` directory

**CRITICAL:** If test AUC < 0.60, review feature engineering and try SMOTE.

---

## 🎯 PHASE 5: SHAP INTERPRETATION (Days 10-11)

### ACTIONS YOU MUST TAKE:

1. **Create Interpretation Notebook**
   ```bash
   # Create: notebooks/06_model_interpretation.ipynb
   ```

2. **Generate SHAP Values**
   Follow the code from MVP_PLAN.md Phase 5:
   ```python
   import shap
   import joblib
   
   # Load best model
   best_model = joblib.load('../models/final_model.pkl')
   metadata = joblib.load('../models/model_metadata.pkl')
   
   # Create explainer
   if metadata['model_name'] in ['Random Forest', 'XGBoost']:
       explainer = shap.TreeExplainer(best_model)
   else:
       explainer = shap.LinearExplainer(best_model, X_train_scaled)
   
   # Calculate SHAP values
   shap_values = explainer(X_test_sample)
   ```

3. **Create Visualizations**
   - Summary bar plot (mean absolute SHAP)
   - Beeswarm plot (distribution of impacts)
   - Dependence plots for top 5 features
   - Waterfall plots for example predictions
   - Save all plots to `reports/figures/`

### ✅ COMPLETION CHECKLIST:
- [ ] `06_model_interpretation.ipynb` completed
- [ ] SHAP values calculated
- [ ] Summary plots created and saved
- [ ] Dependence plots for top 5 features saved
- [ ] Waterfall plots for examples saved
- [ ] Top 10 features documented
- [ ] Written interpretation saved to `reports/shap_interpretation.txt`

---

## 🎯 PHASE 6: FAIRNESS ASSESSMENT (Day 12)

### ACTIONS YOU MUST TAKE:

1. **Create Fairness Notebook**
   ```bash
   # Create: notebooks/07_fairness_assessment.ipynb
   ```

2. **Evaluate Subgroups**
   Follow code from MVP_PLAN.md Phase 6:
   - Define demographic subgroups
   - Calculate AUC for each subgroup
   - Identify disparities (>0.05 difference)
   - Create visualizations

3. **Document Findings**
   - Save fairness results CSV
   - Create plots by demographic group
   - Write fairness summary

### ✅ COMPLETION CHECKLIST:
- [ ] `07_fairness_assessment.ipynb` completed
- [ ] Performance calculated for all demographic subgroups
- [ ] Disparity analysis completed
- [ ] Significant disparities flagged (if any)
- [ ] Visualizations saved to `reports/figures/`
- [ ] `reports/fairness_summary.txt` created

---

## 🎯 PHASE 7: DASHBOARD (Days 13-14)

### ACTIONS YOU MUST TAKE:

1. **Create Dashboard App**
   - Copy dashboard code from MVP_PLAN.md Phase 7
   - Save to: `dashboard/app.py`
   - Update with your actual results

2. **Test Dashboard**
   ```bash
   cd dashboard
   streamlit run app.py
   ```
   
3. **Verify All Pages Work**
   - Overview page displays project info
   - Model Performance shows your results
   - Feature Importance shows SHAP plots
   - Prediction Tool allows input
   - Fairness Assessment shows subgroup analysis
   - Key Insights summarizes findings

### ✅ COMPLETION CHECKLIST:
- [ ] `dashboard/app.py` created with all 6 pages
- [ ] Dashboard runs without errors
- [ ] All visualizations load correctly
- [ ] Prediction tool accepts input and returns predictions
- [ ] Dashboard tested on localhost

---

## 🎯 PHASE 8: REPORT & PRESENTATION (Days 15-16)

### ACTIONS YOU MUST TAKE:

1. **Write IEEE Format Report**
   - Use IEEE conference template: https://www.ieee.org/conferences/publishing/templates.html
   - Follow structure from MVP_PLAN.md Phase 8
   - Minimum 4 pages
   - Include all required sections:
     * Abstract
     * Introduction
     * Methods
     * Results (with tables and figures)
     * Discussion
     * Conclusion
     * References (15-20 citations)

2. **Create Presentation**
   - Create 10-slide PowerPoint/PDF
   - Follow outline from MVP_PLAN.md Phase 8
   - Include:
     * Title slide
     * Motivation
     * Research question & data
     * Methods overview
     * Sample characteristics
     * Model performance
     * Top predictors (SHAP)
     * Example prediction
     * Fairness findings
     * Key takeaways

3. **Write Speaking Notes**
   - Target: 10-minute presentation
   - ~1 minute per slide
   - Save to: `reports/presentation_script.md`

### ✅ COMPLETION CHECKLIST:
- [ ] IEEE format report completed (4+ pages)
- [ ] Report saved as `reports/final_report.pdf`
- [ ] All figures referenced in report
- [ ] All tables formatted properly
- [ ] References section complete (15-20 citations)
- [ ] Presentation created (10 slides)
- [ ] Presentation saved as `reports/presentation.pdf`
- [ ] Speaking notes completed
- [ ] Presentation rehearsed (10 minutes)

---

## 🚨 CRITICAL SUCCESS FACTORS

### You MUST:
1. ✅ Start with Phase 1 immediately - data download is the bottleneck
2. ✅ Split data by `person_id`, NOT by observation (prevent leakage)
3. ✅ Enable class weighting in ALL models (not optional)
4. ✅ Use actual PATH variable names from codebook
5. ✅ Calculate SHAP values (interpretability is crucial)
6. ✅ Report fairness findings honestly (no hiding disparities)
7. ✅ Compare your AUC to published benchmark (0.72)
8. ✅ Document everything in notebooks as you go

### You should AVOID:
1. ❌ Touching test set until final evaluation
2. ❌ Training without class weighting
3. ❌ Splitting by observation instead of person
4. ❌ Skipping SHAP interpretation
5. ❌ Ignoring fairness assessment
6. ❌ Writing documentation at the end

---

## 📊 SUCCESS METRICS

**Minimum Viable Product (MVP) Requirements:**
- ✅ Test Set ROC-AUC > 0.70 (benchmark: 0.72)
- ✅ 25-30 engineered features
- ✅ 3 models compared (LR, RF, XGBoost)
- ✅ SHAP interpretation completed
- ✅ Fairness assessed across 4+ demographic groups
- ✅ Working Streamlit dashboard
- ✅ IEEE format report (4+ pages)
- ✅ 10-slide presentation

---

## 📞 WHEN YOU NEED HELP

If you get stuck:

1. **Data access issues** → Contact ICPSR support
2. **Variable not found** → Check PATH Study codebook carefully
3. **Low model performance (AUC < 0.60)** → Review feature engineering, try SMOTE
4. **Code errors** → Check that all dependencies are installed
5. **High class imbalance** → Verify class weighting is enabled
6. **Test/train leakage** → Verify splitting by person_id

---

## 🎯 TIMELINE SUMMARY

| Days | Phase | Key Deliverable |
|------|-------|-----------------|
| 1 | Setup | Environment ready, data downloaded |
| 2-3 | Sample | `pooled_transitions.csv` |
| 4-5 | Features | `modeling_data.csv` with 25-30 features |
| 6-9 | Modeling | Best model with AUC > 0.70 |
| 10-11 | SHAP | Feature importance visualizations |
| 12 | Fairness | Subgroup performance analysis |
| 13-14 | Dashboard | Working Streamlit app |
| 15-16 | Report | IEEE report + presentation |

**Total: 16 days from start to complete deliverables**

---

## 🚀 GET STARTED NOW

Your immediate next steps:
1. Register at ICPSR (do this TODAY)
2. While waiting for verification, review PATH Study documentation online
3. Set up your Python environment
4. Familiarize yourself with the MVP_PLAN.md

**Good luck! You have all the tools you need to succeed. Follow the plan step by step.**
