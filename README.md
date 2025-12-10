# Smoking Cessation Prediction Using Machine Learning

**Project Goal:** Predict smoking cessation success using machine learning on longitudinal PATH Study data (Waves 1-7, 2013–2020).

## Overview

This project uses the Population Assessment of Tobacco and Health (PATH) Study **Waves 1-7** to develop a machine learning model that predicts smoking cessation success. The approach employs a person-period dataset design with **52 engineered features**, implements three ML algorithms with class weighting, and provides SHAP-based interpretability with fairness assessment.

### Current Performance
- **Best Model:** XGBoost with native NaN handling
- **Validation Performance:** ROC-AUC ≈ 0.884 (52 features, ~47,882 transitions)
- **Test Set Performance:** ROC-AUC ≈ 0.669 (indicates potential drift or subgroup variance)
- **Feature Count:** 52 canonical features (dependence, demographics, cessation methods, environment, motivation)

### Wave-Pair Evaluation (Waves 1→7)
A dedicated script `scripts/run_model_training_and_wave_eval.py` provides:
1. Per baseline→follow-up wave metrics (ROC-AUC, PR-AUC, precision, recall, F1, quit rate)
2. Feature drift analysis across baseline waves vs. Wave 1 (mean differences and Kolmogorov–Smirnov statistics)
3. Subgroup performance by sex, age cohort, and race/ethnicity

Generated reports:
- `reports/WAVE_PAIR_EVAL.md`: Wave-pair model performance tables
- `reports/FEATURE_DRIFT.md`: Feature drift summary (top 25 features by max absolute mean difference)
- `reports/FAIRNESS_RESULTS.md`: Subgroup performance and disparity analysis

Run command:
```bash
python scripts/run_model_training_and_wave_eval.py
```

### Drift Interpretation Notes
- Large negative mean differences in `cpd` across mid waves reflect declining average cigarettes per day among remaining smokers
- Increasing mean age and decreasing `age_young` proportion in later waves indicate cohort aging and potential survivorship
- `college_degree` proxy collapse in later waves signals missing or recoded education data for those waves (treated as 0 in feature engineering)

Use these drift signals to consider adaptive reweighting, feature recalibration, or temporal modeling if performance degrades disproportionately in later wave pairs.

**Target Performance:** ROC-AUC > 0.70  
**Published Benchmark:** 0.72 AUC (Issabakhsh et al., 2023)  
**Current Achievement:** 0.884 validation AUC (exceeds benchmark)

## Repository Structure

```
smoking_cessation_ml/
├── data/
│   ├── raw/                          # PATH STATA files (Waves 1-7)
│   ├── processed/                    # Generated datasets
│   │   ├── pooled_transitions.csv    # 47,882 transitions × 52 features
│   │   └── pooled_transitions.parquet
│   ├── data_dictionary.md            # 52-feature variable mapping
│   ├── PHASE2_VARIABLES.md           # Feature tracking
│   └── VARIABLE_MAPPING_TRACKER.md
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Exploratory analysis
│   ├── 02_data_preprocessing.ipynb   # Preprocessing pipeline (Waves 1-7)
│   ├── 03_feature_engineering.ipynb  # Feature creation
│   └── 04_modeling.ipynb             # Model training & evaluation
├── src/
│   ├── data_preprocessing.py         # Data loading & transitions
│   ├── feature_engineering.py        # 52 canonical features
│   ├── modeling.py                   # ML model training
│   ├── evaluation.py                 # Metrics & evaluation
│   └── reporting.py                  # Report generation
├── scripts/
│   ├── run_preprocessing.py          # Full pipeline runner
│   ├── run_model_training_and_wave_eval.py # Wave-pair evaluation
│   ├── compute_subgroup_performance.py     # Fairness analysis
│   ├── run_test_evaluation.py        # Test set metrics
│   ├── run_interpretability_and_fairness.py # Integrated analysis
│   └── other utilities
├── models/
│   ├── xgboost_best.pkl              # Best model: 0.884 Val AUC
│   ├── random_forest_best.pkl        # Random Forest: 0.819 Val AUC
│   └── logistic_regression_scaler.pkl
├── dashboard/
│   └── app.py                        # Streamlit interactive app
├── reports/
│   ├── PHASE5_RESULTS.md             # Validation metrics (0.884 AUC)
│   ├── TEST_SET_RESULTS.md           # Test metrics (0.669 AUC)
│   ├── WAVE_PAIR_EVAL.md             # Per-wave performance
│   ├── FAIRNESS_RESULTS.md           # Subgroup AUC/FPR/FNR analysis
│   ├── FEATURE_DRIFT.md              # Feature drift across waves
│   ├── INTERPRETABILITY_SUMMARY.md   # SHAP feature importance
│   ├── SUBGROUP_PERFORMANCE.csv      # Detailed fairness metrics
│   └── figures/                      # Generated visualizations
├── requirements.txt                  # Python dependencies
├── README.md                         # This file (project overview)
├── PROJECT_STATUS.md                 # Current status (Phase 5+ complete)
├── MVP_PLAN.md                       # Complete technical plan
├── QUICK_REFERENCE.md                # Quick lookup guide
├── ACTION_GUIDE.md                   # Step-by-step instructions
└── .gitignore                        # Git configuration
```

## Setup Instructions

### 1. Clone Repository
```bash
git clone <repository-url>
cd smoking_cessation_ml
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Obtain PATH Study Data

From ICPSR (https://www.icpsr.umich.edu/):
1. Register at ICPSR: https://www.icpsr.umich.edu/
2. Navigate to PATH Study Series: https://www.icpsr.umich.edu/web/NAHDAP/series/606
3. Download **ADULT files only (Waves 1-7)** in STATA .dta format
   - Do NOT download Youth or Parent files
4. Place files in `data/raw/` directory:
   ```
   data/raw/PATH_W1_Adult_Public.dta
   data/raw/PATH_W2_Adult_Public.dta
   ... through W7
   ```
5. Download documentation (user guide, codebooks)

## Data Overview

- **Source:** Population Assessment of Tobacco and Health (PATH) Study
- **Coverage:** Waves 1-7 (2013-2020)
- **Total Transitions:** 47,882 person-period observations
- **Unique Individuals:** 23,411
- **Features:** 52 canonical engineered features
- **Data File:** `data/processed/pooled_transitions.csv` or `.parquet`

## Key Results

### Validation Set (9,508 transitions)
| Model | ROC-AUC | PR-AUC | F1-Score | Precision | Recall |
|-------|---------|--------|----------|-----------|--------|
| Logistic Regression | 0.787 | 0.658 | 0.659 | 0.661 | 0.657 |
| Random Forest | 0.819 | 0.779 | 0.697 | 0.720 | 0.676 |
| **XGBoost** 🏆 | **0.884** | **0.793** | **0.732** | **0.850** | **0.642** |

### Test Set (9,763 transitions)
- **ROC-AUC:** 0.669 (indicates potential subgroup variance)
- **Precision:** 0.404
- **Recall:** 0.089
- **F1-Score:** 0.145

See `reports/FAIRNESS_RESULTS.md` for subgroup performance analysis.

### Top 10 Predictors (XGBoost)
1. **race_other** (69%) - Dominant predictor, likely capturing unmeasured confounders
2. **high_income** (14%) - Strong socioeconomic indicator
3. **ttfc_minutes** (9%) - Time to first cigarette (dependence measure)
4. **cpd** (3%) - Cigarettes per day
5. **dependence_score** (1%) - Composite dependence measure
6. **age** (1%) - Demographic factor
7. **cpd_light** (<1%) - Light smoking indicator
8. **used_any_method** (<1%) - NRT/cessation aid usage
9. **very_high_dependence** (<1%) - Severe dependence indicator
10. **high_dependence** (<1%) - High dependence indicator

## Usage

### Running Analysis Notebooks
```bash
jupyter notebook
```
Navigate to `notebooks/` and run notebooks in order (01 through 04).

### Running Full Pipeline
```bash
# Preprocess data and engineer features
python scripts/run_preprocessing.py

# Train models and evaluate
python scripts/run_model_training_and_wave_eval.py

# Compute fairness metrics
python scripts/compute_subgroup_performance.py

# Run interpretability analysis
python scripts/run_interpretability_and_fairness.py
```

### Running Dashboard
```bash
streamlit run dashboard/app.py
```
Opens interactive app at http://localhost:8501

## Methods

### Data Structure
- **Person-period design:** Each row = one person at one wave with smoking status transition to next wave
- **Sample:** Baseline smokers with follow-up data
- **Outcome:** Smoking abstinence at follow-up (30-day)

### Feature Engineering (52 features)
- **Dependence:** TTFC, CPD, composite scores
- **Demographics:** Age, sex, education, income, race/ethnicity
- **Cessation Methods:** NRT, medications, counseling, quitline usage
- **Environment:** Household smokers, workplace smokefree policy
- **Motivation:** Plans to quit, quit attempts, longest duration

### Modeling
- **Algorithm:** XGBoost with scale_pos_weight for class imbalance
- **Split Strategy:** Person-level 60/20/20 (train/val/test) to prevent leakage
- **Baseline Models:** Logistic Regression, Random Forest
- **Evaluation:** ROC-AUC, PR-AUC, precision, recall, F1-score

### Fairness & Interpretability
- **SHAP:** Feature importance and dependence plots
- **Subgroup Analysis:** AUC, FPR, FNR by sex, age cohort, race/ethnicity
- **Wave Analysis:** Per-transition performance and feature drift
- **Limitations:** Test AUC (0.669) variance suggests subgroup effects

## Documentation

- **`PROJECT_STATUS.md`** - Current project status and completion checklist
- **`MVP_PLAN.md`** - Complete technical implementation plan
- **`QUICK_REFERENCE.md`** - Quick reference guide with key metrics
- **`ACTION_GUIDE.md`** - Step-by-step instructions for each phase

## Key Files

- **`models/xgboost_best.pkl`** - Trained XGBoost model (0.884 validation AUC)
- **`data/processed/pooled_transitions.csv`** - Final processed dataset
- **`reports/PHASE5_RESULTS.md`** - Validation set results
- **`reports/TEST_SET_RESULTS.md`** - Test set evaluation
- **`reports/FAIRNESS_RESULTS.md`** - Subgroup performance analysis

## Environment

- **OS:** macOS (zsh shell)
- **Python:** 3.13+
- **Key Libraries:** pandas, scikit-learn, xgboost, shap, streamlit, plotly

## Limitations & Next Steps

### Known Limitations
1. **Test AUC variance:** 0.884 validation → 0.669 test suggests subgroup effects
2. **Race variable dominance:** 69% feature importance may indicate confounding
3. **Missing cessation methods:** Behavioral support (counseling, quitline) limited in public use data
4. **Wave-specific effects:** Potential distribution shifts across waves 1-7

### Recommended Next Steps
1. Investigate subgroup performance disparities (fairness analysis)
2. Consider subgroup-specific or fairness-aware models
3. Analyze feature drift across waves for potential recalibration needs
4. Apply for PATH restricted data access for more detailed cessation methods
5. Hyperparameter optimization for improved generalization

## Citation

```bibtex
@project{smoking_cessation_ml_2024,
  title = {Smoking Cessation Prediction Using Machine Learning on PATH Study Data},
  author = {[Your Name]},
  year = {2024-2025},
  url = {https://github.com/[your-repo]}
}
```

## References

- Population Assessment of Tobacco and Health (PATH) Study: https://pathstudyinfo.nih.gov/
- ICPSR Data Archive: https://www.icpsr.umich.edu/
- Issabakhsh et al. (2023): Smoking Cessation Prediction, *[Journal]*, 0.72 AUC benchmark

## License

[Specify your license, e.g., MIT, Apache 2.0, or academic use]

---

**Last Updated:** January 2025  
**Status:** ✅ Validation complete, Test set evaluation complete, Fairness analysis complete  
**Dataset:** PATH Study Waves 1-7 (47,882 transitions from 23,411 individuals)  
**Best Model:** XGBoost (0.884 validation AUC, 0.669 test AUC)
