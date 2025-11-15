# Smoking Cessation Prediction Using Machine Learning

**Project Goal:** Predict smoking cessation success using machine learning on longitudinal PATH Study data.

## Overview

This project uses the Population Assessment of Tobacco and Health (PATH) Study Waves 1-5 to develop a machine learning model that predicts smoking cessation success. The approach employs a person-period dataset design, implements three ML algorithms with class weighting, and provides SHAP-based interpretability with fairness assessment.

### Wave-Pair Evaluation (Expanded Waves 1→7)
After extending preprocessing to include transitions up to Wave 7, a dedicated script `scripts/run_model_training_and_wave_eval.py` now:
1. Retrains all three models on the full pooled transitions dataset.
2. Saves the best validation performer (currently XGBoost) to `models/xgboost_best.pkl`.
3. Produces per baseline→follow-up wave metrics (ROC-AUC, PR-AUC, precision, recall, F1, quit rate) for validation and test partitions.
4. Computes feature drift across baseline waves versus Wave 1 using mean differences and Kolmogorov–Smirnov statistics (SciPy) for the top drifted features.

Generated reports:
- `reports/WAVE_PAIR_EVAL.md`: Wave-pair model performance tables.
- `reports/FEATURE_DRIFT.md`: Feature drift summary (top 25 features by max absolute mean difference).

Run command:
```bash
python scripts/run_model_training_and_wave_eval.py
```

### Drift Interpretation Notes
- Large negative mean differences in `cpd` across mid waves reflect declining average cigarettes per day among remaining smokers.
- Increasing mean age and decreasing `age_young` proportion in later waves indicate cohort aging and potential survivorship.
- `college_degree` proxy collapse in later waves signals missing or recoded education data for those waves (treated as 0 in feature engineering).

Use these drift signals to consider adaptive reweighting, feature recalibration, or temporal modeling if performance degrades disproportionately in later wave pairs.
**Target Performance:** ROC-AUC > 0.70  
**Published Benchmark:** 0.72 AUC (Issabakhsh et al., 2023)

## Project Structure

```
smoking_cessation_ml/
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
1. Register at ICPSR: https://www.icpsr.umich.edu/
2. Navigate to PATH Study Series: https://www.icpsr.umich.edu/web/NAHDAP/series/606
3. Download Public Use Files for Waves 1-5 (STATA .dta or SPSS .sav format)
   - **Download ADULT files only** (ages 18+)
   - **Do NOT download Youth or Parent files** (not relevant for adult smoking cessation)
4. Download the user guide and codebooks
5. Place data files in `data/raw/` directory

**Note:** PATH Study provides data in STATA (.dta) or SPSS (.sav) format, NOT CSV. Pandas can read both formats natively (STATA) or with pyreadstat (SPSS).

## Usage

### Running Analysis Notebooks
```bash
jupyter notebook
```
Navigate to `notebooks/` and run notebooks in order (01 through 07).

### Running Dashboard
```bash
streamlit run dashboard/app.py
```

## Methods

### Data Structure
- **Dataset:** PATH Study Waves 1-5
- **Design:** Pooled person-period dataset across wave transitions (1→2, 2→3, 3→4, 4→5)
- **Outcome:** 30-day smoking abstinence after quit attempt
- **Features:** 25-35 engineered features covering nicotine dependence, demographics, cessation methods, quit history
   - Phase 3 adds dynamic codebook-driven mapping (see Feature Engineering Workflow below)

### Models
- **Logistic Regression** (interpretable baseline)
- **Random Forest** (ensemble method)
- **XGBoost** (gradient boosting)

All models use class weighting to handle imbalanced outcomes.

### Evaluation
- **Primary metric:** ROC-AUC
- **Secondary metrics:** Precision, Recall, F1-Score, PR-AUC
- **Fairness assessment:** Performance across demographic subgroups (gender, age, race, education, income)

### Interpretability
- SHAP values for global and local feature importance
- Feature interaction analysis
- Individual prediction explanations

## Results

*To be updated after analysis completion*

- **Best Model:** [TBD]
- **Test Set AUC:** [TBD]
- **Top Predictors:** [TBD]
- **Fairness Findings:** [TBD]

## Key Resources

- **PATH Study:** https://www.icpsr.umich.edu/web/NAHDAP/series/606
- **Published Benchmark:** Issabakhsh et al. (2023) - https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0286883
- **SHAP Documentation:** https://shap.readthedocs.io/

## Citation

If you use this code or approach, please cite:

```
[Your names]. (2025). Smoking Cessation Prediction Using Machine Learning: 
A Multi-Model Approach with PATH Study Data. [Course/Institution].
```

## License

[Specify license - note that PATH Study data has its own usage terms]

## Authors

Angel Nivar, Ananda Downing

    'age': 'R01R_A_AGE',          # Example PATH variable names
    'sex': 'R01R_A_SEX',
    'education_code': 'R01R_A_EDUC',
    'income': 'R01R_A_INCOME',
    'cpd': 'R01R_A_PERDAY_P30D_CIGS',
    'ttfc_minutes': 'R01R_A_MINFIRST_CIGS',
    'race': 'R01R_A_RACECAT',     # Replace with correct codebook variable
    'hispanic': 'R01R_A_HISP',
    'nrt_any': 'R01R_A_PST12M_LSTQUIT_ECIG_NRT',    # Placeholder examples
    'varenicline': 'R01R_A_PST12M_LSTQUIT_ECIG_RX', # Confirm variable names
    # Add bupropion, counseling, quitline when identified
}
df_feats = engineer_all_features(raw_df, codebook_overrides=codebook_overrides)
```

### Race/Ethnicity Handling
Currently derives a simplified 4-level scheme: `White`, `Black`, `Hispanic`, `Other` with dummies:
`race_white`, `race_black`, `race_hispanic`, `race_other`. Update `_normalize_race_ethnicity` once detailed PATH race coding is confirmed.

### Adding New Variables
Extend `VARIABLE_CANDIDATES` with new raw names; feature engineering will pick them up automatically when present.

### Testing
See `tests/test_feature_engineering.py` (added in Phase 3) for a synthetic DataFrame validation of mapping + derived features.

### Next Refinements
- Confirm and finalize actual PATH variable names for cessation methods and race/ethnicity.
- Document derivation logic for education levels and income ordinal mapping.
- Add SHAP-friendly feature grouping metadata.


- PATH Study Team and ICPSR for data access
- [Course instructor name]
- [Any other acknowledgments]
