# Smoking Cessation Prediction Using Machine Learning

**Project Goal:** Predict smoking cessation success using machine learning on longitudinal PATH Study data.

## Overview

This project uses the Population Assessment of Tobacco and Health (PATH) Study Waves 1-5 to develop a machine learning model that predicts smoking cessation success. The approach employs a person-period dataset design, implements three ML algorithms with class weighting, and provides SHAP-based interpretability with fairness assessment.

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

[Your names and contact information]

## Acknowledgments

- PATH Study Team and ICPSR for data access
- [Course instructor name]
- [Any other acknowledgments]
