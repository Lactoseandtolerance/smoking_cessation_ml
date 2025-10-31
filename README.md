Predicting Smoking Cessation Success Using Machine Learning
A Multi-Model Approach with Longitudinal Survey Data

Authors: Ananda Downing, Angel Nivar
Course: Data Mining
Timeline: 8 weeks (October - December 2025)

Project Overview
Smoking remains a leading cause of preventable death worldwide. This project develops machine learning models to predict smoking cessation success one year after a quit attempt, using the TUS-CPS 2010-2011 Longitudinal Cohort – a nationally representative sample where baseline smokers were re-interviewed after one year to assess cessation outcomes.
Research Question
What individual characteristics, behavioral patterns, and quit strategies best predict successful smoking cessation one year after a quit attempt?
Key Contributions

Novel application of SHAP values for interpretable cessation prediction using longitudinal data
Evaluation of quit method combinations (e.g., NRT + counseling) rather than single interventions
Fairness assessment across demographic subgroups to ensure equitable public health application
Interactive dashboard for exploring model predictions and explanations


Data Source
Dataset: Tobacco Use Supplement to the Current Population Survey (TUS-CPS) 2010-2011 Longitudinal Cohort

Design: True longitudinal cohort (baseline May 2010 → follow-up May 2011)
Sample Size: Approximately 2,400-2,500 baseline smokers with complete follow-up data
Target Variable: 30+ days smoking abstinence at follow-up
Features: Demographics, smoking history, nicotine dependence, cessation methods
Data Access: Publicly available from NCI TUS-CPS

Citation:
US Department of Commerce, Census Bureau. National Cancer Institute and Food and Drug 
Administration co-sponsored Tobacco Use Supplement to the Current Population Survey 
(TUS-CPS), 2010-2011 Longitudinal Cohort.

Technical Stack
Core Libraries

Data Processing: pandas, numpy
Machine Learning: scikit-learn, xgboost, imbalanced-learn
Model Interpretation: shap
Visualization: matplotlib, seaborn, plotly
Dashboard: streamlit

Development Environment

Language: Python 3.9+
Version Control: Git/GitHub
Environment Manager: conda or venv


Installation
Prerequisites

Python 3.9 or higher
Git

Setup Instructions
1. Clone the repository
bashgit clone https://github.com/[your-username]/smoking-cessation-ml.git
cd smoking-cessation-ml
2. Create virtual environment
Using conda:
bashconda create -n smoking-cessation python=3.9
conda activate smoking-cessation
pip install -r requirements.txt
Using venv:
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Download data

Visit TUS-CPS 2010-2011 Data
Download the "2010-2011 Longitudinal Cohort" file
Download the technical documentation and codebook
Place data files in data/raw/ directory

4. Verify installation
bashpython -c "import pandas, sklearn, xgboost, shap; print('Success!')"
```

---

## Project Structure
```
smoking-cessation-ml/
│
├── data/
│   ├── raw/                                # Original TUS-CPS data (not in Git)
│   ├── processed/                          # Cleaned datasets
│   └── data_dictionary.md                  # Variable descriptions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling_baseline.ipynb
│   ├── 05_modeling_advanced.ipynb
│   ├── 06_model_interpretation.ipynb
│   └── 07_fairness_assessment.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── visualization.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── dashboard/
│   ├── app.py
│   ├── pages/
│   └── assets/
│
├── reports/
│   ├── figures/
│   ├── final_report.pdf
│   └── presentation.pdf
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE

Usage
Run Notebooks
bash# Data exploration and cleaning
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_data_cleaning.ipynb

# Feature engineering
jupyter notebook notebooks/03_feature_engineering.ipynb

# Model training
jupyter notebook notebooks/04_modeling_baseline.ipynb
jupyter notebook notebooks/05_modeling_advanced.ipynb

# Model interpretation
jupyter notebook notebooks/06_model_interpretation.ipynb
Launch Dashboard
bashstreamlit run dashboard/app.py

Methodology
Data Preprocessing

Handle missing data using multiple imputation
Outlier detection using IQR and z-score methods
Train/validation/test splits (60%/20%/20%) with stratification
Apply SMOTE to address class imbalance

Feature Engineering
Demographic Features

Age groups, education level, income, employment status, geographic region

Smoking History

Cigarettes per day, pack-years, age started smoking, years as smoker
Previous quit attempts, time to first cigarette

Cessation Methods

NRT usage (patch, gum, lozenge, inhaler, nasal spray)
Prescription medications (varenicline, bupropion)
Behavioral support (counseling, quitline)
Multi-method combinations

Contextual Factors

Household smoking, workplace policies, health insurance coverage

Interaction Features

High dependence × NRT use
Prescription medication × counseling

Machine Learning Models

Logistic Regression (baseline interpretable model)
Random Forest (ensemble method with feature importance)
XGBoost (gradient boosting for performance)

Evaluation Metrics

ROC-AUC and Precision-Recall AUC
Confusion matrices
F1 score, sensitivity, specificity, balanced accuracy
Fairness metrics across demographic subgroups

Model Interpretation

SHAP values for global and local explanations
Feature importance rankings
Interaction effect analysis


Timeline
WeekPhaseDeliverables1-3Data Preparation & Feature EngineeringCleaned dataset, 25-35 engineered features4-6Modeling & Evaluation3 trained models, SHAP analysis, fairness assessment7-8Deployment & CommunicationInteractive dashboard, IEEE report, presentation

Expected Outcomes
Performance Target: ROC-AUC > 0.70
Hypothesized Key Predictors:

Nicotine dependence (time to first cigarette)
Multi-method cessation approaches (NRT + counseling)
Socioeconomic factors (education, income)
Previous quit attempt history

Public Health Insights:

Identify high-risk relapse profiles
Recommend optimal cessation method combinations
Inform targeted intervention strategies


Limitations
Data Limitations

Temporal: Data from 2010-2011 predates widespread e-cigarette use and newer cessation products
Self-reported outcomes: No biochemical verification (cotinine testing)
Follow-up duration: 30-day abstinence is an early marker, not long-term sustained cessation
Generalizability: US population sample may not apply internationally

Ethical Considerations
This model is for educational and research purposes only. It does not constitute medical advice. Individuals seeking to quit smoking should consult qualified healthcare professionals.
Fairness audits will be conducted to identify performance disparities across demographic groups. All limitations will be transparently documented in the final report.

Team
Ananda Downing
Focus: Feature engineering, Logistic Regression, XGBoost, SHAP analysis, report writing
Email: ananda.email@university.edu
Angel Nivar
Focus: Data preprocessing, Random Forest, model evaluation, dashboard development
Email: angel.email@university.edu

 pip install -r requirements.txt -c constraints.txt

National Cancer Institute (NCI) for providing publicly accessible TUS-CPS data
 
Notes on reproducible installs
-----------------------------
- This repository includes a pinned constraints file at `constraints.txt` created from a known-good environment.
- To install reproducibly, create and activate a virtualenv and run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt
```

Platform notes:
- On macOS, `xgboost` may require the OpenMP runtime. If you see errors importing `xgboost`, install libomp:

```bash
brew install libomp
```

Regenerating constraints
-----------------------
- To regenerate `constraints.txt` after updating `requirements.txt`, create a clean venv, install `requirements.txt`, and run:

```bash
pip freeze --exclude-editable > constraints.txt
```

US Census Bureau for conducting the Current Population Survey
TUS-CPS participants for contributing data to advance public health research


Key References

Kalkhoran S, et al. (2016). E-cigarette use and smoking reduction or cessation in the 2010/2011 TUS-CPS longitudinal cohort. BMC Public Health, 16(1), 1105.
Pierce JP, et al. (2018). Effectiveness of Pharmaceutical Smoking Cessation Aids in a Nationally Representative Cohort of American Smokers. JNCI: Journal of the National Cancer Institute, 110(6), 581-587.
Hughes JR, et al. (2003). Measures of abstinence in clinical trials: issues and recommendations. Nicotine & Tobacco Research, 5(1), 13-25.


Important Links

TUS-CPS Official Website
TUS-CPS 2010-2011 Data Download
SHAP Documentation
Scikit-learn Documentation
XGBoost Documentation


License
This project is for educational purposes as part of a university data mining course. Data is publicly available from NCI. Code is available under MIT License.

Contact
For questions about this project:


Disclaimer: This model is for educational and research purposes only and does not constitute medical advice. Individuals seeking to quit smoking should consult qualified healthcare professionals.
Last Updated: October 2025
