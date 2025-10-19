Predicting Smoking Cessation Success Using Machine Learning: A Multi-Model Approach with Longitudinal Survey Data
Authors: Ananda Downing, Angel Nivar
Course: Data Mining
Institution: [Your University]
Timeline: 8 weeks (October - December 2025)

📋 Project Overview
Smoking remains a leading cause of preventable death worldwide. This project develops machine learning models to predict smoking cessation success one year after a quit attempt, using the TUS-CPS 2010-2011 Longitudinal Cohort – a nationally representative sample where baseline smokers were re-interviewed after one year to assess cessation outcomes.
Research Question
What individual characteristics, behavioral patterns, and quit strategies best predict successful smoking cessation one year after a quit attempt?
Key Contributions

Novel application of SHAP values for interpretable cessation prediction using longitudinal data
Evaluation of quit method combinations (e.g., NRT + counseling) rather than single interventions
Fairness assessment across demographic subgroups to ensure equitable public health application
Interactive dashboard for exploring model predictions and explanations


🎯 Project Goals

Predict 30+ day smoking abstinence at 1-year follow-up among baseline smokers who made a quit attempt
Engineer 25-35 features from demographics, smoking history, nicotine dependence indicators, and cessation methods
Compare multiple ML models: Logistic Regression, Random Forest, XGBoost
Interpret model predictions using SHAP (SHapley Additive exPlanations)
Develop actionable insights for public health interventions


📊 Data Source
Dataset: Tobacco Use Supplement to the Current Population Survey (TUS-CPS) 2010-2011 Longitudinal Cohort

Design: True longitudinal cohort (baseline May 2010 → follow-up May 2011)
Sample Size: ~2,400-2,500 baseline smokers with complete follow-up data
Target Variable: 30+ days smoking abstinence at follow-up
Features: Demographics, smoking history, nicotine dependence, cessation methods (NRT, medications, counseling)
Data Access: Publicly available from NCI TUS-CPS

Citation:
US Department of Commerce, Census Bureau. National Cancer Institute and Food and Drug 
Administration co-sponsored Tobacco Use Supplement to the Current Population Survey 
(TUS-CPS), 2010-2011 Longitudinal Cohort.

🛠️ Technical Stack
Core Libraries:

Data Processing: pandas, numpy
Machine Learning: scikit-learn, xgboost, imbalanced-learn
Model Interpretation: shap
Visualization: matplotlib, seaborn, plotly
Dashboard: streamlit

Development:

Language: Python 3.9+
Version Control: Git/GitHub
Environment: Virtual environment (conda or venv)


🚀 Getting Started
Prerequisites

Python 3.9 or higher
Git

Installation

Clone the repository:

bashgit clone https://github.com/[your-username]/smoking-cessation-ml.git
cd smoking-cessation-ml

Set up Python environment:

Option A: Using conda (recommended)
bashconda create -n smoking-cessation python=3.9
conda activate smoking-cessation
pip install -r requirements.txt
Option B: Using venv
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Download the data:


Visit TUS-CPS 2010-2011 Data
Download the "2010-2011 Longitudinal Cohort" file
Download the technical documentation and codebook
Place data files in data/raw/ directory


Verify installation:

bashpython -c "import pandas, sklearn, xgboost, shap; print('All dependencies installed successfully!')"
```

---

## 📁 Project Structure
```
smoking-cessation-ml/
│
├── data/
│   ├── raw/                      # Original TUS-CPS data files (not committed to Git)
│   ├── processed/                # Cleaned and processed datasets
│   └── data_dictionary.md        # Variable descriptions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb           # Initial EDA
│   ├── 02_data_cleaning.ipynb              # Preprocessing and cleaning
│   ├── 03_feature_engineering.ipynb        # Feature creation
│   ├── 04_modeling_baseline.ipynb          # Baseline models
│   ├── 05_modeling_advanced.ipynb          # Hyperparameter tuning
│   ├── 06_model_interpretation.ipynb       # SHAP analysis
│   └── 07_fairness_assessment.ipynb        # Subgroup analysis
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Data cleaning functions
│   ├── feature_engineering.py    # Feature creation functions
│   ├── modeling.py               # Model training utilities
│   ├── evaluation.py             # Evaluation metrics
│   └── visualization.py          # Plotting functions
│
├── models/
│   ├── logistic_regression.pkl   # Trained models
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── dashboard/
│   ├── app.py                    # Streamlit dashboard
│   ├── pages/                    # Dashboard pages
│   └── assets/                   # Images, CSS
│
├── reports/
│   ├── figures/                  # Generated plots for report
│   ├── final_report.pdf          # IEEE format technical report
│   └── presentation.pdf          # Presentation slides
│
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE

📝 Usage
1. Data Preprocessing
bashjupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_data_cleaning.ipynb
2. Feature Engineering
bashjupyter notebook notebooks/03_feature_engineering.ipynb
3. Model Training
bashjupyter notebook notebooks/04_modeling_baseline.ipynb
jupyter notebook notebooks/05_modeling_advanced.ipynb
4. Model Interpretation
bashjupyter notebook notebooks/06_model_interpretation.ipynb
5. Run Dashboard
bashstreamlit run dashboard/app.py

🔬 Methodology
Data Preprocessing

Handle missing data using multiple imputation
Outlier detection using IQR and z-score methods
Create train/validation/test splits (60%/20%/20%) with stratification
Apply SMOTE to address class imbalance (~18-21% success rate)

Feature Engineering (25-35 features)
Demographic Features:

Age groups, education level, income, employment status, geographic region

Smoking History:

Cigarettes per day, pack-years, age started smoking, years as smoker
Previous quit attempts, time to first cigarette (nicotine dependence proxy)

Cessation Methods:

NRT usage (patch, gum, lozenge, inhaler, nasal spray)
Prescription medications (varenicline, bupropion)
Behavioral support (counseling, quitline)
Multi-method combinations

Contextual Factors:

Household smoking, workplace policies, health insurance coverage

Interaction Features:

High dependence × NRT use
Prescription medication × counseling

Models

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


📊 Expected Outcomes
Model Performance Target: ROC-AUC > 0.70
Key Predictors (Hypothesized):

Nicotine dependence (time to first cigarette)
Multi-method cessation approaches (NRT + counseling)
Socioeconomic factors (education, income)
Previous quit attempt history

Public Health Insights:

Identify high-risk relapse profiles
Recommend optimal cessation method combinations
Inform targeted intervention strategies


📅 Project Timeline
WeekPhaseDeliverables1-3Data Preparation & Feature EngineeringCleaned dataset, 25-35 engineered features4-6Modeling & Evaluation3 trained models, SHAP analysis, fairness assessment7-8Deployment & CommunicationInteractive dashboard, IEEE report, presentation

⚠️ Limitations & Ethical Considerations
Data Limitations

Temporal: Data from 2010-2011 predates widespread e-cigarette use and newer cessation products
Self-reported outcomes: No biochemical verification (cotinine testing)
30-day abstinence: Early marker, not long-term sustained cessation (6+ months)
Generalizability: US population sample may not generalize internationally

Ethical Considerations

Model is for educational and research purposes only
Not medical advice – individuals should consult healthcare professionals
Fairness audits conducted to identify performance disparities across demographic groups
All limitations transparently documented

Responsible Use
This model should inform, not replace, clinical judgment. Predictions should be interpreted alongside patient preferences, clinical context, and evidence-based guidelines.

👥 Team
Ananda Downing

Focus: Feature engineering, Logistic Regression, XGBoost, SHAP analysis, report writing
Email: [ananda.email@university.edu]

Angel Nivar

Focus: Data preprocessing, Random Forest, model evaluation, dashboard development
Email: [angel.email@university.edu]


🙏 Acknowledgments

National Cancer Institute (NCI) for providing publicly accessible TUS-CPS data
US Census Bureau for conducting the Current Population Survey
TUS-CPS participants for contributing data to advance public health research
Course Instructor: [Professor Name] for project guidance


📚 References
Key papers using TUS-CPS 2010-2011 longitudinal cohort:

Kalkhoran S, et al. (2016). E-cigarette use and smoking reduction or cessation in the 2010/2011 TUS-CPS longitudinal cohort. BMC Public Health, 16(1), 1105.
Pierce JP, et al. (2018). Effectiveness of Pharmaceutical Smoking Cessation Aids in a Nationally Representative Cohort of American Smokers. JNCI: Journal of the National Cancer Institute, 110(6), 581-587.
Hughes JR, et al. (2003). Measures of abstinence in clinical trials: issues and recommendations. Nicotine & Tobacco Research, 5(1), 13-25.


📄 License
This project is for educational purposes as part of a university data mining course. Data is publicly available from NCI. Code is available under MIT License.

📞 Contact
For questions about this project, please contact:

Ananda Downing: [ananda.email@university.edu]
Angel Nivar: [angel.email@university.edu]


🔗 Useful Links

TUS-CPS Official Website
TUS-CPS 2010-2011 Data Download
SHAP Documentation
Scikit-learn Documentation
XGBoost Documentation


Disclaimer: This model is for educational and research purposes only and does not constitute medical advice. Individuals seeking to quit smoking should consult qualified healthcare professionals.

Last Updated: October 2025
