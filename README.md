# Predicting Smoking Cessation Success Using Machine Learning: A Multi-Model Approach with Public Health Survey Data

**By: Ananda Downing, Angel Nivar**  
**Timeline: 8 weeks | Course: Data Mining**

## Abstract

Smoking remains a leading cause of preventable death worldwide, and understanding factors that influence successful cessation is critical for public health interventions. This project develops machine learning models to predict smoking cessation success based on individual characteristics, behavioral patterns, and quit strategies using publicly available health survey data. We will utilize the Tobacco Use Supplement to the Current Population Survey (TUS-CPS) 2018-2019 data, representing approximately 15,000+ current and former smokers.

Our primary research question is: What individual characteristics, behavioral patterns, and quit strategies best predict successful smoking cessation (defined as sustained abstinence for 6+ months)? We will engineer 25-35 features spanning demographics, smoking history, nicotine dependence indicators, cessation methods used, and socioeconomic factors. Multiple machine learning algorithms including Logistic Regression, Random Forest, and XGBoost will be trained and compared using appropriate evaluation metrics (ROC-AUC, Precision-Recall curves, F1 scores) while addressing class imbalance through SMOTE and stratified sampling techniques. Model interpretation will be conducted using SHAP (SHapley Additive exPlanations) values to identify the most influential predictors of cessation success.

The final deliverables include: (1) a cleaned analytical dataset with engineered features and documentation, (2) trained and evaluated machine learning models with rigorous performance comparisons, (3) an interactive dashboard for exploring predictions and model explanations, and (4) a comprehensive IEEE-format technical report. This project demonstrates advanced data preprocessing, feature engineering, model evaluation, and interpretation techniques while addressing a meaningful public health question.

## Data Strategy and Rationale

**Primary Dataset: TUS-CPS (Tobacco Use Supplement to Current Population Survey)**

We have selected TUS-CPS as our sole data source for the following methodological reasons:
- Specifically designed to measure smoking cessation behaviors and outcomes
- Contains validated quit success indicators (6-month and 12-month abstinence measures)
- Large sample size of smokers and recent quitters (~30,000+ tobacco users per wave)
- Includes comprehensive cessation method variables (NRT, counseling, medication, cold turkey)
- Well-documented survey weights for population-level inference
- Publicly available with detailed codebooks and user guides

**Target Variable Definition:**
- **Cessation Success**: Self-reported abstinence from smoking for 6+ consecutive months among individuals who made a quit attempt in the past 12 months
- **Binary outcome**: Success (1) vs. Relapse/Ongoing smoking (0)
- Expected class distribution: ~30% success rate based on literature (moderate imbalance)

**Expected Sample Size:**
- Full TUS-CPS sample: ~240,000 respondents
- Current/former smokers: ~30,000
- Recent quit attempters with outcome data: 8,000-12,000 (analytical sample)

## Weekly Timeline and Milestones

### Week 1: Project Setup and Initial Data Exploration
**Checkpoint: Environment ready, data downloaded, initial EDA complete**

**Angel:**
- Set up GitHub repository with proper structure (data/, notebooks/, src/, docs/, reports/)
- Download TUS-CPS 2018-2019 data files and documentation
- Create data dictionary mapping TUS-CPS variable codes to readable names
- Initial data loading: verify file formats, check for corruption

**Ananda:**
- Set up Python environment (Python 3.9+, conda environment with requirements.txt)
- Install core libraries: pandas, numpy, scikit-learn, xgboost, shap, imbalanced-learn, matplotlib, seaborn
- Create initial EDA notebook: dataset shape, column types, missing data overview
- Document survey structure and weighting methodology from TUS-CPS documentation

**Both:**
- Team meeting (end of week): Review data structure, confirm target variable availability, finalize feature list

---

### Week 2: Data Cleaning and Preprocessing
**Checkpoint: Clean dataset ready for feature engineering**

**Angel:**
- Handle missing data:
  - Document missingness patterns (MCAR, MAR, MNAR assessment)
  - Implement appropriate imputation strategies (mode for categorical, median for numerical, or multiple imputation if needed)
  - Create missingness indicator flags for features with >10% missing
- Remove duplicate records and resolve inconsistencies
- Filter dataset to analytical sample (recent quit attempters with outcome data)
- Document all cleaning decisions in preprocessing log

**Ananda:**
- Define and extract target variable (6-month cessation success)
- Verify target variable: check distribution, examine edge cases
- Create train/validation/test split (60%/20%/20%) with stratification on outcome
- Document final sample size and basic demographic characteristics
- Initial outlier detection using IQR and z-score methods for continuous variables

**Both:**
- Peer review each other's preprocessing code
- Team meeting: Review cleaned data statistics, discuss any data quality concerns

---

### Week 3: Feature Engineering and Exploratory Analysis
**Checkpoint: Final feature set created, bivariate relationships analyzed**

**Angel:**
- Engineer demographic features:
  - Age groups, education levels (ordinal encoding)
  - Income categories, employment status
  - Geographic region indicators
- Engineer smoking history features:
  - Smoking intensity (cigarettes per day, pack-years calculation)
  - Age started smoking, years as smoker
  - Previous quit attempts (count, longest duration)
  - Time to first cigarette (nicotine dependence proxy)

**Ananda:**
- Engineer cessation method features:
  - NRT usage (patch, gum, lozenge - one-hot encode)
  - Prescription medication (varenicline, bupropion)
  - Behavioral support (counseling, quitline, app)
  - Quit method combinations (multi-method vs. single method)
- Engineer contextual features:
  - Household smoking (other smokers present)
  - Workplace smoking policies
  - Health insurance coverage
- Create interaction features (e.g., high_dependence × NRT_use)

**Both:**
- Exploratory data analysis:
  - Univariate distributions (histograms, bar charts)
  - Bivariate analysis (cessation success vs. each feature)
  - Correlation matrix and multicollinearity check (VIF)
  - Feature selection: remove highly correlated or zero-variance features
- Apply scaling/normalization (StandardScaler for tree-based models, RobustScaler if outliers present)
- Team meeting: Finalize feature set (~25-35 features), document feature engineering rationale

---

### Week 4: Baseline Models and Class Imbalance Handling
**Checkpoint: Three baseline models trained, class imbalance strategy selected**

**Ananda:**
- Implement class imbalance strategies:
  - Calculate baseline class distribution
  - Apply SMOTE to training data (preserve validation/test sets)
  - Experiment with class_weight='balanced' parameter
  - Compare strategies using cross-validation
- Build Model 1: Logistic Regression
  - Train baseline with all features
  - L2 regularization with hyperparameter tuning (C values: 0.001, 0.01, 0.1, 1, 10)
  - Generate coefficient interpretation

**Angel:**
- Build Model 2: Random Forest
  - Train baseline (n_estimators=100, 200, 500)
  - Tune max_depth, min_samples_split, min_samples_leaf
  - Extract feature importance (Gini importance)
  - Analyze out-of-bag error estimates
- Implement 5-fold stratified cross-validation framework
  - Create reusable CV function
  - Calculate mean CV scores with standard deviations

**Both:**
- Model evaluation on validation set:
  - ROC curves and AUC
  - Precision-Recall curves and PR-AUC
  - Confusion matrices
  - F1 scores, sensitivity, specificity, balanced accuracy
- Team meeting: Compare baseline results, select class imbalance strategy, plan hyperparameter optimization

---

### Week 5: Advanced Models and Hyperparameter Tuning
**Checkpoint: Optimized models trained, performance documented**

**Ananda:**
- Build Model 3: XGBoost
  - Tune learning_rate, max_depth, n_estimators, subsample, colsample_bytree
  - Implement early stopping to prevent overfitting
  - Use RandomizedSearchCV or GridSearchCV for efficient tuning
- Document all hyperparameter search spaces and final selections
- Create hyperparameter tuning visualizations (validation curves)

**Angel:**
- Refine Random Forest and Logistic Regression with optimized hyperparameters
- Perform threshold optimization:
  - Plot precision-recall tradeoff curves
  - Select optimal decision threshold based on use case (balance sensitivity/specificity)
- Conduct statistical comparison of models:
  - McNemar's test for paired model comparison
  - Bootstrapped confidence intervals for performance metrics

**Both:**
- Final model evaluation on held-out test set
  - Generate comprehensive performance report for all three models
  - Compare test performance to validation performance (check for overfitting)
- Team meeting: Select final production model(s), discuss interpretation strategy

---

### Week 6: Model Interpretation and Validation
**Checkpoint: SHAP analysis complete, model insights documented**

**Ananda:**
- SHAP (SHapley Additive exPlanations) analysis:
  - Calculate SHAP values for final model(s)
  - Generate global feature importance plots (bar plots, beeswarm plots)
  - Create individual prediction explanations (waterfall plots for sample cases)
  - Analyze feature interactions using SHAP dependence plots
- Document top 10 most influential features with interpretation

**Angel:**
- Subgroup analysis and fairness assessment:
  - Evaluate model performance across demographic groups (age, gender, income, race/ethnicity)
  - Check for performance disparities (disparate impact analysis)
  - Document any fairness concerns and limitations
- Error analysis:
  - Examine false positives and false negatives
  - Identify systematic prediction errors
  - Create case studies of misclassified examples

**Both:**
- Compile visualization library:
  - Create publication-quality figures for report and presentation
  - Ensure all plots have proper labels, legends, and captions
- Team meeting: Review interpretation findings, outline report structure

---

### Week 7: Dashboard Development and Report Writing
**Checkpoint: Dashboard deployed, draft report complete**

**Angel:**
- Build interactive dashboard (using Streamlit or Plotly Dash):
  - **Page 1**: Project overview, motivation, and data description
  - **Page 2**: Data explorer with filters (demographics, smoking history) and dynamic visualizations
  - **Page 3**: Model performance comparison (ROC curves, metrics table, confusion matrices)
  - **Page 4**: Feature importance dashboard (SHAP summary, individual feature deep-dives)
  - **Page 5**: Prediction interface (input features, get prediction with explanation)
  - **Page 6**: Key insights, limitations, ethical considerations, and references
- Deploy dashboard (Streamlit Cloud or local hosting)

**Ananda:**
- Write IEEE-format report (4-6 pages):
  - **Abstract** (150-200 words)
  - **Introduction**: Motivation, public health context, research question
  - **Related Work**: Brief literature review on ML for smoking cessation (3-5 key papers)
  - **Methods**: Data description, preprocessing, feature engineering, models, evaluation metrics
  - **Results**: Model performance, SHAP analysis, key findings
  - **Discussion**: Interpretation, limitations, public health implications, future work
  - **Conclusion**: Summary of contributions
  - **References**: APA or IEEE citation style
- Create infographic summarizing key findings (one-page visual summary)

**Both:**
- Peer review report draft
- Test dashboard functionality and fix bugs

---

### Week 8: Finalization and Presentation Preparation
**Checkpoint: Final deliverables ready, presentation rehearsed**

**Angel:**
- Finalize dashboard:
  - Incorporate feedback
  - Add disclaimers ("Educational purposes only, not medical advice")
  - Test on multiple browsers/devices
  - Create brief user guide
- Polish visualizations for presentation

**Ananda:**
- Finalize report:
  - Incorporate peer feedback
  - Proofread for grammar, spelling, formatting
  - Ensure IEEE template compliance
  - Generate final PDF
- Clean and document codebase:
  - Add docstrings and comments
  - Create README.md with project overview and setup instructions
  - Ensure reproducibility (requirements.txt, random seeds set)

**Both:**
- Create 10-minute presentation (8-10 slides):
  - Slide 1: Title, authors, motivation
  - Slide 2: Research question and data overview
  - Slide 3: Methodology overview (pipeline diagram)
  - Slide 4: Feature engineering highlights
  - Slide 5: Model comparison results
  - Slide 6: SHAP analysis and key predictors
  - Slide 7: Dashboard demo (live or screenshots)
  - Slide 8: Public health implications and limitations
  - Slide 9: Conclusions and future work
  - Slide 10: Questions
- Rehearse presentation multiple times (practice timing, transitions, Q&A)
- Final team meeting: Last-minute checks, assign presentation roles

---

## Technical Infrastructure

**Version Control:**
- GitHub repository (private during development, public after submission if permitted)
- Branching strategy: `main` (stable), `dev` (integration), feature branches for each major task
- Commit frequently with descriptive messages

**Computing Resources:**
- Local development on personal laptops (sufficient for TUS-CPS dataset size)
- Google Colab Pro (optional backup for model training if local resources insufficient)

**Environment Management:**
- Conda environment with Python 3.9+
- `requirements.txt` with pinned versions for reproducibility

**Collaboration Tools:**
- Slack/Discord for daily communication
- Weekly in-person/Zoom meetings
- Google Docs for collaborative writing (then transfer to LaTeX/Word for IEEE format)
- Jupyter notebooks for exploration, Python scripts for production code

---

## Ethical Considerations and Limitations

**IRB and Data Ethics:**
- Confirm TUS-CPS data is de-identified and publicly available (no IRB required for secondary analysis)
- Acknowledge survey participants and data stewards in report

**Bias and Fairness:**
- Assess model performance across demographic subgroups
- Document any performance disparities and discuss implications
- Avoid making causal claims (observational data, confounding variables present)

**Responsible Communication:**
- Include clear disclaimer: "This model is for educational and research purposes only and does not constitute medical advice. Individuals seeking to quit smoking should consult healthcare professionals."
- Acknowledge limitations: self-reported data, potential recall bias, generalizability constraints

**Limitations to Explicitly Document:**
- Cross-sectional survey design limits causal inference
- Self-reported cessation success (no biochemical verification)
- Missing data and potential selection bias
- Model trained on US population may not generalize internationally
- Temporal factors (quit date, seasonality) not fully captured

---

## Risk Mitigation

**Potential Risks and Mitigation Strategies:**

1. **Data quality issues discovered late:**
   - Mitigation: Complete thorough EDA in Weeks 1-2, maintain flexibility in feature selection

2. **Class imbalance more severe than expected:**
   - Mitigation: Multiple strategies planned (SMOTE, class weights, threshold tuning), early testing in Week 4

3. **Models perform poorly (AUC < 0.65):**
   - Mitigation: Focus on model interpretation and feature analysis; reframe as exploratory study

4. **Time overruns in specific phases:**
   - Mitigation: Weekly checkpoints allow early detection, can adjust scope (e.g., reduce to 2 models, simplify dashboard)

5. **Technical difficulties with dashboard deployment:**
   - Mitigation: Start dashboard early in Week 7, have backup option (static HTML report with interactive plots)

---

## Success Metrics (Aligned with Rubric)

**Questions (Novel, Motivated, Insightful):**
- Research question addresses real public health need
- Clear motivation tied to reducing smoking-related mortality
- Novel application of SHAP for cessation prediction interpretability

**Analysis (Complete, Advanced, Informative):**
- All data mining stages covered: preprocessing, analysis, model evaluation, interpretation
- Advanced techniques: SMOTE, hyperparameter tuning, SHAP, fairness assessment
- Multiple models compared rigorously

**Results/Understanding (Insightful, Tied to Context):**
- SHAP analysis provides actionable insights for interventions
- Discussion explicitly connects findings to public health implications
- All plots properly labeled with context

**Presentation (Correct, Complete, Convincing):**
- 10-minute presentation with clear narrative arc
- Visual slides complement verbal explanation
- Live dashboard demo showcases technical implementation

**Writing (IEEE Format, Clear, Grammatically Correct):**
- Minimum 4 pages in IEEE format
- Clear explanation of methods and results
- Outstanding structure and organization

**Code (Well-Organized, Readable, Clear Variable Names):**
- Modular code structure, functions with docstrings
- Meaningful variable/function names
- No irrelevant code, clean repository

---

## Deliverables Checklist

- [ ] Cleaned analytical dataset (CSV with data dictionary)
- [ ] Jupyter notebooks documenting EDA, preprocessing, modeling
- [ ] Trained model files (pickled sklearn/xgboost models)
- [ ] Interactive dashboard (deployed or local)
- [ ] IEEE-format technical report (PDF, 4-6 pages)
- [ ] Presentation slides (PDF or PPTX)
- [ ] GitHub repository with README and documentation
- [ ] Infographic summary (optional but impressive)

---

**Final Notes:**
This plan prioritizes rigor, reproducibility, and clear communication. By focusing on a single, well-suited dataset (TUS-CPS) rather than attempting complex multi-survey integration, we ensure methodological soundness while maintaining ambitious analytical goals. Weekly checkpoints enable course correction, and the distributed workload balances team contributions across all project phases.
