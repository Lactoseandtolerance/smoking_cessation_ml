# Model Interpretability Summary

## Top Predictive Features (XGBoost Feature Importance)

| Rank | Feature | Gain (%) | Weight (%) | Clinical Interpretation |
|------|---------|----------|------------|---------------------------|
| 5 | `cpd_light` | 20.0 | 0.6 | Light smokers (<10/day) have higher quit rates |
| 1 | `high_dependence` | 18.4 | 1.1 | Higher nicotine dependence reduces quit success |
| 7 | `ttfc_minutes` | 14.1 | 17.1 | Time to first cigarette measures dependence |
| 4 | `cpd_heavy` | 11.2 | 0.1 | Heavy smokers (20+/day) face more challenges |
| 3 | `cpd` | 4.4 | 15.3 | Cigarettes per day indicates addiction severity |
| 12 | `high_income` | 4.1 | 2.0 | Feature impact on quit prediction |
| 6 | `dependence_score` | 2.6 | 0.8 | Feature impact on quit prediction |
| 42 | `quit_timeframe_code` | 2.2 | 7.9 | Quit timing plans indicate motivation |
| 9 | `age_young` | 1.9 | 0.9 | Younger smokers may have different patterns |
| 40 | `motivation_high` | 1.8 | 1.4 | High motivation predicts success |
| 44 | `household_smokers` | 1.7 | 3.5 | Feature impact on quit prediction |
| 45 | `smokefree_home` | 1.6 | 2.8 | Supportive environment aids quitting |
| 41 | `plans_to_quit` | 1.4 | 1.3 | Readiness/motivation is predictive |
| 8 | `age` | 1.2 | 10.1 | Feature impact on quit prediction |
| 14 | `race_black` | 0.9 | 2.1 | Feature impact on quit prediction |

## Key Insights

### Methodology Note

**Importance Metrics:**
- **Gain**: Average improvement in model accuracy when this feature is used for splitting
- **Weight**: How frequently the feature is selected for splits across all trees
- **Cover**: Average number of samples affected by splits on this feature

### Actionable Recommendations

2. **Dependence Screening**: Nicotine dependence measures (TTFC, CPD, dependence score) are highly predictive. Screen for dependence level and tailor intervention intensity accordingly.

### Model Limitations

- Feature importance reflects associations, not causal effects.
- Missing data (especially CPD: 77% missing) handled via XGBoost native splitting.
- Interactions between features (e.g., medication × dependence) captured in model but not shown separately.

## Visualization Guide

- **Feature Importance**: Shows which features drive model decisions
- **Partial Dependence**: Shows how changing one feature affects predicted quit probability
- **Distribution Comparison**: Shows how feature values differ between quit/no-quit groups
- **Individual Examples**: Shows how features combine to produce predictions for specific cases

