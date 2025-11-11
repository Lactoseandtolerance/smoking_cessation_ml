# PATH Study Data Dictionary
## Variable Mapping for Smoking Cessation Analysis

*This document maps PATH Study variables to features needed for the analysis.*

---

## Core Outcome Variable

| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| quit_success | TBD | 30-day smoking abstinence at follow-up | 0 = Still smoking, 1 = Abstinent |

---

## Smoking Status & Behavior

| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| current_smoker | TBD | Current smoking status | 0 = No, 1 = Yes |
| quit_attempt | TBD | Made quit attempt since last wave | 0 = No, 1 = Yes |
| cpd | TBD | Cigarettes per day | Numeric (1-60+) |
| ttfc_minutes | TBD | Time to first cigarette after waking | Numeric (minutes) |

---

## Demographics

| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| person_id | TBD | Unique person identifier | String/Numeric |
| age | TBD | Age at interview | Numeric (18-99) |
| sex | TBD | Biological sex | 1 = Male, 2 = Female |
| education_years | TBD | Years of education completed | Numeric (0-20+) |
| education_cat | DERIVED | Education category | <HS, HS, Some College, College+ |
| income | TBD | Household income | Numeric or categorical |
| race_ethnicity | TBD | Race/ethnicity | Categories vary by PATH coding |

---

## Cessation Methods

| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| nrt_any | TBD | Used any NRT product | 0 = No, 1 = Yes |
| nrt_patch | TBD | Used nicotine patch | 0 = No, 1 = Yes |
| nrt_gum | TBD | Used nicotine gum | 0 = No, 1 = Yes |
| nrt_lozenge | TBD | Used nicotine lozenge | 0 = No, 1 = Yes |
| varenicline | TBD | Used varenicline (Chantix) | 0 = No, 1 = Yes |
| bupropion | TBD | Used bupropion (Zyban/Wellbutrin) | 0 = No, 1 = Yes |
| counseling | TBD | Received counseling/behavioral support | 0 = No, 1 = Yes |
| quitline | TBD | Used telephone quitline | 0 = No, 1 = Yes |

---

## Quit History

| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| lifetime_quit_attempts | TBD | Number of lifetime quit attempts | Numeric (0-99) |
| longest_abstinence_days | TBD | Longest period of abstinence | Numeric (days) |
| previous_successful_quit | DERIVED | Ever successfully quit for 6+ months | 0 = No, 1 = Yes |

---

## Motivation & Intention

| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| readiness_to_quit | TBD | Readiness/interest in quitting | Scale (e.g., 1-10) |
| plans_quit_next_month | TBD | Plans to quit in next 30 days | 0 = No, 1 = Yes |
| plans_quit_next_6months | TBD | Plans to quit in next 6 months | 0 = No, 1 = Yes |

---

## Environmental Factors

| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| num_household_smokers | TBD | Number of smokers in household | Numeric (0-10+) |
| home_smoking_rules | TBD | Smoking allowed in home | 0 = Allowed, 1 = Not allowed |
| workplace_policy | TBD | Workplace smoking policy | 0 = Allowed, 1 = Restricted/banned |

---

## Additional Variables (if available in Public Use Files)

| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| mental_health_diagnosis | TBD | Mental health condition | 0 = No, 1 = Yes |
| alcohol_frequency | TBD | Alcohol consumption frequency | Scale or categorical |
| wake_up_urge | TBD | Urge to smoke upon waking | Scale (e.g., 1-5) |

---

## Missing Value Codes in PATH Study

PATH Study uses negative values for missing data:
- `-9` = Refused
- `-8` = Don't know / Not ascertained
- `-7` = Don't know
- `-4` = Multiple responses (invalid)
- `-1` = Inapplicable (skip pattern)

**These should be recoded to `NaN` during preprocessing.**

---

## Survey Weights

| Variable | PATH Variable | Description |
|----------|---------------|-------------|
| survey_weight | TBD | Person-level survey weight for analysis |

*Note: Compare weighted vs. unweighted models. Use whichever performs better.*

---

## Notes for Implementation

1. **Variable Names:** Fill in actual PATH variable names after reviewing the codebook
2. **Wave-Specific Naming:** PATH uses wave prefixes (e.g., R01_ for Wave 1, R02_ for Wave 2)
3. **Longitudinal Merging:** Variables will need suffixes (_t for baseline, _t1 for follow-up)
4. **Derivation Rules:** Document any complex derivations or calculations
5. **Version Control:** Note PATH Study version and wave-specific differences

---

## TODO: Update This File

- [ ] Review PATH Study codebook (565-page user guide)
- [ ] Identify exact variable names for each feature
- [ ] Document wave-specific variable naming conventions
- [ ] Note any variables that are restricted (not in Public Use Files)
- [ ] Document any derivation logic for computed variables
- [ ] Verify coding schemes match documentation

---

*Last Updated: [Date]*  
*PATH Study Version: Public Use Files, Waves 1-5*
