# PATH Study Data Dictionary
## Variable Mapping for Smoking Cessation Analysis

*This document maps PATH Study variables to features needed for the analysis.*

---

## Core Outcome Variable

| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| quit_success | DERIVED (from R0{w}_AC1002/AC1003 at follow-up) | 30-day smoking abstinence at follow-up | 0 = Still smoking, 1 = Abstinent |

---

## Smoking Status & Behavior

| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| current_smoker | R0{w}_AC1002, R0{w}_AC1003 | Current smoking status | 0 = No, 1 = Yes |
| quit_attempt | DERIVED (from current_smoker baseline→follow-up) | Made quit attempt since last wave | 0 = No, 1 = Yes |
| cpd | R0{w}R_A_PERDAY_P30D_CIGS | Cigarettes per day | Numeric (1-60+) |
| ttfc_minutes | R0{w}R_A_MINFIRST_CIGS | Time to first cigarette after waking | Numeric (minutes) |

---

## Demographics

| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| person_id | PERSONID | Unique person identifier | String/Numeric |
| age | R0{w}R_A_AGE, R0{w}R_A_AGECAT7, R0{w}R_A_AGECAT6 | Age at interview | Numeric (18-99) or derived from categories |
| sex | R0{w}R_A_SEX | Biological sex | 1 = Male, 2 = Female |
| education_code | R0{w}R_A_AM0018 | Highest grade or level of school completed (6 levels) | 1=<HS, 2=GED, 3=HS, 4=Some college/Assoc, 5=Bachelor's, 6=Advanced |
| education_cat | DERIVED from education_code | Collapsed education category | <HS, HS (incl. GED), Some College, College+ |
| income | R0{w}R_POVCAT3, R0{w}R_A_INCOME | Household income | Numeric or categorical |
| race_ethnicity | R0{w}R_A_RACECAT3, R0{w}R_A_RACE, R0{w}R_A_HISP | Race/ethnicity | White, Black, Hispanic, Other (collapsed) |

---

## Cessation Methods

| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| nrt_any | R0{w}R_A_PST12M_LSTQUIT_NRT, R0{w}R_A_PST12M_LSTQUIT_ECIG_NRT | Used any NRT product | 0 = No, 1 = Yes (duration > 0) |
| nrt_patch | Not currently mapped (aggregated to nrt_any) | Used nicotine patch | 0 = No, 1 = Yes |
| nrt_gum | Not currently mapped (aggregated to nrt_any) | Used nicotine gum | 0 = No, 1 = Yes |
| nrt_lozenge | Not currently mapped (aggregated to nrt_any) | Used nicotine lozenge | 0 = No, 1 = Yes |
| varenicline | R0{w}R_A_PST12M_LSTQUIT_RX, R0{w}R_A_PST12M_LSTQUIT_ECIG_RX | Used varenicline (Chantix) | 0 = No, 1 = Yes (aggregated RX duration > 0) |
| bupropion | R0{w}R_A_PST12M_LSTQUIT_RX (aggregated) | Used bupropion (Zyban/Wellbutrin) | 0 = No, 1 = Yes |
| counseling | R0{w}_AN0215 | Used counseling (in-person, telephone, or web) or self-help materials | 0 = No, 1 = Yes |
| quitline | Not currently mapped | Used telephone quitline | 0 = No, 1 = Yes |

---

## Quit History

| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| lifetime_quit_attempts | Not currently mapped | Number of lifetime quit attempts | Numeric (0-99) |
| longest_abstinence_days | R0{w}R_A_PST12M_LNQUIT_DUR, R0{w}R_A_PST12M_LSTQUIT_DUR | Longest period of abstinence | Numeric (minutes → days) |
| previous_successful_quit | DERIVED | Ever successfully quit for 6+ months | 0 = No, 1 = Yes |

---

## Motivation & Intention

| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| readiness_to_quit | Not currently mapped | Readiness/interest in quitting | Scale (e.g., 1-10) |
| plans_quit_next_month | Not currently mapped | Plans to quit in next 30 days | 0 = No, 1 = Yes |
| plans_quit_next_6months | Not currently mapped | Plans to quit in next 6 months | 0 = No, 1 = Yes |

---

## Environmental Factors

| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| num_household_smokers | DERIVED (from R0{w}R_HHSIZE5, R0{w}R_HHYOUTH) | Number of smokers in household | Proxy: 1 if hhsize>=3 or youth present, else 0 |
| home_smoking_rules | Not currently mapped | Smoking allowed in home | 0 = Allowed, 1 = Not allowed |
| workplace_policy | Not currently mapped | Workplace smoking policy | 0 = Allowed, 1 = Restricted/banned |

---

## Additional Variables (if available in Public Use Files)

| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| mental_health_diagnosis | Not currently mapped | Mental health condition | 0 = No, 1 = Yes |
| alcohol_frequency | Not currently mapped | Alcohol consumption frequency | Scale or categorical |
| wake_up_urge | Not currently mapped | Urge to smoke upon waking | Scale (e.g., 1-5) |

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
| survey_weight | Not currently used | Person-level survey weight for analysis |

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

*Last Updated: 2025-11-12*  
*PATH Study Version: Public Use Files, Waves 1-5*

---

## Wave-aware PATH variables used by the pipeline

The preprocessing and feature engineering code uses the following PATH variables (wave-aware). Replace `{w}` with the wave number 1–5.

- Person ID
  - `PERSONID` — unique respondent identifier

- Current smoking status (baseline selection, follow-up outcome)
  - `R0{w}_AC1002` — current smoking indicator (past 30-day smoking item; used to detect smokers)
  - `R0{w}_AC1003` — smoking frequency (values 1–2 interpreted as current smoker)
  - Outcome `quit_success` is derived as NOT current smoker at follow-up: 1 = abstinent at wave w+1, 0 = still smoking

- Demographics
  - `R0{w}R_A_AGE` — numeric age in years (preferred when available)
  - `R0{w}R_A_AGECAT7`, `R0{w}R_A_AGECAT6` — age categories (used to derive numeric age midpoints when `A_AGE` missing)
  - `R0{w}R_A_SEX` — 1 = Male, 2 = Female (used to derive `sex` and `female`)
    - `R0{w}R_A_AM0018` — highest education (6 levels) used to derive `education_cat`
    - `R0{w}R_POVCAT3` (and sometimes `R0{w}R_POVCAT2`) — poverty ratio category (income proxy)
  - `R0{w}R_A_INCOME` — household income (alternative income source)
  - `R0{w}R_A_RACECAT3` or `R0{w}R_A_RACE` — race code (mapped to White/Black/Asian/Other and collapsed)
  - `R0{w}R_A_HISP` — Hispanic indicator (overrides race to Hispanic)

- Smoking behavior / dependence
  - `R0{w}R_A_PERDAY_P30D_CIGS` — cigarettes per day (`cpd`)
  - `R0{w}R_A_MINFIRST_CIGS` — minutes to first cigarette (`ttfc_minutes`)

- Quit history
  - `R0{w}R_A_PST12M_LSTQUIT_DUR` — minutes in last quit attempt (used to help derive longest duration)
  - `R0{w}R_A_PST12M_LNQUIT_DUR` — minutes in longest quit attempt (`longest_abstinence_days`, converted to days)

- Cessation methods (past 12 months quit attempt, duration-coded; any positive duration interpreted as used)
  - `R0{w}R_A_PST12M_LSTQUIT_NRT` and `R0{w}R_A_PST12M_LSTQUIT_ECIG_NRT` — any NRT duration (`nrt_any`/`used_nrt`)
  - `R0{w}R_A_PST12M_LSTQUIT_RX` and `R0{w}R_A_PST12M_LSTQUIT_ECIG_RX` — prescription meds duration (aggregated to `used_varenicline` proxy and `used_any_medication`)
  - `R0{w}_AN0215` — counseling or self-help (mapped to `used_counseling`)

- Household environment
  - `R0{w}R_HHSIZE5` — household size (used to derive household smoker proxy)
  - `R0{w}R_HHYOUTH` — presence of youth in household (used in proxy for household smokers)

Notes:
- Variables with the `ECIG_` alternates are used as fallbacks when present.
- Education category is derived from `R0{w}R_A_AM0018` (not SES proxy).

---

## Canonical feature ↔ PATH mapping (summary)

Below is a high-level mapping between engineered features and the PATH inputs above. Engineered features are saved to `data/processed/pooled_transitions.(csv|parquet)`.

- Dependence and behavior
  - `cpd` ← `R0{w}R_A_PERDAY_P30D_CIGS`
  - `ttfc_minutes` ← `R0{w}R_A_MINFIRST_CIGS`
  - `high_dependence`/`very_high_dependence` ← thresholds on `ttfc_minutes`
  - `cpd_heavy`/`cpd_light` ← thresholds on `cpd`

- Demographics
  - `age` ← `R0{w}R_A_AGE` or derived from `AGECAT7/6`
  - `female` ← from `R0{w}R_A_SEX`
  - `high_income` ← relative (median) on derived `income` from `POVCAT3`/`A_INCOME`
  - `race_white`/`race_black`/`race_hispanic`/`race_other` ← from `RACE` + `HISP`

- Methods
  - `used_nrt` ← any positive duration in NRT fields
  - `used_varenicline`, `used_bupropion`, `used_any_medication` ← from RX duration (aggregated)
  - `used_counseling` ← from `R0{w}_AN0215` (1/0)
  - `used_quitline` ← placeholder unless mapped; present as column (0/1)
  - Combinations: `med_plus_counseling`, `nrt_plus_med`, `nrt_plus_counseling`, `nrt_plus_quitline`, `med_plus_quitline`

- History and motivation
  - `longest_quit_duration` (days) ← longest/last quit durations (minutes → days)
  - `num_previous_quits`, `previous_quit_success`, `motivation_high`, `plans_to_quit` — present as columns; mapped when source items are confirmed

- Environment
  - `household_smokers` (proxy) ← from `HHSIZE5`, `HHYOUTH`
  - `smokefree_home`/`workplace_smokefree` — present as columns; mapped when source items are confirmed

---

## PATH missing value codes handled in code

During preprocessing, the following codes are normalized to missing (NaN):

- Common public-use codes: `-9, -8, -7, -4, -1`
- Extended set observed in PATH documentation: `-99999, -99988, -99977, -99955, -99911, -97777`

These are applied broadly before numeric coercions and derivations.
