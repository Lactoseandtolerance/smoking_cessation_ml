# PATH Study Data Dictionary
## Variable Mapping for Smoking Cessation Analysis (52 Canonical Features)

*This document maps PATH Study variables (Waves 1-7, 2013–2020) to the 52 engineered features for smoking cessation prediction.*

**Feature Count**: 52 canonical engineered features (excluding raw alias columns)  
**Data Coverage**: Waves 1-7 pooled transitions (47,882 person-period observations from 23,411 individuals)  
**Outcome**: Smoking abstinence at follow-up wave (30-day abstinence indicator)

---

## Core Outcome Variable
| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| quit_success | DERIVED (from R0{w}_AC1002/AC1003 at follow-up) | 30-day smoking abstinence at follow-up | 0 = Still smoking, 1 = Abstinent |

|--------------|---------------|-------------|--------|
| current_smoker | R0{w}_AC1002, R0{w}_AC1003 | Current smoking status | 0 = No, 1 = Yes |
| quit_attempt | DERIVED (from current_smoker baseline→follow-up) | Made quit attempt since last wave | 0 = No, 1 = Yes |
| ttfc_minutes | R0{w}R_A_MINFIRST_CIGS | Time to first cigarette after waking | Numeric (minutes) |

---

## Demographics

| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| person_id | PERSONID | Unique person identifier | String/Numeric |
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
| quitline | Not currently mapped | Used telephone quitline | 0 = No, 1 = Yes |
---

## Quit History

|--------------|---------------|-------------|--------|
| lifetime_quit_attempts | Not currently mapped | Number of lifetime quit attempts | Numeric (0-99) |
| longest_abstinence_days | R0{w}R_A_PST12M_LNQUIT_DUR, R0{w}R_A_PST12M_LSTQUIT_DUR | Longest period of abstinence | Numeric (minutes → days) |
| previous_successful_quit | DERIVED | Ever successfully quit for 6+ months | 0 = No, 1 = Yes |


## Motivation & Intention
| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| plans_to_quit | R0{w}_AN0235 + R0{w}_AN0240 | Plans to quit within 30 days | 1 if AN0235==1 and AN0240 in {1,2}; else 0 |
| motivation_high | R0{w}_AN0235 + R0{w}_AN0240 | Plans to quit within 6 months | 1 if AN0235==1 and AN0240 in {1,2,3}; else 0 |
| quit_timeframe_raw | R0{w}_AN0240 | Reported timeframe for quitting | 1=7d,2=30d,3=6mo,4=1y,5=>1y |

---

## Environmental Factors

| Feature Name | PATH Variable | Description | Coding |
|--------------|---------------|-------------|--------|
| household_smokers | R0{w}_AX0066_01 (W1–W3) / Proxy (W4–W5) | Presence of other cigarette smokers in household | 1 if AX0066_01==1 (W1–W3); else proxy (derived); 0 otherwise |
| workplace_smokefree | Not currently mapped | Workplace smoking policy | 0 = Allowed, 1 = Restricted/banned (placeholder) |
| household_smokers_explicit | R0{w}_AX0066_01 | Explicit household smoker indicator (W1–W3 only) | 1=Marked, 0=Not marked |


## Additional Variables (if available in Public Use Files)

|--------------|---------------|-------------|--------|
| mental_health_diagnosis | Not currently mapped | Mental health condition | 0 = No, 1 = Yes |
| alcohol_frequency | Not currently mapped | Alcohol consumption frequency | Scale or categorical |
| wake_up_urge | Not currently mapped | Urge to smoke upon waking | Scale (e.g., 1-5) |
---

## Missing Value Codes in PATH Study

- `-8` = Don't know / Not ascertained
- `-4` = Multiple responses (invalid)
- `-1` = Inapplicable (skip pattern)

**These should be recoded to `NaN` during preprocessing.**


## Survey Weights

| Variable | PATH Variable | Description |
|----------|---------------|-------------|

*Note: Compare weighted vs. unweighted models. Use whichever performs better.*

---


1. **Variable Names:** Fill in actual PATH variable names after reviewing the codebook
2. **Wave-Specific Naming:** PATH uses wave prefixes (e.g., R01_ for Wave 1, R02_ for Wave 2)
3. **Longitudinal Merging:** Variables will need suffixes (_t for baseline, _t1 for follow-up)
4. **Derivation Rules:** Document any complex derivations or calculations
5. **Version Control:** Note PATH Study version and wave-specific differences
---

## TODO: Update This File
- [x] Identified all 52 canonical features (complete)
- [x] Documented feature categories (dependence, demographics, methods, environment, motivation)
- [x] Mapped to PATH Study variables (Waves 1-7)
- [x] Documented missing value handling
- [x] Added raw alias columns for audit trail (Waves 1-4)

**Status**: Data dictionary complete and tested on 47,882 real transitions from PATH Study Waves 1-7

---


---

---

## Raw Alias Columns (Waves 1–4)

To improve transparency and auditability, non-engineered "alias" columns have been added for selected raw PATH items across Waves 1–4. These aliases preserve original coding (including negative missing codes) and follow the naming pattern:

`w{wave}_{semantic_suffix}`

Where `{wave}` ∈ {1,2,3,4}. Wave 5 aliases are pending. Aliases are appended after feature engineering and are not part of the canonical modeling feature list (they are excluded when constructing `X` for training/evaluation).

### Purpose
- Enable quick human-readable inspection of raw inputs without searching for full PATH variable names.
- Preserve original categorical/missing encodings for auditing recode logic.
- Facilitate documentation of wave-to-wave availability (e.g., household smokers explicit only Waves 1–3).

### Coverage Summary
| Wave | Alias Count | Notes |
|------|-------------|-------|
| 1 | 15 | Full baseline set including dependence, demographics, quit plans, methods durations |
| 2 | 7  | Subset present in public-use Wave 2 for selected dependence & motivation items |
| 3 | 7  | Mirrors Wave 2 availability |
| 4 | 6  | Household smoker explicit item absent; proxy derivation only |

### Naming Suffixes (examples)
- `age_cat7` → PATH `R0{w}R_A_AGECAT7`
- `sex_raw` → `R0{w}R_A_SEX`
- `race_cat3_raw` → `R0{w}R_A_RACECAT3`
- `hisp_raw` → `R0{w}R_A_HISP`
- `cpd_raw` → `R0{w}R_A_PERDAY_P30D_CIGS`
- `ttfc_raw` → `R0{w}R_A_MINFIRST_CIGS`
- `quit_plan_any_raw` → `R0{w}_AN0235`
- `quit_timeframe_raw` → `R0{w}_AN0240`
- `nrt_duration_raw` → aggregated duration across `R0{w}R_A_PST12M_LSTQUIT_NRT` + ECIG variant
- `rx_duration_raw` → aggregated prescription med duration (`R0{w}R_A_PST12M_LSTQUIT_RX` + ECIG variant)
- `counseling_raw` → `R0{w}_AN0215`
- `household_smokers_explicit_raw` → `R0{w}_AX0066_01` (Waves 1–3 only)
- `smokefree_home_raw` → `R0{w}_AR1045`

### Handling & Usage Notes
1. Negative codes (e.g., -9, -8, -99911) are intentionally retained in alias columns; downstream preprocessing should reference engineered counterparts for modeling.
2. Aliases enable cross-wave consistency checks (e.g., verifying absence of `AX0066_01` in Wave 4+).
3. When adding Wave 5 (and later) aliases, extend the loop in `scripts/run_preprocessing.py` to include `raw_wave_cols` pattern for `{w}=5` and append new `w5_` columns using the same suffix mapping.
4. Do not feed alias columns directly into models unless performing robustness or sensitivity analyses.

### Future Extensions
- Wave 5 alias set.
- Restricted-use variables (if access obtained) for workplace policies or additional cessation aids.
- Harmonized multi-wave duration normalization (minutes → days) exposed via both engineered and raw alias forms for traceability.

*Alias section added: 2025-11-14*

```

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
