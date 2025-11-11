# Phase 2: Variables Identified

## Date: November 10, 2025

This document summarizes the actual PATH Study variables identified during Phase 2 data exploration.

---

## Person Identifier

| Project Variable | PATH Variable | Description | Notes |
|------------------|---------------|-------------|-------|
| person_id | **PERSONID** | Unique person identifier | Consistent across all waves, format: P000000XXX |

---

## Smoking Status Variables (Current Established Cigarette Smoker)

| Wave | PATH Variable | Coding | Sample Size | Smokers | Prevalence |
|------|---------------|--------|-------------|---------|------------|
| 1 | **R01R_A_CUR_ESTD_CIGS** | 1=Yes, 2=No, <0=Missing | 32,320 | 11,402 | 35.3% |
| 2 | **R02R_A_CUR_ESTD_CIGS** | 1=Yes, 2=No, <0=Missing | 28,362 | 9,694 | 34.2% |
| 3 | **R03R_A_CUR_ESTD_CIGS** | 1=Yes, 2=No, <0=Missing | 28,148 | 9,013 | 32.0% |
| 4 | **R04R_A_CUR_ESTD_CIGS** | 1=Yes, 2=No, <0=Missing | 33,822 | 9,915 | 29.3% |
| 5 | **R05R_A_CUR_ESTD_CIGS** | 1=Yes, 2=No, <0=Missing | 34,309 | 8,590 | 25.0% |

---

## Every Day Smoker Variables

| Wave | PATH Variable | Description |
|------|---------------|-------------|
| 1 | **R01R_A_EDY_CIGS** | Smokes every day (vs some days) |
| 2 | **R02R_A_EDY_CIGS** | Smokes every day (vs some days) |
| 3 | **R03R_A_EDY_CIGS** | Smokes every day (vs some days) |
| 4 | **R04R_A_EDY_CIGS** | Smokes every day (vs some days) |
| 5 | **R05R_A_EDY_CIGS** | Smokes every day (vs some days) |

Coding: 1=Yes, 2=No, <0=Missing

---

## Cigarettes Per Day (Past 30 Days)

| Wave | PATH Variable |
|------|---------------|
| 1 | **R01R_A_PERDAY_P30D_CIGS** |
| 2 | **R02R_A_PERDAY_P30D_CIGS** |
| 3 | **R03R_A_PERDAY_P30D_CIGS** |
| 4 | **R04R_A_PERDAY_P30D_CIGS** |
| 5 | **R05R_A_PERDAY_P30D_CIGS** |

Coding: Numeric (count), <0=Missing

---

## Time to First Cigarette After Waking

| Wave | PATH Variable |
|------|---------------|
| 1 | **R01R_A_MINFIRST_CIGS** |
| 2 | **R02R_A_MINFIRST_CIGS** |
| 3 | **R03R_A_MINFIRST_CIGS** |
| 4 | **R04R_A_MINFIRST_CIGS** |
| 5 | **R05R_A_MINFIRST_CIGS** |

Coding: Minutes (numeric), <0=Missing

---

## Missing Data Codes

PATH Study uses negative values for missing data:

| Code | Meaning |
|------|---------|
| -99988 | Missing due to don't know response on component variables |
| -99977 | Missing due to refused response on component variables |
| -99955 | Missing due to improbable response on component variables |
| -99911 | Missing due to instrument skip pattern |
| -97777 | Missing due to data removed per respondent request |

**Action for Phase 3:** Recode all negative values to `NaN` during feature engineering.

---

## Person-Period Dataset Created

**File**: `data/processed/pooled_transitions.csv`

### Summary Statistics:
- **Total person-periods**: 34,051
- **Unique individuals**: 12,993
- **Overall quit success rate**: 11.8%
- **Variables**: 10 (PERSONID, baseline_smoker, baseline_cpd, baseline_ttfc, baseline_everyday, followup_smoker, quit_success, transition, baseline_wave, followup_wave)

### Quit Rates by Transition:
| Transition | N Observations | Quit Success Rate |
|------------|----------------|-------------------|
| W1→W2 | 9,282 | 11.2% |
| W2→W3 | 8,589 | 10.5% |
| W3→W4 | 7,913 | 10.2% |
| W4→W5 | 8,267 | 15.4% |

✅ **Validation**: Overall rate of 11.8% is within expected range (7-15%) based on literature.

---

## Outcome Variable Definition

**quit_success**: Binary indicator of successful cessation
- **1** = Smoker at baseline (time t) who is NOT smoking at follow-up (time t+1)
- **0** = Smoker at baseline who continues smoking at follow-up

Operationalized as:
- Baseline: `R0XR_A_CUR_ESTD_CIGS == 1` (current smoker)
- Follow-up: `R0(X+1)R_A_CUR_ESTD_CIGS == 2` (not current smoker)

---

## Additional Variables Available (37 smoking-related variables found)

Variables containing 'smoke' or 'cig' in Wave 1:
- R01R_A_CUR_ESTD_CIGS ✅ (used)
- R01R_A_EDY_CIGS ✅ (used)
- R01R_A_P30D_CIGS
- R01R_A_SDY_CIGS
- R01R_A_EVR_THRSH_CIGS
- R01R_A_CUR_EXPR_CIGS
- R01R_A_FMR_ESTD_CIGS
- R01R_A_FMR_ESTD_P12M_CIGS
- R01R_A_FMR_EXPR_CIGS
- R01R_A_NVR_CIGS
- R01R_A_CUR_ESTD_ECIG
- R01R_A_EDY_ECIG
- R01R_A_P30D_ECIG
- R01R_A_SDY_ECIG
- R01R_A_EVR_THRSH_ECIG
- R01R_A_CUR_EXPR_ECIG
- R01R_A_FMR_ESTD_ECIG
- R01R_A_FMR_ESTD_P12M_ECIG
- R01R_A_FMR_EXPR_ECIG
- R01R_A_NVR_ECIG
- R01R_A_PERDAY_EDY_CIGS
- R01R_A_PERDAY_P30D_CIGS ✅ (used)
- R01R_A_PERDAY_12MA_CIGS
- R01R_A_PERDAY_FMR_CIGS
- R01R_A_PERDAY_PAST_CIGS
- R01R_A_MINFIRST_CIGS ✅ (used)
- R01R_A_MNTHSMK_CIGS
- R01R_A_DAYSBRAND_CIGSRYO
- R01R_A_DAYSBRAND_CIGSMFG
- R01R_A_DAYSQUIT_CIGS
- R01R_A_MINFIRST_ECIG
- R01R_A_DAYSBRAND_ECIG
- R01R_A_DAYSQUIT_ECIG
- R01R_A_PST12M_LNQUIT_ECIG_DUR
- R01R_A_PST12M_LSTQUIT_ECIG_DUR
- R01R_A_PST12M_LSTQUIT_ECIG_NRT
- R01R_A_PST12M_LSTQUIT_ECIG_RX

---

## Next Steps for Phase 3

To engineer features, we need to identify additional PATH variables for:

1. **Demographics**: age, sex, education, income, race/ethnicity
2. **Cessation methods**: NRT use, varenicline, bupropion, counseling
3. **Quit history**: past quit attempts, longest abstinence duration
4. **Motivation**: readiness to quit, plans to quit
5. **Environmental factors**: household smoking rules, workplace policies

**Action**: Review PATH codebooks to map these variables, then update `src/feature_engineering.py` with actual variable names.

---

## Files Created in Phase 2

1. ✅ `notebooks/01_data_exploration.ipynb` - Data exploration and person-period dataset creation
2. ✅ `data/processed/pooled_transitions.csv` - Pooled person-period dataset
3. ✅ `reports/figures/phase2_cessation_rates.png` - Visualization of quit rates
4. ✅ `data/PHASE2_VARIABLES.md` - This document

**Phase 2 Status**: ✅ COMPLETE
