# PATH Variable Mapping Tracker
## Date: 2025-11-12

This document tracks the mapping of PATH Study variables across waves for smoking cessation prediction.

---

## ✅ CONFIRMED MAPPED VARIABLES

### Demographics

| Feature | Wave 1 | Wave 2 | Wave 3 | Wave 4 | Wave 5 | Notes |
|---------|--------|--------|--------|--------|--------|-------|
| **Person ID** | PERSONID | PERSONID | PERSONID | PERSONID | PERSONID | ✅ Consistent |
| **Age (numeric)** | R01R_A_AGE | R02R_A_AGE | R03R_A_AGE | R04R_A_AGE | R05R_A_AGE | ✅ Derived |
| **Age (category 7)** | R01R_A_AGECAT7 | R02R_A_AGECAT7 | R03R_A_AGECAT7 | R04R_A_AGECAT7 | R05R_A_AGECAT7 | ✅ Consistent |
| **Age (category 6)** | R01R_A_AGECAT6 | R02R_A_AGECAT6 | R03R_A_AGECAT6 | R04R_A_AGECAT6 | R05R_A_AGECAT6 | ✅ Fallback |
| **Sex** | R01R_A_SEX | R02R_A_SEX | R03R_A_SEX | R04R_A_SEX | R05R_A_SEX | ✅ 1=Male, 2=Female |
| **Race (3 categories)** | R01R_A_RACECAT3 | R02R_A_RACECAT3 | R03R_A_RACECAT3 | R04R_A_RACECAT3 | R05R_A_RACECAT3 | ✅ Consistent |
| **Hispanic** | R01R_A_HISP | R02R_A_HISP | R03R_A_HISP | R04R_A_HISP | R05R_A_HISP | ✅ 1=Yes |
| **Income (poverty category 3)** | R01R_POVCAT3 | R02R_POVCAT3 | R03R_POVCAT3 | R04R_POVCAT3 | R05R_POVCAT3 | ✅ 1-3 scale |
| **Income (poverty category 2)** | R01R_POVCAT2 | R02R_POVCAT2 | R03R_POVCAT2 | R04R_POVCAT2 | R05R_POVCAT2 | ✅ 1-2 scale |

### Smoking Behavior

| Feature | Wave 1 | Wave 2 | Wave 3 | Wave 4 | Wave 5 | Notes |
|---------|--------|--------|--------|--------|--------|-------|
| **Current established smoker** | R01R_A_CUR_ESTD_CIGS | R02R_A_CUR_ESTD_CIGS | R03R_A_CUR_ESTD_CIGS | R04R_A_CUR_ESTD_CIGS | R05R_A_CUR_ESTD_CIGS | ✅ 1=Yes, 2=No |
| **Every day smoker** | R01R_A_EDY_CIGS | R02R_A_EDY_CIGS | R03R_A_EDY_CIGS | R04R_A_EDY_CIGS | R05R_A_EDY_CIGS | ✅ 1=Yes, 2=No |
| **Cigarettes per day (past 30d)** | R01R_A_PERDAY_P30D_CIGS | R02R_A_PERDAY_P30D_CIGS | R03R_A_PERDAY_P30D_CIGS | R04R_A_PERDAY_P30D_CIGS | R05R_A_PERDAY_P30D_CIGS | ✅ Numeric |
| **Time to first cigarette** | R01R_A_MINFIRST_CIGS | R02R_A_MINFIRST_CIGS | R03R_A_MINFIRST_CIGS | R04R_A_MINFIRST_CIGS | R05R_A_MINFIRST_CIGS | ✅ Minutes |

### Quit History

| Feature | Wave 1 | Wave 2 | Wave 3 | Wave 4 | Wave 5 | Notes |
|---------|--------|--------|--------|--------|--------|-------|
| **Last quit duration** | R01R_A_PST12M_LSTQUIT_DUR | R02R_A_PST12M_LSTQUIT_DUR | R03R_A_PST12M_LSTQUIT_DUR | R04R_A_PST12M_LSTQUIT_DUR | R05R_A_PST12M_LSTQUIT_DUR | ✅ Minutes |
| **Longest quit duration** | R01R_A_PST12M_LNQUIT_DUR | R02R_A_PST12M_LNQUIT_DUR | R03R_A_PST12M_LNQUIT_DUR | R04R_A_PST12M_LNQUIT_DUR | R05R_A_PST12M_LNQUIT_DUR | ✅ Minutes |

### Cessation Methods (Aggregated Duration)

| Feature | Wave 1 | Wave 2 | Wave 3 | Wave 4 | Wave 5 | Notes |
|---------|--------|--------|--------|--------|--------|-------|
| **NRT use (any)** | R01R_A_PST12M_LSTQUIT_NRT | R02R_A_PST12M_LSTQUIT_NRT | R03R_A_PST12M_LSTQUIT_NRT | R04R_A_PST12M_LSTQUIT_NRT | R05R_A_PST12M_LSTQUIT_NRT | ✅ Days used |
| **NRT use (e-cig context)** | R01R_A_PST12M_LSTQUIT_ECIG_NRT | R02R_A_PST12M_LSTQUIT_ECIG_NRT | R03R_A_PST12M_LSTQUIT_ECIG_NRT | R04R_A_PST12M_LSTQUIT_ECIG_NRT | R05R_A_PST12M_LSTQUIT_ECIG_NRT | ✅ Fallback |
| **Prescription meds** | R01R_A_PST12M_LSTQUIT_RX | R02R_A_PST12M_LSTQUIT_RX | R03R_A_PST12M_LSTQUIT_RX | R04R_A_PST12M_LSTQUIT_RX | R05R_A_PST12M_LSTQUIT_RX | ✅ Days used |
| **Rx (e-cig context)** | R01R_A_PST12M_LSTQUIT_ECIG_RX | R02R_A_PST12M_LSTQUIT_ECIG_RX | R03R_A_PST12M_LSTQUIT_ECIG_RX | R04R_A_PST12M_LSTQUIT_ECIG_RX | R05R_A_PST12M_LSTQUIT_ECIG_RX | ✅ Fallback |

### Household

| Feature | Wave 1 | Wave 2 | Wave 3 | Wave 4 | Wave 5 | Notes |
|---------|--------|--------|--------|--------|--------|-------|
| **Household size (5 categories)** | R01R_HHSIZE5 | R02R_HHSIZE5 | R03R_HHSIZE5 | R04R_HHSIZE5 | R05R_HHSIZE5 | ✅ 1-5 scale (confirmed in data) |
| **Youth in household** | R01R_HHYOUTH | R02R_HHYOUTH | R03R_HHYOUTH | R04R_HHYOUTH | R05R_HHYOUTH | ✅ 0/1 |

---

### Income and SES

| Feature | Wave 1 | Wave 2 | Wave 3 | Wave 4 | Wave 5 | Notes |
|---------|--------|--------|--------|--------|--------|-------|
| **Poverty category (3-level)** | R01R_POVCAT3 | R02R_POVCAT3 | R03R_POVCAT3 | R04R_POVCAT3 | R05R_POVCAT3 | ✅ Used as `income` proxy in features |
| **Poverty category (2-level)** | R01R_POVCAT2 | R02R_POVCAT2 | R03R_POVCAT2 | R04R_POVCAT2 | R05R_POVCAT2 | Optional fallback |

---

## 🔍 VARIABLES TO LOCATE (UPDATED 2025-11-14)

### Education (HIGH PRIORITY)

**Status**: ✅ FOUND in public use files

**Confirmed mapping**:
- `R0{w}R_A_AM0018` = DERIVED - Highest grade or level of school completed (6 levels)

**Project mapping**:
- We now map `education_code` from `R0{w}R_A_AM0018` and derive 4-level `education_cat`:
   - 1 → <HS
   - 2-3 → HS (GED + HS grad)
   - 4 → Some College
   - 5-6 → College+

Note: We still keep fallback to `R0{w}R_A_EDUC` if it exists in any wave, but AM0018 is the primary source.

### Individual NRT Products (MEDIUM PRIORITY)

**Status**: ⏳ AGGREGATED in current data (all NRT types combined)

**Search needed**:
- R01_AC section for specific questions about:
  - Nicotine patch
  - Nicotine gum
  - Nicotine lozenge
  - Nicotine inhaler
  - Nicotine nasal spray

**Note**: May require disaggregating R0{w}R_A_PST12M_LSTQUIT_NRT or finding questionnaire items

### Counseling/Behavioral Support (HIGH PRIORITY)

**Status**: ✅ FOUND

**Confirmed mapping**:
- `R0{w}_AN0215` = Used counseling (in-person, telephone or web) or self-help materials (1=Yes, 2=No)

This is now mapped wave-aware to the canonical `counseling` feature (1/0).

### Quitline (MEDIUM PRIORITY)

**Status**: NOT FOUND (no explicit quitline variable label found)

**Search needed**:
- Telephone quitline use
- 1-800-QUIT-NOW or state quitlines

**Possible patterns**:
- May be captured under AN0215 (telephone or web counseling)
- Continue searching for explicit quitline item; look for "phone", "hotline", "800" in labels

### Plans to Quit / Motivation (HIGH PRIORITY)

**Status**: ✅ FOUND & IMPLEMENTED

**Confirmed mapping**:
- `R0{w}_AN0235` – Plans to quit smoking (1=Yes, 2=No; negative codes missing)
- `R0{w}_AN0240` – Time-frame for plan (1=7 days, 2=30 days, 3=6 months, 4=1 year, 5=>1 year)

**Engineered features**:
- `plans_to_quit` ← (AN0235==1) & (AN0240 <= 2) [within 30 days]
- `motivation_high` ← (AN0235==1) & (AN0240 <= 3) [within 6 months]

**Notes**:
- Negative PATH missing codes (-1, -8, -9, etc.) are recoded to NaN before logic.
- Additional e-cigarette specific variants (AN0235E / AN0240E) currently not used.

### Lifetime Quit Attempts (MEDIUM PRIORITY)

**Status**: NOT FOUND

**Search needed**:
- Number of times tried to quit (numeric count)
- "Have you ever tried to quit" with follow-up count

**Possible patterns**:
- R01_AC variables in quit history section
- May be computed from wave-to-wave transitions

### Home Smoking Rules (HIGH PRIORITY)

**Status**: ✅ FOUND & IMPLEMENTED

**Confirmed mapping**:
- `R0{w}_AR1045` – Rules about smoking combustible tobacco inside home:
   - 1 = Not allowed anywhere (mapped to `smokefree_home`=1)
   - 2 = Allowed in some places/times
   - 3 = Allowed anywhere/anytime (mapped to `smokefree_home`=0)

**Engineered feature**:
- `smokefree_home` ← (AR1045 == 1)

**Notes**:
- Consistent across Waves 1–5.

### Workplace Smoking Policy (LOW PRIORITY)

**Status**: ❌ NOT FOUND (unchanged)

**Update**:
- No explicit workplace smoking policy variable identified in public use files.
- Placeholder `workplace_smokefree` retained in engineered features (always 0 unless mapped).

---

## 🔧 NEXT STEPS (UPDATED 2025-11-14)

### Immediate Actions

1. Confirm whether individual NRT product items are accessible (patch/gum/lozenge/inhaler/nasal).
2. Decide on inclusion of e-cigarette-specific quit planning variables (AN0235E/AN0240E) for dual users.
3. Explore adding interaction terms using motivation and environment (e.g., `motivation_high * smokefree_home`).

2. **Check Wave-to-Wave Consistency**
   - Run script to verify all found variables exist in Waves 2-5
   - Document any variable name changes across waves
   - Note any variables added/removed in later waves

3. **Examine Questionnaire Structure**
   - AC sections appear to be Adult Core questions
   - Look for section numbering patterns (AC10=smoking, AC11=quit attempts, AC12=? AC13=?)

### Cross-Wave Verification Script Needed

```python
# Verify variables exist in all waves
for wave in [1, 2, 3, 4, 5]:
    df = load_wave(wave)
    check_variables = [
        f'R0{wave}_AC9010',  # Plans to quit
        f'R0{wave}_AC9011',  # ?
        # etc...
    ]
```

### Documentation Updates Completed
- Education mapping finalized (AM0018)
- Counseling mapping finalized (AN0215)
- Plans to quit & motivation finalized (AN0235/AN0240)
- Home smoking rules finalized (AR1045)
- Household smokers partial (AX0066_01 Waves 1–3; proxy for Waves 4–5)

### Remaining Documentation
- Individual NRT product disaggregation (if feasible)
- Quitline explicit variable (still unknown; may rely on counseling)
- Lifetime quit attempts (not located; may approximate via duration metrics)
- Workplace policy (absent)

---

## 📊 WAVE DIFFERENCES TRACKER (UPDATED 2025-11-14)

### Variables Added in Later Waves
- None among mapped core cessation predictors (W1–W5 consistent for AN0235, AN0240, AR1045).

### Variables Removed in Later Waves
- AX0066_01 (Household smoker indicator) present only W1–W3; absent W4–W5.

### Coding Changes Across Waves
- No coding changes detected for confirmed variables (AN0235, AN0240, AR1045, AM0018, AN0215).

---

## 🎯 PRIORITY RANKING FOR DASHBOARD

1. **CRITICAL (COMPLETED)**:
   - Education
   - Plans to quit / motivation
   - Home smoking rules
   - Counseling

2. **IMPORTANT (IN PROGRESS)**:
   - Individual NRT products
   - Quitline (may be embedded in counseling)
   - Lifetime quit attempts (still absent)

3. **NICE TO HAVE**:
   - Workplace policy
   - Mental health variables
   - Alcohol use
   - E-cigarette specific quit planning (AN0235E/AN0240E)

---

*Next update: After codebook review and wave verification*
