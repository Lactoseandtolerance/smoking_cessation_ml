# PATH Study Data Download Guide

## Quick Answers to Common Questions

### Q: What format is the PATH Study data?
**A:** PATH Study data is provided in **STATA (.dta)** or **SPSS (.sav)** format, NOT CSV.
- ✅ **Best choice:** Download STATA (.dta) format - pandas can read it natively
- ✅ **Alternative:** SPSS (.sav) format - requires pyreadstat (already in requirements.txt)

### Q: Which files should I download?
**A:** Download **ADULT files only** for Waves 1-5.

**DO download:**
- ✅ Wave 1: Adult Public Use Files
- ✅ Wave 2: Adult Public Use Files
- ✅ Wave 3: Adult Public Use Files
- ✅ Wave 4: Adult Public Use Files
- ✅ Wave 5: Adult Public Use Files

**DO NOT download:**
- ❌ Youth files (ages 12-17) - not relevant for adult smoking cessation
- ❌ Parent files - not relevant for this analysis

### Q: Why only Adult files?
**A:** Your analysis focuses on **adult smoking cessation** (ages 18+). The youth and parent questionnaires:
- Ask different questions (different variables)
- Have different outcomes (youth initiation vs. adult cessation)
- Would complicate the analysis without adding value

---

## Step-by-Step Download Instructions

### 1. Register and Log In
1. Go to: https://www.icpsr.umich.edu/
2. Click "Create Account"
3. Complete registration (may take 1-2 business days for verification)
4. Log in to your account

### 2. Navigate to PATH Study
1. Go to: https://www.icpsr.umich.edu/web/NAHDAP/series/606
2. You'll see a list of all PATH Study datasets

### 3. Download Each Wave (Repeat for Waves 1-5)

**For each wave, look for entries like:**
- "Population Assessment of Tobacco and Health (PATH) Study [United States] Public-Use Files (PUF), Wave X (2013-2014)"

**What to download:**
1. Click on the wave (e.g., "Wave 1")
2. Look for **"Adult"** data file
3. Choose **STATA (.dta)** format (recommended) or SPSS (.sav)
4. Download the data file
5. Download the **Adult Questionnaire Codebook** (PDF)

**File naming:**
- Files may be named like: `36498-0001-Data.dta` or similar
- **Rename them** to something clear, like: `PATH_W1_Adult.dta`

### 4. Download Documentation
**Essential documents:**
- [ ] PATH Study User Guide (565-page PDF) - comprehensive documentation
- [ ] Adult Questionnaire Codebook for each wave
- [ ] Variable Codebook (if available)

**Where to find:**
- Usually available on each wave's download page
- Or in the main PATH Study series page under "Documentation"

### 5. Organize Your Files

Place all downloaded files in: `data/raw/`

**Recommended naming convention:**
```
data/raw/
├── PATH_W1_Adult.dta          # Wave 1 adult data
├── PATH_W2_Adult.dta          # Wave 2 adult data
├── PATH_W3_Adult.dta          # Wave 3 adult data
├── PATH_W4_Adult.dta          # Wave 4 adult data
├── PATH_W5_Adult.dta          # Wave 5 adult data
├── PATH_UserGuide.pdf         # Main documentation
├── PATH_W1_Codebook.pdf       # Wave 1 variable codebook
├── PATH_W2_Codebook.pdf       # Wave 2 variable codebook
├── PATH_W3_Codebook.pdf       # Wave 3 variable codebook
├── PATH_W4_Codebook.pdf       # Wave 4 variable codebook
└── PATH_W5_Codebook.pdf       # Wave 5 variable codebook
```

---

## Working with STATA/SPSS Files in Python

### Option 1: STATA format (.dta) - RECOMMENDED
```python
import pandas as pd

# Pandas can read STATA files natively
df = pd.read_stata('data/raw/PATH_W1_Adult.dta')
print(df.head())
```

### Option 2: SPSS format (.sav)
```python
import pandas as pd

# Requires pyreadstat (already in requirements.txt)
df = pd.read_spss('data/raw/PATH_W1_Adult.sav')
print(df.head())
```

### Updated data_preprocessing.py
The `load_wave_data()` function in `src/data_preprocessing.py` has been updated to:
- ✅ Handle both STATA (.dta) and SPSS (.sav) formats
- ✅ Try multiple common naming conventions
- ✅ Provide helpful error messages if files not found

**Usage:**
```python
from src.data_preprocessing import load_wave_data

# Will automatically try to find and load the file
wave1 = load_wave_data(1, data_dir='data/raw', file_format='dta')
```

---

## Key PATH Study Variables to Look For

When you open the codebooks, look for these variable categories:

### Essential Variables:
1. **Person ID** - Unique identifier (usually named `R01_PERSONID` or similar)
2. **Current smoking status** - Binary indicator
3. **Cigarettes per day** - Numeric
4. **Time to first cigarette** - Numeric (minutes)
5. **Quit attempts** - Binary or count
6. **Cessation methods used** - Binary indicators for each method
7. **Demographics** - Age, sex, education, income, race/ethnicity

### Variable Naming Convention:
PATH Study uses wave-specific prefixes:
- Wave 1: `R01_...`
- Wave 2: `R02_...`
- Wave 3: `R03_...`
- Wave 4: `R04_...`
- Wave 5: `R05_...`

**Example:**
- Wave 1 smoking status: `R01_CURRENT_SMOKER`
- Wave 2 smoking status: `R02_CURRENT_SMOKER`

---

## Troubleshooting

### "Can't find the adult files"
- PATH Study datasets may be labeled as "Public Use Files (PUF)"
- Look for entries with "Adult" or "Ages 18+"
- Each wave might have multiple datasets - you want the main adult questionnaire data

### "File won't load in Python"
- Make sure you installed pyreadstat: `pip install pyreadstat`
- For STATA files: pandas can read natively (pandas >= 2.0)
- Check file isn't corrupted: try re-downloading

### "Too many variables in the dataset"
- PATH Study adult files have 1000+ variables
- Don't worry - you'll only use 30-50 variables
- The data_dictionary.md will help you identify which ones to use

### "Youth and Adult files have different variables"
- This is expected and why you only download Adult files
- Adult files have questions specific to adult smoking and cessation
- Youth files focus on initiation and different behaviors

---

## After Download Checklist

- [ ] Downloaded Wave 1-5 Adult data files (STATA or SPSS format)
- [ ] Renamed files to clear names (e.g., PATH_W1_Adult.dta)
- [ ] Placed all files in `data/raw/` directory
- [ ] Downloaded PATH Study User Guide (565-page PDF)
- [ ] Downloaded adult questionnaire codebooks for each wave
- [ ] Verified files can be loaded in Python
- [ ] Updated `data/data_dictionary.md` with actual variable names from codebooks

---

## Next Steps

Once files are downloaded:
1. **Test loading:** Run `wave1 = pd.read_stata('data/raw/PATH_W1_Adult.dta')`
2. **Explore structure:** Check `wave1.columns` and `wave1.head()`
3. **Review codebook:** Identify actual variable names for smoking status, quit attempts, etc.
4. **Update data_dictionary.md:** Map PATH variables to your feature names
5. **Start Phase 2:** Follow ACTION_GUIDE.md Phase 2 instructions

---

## Additional Resources

- **PATH Study Website:** https://pathstudyinfo.nih.gov/
- **ICPSR PATH Series:** https://www.icpsr.umich.edu/web/NAHDAP/series/606
- **PATH Study Publications:** Search PubMed for "PATH Study smoking cessation"
- **Pandas STATA Documentation:** https://pandas.pydata.org/docs/reference/api/pandas.read_stata.html

---

**Last Updated:** November 10, 2025  
**Data Format:** STATA (.dta) or SPSS (.sav)  
**Required Files:** Adult data for Waves 1-5 only
