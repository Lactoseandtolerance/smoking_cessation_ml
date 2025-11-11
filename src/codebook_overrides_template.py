"""
Template for Phase 3 codebook overrides.

Fill in actual PATH variable names per wave/user guide and import this dict in
notebooks to pass into engineer_all_features(..., codebook_overrides=OVERRIDES).

Note: For longitudinal person-period datasets, you may want to suffix with _t for
baseline and pass baseline variables specifically. Adjust as needed.
"""

OVERRIDES = {
    # Demographics
    'age': 'R01R_A_AGE',
    'sex': 'R01R_A_SEX',          # 1=Male, 2=Female
    'education_code': 'R01R_A_EDUC',
    'income': 'R01R_A_INCOME',

    # Race/Ethnicity (confirm names)
    'race': 'R01R_A_RACECAT',     # or R01R_A_RACE
    'hispanic': 'R01R_A_HISP',    # 1=Yes, 2=No

    # Smoking behavior
    'cpd': 'R01R_A_PERDAY_P30D_CIGS',
    'ttfc_minutes': 'R01R_A_MINFIRST_CIGS',

    # Cessation methods (placeholders – update to actual PATH variables for method use)
    'nrt_any': None,
    'nrt_patch': None,
    'nrt_gum': None,
    'nrt_lozenge': None,
    'varenicline': None,
    'bupropion': None,
    'counseling': None,
    'quitline': None,
}
