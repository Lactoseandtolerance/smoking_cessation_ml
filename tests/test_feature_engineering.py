import unittest
import pandas as pd

from src.feature_engineering import engineer_all_features


class TestFeatureEngineeringPhase3(unittest.TestCase):
    def setUp(self):
        # Synthetic minimal DataFrame with typical PATH-like encodings
        self.df = pd.DataFrame({
            'person_id': ['P1','P2','P3','P4'],
            'baseline_cpd': [15, 25, 8, -9],
            'baseline_ttfc': [10, 60, 3, 30],
            'R01R_A_SEX': [1, 2, 2, 1],  # 1=Male, 2=Female
            'R01R_A_AGE': [28, 52, 67, 40],
            'R01R_A_EDUC': [3, 7, 5, -9],  # placeholder coding
            'R01R_A_INCOME': [3, 7, 5, 2],
            'R01R_A_RACECAT': [1, 2, 4, 3],
            'R01R_A_HISP': [2, 2, 2, 1],  # P4 Hispanic
            'nrt_any': [1, 0, 2, 1],
            'nrt_patch': [0, 1, 2, 0],
            'varenicline': [2, 1, 2, 2],
            'bupropion': [0, 0, 1, 0],
            'counseling': [0, 1, 0, 1],
            'quitline': [0, 0, 1, 0],
        })

    def test_mapping_and_features(self):
        out = engineer_all_features(self.df.copy(), recode_missing=True)

        # Canonical mappings
        self.assertTrue({'cpd','ttfc_minutes','age','sex','income'}.issubset(out.columns))

        # Dependence features
        self.assertIn('high_dependence', out.columns)
        self.assertIn('very_high_dependence', out.columns)

        # Demographics
        self.assertIn('female', out.columns)
        self.assertIn('college_degree', out.columns)
        self.assertIn('high_income', out.columns)
        self.assertIn('race_ethnicity', out.columns)
        for c in ['race_white','race_black','race_hispanic','race_other']:
            self.assertIn(c, out.columns)

        # Cessation method features
        for c in ['used_nrt','used_patch','used_varenicline','used_bupropion','used_counseling','used_quitline','used_any_behavioral','used_any_method','cold_turkey']:
            self.assertIn(c, out.columns)

        # Interaction features
        for c in ['med_plus_counseling','nrt_plus_med','highdep_x_varenicline','highdep_x_nrt','young_x_counseling']:
            self.assertIn(c, out.columns)

        # Basic sanity on values
        self.assertEqual(int(out.loc[0, 'female']), 0)  # R01R_A_SEX=1 male
        self.assertEqual(int(out.loc[1, 'female']), 1)  # R01R_A_SEX=2 female

        # Hispanic override
        self.assertEqual(int(out.loc[3, 'race_hispanic']), 1)


if __name__ == '__main__':
    unittest.main()
