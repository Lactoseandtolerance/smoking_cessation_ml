import unittest
import pandas as pd

from src.reporting import ModelSummary, summarize_results, render_text_summary


class TestReporting(unittest.TestCase):
    def test_summarize_with_minimal_inputs(self):
        comp = pd.DataFrame({
            'model': ['A','B'],
            'roc_auc': [0.70, 0.73],
            'n_transitions': [1000, 1000],
        })
        best = ModelSummary(model_name='B', model_index=1, metrics={'roc_auc': 0.73, 'pr_auc': 0.40})
        splits = {
            'X_train': list(range(600)),
            'X_val': list(range(200)),
            'X_test': list(range(200)),
            'y_train': [0]*450 + [1]*150,
            'y_val': [0]*160 + [1]*40,
            'y_test': [0]*150 + [1]*50,
        }
        feature_cols = [f'f{i}' for i in range(25)]

        summary = summarize_results(comp, best, splits, feature_cols)
        self.assertEqual(summary['comparison']['n_models'], 2)
        self.assertEqual(summary['data_splits']['n_train'], 600)
        self.assertEqual(summary['data_splits']['n_features'], 25)
        self.assertAlmostEqual(summary['data_splits']['quit_rate_train'], 0.25, places=6)

        text = render_text_summary(summary)
        self.assertIn('BEST MODEL', text)
        self.assertIn('Models compared: 2', text)
        self.assertIn('Train size: 600', text)


if __name__ == '__main__':
    unittest.main()
