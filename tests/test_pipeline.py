import unittest
import pandas as pd
import numpy as np
from src.data_loader import load_german_credit_data
from src.feature_engineering import engineer_features
from src.models import prepare_data, get_preprocessor, evaluate_model, calculate_financial_cost

class TestCreditScoringPipeline(unittest.TestCase):

    def test_data_loader(self):
        df = load_german_credit_data()
        self.assertEqual(len(df), 1000)
        self.assertIn('default', df.columns)
        self.assertIn('checking_status_desc', df.columns)
        self.assertTrue(set(df['default'].unique()).issubset({0, 1}))

    def test_feature_engineering(self):
        df = load_german_credit_data()
        df_fe = engineer_features(df)
        self.assertIn('monthly_installment_est', df_fe.columns)
        self.assertIn('credit_to_age_ratio', df_fe.columns)
        self.assertIn('is_high_risk_checking', df_fe.columns)
        self.assertFalse(df_fe['monthly_installment_est'].isnull().any())

    def test_prepare_data_and_cost_calc(self):
        df = load_german_credit_data()
        df_fe = engineer_features(df)
        X_train, X_test, y_train, y_test = prepare_data(df_fe)
        self.assertEqual(len(X_train) + len(X_test), 1000)

        cost_info = calculate_financial_cost([0, 1, 0, 1], [0, 0, 1, 1], cost_fp=1, cost_fn=5)
        # y_true = [0, 1, 0, 1], y_pred = [0, 0, 1, 1]
        # TN (0->0): 1, FP (0->1): 1, FN (1->0): 1, TP (1->1): 1
        # Cost = 1*1 + 1*5 = 6
        self.assertEqual(cost_info['total_cost'], 6)

if __name__ == '__main__':
    unittest.main()
