import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# Define feature columns
NUMERICAL_FEATURES = [
    'duration_months', 'credit_amount', 'installment_rate', 'residence_since',
    'age_years', 'existing_credits', 'people_liable',
    'monthly_installment_est', 'credit_to_age_ratio', 'monthly_burden_to_age_ratio',
    'credit_per_existing_credit', 'is_high_risk_checking', 'is_critical_credit_history',
    'is_low_savings', 'is_young_borrower', 'is_long_duration', 'is_foreign_worker'
]

CATEGORICAL_FEATURES = [
    'checking_status', 'credit_history', 'purpose', 'savings_status',
    'employment_since', 'personal_status_sex', 'other_debtors',
    'property', 'other_installment_plans', 'housing', 'job',
    'telephone', 'foreign_worker'
]

def prepare_data(df, test_size=0.2, random_state=42):
    """
    Splits features and target with stratification.
    """
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df['default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def get_preprocessor():
    """
    Returns a Sklearn ColumnTransformer for numerical scaling and categorical one-hot encoding.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

def calculate_financial_cost(y_true, y_pred, cost_fp=1, cost_fn=5):
    """
    Calculates total financial cost based on confusion matrix.
    FP: Granting credit to a default applicant (Bank loses principal / loan amount).
    FN: Denying credit to a good applicant (Bank loses interest income opportunity).
    By convention in financial literature (and UCI German dataset note), FP (predicting Good when actual is Bad)
    costs significantly more than FN (predicting Bad when actual is Good).
    Note on labels: Target 1 = Bad/Default, Target 0 = Good/Non-default.
    If model predicts 1 (Bad/Deny) when actual is 0 (Good): False Positive in risk detection = Opportunity cost (cost_fn in banking terms, say 1 unit).
    If model predicts 0 (Good/Approve) when actual is 1 (Bad): False Negative in risk detection = Default loss (cost_fp in banking terms, say 5 units).
    """
    cm = confusion_matrix(y_true, y_pred)
    # cm: [[TN, FP], [FN, TP]]
    # TN: Actual Good, Predicted Good
    # FP: Actual Good, Predicted Bad (False Alarm / Denied Solvent Applicant) -> Opportunity Cost
    # FN: Actual Bad, Predicted Good (Missed Risk / Approved Defaulter) -> Default Loss
    # TP: Actual Bad, Predicted Bad (Caught Defaulter)
    tn, fp, fn, tp = cm.ravel()
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    return {
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'cost_fp_opportunity': fp * cost_fp,
        'cost_fn_default': fn * cost_fn,
        'total_cost': total_cost
    }

def evaluate_model(model, X_test, y_test, threshold=0.5, cost_fp=1, cost_fn=5):
    """
    Evaluates a trained model on test data with a given decision threshold.
    """
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_probs = model.decision_function(X_test)

    y_pred = (y_probs >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    cost_info = calculate_financial_cost(y_test, y_pred, cost_fp=cost_fp, cost_fn=cost_fn)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'threshold': threshold,
        **cost_info
    }
    return metrics, y_probs
