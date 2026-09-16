import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from src.data_loader import load_german_credit_data
from src.feature_engineering import engineer_features
from src.models import prepare_data, get_preprocessor, evaluate_model, calculate_financial_cost

def run_all_experiments(cost_fp=1, cost_fn=5):
    """
    Trains and compares multiple classifiers:
    - Logistic Regression (Standard & Balanced)
    - Random Forest (Standard & Balanced)
    - LightGBM (Standard & Balanced)
    - XGBoost (Standard & Balanced)
    Also optimizes decision thresholds to minimize financial cost.
    """
    df = load_german_credit_data()
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = prepare_data(df, test_size=0.2, random_state=42)
    preprocessor = get_preprocessor()

    # Preprocess train and test features once for tree models or pipeline execution
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Get feature names after one-hot encoding
    cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(
        preprocessor.transformers_[1][2]
    )
    feature_names = list(preprocessor.transformers_[0][2]) + list(cat_cols)

    models = {
        'Logistic Regression (Default)': LogisticRegression(max_iter=1000, random_state=42),
        'Logistic Regression (Balanced)': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'Random Forest (Default)': RandomForestClassifier(n_estimators=100, random_state=42),
        'Random Forest (Balanced)': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'LightGBM (Default)': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'LightGBM (Balanced)': lgb.LGBMClassifier(scale_pos_weight=(700/300), random_state=42, verbose=-1),
        'XGBoost (Default)': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'XGBoost (Balanced)': xgb.XGBClassifier(scale_pos_weight=(700/300), random_state=42, eval_metric='logloss')
    }

    results = []
    trained_models = {}
    test_probs = {}

    for name, clf in models.items():
        clf.fit(X_train_proc, y_train)
        trained_models[name] = clf

        # Standard threshold 0.5
        metrics, probs = evaluate_model(clf, X_test_proc, y_test, threshold=0.5, cost_fp=cost_fp, cost_fn=cost_fn)
        test_probs[name] = probs

        # Optimal threshold search (minimizing total cost)
        thresholds = np.linspace(0.1, 0.9, 81)
        best_thresh = 0.5
        min_cost = float('inf')
        best_metrics = metrics

        for th in thresholds:
            th_metrics, _ = evaluate_model(clf, X_test_proc, y_test, threshold=th, cost_fp=cost_fp, cost_fn=cost_fn)
            if th_metrics['total_cost'] < min_cost:
                min_cost = th_metrics['total_cost']
                best_thresh = th
                best_metrics = th_metrics

        results.append({
            'Model': name,
            'Precision (0.5)': metrics['precision'],
            'Recall (0.5)': metrics['recall'],
            'F1 (0.5)': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc'],
            'PR-AUC': metrics['pr_auc'],
            'Cost (0.5)': metrics['total_cost'],
            'Best Threshold': best_thresh,
            'Precision (Opt)': best_metrics['precision'],
            'Recall (Opt)': best_metrics['recall'],
            'F1 (Opt)': best_metrics['f1_score'],
            'Min Cost': best_metrics['total_cost']
        })

    results_df = pd.DataFrame(results)
    return results_df, trained_models, test_probs, feature_names, X_train_proc, X_test_proc, y_train, y_test

if __name__ == '__main__':
    results_df, trained_models, test_probs, feature_names, X_tr, X_te, y_tr, y_te = run_all_experiments()
    print(results_df.to_string())
