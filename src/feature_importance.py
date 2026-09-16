import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_german_credit_data
from src.feature_engineering import engineer_features
from src.models import prepare_data, get_preprocessor

def analyze_feature_importance():
    """
    Extracts feature importances for Logistic Regression (coefficients/odds ratios)
    and Random Forest (Gini feature importances).
    """
    df = load_german_credit_data()
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = prepare_data(df, test_size=0.2, random_state=42)
    preprocessor = get_preprocessor()

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Extract feature names after encoding
    num_cols = list(preprocessor.transformers_[0][2])
    cat_ohe = preprocessor.named_transformers_['cat']
    cat_cols = list(cat_ohe.get_feature_names_out(preprocessor.transformers_[1][2]))
    all_feature_names = num_cols + cat_cols

    # 1. Logistic Regression Coefficients
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_proc, y_train)

    lr_coefs = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient': lr.coef_[0],
        'Odds_Ratio': np.exp(lr.coef_[0])
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    # 2. Random Forest Feature Importances
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train_proc, y_train)

    rf_importances = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return lr_coefs, rf_importances

if __name__ == '__main__':
    lr_coefs, rf_importances = analyze_feature_importance()
    print("--- Top 15 Logistic Regression Coefficients ---")
    print(lr_coefs.head(15).to_string(index=False))
    print("\n--- Top 15 Random Forest Feature Importances ---")
    print(rf_importances.head(15).to_string(index=False))
