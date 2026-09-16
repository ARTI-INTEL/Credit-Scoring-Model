import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers domain-specific financial features from the German Credit Dataset.
    """
    data = df.copy()

    # 1. Monthly installment estimate (loan amount divided by duration)
    data['monthly_installment_est'] = data['credit_amount'] / np.maximum(data['duration_months'], 1)

    # 2. Credit amount relative to age
    data['credit_to_age_ratio'] = data['credit_amount'] / data['age_years']

    # 3. Monthly payment burden relative to age
    data['monthly_burden_to_age_ratio'] = data['monthly_installment_est'] / data['age_years']

    # 4. Credit per existing credit line
    data['credit_per_existing_credit'] = data['credit_amount'] / np.maximum(data['existing_credits'], 1)

    # 5. Domain risk indicators
    # Checking account risk: < 0 DM (A11) or low positive balance (A12)
    data['is_high_risk_checking'] = data['checking_status'].isin(['A11', 'A12']).astype(int)

    # Critical credit history indicator: A34 (critical account) or A33 (delay in paying)
    data['is_critical_credit_history'] = data['credit_history'].isin(['A33', 'A34']).astype(int)

    # Low or unknown savings indicator: A61 (<100 DM) or A65 (unknown)
    data['is_low_savings'] = data['savings_status'].isin(['A61', 'A65']).astype(int)

    # Young borrower flag (< 25 years old)
    data['is_young_borrower'] = (data['age_years'] < 25).astype(int)

    # Long loan duration flag (> 36 months)
    data['is_long_duration'] = (data['duration_months'] > 36).astype(int)

    # Foreign worker indicator binary
    data['is_foreign_worker'] = (data['foreign_worker'] == 'A201').astype(int)

    return data
