import pandas as pd

def load_german_credit_data():
    """
    Downloads and maps the UCI German Credit dataset.
    Returns a DataFrame with human-readable column names and mapped categorical values.
    Target variable: 'default' (1 = Default / High Risk, 0 = Non-Default / Good Risk)
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    col_names = [
        'checking_status', 'duration_months', 'credit_history', 'purpose',
        'credit_amount', 'savings_status', 'employment_since', 'installment_rate',
        'personal_status_sex', 'other_debtors', 'residence_since', 'property',
        'age_years', 'other_installment_plans', 'housing', 'existing_credits',
        'job', 'people_liable', 'telephone', 'foreign_worker', 'target'
    ]

    df = pd.read_csv(url, sep=' ', header=None, names=col_names)

    # Original target: 1 = Good, 2 = Bad. Binary classification: 1 = Bad (Default), 0 = Good
    df['default'] = (df['target'] == 2).astype(int)
    df = df.drop(columns=['target'])

    # Mapping dicts for categorical variables
    checking_map = {
        'A11': '< 0 DM',
        'A12': '0 <= x < 200 DM',
        'A13': '>= 200 DM',
        'A14': 'no checking account'
    }

    credit_history_map = {
        'A30': 'no credits / all paid',
        'A31': 'all credits at this bank paid back',
        'A32': 'existing credits paid back till now',
        'A33': 'delay in paying past credit',
        'A34': 'critical account / other credits existing'
    }

    purpose_map = {
        'A40': 'car (new)',
        'A41': 'car (used)',
        'A42': 'furniture/equipment',
        'A43': 'radio/television',
        'A44': 'domestic appliances',
        'A45': 'repairs',
        'A46': 'education',
        'A47': 'vacation',
        'A48': 'retraining',
        'A49': 'business',
        'A410': 'others'
    }

    savings_map = {
        'A61': '< 100 DM',
        'A62': '100 <= x < 500 DM',
        'A63': '500 <= x < 1000 DM',
        'A64': '>= 1000 DM',
        'A65': 'unknown / no savings account'
    }

    employment_map = {
        'A71': 'unemployed',
        'A72': '< 1 year',
        'A73': '1 <= x < 4 years',
        'A74': '4 <= x < 7 years',
        'A75': '>= 7 years'
    }

    personal_sex_map = {
        'A91': 'male : divorced/separated',
        'A92': 'female : divorced/separated/married',
        'A93': 'male : single',
        'A94': 'male : married/widowed',
        'A95': 'female : single'
    }

    other_debtors_map = {
        'A101': 'none',
        'A102': 'co-applicant',
        'A103': 'guarantor'
    }

    property_map = {
        'A121': 'real estate',
        'A122': 'building society savings / life insurance',
        'A123': 'car or other',
        'A124': 'unknown / no property'
    }

    other_plans_map = {
        'A141': 'bank',
        'A142': 'stores',
        'A143': 'none'
    }

    housing_map = {
        'A151': 'rent',
        'A152': 'own',
        'A153': 'for free'
    }

    job_map = {
        'A171': 'unemployed / unskilled - non-resident',
        'A172': 'unskilled - resident',
        'A173': 'skilled employee / official',
        'A174': 'management / self-employed / highly qualified'
    }

    telephone_map = {
        'A191': 'none',
        'A192': 'yes'
    }

    foreign_map = {
        'A201': 'yes',
        'A202': 'no'
    }

    df['checking_status_desc'] = df['checking_status'].map(checking_map)
    df['credit_history_desc'] = df['credit_history'].map(credit_history_map)
    df['purpose_desc'] = df['purpose'].map(purpose_map)
    df['savings_status_desc'] = df['savings_status'].map(savings_map)
    df['employment_since_desc'] = df['employment_since'].map(employment_map)
    df['personal_status_sex_desc'] = df['personal_status_sex'].map(personal_sex_map)
    df['other_debtors_desc'] = df['other_debtors'].map(other_debtors_map)
    df['property_desc'] = df['property'].map(property_map)
    df['other_installment_plans_desc'] = df['other_installment_plans'].map(other_plans_map)
    df['housing_desc'] = df['housing'].map(housing_map)
    df['job_desc'] = df['job'].map(job_map)
    df['telephone_desc'] = df['telephone'].map(telephone_map)
    df['foreign_worker_desc'] = df['foreign_worker'].map(foreign_map)

    return df
