# Executive Summary: Credit Scoring & Risk Assessment Model

## Overview & Business Context
Evaluating credit applications requires balancing two competing goals: expanding loan originations to drive revenue while protecting the institution from credit defaults. Using historical financial data from 1,000 credit applicants, we developed and benchmarked predictive risk models—comparing **Logistic Regression** against **Tree Ensembles** (Random Forest, LightGBM, and XGBoost)—to automate creditworthiness assessments and minimize overall financial loss.

---

## 1. Why Standard Accuracy is Deceptive
In typical credit portfolios, the vast majority of applicants are creditworthy. In our dataset, **70% of applicants were good credit risks and 30% defaulted**.

A naive "dumb" model that approves every single applicant would achieve **70% overall accuracy**, yet it would fail to detect a single defaulting borrower, leading to catastrophic credit losses. Therefore, evaluating models using accuracy alone is highly misleading. Instead, we evaluate models using **Recall** (ability to catch defaulters), **Precision** (avoiding false alarms), **ROC-AUC** (overall risk ranking ability), and **Total Expected Financial Cost**.

---

## 2. Asymmetric Cost of Decision Errors
Not all mistakes cost the bank the same amount of money:

* **False Negative (Approving a Defaulter)**: The model predicts an applicant is safe, but they default. The institution suffers a **direct capital loss** of unrecovered loan principal and interest (e.g., thousands of euros).
* **False Positive (Rejecting a Solvent Applicant)**: The model predicts an applicant is high-risk, but they would have repaid. The institution suffers an **opportunity cost** of lost interest profit (e.g., hundreds of euros) and potential customer dissatisfaction.

**In retail banking, approving a defaulting borrower is typically 5 to 10 times more expensive than rejecting a solvent applicant.**

By tuning our decision models specifically to penalize default errors 5x more heavily than false rejections, we lowered total expected financial losses by over **35%** compared to standard decision thresholds.

---

## 3. Key Risk Drivers Explained in Plain Language
Our model interpretability analysis highlights four main financial drivers that govern creditworthiness:

1. **Checking Account Balance (Highest Impact)**:
   * Applicants with **no checking account** or **healthy positive balances** have significantly lower default rates.
   * Conversely, applicants with negative balances (`< 0 DM`) or very low checking balances are the most likely to default due to immediate liquidity strain.
2. **Loan Duration & Monthly Debt Burden**:
   * Longer loan terms (e.g., exceeding **36 months**) and higher monthly repayment relative to age dramatically increase the likelihood of default over time.
3. **Credit History**:
   * Applicants with past payment delays or critical existing credit lines carry a significantly higher probability of defaulting again.
4. **Savings Buffer & Financial Stability**:
   * Substantial savings (>1,000 DM) serve as a safety cushion against unexpected personal financial shocks. Older borrowers with stable job tenure show markedly higher repayment reliability.

---

## 4. Model Benchmarking & Comparison

| Model Architecture | Class Balancing | Default Recall (Catching Defaulters) | Precision | ROC-AUC | Total Financial Cost Units (Lower = Better) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Standard)** | No | 56.7% | 64.2% | 0.790 | 149 |
| **Logistic Regression (Balanced)** | **Yes** | **80.0%** | **57.8%** | **0.794** | **95** |
| **Random Forest (Standard)** | No | 56.7% | 69.4% | 0.809 | 145 |
| **Random Forest (Balanced)** | **Yes** | **68.3%** | **62.1%** | **0.802** | **120** |
| **LightGBM (Balanced)** | **Yes** | 55.0% | 56.9% | 0.779 | 160 |
| **XGBoost (Standard)** | No | 60.0% | 69.2% | 0.784 | 136 |

### Summary Findings:
* **Logistic Regression with Class Balancing** achieved the lowest overall financial cost (**95 cost units**) and caught **80% of defaulting applicants**, making it highly effective and transparent.
* **Random Forest** achieved the highest overall ranking score (**ROC-AUC = 0.809**), proving excellent at differentiating credit tiers across complex non-linear patterns.

---

## 5. Strategic Business Recommendations
1. **Deploy Class-Weighted Decision Models**:
   Transition from unweighted models to class-balanced models to ensure default risk is actively prioritized.
2. **Optimize Probability Thresholds ($p = 0.30 - 0.45$)**:
   Lower the risk acceptance threshold to flag high-risk applicants early, reducing total portfolio loss.
3. **Automated Tiered Underwriting Policy**:
   * **Tier 1 (Auto-Approve)**: Low predicted risk ($p < 0.25$), positive checking balance, low duration.
   * **Tier 2 (Manual Underwriting)**: Moderate risk ($0.25 \le p \le 0.45$) or loans $>36$ months—require proof of income and collateral.
   * **Tier 3 (Auto-Decline / High Risk)**: High predicted risk ($p > 0.45$) or negative checking balance with past payment delays.
