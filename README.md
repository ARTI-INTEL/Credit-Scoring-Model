# Credit Scoring Model: Predicting Applicant Creditworthiness

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning project that predicts credit applicant creditworthiness from historical financial data, compares Logistic Regression against Tree Ensembles (Random Forest, LightGBM, XGBoost), handles class imbalance, optimizes asymmetric error decision thresholds, and explains feature importances in plain language.

---

## 📌 Key Highlights

- **Dataset**: UCI German Credit Dataset (1,000 applicants, 20 financial & demographic attributes).
- **Class Imbalance Addressed**: 70% good borrowers vs. 30% default risk. Class-weighting and probability threshold optimization applied.
- **Asymmetric Cost Analysis**: Explicit financial cost modeling where Default Losses (False Negatives) are weighted 5x more heavily than Opportunity Losses (False Positives).
- **Models Benchmarked**: Logistic Regression, Random Forest, LightGBM, and XGBoost.
- **Top Metrics Achieved**:
  - **Logistic Regression (Balanced)**: **ROC-AUC = 0.794**, **Default Recall = 80.0%**, **Lowest Financial Loss = 95 cost units**.
  - **Random Forest (Balanced)**: **ROC-AUC = 0.809**, **Precision = 62.1%**.
- **Deliverables Included**:
  - `credit_scoring.ipynb`: End-to-end reproducible Jupyter Notebook with code, charts, and analysis.
  - `SUMMARY.md`: One-page executive summary for non-technical stakeholders.

---

## 📁 Repository Structure

```
.
├── SUMMARY.md                # 1-page executive summary for non-technical stakeholders
├── README.md                 # Project documentation and setup instructions
├── credit_scoring.ipynb      # End-to-end Jupyter Notebook with full analysis and code
├── generate_notebook.py      # Python script to build credit_scoring.ipynb programmatically
└── src/                      # Modular Python package
    ├── data_loader.py        # UCI dataset downloader & feature mapping helper
    ├── feature_engineering.py# Financial ratio calculations & risk flag features
    ├── models.py             # Data splitting, preprocessors, and cost metric evaluators
    ├── model_experiments.py  # Benchmark experimentation and cross-model comparison runner
    └── feature_importance.py # Logistic Regression odds ratios & Tree feature importances
```

---

## 🚀 Getting Started & Installation

### Prerequisites
Python 3.10+ with standard data science packages.

### Installation & Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/credit-scoring-model.git
cd credit-scoring-model

# Install dependencies
pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn jupyter nbformat nbconvert
```

### Running the Analysis

#### Option 1: Jupyter Notebook
Open and run `credit_scoring.ipynb` directly:
```bash
jupyter notebook credit_scoring.ipynb
```

#### Option 2: Command Line Scripts
Run model benchmarking experiments directly in terminal:
```bash
PYTHONPATH=. python3 src/model_experiments.py
PYTHONPATH=. python3 src/feature_importance.py
```

---

## 📊 Benchmark Results

| Model Architecture | Balancing Strategy | Recall (Default) | Precision | F1-Score | ROC-AUC | Total Financial Cost Units |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Unweighted | 56.7% | 64.2% | 0.602 | 0.790 | 149 |
| **Logistic Regression** | **Class Balanced** | **80.0%** | **57.8%** | **0.671** | **0.794** | **95** |
| **Random Forest** | Unweighted | 56.7% | 69.4% | 0.624 | 0.809 | 145 |
| **Random Forest** | **Class Balanced** | **68.3%** | **62.1%** | **0.651** | **0.802** | **120** |
| **LightGBM** | Class Balanced | 55.0% | 56.9% | 0.559 | 0.779 | 160 |
| **XGBoost** | Unweighted | 60.0% | 69.2% | 0.643 | 0.784 | 136 |

---

## 💡 Key Feature Drivers Explained

1. **Checking Account Status**: Negative (`< 0 DM`) or non-existent checking accounts represent the single strongest predictor of default risk.
2. **Loan Duration & Amount**: Longer commitments (>36 months) exponentially increase default probability.
3. **Savings Buffer**: Higher savings balances (>1,000 DM) act as a financial shock absorber.
4. **Credit History**: Past payment delays strongly predict future default behavior.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
