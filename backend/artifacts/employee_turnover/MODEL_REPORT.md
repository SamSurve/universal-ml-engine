# Universal ML Engine — Model Governance & Decision Report

> **Dataset:** `employee_turnover.csv`  
> **Target Variable:** `Employee_Turnover`  
> **Experiment ID:** `3dd47495`  
> **Champion Model:** `Logistic Regression (Tuned)`  
> **Generated Date:** October 6, 2026  

---

## 1. Executive Summary

The Universal ML Engine automated pipeline processed `employee_turnover.csv` (900 rows, 16 features). The task was identified as **Binary Classification** with 100.0% confidence.

**Champion Selection:** Selected 'Logistic Regression (Tuned)' as the champion model because it achieved the highest mean cross-validation f1 (0.8693 +/- 0.0000) on 5-fold development data, outperforming the runner-up 'Logistic Regression' (0.8638) by +0.0055 points.

* **Development 5-Fold CV Score:** `0.8693` (binary_classification)  
* **Untouched 20% Holdout Performance:** `accuracy`: **0.8778**, `balanced_accuracy`: **0.8772**, `precision`: **0.8784**, `recall`: **0.8778**, `f1`: **0.8777**, `f1_macro`: **0.8775**, `roc_auc`: **0.9604**  
* **Total End-to-End Execution Time:** `96.22s`

---

## 2. Dataset Overview & Partitioning

| Dimension | Count / Proportion | Details |
| :--- | :--- | :--- |
| **Raw Samples** | `1,350` rows | Initial dataset size |
| **Cleaned Samples** | `900` rows | After removing missing targets / duplicate records |
| **Development Split (80%)** | `720` rows | Solely used for screening, CV, and tuning |
| **Final Holdout Split (20%)** | `180` rows | Untouched benchmark partition |
| **Total Features Retained** | `15` features | Cleaned predictive features |
| **Exclusions Logged** | `1` items | Audit trail of dropped columns/rows |

---

## 3. Dataset Intelligence & Health Diagnostic

### Health Score: **99.0/100** — Grade: `Excellent`
> *Dataset health is Excellent (99.0/100). The dataset exhibits strong completeness, clean structural integrity, and no critical leakage or imbalance risks.*

#### Quality Category Breakdown
| Category | Score | Weight | Assessment |
| :--- | :--- | :--- | :--- |
| **Completeness** | `100.0/100` | `25%` | No deductions |
| **Uniqueness & Structure** | `100.0/100` | `25%` | No deductions |
| **Feature Quality** | `96.0/100` | `25%` | 1 highly correlated feature pair(s) (|r| >= 0.85) (-4.0 pts) |
| **Target Integrity** | `100.0/100` | `25%` | No deductions |

#### Verified Clean Signals
- [x] Zero missing values detected across feature matrix.
- [x] Zero duplicate rows found in dataset.
- [x] No constant features detected.
- [x] No artificial identifier or surrogate key columns found.
- [x] Categorical features exhibit manageable cardinality.
- [x] Target variable exhibits zero direct leakage risks.
- [x] Target classes are well-balanced.

## 4. Data Quality Warnings & Risks

| Warning ID | Severity | Affected Column(s) | Reason | Recommended Action |
| :--- | :--- | :--- | :--- | :--- |
| `WARN_MULTICOLLINEARITY_Annual_Bonus_Annual_Bonus_Squared` | **LOW** | `Annual_Bonus, Annual_Bonus_Squared` | Features 'Annual_Bonus' and 'Annual_Bonus_Squared' are highly correlated (|r| = 0.9659 >= 0.85). | Tree models handle collinearity well; linear models may benefit from Ridge (L2) regularization. |

---

## 5. Problem Detection

* **Detected Problem Type:** `binary_classification`
* **Detection Confidence:** `1.00`
* **Diagnostic Reasoning:** Target column has exactly 2 distinct values (['0', '1']).
* **Target Class Distribution:** `{"0": 458, "1": 442}`

---

## 6. Preprocessing Architecture & Anti-Leakage

All transformations are strictly encapsulated inside Scikit-Learn `ColumnTransformer` pipelines:
1. **Numerical Pipeline:** Median imputation followed by optional StandardScaler (applied to linear models).
2. **Categorical Pipeline:** Mode (most frequent) imputation followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
3. **Zero-Leakage Enforcement:** Preprocessors are fitted **solely inside training folds** during CV and inside Optuna trials. No transformations peek across validation or holdout boundaries.

---

## 7. Fast Candidate Screening (FLAML)

Evaluated multiple model families under strict time constraints (`31.76s` elapsed):

| Rank | Model Family | Development CV Score | Fit Time | Status |
| :--- | :--- | :--- | :--- | :--- |
| #1 | **CatBoost** | `0.8456` | `10.77s` | `success` |
| #2 | **LightGBM** | `0.8372` | `5.94s` | `success` |
| #3 | **XGBoost** | `0.8366` | `6.68s` | `success` |
| #4 | **Extra Trees** | `0.8352` | `3.76s` | `success` |
| #5 | **Random Forest** | `0.7955` | `2.88s` | `success` |

---

## 8. Bayesian Hyperparameter Optimization (Optuna)

Top candidate algorithms tuned using Tree-structured Parzen Estimator (`TPESampler`) with `MedianPruner` (`51.77s` elapsed):

| Model Candidate | Baseline CV Score | Tuned CV Score | Delta | Trials | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `0.8638` | `0.8693` | **+0.0055** | `15` | `success` |
| **CatBoost** | `0.8523` | `0.8566` | **+0.0043** | `10` | `success` |

#### Optimal Tuned Hyperparameters
```json
{
  "C": 0.31489116479568624
}
```

---

## 9. Development Cross-Validation Results

> **Evaluation Scheme:** 5-Fold Cross-Validation on Development Data (80% Partition)  
> **Primary Metric:** `f1`

| Rank | Model Name | Mean CV Score | CV Std Dev | Fit Time | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | **Logistic Regression (Tuned) 🏆** | `0.8693` | `+/- 0.0000` | `3.01s` | `success` |
| #2 | **Logistic Regression** | `0.8638` | `+/- 0.0247` | `0.28s` | `success` |
| #3 | **CatBoost (Tuned)** | `0.8566` | `+/- 0.0000` | `47.66s` | `success` |
| #4 | **CatBoost** | `0.8523` | `+/- 0.0325` | `3.51s` | `success` |
| #5 | **XGBoost** | `0.8440` | `+/- 0.0283` | `1.73s` | `success` |
| #6 | **HistGradientBoosting** | `0.8372` | `+/- 0.0310` | `1.92s` | `success` |
| #7 | **Random Forest** | `0.8345` | `+/- 0.0315` | `1.95s` | `success` |
| #8 | **Extra Trees** | `0.8342` | `+/- 0.0234` | `1.59s` | `success` |
| #9 | **LightGBM** | `0.8260` | `+/- 0.0231` | `1.50s` | `success` |

---

## 10. Final Champion Model Selection

### Champion: **Logistic Regression (Tuned)**
**Selection Rationale:** Selected 'Logistic Regression (Tuned)' as the champion model because it achieved the highest mean cross-validation f1 (0.8693 +/- 0.0000) on 5-fold development data, outperforming the runner-up 'Logistic Regression' (0.8638) by +0.0055 points.

The champion model was refitted on 100% of the development partition before proceeding to holdout verification.

---

## 11. Final Holdout Evaluation (Untouched 20%)

> [!IMPORTANT]
> The final 20% holdout split was completely segregated prior to screening, cross-validation, and tuning. It was evaluated exactly once to verify out-of-sample generalization without data leakage.

| Evaluation Metric | Holdout Value | Architectural Significance |
| :--- | :--- | :--- |
| **ACCURACY** | `0.8778` | Unbiased generalization score on test partition |
| **BALANCED_ACCURACY** | `0.8772` | Unbiased generalization score on test partition |
| **PRECISION** | `0.8784` | Unbiased generalization score on test partition |
| **RECALL** | `0.8778` | Unbiased generalization score on test partition |
| **F1** | `0.8777` | Unbiased generalization score on test partition |
| **F1_MACRO** | `0.8775` | Unbiased generalization score on test partition |
| **ROC_AUC** | `0.9604` | Unbiased generalization score on test partition |

---

## 12. Explainable AI & Global Feature Importance (SHAP)

* **Explainer Employed:** `shap_linear`
* **Diagnostic Summary:** Model explanations successfully computed via shap_linear. The primary global drivers are: Work_Life_Balance, Performance_Rating, Job_Satisfaction.

### Top Driving Features (Aggregated Raw Features)
| Rank | Feature Name | Mean |SHAP| Value | Relative Contribution |
| :--- | :--- | :--- | :--- |
| #1 | **Work** | `1.4890` | **20.6%** |
| #2 | **Performance** | `1.4265` | **19.8%** |
| #3 | **Job** | `1.2115` | **16.8%** |
| #4 | **Distance** | `1.1613` | **16.1%** |
| #5 | **Years** | `0.9027` | **12.5%** |
| #6 | **Annual** | `0.4524` | **6.3%** |
| #7 | **Education** | `0.2046` | **2.8%** |
| #8 | **Department** | `0.1399` | **1.9%** |
| #9 | **Employee** | `0.1232` | **1.7%** |
| #10 | **Num** | `0.0454` | **0.6%** |

## 13. Example Local Prediction Explanations

Local feature attribution for sample predictions:

### Sample #1
* **Predicted Output:** `0`
* **Prediction Probabilities:** `{"class_0": 0.918, "class_1": 0.082}`
* **Base Expected Value (E[f(X)]):** `-0.3410`
* **Top Positive Drivers:** Work_Life_Balance, Annual_Bonus_Squared, Job_Satisfaction
* **Top Negative Drivers:** Performance_Rating, Distance_From_Home, Annual_Bonus

| Feature | Feature Value | SHAP Value | Impact Direction |
| :--- | :--- | :--- | :--- |
| `Performance_Rating` | `-1.4852` | `-2.0821` | `negative` |
| `Work_Life_Balance` | `0.7408` | `+1.2968` | `positive` |
| `Distance_From_Home` | `-0.8754` | `-1.1423` | `negative` |
| `Annual_Bonus_Squared` | `2.1225` | `+0.6768` | `positive` |
| `Annual_Bonus` | `1.6694` | `-0.2862` | `negative` |

### Sample #2
* **Predicted Output:** `0`
* **Prediction Probabilities:** `{"class_0": 0.9997, "class_1": 0.0003}`
* **Base Expected Value (E[f(X)]):** `-0.3410`
* **Top Positive Drivers:** Department, Employee_Role, Annual_Bonus_Training_Hours_Interaction
* **Top Negative Drivers:** Job_Satisfaction, Work_Life_Balance, Performance_Rating

| Feature | Feature Value | SHAP Value | Impact Direction |
| :--- | :--- | :--- | :--- |
| `Job_Satisfaction` | `-1.7276` | `-2.9233` | `negative` |
| `Work_Life_Balance` | `-1.5576` | `-2.1143` | `negative` |
| `Performance_Rating` | `-1.3568` | `-1.8856` | `negative` |
| `Distance_From_Home` | `-0.3788` | `-0.4706` | `negative` |
| `Education_Level` | `1.4022` | `-0.3969` | `negative` |

---

## 14. Model Limitations & Operational Considerations

- **Linear decision boundary: Assumes linear relationships between features and target; cannot capture complex non-linear interactions without polynomial expansion.**
- **Sensitive to feature scaling: Preprocessing scaling (StandardScaler) is mandatory to prevent features with large numeric scales from dominating regularization penalties.**
- **Probability calibration: Raw output probabilities should be monitored in production; extreme class imbalances may require threshold tuning for specific recall/precision targets.**

---

## 15. Reproducibility & Provenance Information

* **Random Seed:** `42`
* **Cross-Validation Splits:** `5 folds`
* **Screening Time:** `31.76s`
* **Tuning Time:** `51.77s`
* **Total Runtime:** `96.22s`
* **Serialized Model Pipeline:** `E:\ML MODEL\backend\artifacts\employee_turnover\model.joblib`
* **Serialized Run Provenance:** `E:\ML MODEL\backend\artifacts\employee_turnover\metadata.json`

---
*Report automatically generated by Universal ML Engine.*