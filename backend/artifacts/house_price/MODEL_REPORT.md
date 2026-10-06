# Universal ML Engine — Model Governance & Decision Report

> **Dataset:** `HousePricePrediction.csv`  
> **Target Variable:** `SalePrice`  
> **Experiment ID:** `a368a718`  
> **Champion Model:** `CatBoost (Tuned)`  
> **Generated Date:** October 6, 2026  

---

## 1. Executive Summary

The Universal ML Engine automated pipeline processed `HousePricePrediction.csv` (1,460 rows, 12 features). The task was identified as **Regression** with 95.0% confidence.

**Champion Selection:** Selected 'CatBoost (Tuned)' as the champion model because it achieved the highest mean cross-validation r2 (0.7967 +/- 0.0000) on 5-fold development data, outperforming the runner-up 'CatBoost' (0.7843) by +0.0124 points.

* **Development 5-Fold CV Score:** `0.7967` (regression)  
* **Untouched 20% Holdout Performance:** `mae`: **23221.1238**, `mse`: **1308053502.5761**, `rmse`: **36167.0223**, `r2`: **0.8295**  
* **Total End-to-End Execution Time:** `136.91s`

---

## 2. Dataset Overview & Partitioning

| Dimension | Count / Proportion | Details |
| :--- | :--- | :--- |
| **Raw Samples** | `2,919` rows | Initial dataset size |
| **Cleaned Samples** | `1,460` rows | After removing missing targets / duplicate records |
| **Development Split (80%)** | `1,168` rows | Solely used for screening, CV, and tuning |
| **Final Holdout Split (20%)** | `292` rows | Untouched benchmark partition |
| **Total Features Retained** | `11` features | Cleaned predictive features |
| **Exclusions Logged** | `2` items | Audit trail of dropped columns/rows |

---

## 3. Dataset Intelligence & Health Diagnostic

### Health Score: **98.8/100** — Grade: `Excellent`
> *Dataset health is Excellent (98.8/100). The dataset exhibits strong completeness, clean structural integrity, and no critical leakage or imbalance risks.*

#### Quality Category Breakdown
| Category | Score | Weight | Assessment |
| :--- | :--- | :--- | :--- |
| **Completeness** | `100.0/100` | `25%` | No deductions |
| **Uniqueness & Structure** | `95.0/100` | `25%` | 1 artificial ID / index column(s) detected (-5.0 pts) |
| **Feature Quality** | `100.0/100` | `25%` | No deductions |
| **Target Integrity** | `100.0/100` | `25%` | No deductions |

#### Verified Clean Signals
- [x] Zero missing values detected across feature matrix.
- [x] Zero duplicate rows found in dataset.
- [x] No constant features detected.
- [x] Categorical features exhibit manageable cardinality.
- [x] No severe multicollinearity (|r| >= 0.85) detected between features.
- [x] Target variable exhibits zero direct leakage risks.
- [x] Target distribution is valid.

## 4. Data Quality Warnings & Risks

| Warning ID | Severity | Affected Column(s) | Reason | Recommended Action |
| :--- | :--- | :--- | :--- | :--- |
| `WARN_SUSPICIOUS_ID_Id` | **HIGH** | `Id` | Column 'Id' exhibits artificial identifier characteristics (ID pattern match or 100% unique key). | Drop column; training on artificial keys causes memorization and severe test overfitting. |

---

## 5. Problem Detection

* **Detected Problem Type:** `regression`
* **Detection Confidence:** `0.95`
* **Diagnostic Reasoning:** Target column is numeric with 663 unique continuous values (float: True, has fractional values: False).

---

## 6. Preprocessing Architecture & Anti-Leakage

All transformations are strictly encapsulated inside Scikit-Learn `ColumnTransformer` pipelines:
1. **Numerical Pipeline:** Median imputation followed by optional StandardScaler (applied to linear models).
2. **Categorical Pipeline:** Mode (most frequent) imputation followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
3. **Zero-Leakage Enforcement:** Preprocessors are fitted **solely inside training folds** during CV and inside Optuna trials. No transformations peek across validation or holdout boundaries.

---

## 7. Fast Candidate Screening (FLAML)

Evaluated multiple model families under strict time constraints (`30.67s` elapsed):

| Rank | Model Family | Development CV Score | Fit Time | Status |
| :--- | :--- | :--- | :--- | :--- |
| #1 | **CatBoost** | `0.7816` | `23.20s` | `success` |
| #2 | **LightGBM** | `0.7763` | `3.43s` | `success` |
| #3 | **XGBoost** | `0.7183` | `1.58s` | `success` |
| #4 | **Random Forest** | `0.6554` | `0.67s` | `success` |
| #5 | **Extra Trees** | `0.5948` | `1.03s` | `success` |

---

## 8. Bayesian Hyperparameter Optimization (Optuna)

Top candidate algorithms tuned using Tree-structured Parzen Estimator (`TPESampler`) with `MedianPruner` (`94.43s` elapsed):

| Model Candidate | Baseline CV Score | Tuned CV Score | Delta | Trials | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CatBoost** | `0.7843` | `0.7967` | **+0.0124** | `15` | `success` |
| **Extra Trees** | `0.7671` | `0.7710` | **+0.0039** | `15` | `success` |

#### Optimal Tuned Hyperparameters
```json
{
  "iterations": 150,
  "learning_rate": 0.15218560699686326,
  "depth": 7,
  "l2_leaf_reg": 6.2186601610043635
}
```

---

## 9. Development Cross-Validation Results

> **Evaluation Scheme:** 5-Fold Cross-Validation on Development Data (80% Partition)  
> **Primary Metric:** `r2`

| Rank | Model Name | Mean CV Score | CV Std Dev | Fit Time | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | **CatBoost (Tuned) 🏆** | `0.7967` | `+/- 0.0000` | `52.69s` | `success` |
| #2 | **CatBoost** | `0.7843` | `+/- 0.0329` | `1.47s` | `success` |
| #3 | **Extra Trees (Tuned)** | `0.7710` | `+/- 0.0000` | `39.72s` | `success` |
| #4 | **Extra Trees** | `0.7671` | `+/- 0.0364` | `2.14s` | `success` |
| #5 | **LightGBM** | `0.7669` | `+/- 0.0364` | `0.97s` | `success` |
| #6 | **HistGradientBoosting** | `0.7656` | `+/- 0.0358` | `2.71s` | `success` |
| #7 | **Random Forest** | `0.7450` | `+/- 0.0448` | `2.13s` | `success` |
| #8 | **XGBoost** | `0.7275` | `+/- 0.0738` | `1.41s` | `success` |
| #9 | **Ridge** | `0.5776` | `+/- 0.1014` | `0.13s` | `success` |
| #10 | **Linear Regression** | `0.5737` | `+/- 0.1068` | `0.19s` | `success` |

---

## 10. Final Champion Model Selection

### Champion: **CatBoost (Tuned)**
**Selection Rationale:** Selected 'CatBoost (Tuned)' as the champion model because it achieved the highest mean cross-validation r2 (0.7967 +/- 0.0000) on 5-fold development data, outperforming the runner-up 'CatBoost' (0.7843) by +0.0124 points.

The champion model was refitted on 100% of the development partition before proceeding to holdout verification.

---

## 11. Final Holdout Evaluation (Untouched 20%)

> [!IMPORTANT]
> The final 20% holdout split was completely segregated prior to screening, cross-validation, and tuning. It was evaluated exactly once to verify out-of-sample generalization without data leakage.

| Evaluation Metric | Holdout Value | Architectural Significance |
| :--- | :--- | :--- |
| **MAE** | `23221.1238` | Unbiased generalization score on test partition |
| **MSE** | `1308053502.5761` | Unbiased generalization score on test partition |
| **RMSE** | `36167.0223` | Unbiased generalization score on test partition |
| **R2** | `0.8295` | Unbiased generalization score on test partition |

---

## 12. Explainable AI & Global Feature Importance (SHAP)

* **Explainer Employed:** `shap_tree`
* **Diagnostic Summary:** Model explanations successfully computed via shap_tree. The primary global drivers are: TotalBsmtSF, YearBuilt, MSSubClass.

### Top Driving Features (Aggregated Raw Features)
| Rank | Feature Name | Mean |SHAP| Value | Relative Contribution |
| :--- | :--- | :--- | :--- |
| #1 | **TotalBsmtSF** | `26802.4993` | **24.8%** |
| #2 | **YearBuilt** | `19863.4507` | **18.4%** |
| #3 | **MSSubClass** | `18856.7342` | **17.4%** |
| #4 | **YearRemodAdd** | `13196.8327` | **12.2%** |
| #5 | **LotArea** | `12674.8865` | **11.7%** |
| #6 | **OverallCond** | `5701.1550` | **5.3%** |
| #7 | **Exterior1st** | `4241.0609` | **3.9%** |
| #8 | **MSZoning** | `3604.1383` | **3.3%** |
| #9 | **BldgType** | `1801.9727` | **1.7%** |
| #10 | **LotConfig** | `1055.0044` | **1.0%** |

## 13. Example Local Prediction Explanations

Local feature attribution for sample predictions:

### Sample #1
* **Predicted Output:** `150018.9817695474`
* **Base Expected Value (E[f(X)]):** `181438.3923`
* **Top Positive Drivers:** OverallCond, YearRemodAdd, Exterior1st_VinylSd
* **Top Negative Drivers:** MSSubClass, YearBuilt, LotArea

| Feature | Feature Value | SHAP Value | Impact Direction |
| :--- | :--- | :--- | :--- |
| `MSSubClass` | `20.0` | `-19496.2653` | `negative` |
| `YearBuilt` | `1963.0` | `-13964.1501` | `negative` |
| `LotArea` | `8414.0` | `-11185.9627` | `negative` |
| `OverallCond` | `8.0` | `+10060.2403` | `positive` |
| `YearRemodAdd` | `2003.0` | `+8118.7287` | `positive` |

### Sample #2
* **Predicted Output:** `313963.7325974433`
* **Base Expected Value (E[f(X)]):** `181438.3923`
* **Top Positive Drivers:** TotalBsmtSF, MSSubClass, YearBuilt
* **Top Negative Drivers:** OverallCond, Exterior1st_HdBoard, Exterior1st_BrkFace

| Feature | Feature Value | SHAP Value | Impact Direction |
| :--- | :--- | :--- | :--- |
| `TotalBsmtSF` | `1463.0` | `+44977.6610` | `positive` |
| `MSSubClass` | `60.0` | `+42158.9391` | `positive` |
| `YearBuilt` | `1994.0` | `+20850.2713` | `positive` |
| `LotArea` | `12256.0` | `+13218.4809` | `positive` |
| `YearRemodAdd` | `1995.0` | `+11812.0623` | `positive` |

---

## 14. Model Limitations & Operational Considerations

- **Tree extrapolation limitation: Oblivious decision trees cannot extrapolate beyond minimum/maximum feature values observed in training data.**
- **Inference overhead: Categorical hashing and deep ensembles require marginally higher single-record latency compared to simple linear models.**

---

## 15. Reproducibility & Provenance Information

* **Random Seed:** `42`
* **Cross-Validation Splits:** `5 folds`
* **Screening Time:** `30.67s`
* **Tuning Time:** `94.43s`
* **Total Runtime:** `136.91s`
* **Serialized Model Pipeline:** `E:\ML MODEL\backend\artifacts\house_price\model.joblib`
* **Serialized Run Provenance:** `E:\ML MODEL\backend\artifacts\house_price\metadata.json`

---
*Report automatically generated by Universal ML Engine.*