# Universal ML Engine — Core Engine Verification Report

> **Milestone:** Milestone 2 Verification & Checkpoint  
> **Status:** 100% VERIFIED & BENCHMARK VALIDATED  
> **Date:** October 6, 2026  
> **Python Runtime:** Python 3.12.7 (64-bit)  
> **System:** Windows 11  

---

## 1. Executive Summary

The foundational headless Machine Learning Engine for **Universal ML Engine** is complete, runnable, and verified against both synthetic unit tests and real-world production datasets.

* **Dependencies:** All 12 required core and booster packages installed and import-verified.
* **Unit Tests:** **20 of 20 tests PASSED (100% pass rate)** in 16.17 seconds.
* **Real-World Benchmarks:** Both Classification (`employee_turnover.csv`) and Regression (`HousePricePrediction.csv`) ran through complete 5-fold cross-validation with zero data leakage.
* **Multi-Model Training:** Successfully trained and benchmarked 7 classification model families and 8 regression model families including Scikit-Learn baselines, LightGBM, XGBoost, and CatBoost.
* **Artifact Generation & Reloading:** Model pipelines serialized to `.joblib` with companion `metadata.json` files and verified for real-time inference reloading.

---

## 2. Dependency Verification Status

Python environment verified at `C:\Users\surve\AppData\Local\Programs\Python\Python312\python.exe`:

| Package | Version | Status | Architectural Role |
| :--- | :--- | :--- | :--- |
| **NumPy** | `2.5.3` | Installed & Verified | Vectorized tensor & numerical math |
| **Pandas** | `3.0.6` | Installed & Verified | Dataframe loading, profiling & sniffing |
| **SciPy** | `1.18.1` | Installed & Verified | Mathematical utilities & distributions |
| **Scikit-learn** | `1.9.1` | Installed & Verified | Pipelines, ColumnTransformers, CV, baselines |
| **Joblib** | `1.6.0` | Installed & Verified | Pipeline serialization & persistence |
| **Openpyxl** | `3.1.5` | Installed & Verified | Excel `.xlsx` ingestion |
| **Xlrd** | `2.0.2` | Installed & Verified | Legacy Excel `.xls` ingestion |
| **XGBoost** | `3.4.1` | Installed & Verified | Gradient boosted decision trees |
| **LightGBM** | `4.7.0` | Installed & Verified | Tabular gradient boosting |
| **CatBoost** | `1.2.10` | Installed & Verified | Categorical gradient boosting |
| **FLAML** | `2.7.0` | Installed & Verified | Fast AutoML screening (planned M3) |
| **Pytest** | `9.1.1` | Installed & Verified | Test execution framework |

---

## 3. Test Suite Verification Results

Command:
```powershell
python -m pytest backend/tests/ -v
```

**Results:** **20 Passed / 0 Failed / 0 Warnings** (Execution time: 16.17s)

| Test Module | Test Name | Status |
| :--- | :--- | :--- |
| `test_ingestion_and_validation.py` | `test_load_csv_valid` | **PASSED** |
| `test_ingestion_and_validation.py` | `test_load_csv_semicolon_delimited` | **PASSED** |
| `test_ingestion_and_validation.py` | `test_validation_drops_missing_target_and_duplicates` | **PASSED** |
| `test_ingestion_and_validation.py` | `test_validation_drops_constant_and_empty_columns` | **PASSED** |
| `test_ingestion_and_validation.py` | `test_validation_missing_target_column` | **PASSED** |
| `test_leakage_guard.py` | `test_leakage_guard_drops_id_columns` | **PASSED** |
| `test_leakage_guard.py` | `test_leakage_guard_drops_perfectly_correlated_features` | **PASSED** |
| `test_models_and_cv.py` | `test_model_registry_returns_models` | **PASSED** |
| `test_models_and_cv.py` | `test_cross_validation_runner_classification` | **PASSED** |
| `test_models_and_cv.py` | `test_leaderboard_ranking_deterministic` | **PASSED** |
| `test_orchestrator_end_to_end.py` | `test_orchestrator_end_to_end_classification` | **PASSED** |
| `test_orchestrator_end_to_end.py` | `test_orchestrator_end_to_end_regression` | **PASSED** |
| `test_orchestrator_end_to_end.py` | `test_orchestrator_reproducibility` | **PASSED** |
| `test_preprocessing_pipeline.py` | `test_preprocessing_pipeline_handles_missing_and_categories` | **PASSED** |
| `test_preprocessing_pipeline.py` | `test_preprocessing_handles_unseen_categories` | **PASSED** |
| `test_problem_detection.py` | `test_detect_binary_classification_boolean` | **PASSED** |
| `test_problem_detection.py` | `test_detect_binary_classification_strings` | **PASSED** |
| `test_problem_detection.py` | `test_detect_multiclass_classification` | **PASSED** |
| `test_problem_detection.py` | `test_detect_regression_continuous_floats` | **PASSED** |
| `test_problem_detection.py` | `test_detect_unknown_on_missing_column` | **PASSED** |

---

## 4. Real-World Dataset Benchmark Results

Execution command:
```powershell
python backend/run_real_datasets.py
```

### Dataset 1: Employee Turnover (Binary Classification)
* **File:** `employee_turnover.csv`
* **Target:** `Employee_Turnover`
* **Detected Problem:** `binary_classification` (Confidence: 1.00)
* **Partitions:** 720 Dev samples, 180 Holdout samples (80/20 split)
* **Exclusions Tracked:** 450 duplicate rows dropped transparently
* **Leaderboard (5-Fold Stratified CV):**

| Rank | Model Name | Primary Metric (CV F1) | CV Std | Fit Time | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Logistic Regression** | **0.8638** | **0.0247** | **0.19s** | **success** |
| 2 | CatBoost | 0.8523 | 0.0325 | 2.62s | success |
| 3 | XGBoost | 0.8440 | 0.0283 | 1.54s | success |
| 4 | HistGradientBoosting | 0.8372 | 0.0310 | 1.42s | success |
| 5 | Random Forest | 0.8345 | 0.0315 | 1.74s | success |
| 6 | Extra Trees | 0.8342 | 0.0234 | 1.43s | success |
| 7 | LightGBM | 0.8260 | 0.0231 | 0.72s | success |

* **Selected Model:** **Logistic Regression**
* **Holdout Evaluation Metrics (Untouched 20%):**
  * `accuracy`: 0.8778 (87.78%)
  * `balanced_accuracy`: 0.8772
  * `precision`: 0.8784
  * `recall`: 0.8778
  * `f1`: 0.8777
  * `f1_macro`: 0.8775
  * `roc_auc`: **0.9605**
* **Generated Artifacts:**
  * Pipeline: `backend/artifacts/employee_turnover/model.joblib` (4.6 KB)
  * Metadata: `backend/artifacts/employee_turnover/metadata.json` (4.2 KB)

---

### Dataset 2: House Price Prediction (Regression)
* **File:** `HousePricePrediction.csv`
* **Target:** `SalePrice`
* **Detected Problem:** `regression` (Confidence: 0.95)
* **Partitions:** 1,168 Dev samples, 292 Holdout samples (80/20 split)
* **Exclusions Tracked:** 1,459 rows with missing target values dropped; `Id` column identified and purged as an artificial identifier.
* **Leaderboard (5-Fold K-Fold CV):**

| Rank | Model Name | Primary Metric (CV R²) | CV Std | Fit Time | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **CatBoost** | **0.7843** | **0.0329** | **1.34s** | **success** |
| 2 | Extra Trees | 0.7671 | 0.0364 | 1.59s | success |
| 3 | LightGBM | 0.7669 | 0.0364 | 0.82s | success |
| 4 | HistGradientBoosting | 0.7656 | 0.0358 | 1.75s | success |
| 5 | Random Forest | 0.7450 | 0.0448 | 1.63s | success |
| 6 | XGBoost | 0.7275 | 0.0738 | 1.43s | success |
| 7 | Ridge | 0.5776 | 0.1014 | 0.12s | success |
| 8 | Linear Regression | 0.5737 | 0.1068 | 0.15s | success |

* **Selected Model:** **CatBoost Regressor**
* **Holdout Evaluation Metrics (Untouched 20%):**
  * `r2`: **0.8476** (84.76% variance explained)
  * `rmse`: \$34,189.74
  * `mae`: \$22,420.33
  * `mse`: 1,168,938,528.88
* **Generated Artifacts:**
  * Pipeline: `backend/artifacts/house_price/model.joblib` (124.2 KB)
  * Metadata: `backend/artifacts/house_price/metadata.json` (3.8 KB)

---

## 5. Machine Learning Quality & Safety Verifications

1. **Preprocessing Inside Pipeline:** Preprocessors (`ColumnTransformer`) are instantiated and fitted strictly inside each fold in `CrossValidationRunner`. No transformation occurs globally across train/validation splits.
2. **Zero Transformation Leakage:** Numeric imputers (medians), scalers, and categorical encoders (`OneHotEncoder`) learn statistics only from training folds.
3. **Strict Holdout Isolation:** The 20% holdout split is segregated prior to cross-validation and evaluated **exactly once** on the refitted best model. It is never used to rank models or choose hyperparameters.
4. **Fault Isolation:** Individual model candidate exceptions during cross-validation are caught, logged, and isolated without aborting the experiment.
5. **Reproducibility:** Confirmed by `test_orchestrator_reproducibility` with `random_state=42`, producing identical best model selection and CV scores across repeated runs.

---

## 6. Open-Source Components Actively Used

* **Scikit-Learn:** Core pipeline interfaces, splitters, baselines, and evaluation metrics.
* **XGBoost:** `XGBClassifier` and `XGBRegressor` actively trained and evaluated on both datasets.
* **LightGBM:** `LGBMClassifier` and `LGBMRegressor` actively trained and evaluated on both datasets.
* **CatBoost:** `CatBoostClassifier` and `CatBoostRegressor` actively trained; won Rank 1 in regression.
* **Joblib:** Used for pipeline serialization and deserialization.
* **Pandas / NumPy:** Tabular processing, profiling, and linear algebra.
* **FLAML:** Installed in runtime; integration into active training search space planned for Milestone 3.

---

## 7. Remaining Limitations & Next Steps

1. **Hyperparameter Tuning:** Current models use standardized baseline configurations; automated Bayesian tuning via Optuna and fast screening via FLAML are scheduled for Milestone 3.
2. **Explainability:** SHAP feature importance trees and local prediction waterfall explanations are scheduled for Milestone 4.
3. **Web API & UI:** The headless engine is currently executed via Python scripts; FastAPI endpoints (Milestone 5) and React + TypeScript SaaS UI (Milestone 6) will expose this functionality to end users.
