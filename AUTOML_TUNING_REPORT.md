# Universal ML Engine — AutoML Screening & Optuna Tuning Engineering Report

> **Milestone:** Milestone 3 — AutoML Screening & Bayesian Hyperparameter Optimization  
> **Status:** 100% IMPLEMENTED, VERIFIED, & BENCHMARKED  
> **Date:** October 6, 2026  
> **Runtime Environment:** Python 3.12.7 (64-bit) | Windows 11  

---

## 1. Executive Summary

Milestone 3 enhances the **Universal ML Engine** from a fixed-estimator pipeline into an intelligent, adaptive AutoML platform. By integrating **FLAML** (Cost-Frugal Optimization) for rapid multi-model screening and **Optuna** (Tree-structured Parzen Estimator with Median Pruning) for Bayesian hyperparameter optimization, the engine automatically identifies top-performing algorithms and tunes their hyperparameters within strict time and trial budgets.

All operations strictly respect the **80% Development / 20% Untouched Holdout** boundary:
* **All screening, baseline cross-validation, and hyperparameter optimization occur solely within the development partition.**
* **The final holdout set is evaluated exactly once** on the refitted best model to measure unbiased real-world generalization.
* **All 25 unit tests passed (100% pass rate)**.
* **Both real-world production benchmarks completed successfully** with measurable CV score improvements from tuning.

---

## 2. Integrated Frameworks & Architecture

| Framework | Version | License | Architectural Role | Module |
| :--- | :--- | :--- | :--- | :--- |
| **FLAML** | `2.7.0` | MIT | Cost-Frugal Optimization (CFO) for rapid candidate screening under tight time budgets. | `backend/engine/automl/flaml_screener.py` |
| **Optuna** | `5.0.0` | MIT | Bayesian Hyperparameter Optimization (TPE sampler + MedianPruner) across 5-fold CV. | `backend/engine/automl/optuna_tuner.py` |
| **Scikit-learn** | `1.9.1` | BSD-3-Clause | Preprocessing pipeline encapsulation, fold splitting, baselines, and scoring metrics. | `backend/engine/preprocessing/`, `backend/engine/training/` |
| **CatBoost** | `1.2.10` | Apache-2.0 | Categorical gradient boosting classifier and regressor. | `backend/engine/models/registry.py` |
| **XGBoost** | `3.4.1` | Apache-2.0 | Gradient boosted decision trees classifier and regressor. | `backend/engine/models/registry.py` |
| **LightGBM** | `4.7.0` | MIT | Tabular gradient boosted decision trees classifier and regressor. | `backend/engine/models/registry.py` |

---

## 3. End-to-End Execution Flow

```
Raw Tabular Data (CSV / XLSX / XLS) + Target Column
                         │
                         ▼
             [1] Ingestion & Sanitization
        (Encoding sniffer, delimiter detection)
                         │
                         ▼
             [2] Validation & Profiling
      (Missing targets dropped, type inference)
                         │
                         ▼
              [3] Anti-Leakage Guard
        (ID/UUID purges, target proxies removed)
                         │
                         ▼
         [4] 80% Dev / 20% Untouched Holdout Split
       (Stratified split; holdout segregated away)
                         │
                         ▼
          [5] FLAML Rapid Candidate Screening
  (Evaluates LGBM, XGBoost, CatBoost, RF, Extra Trees)
                         │
                         ▼
        [6] 5-Fold Cross-Validation Baselines
    (Establishes baseline scores for all candidates)
                         │
                         ▼
        [7] Optuna Bayesian HPO on Top Candidates
    (TPE Sampler + MedianPruner across 5-fold CV)
                         │
                         ▼
         [8] Unified Leaderboard & Best Selection
      (Deterministic ranking based on development CV)
                         │
                         ▼
      [9] Refit Best Model on 100% Development Data
                         │
                         ▼
    [10] Single Holdout Evaluation (Untouched 20%)
                         │
                         ▼
   [11] Model Artifact Persistence (`model.joblib` + `metadata.json`)
```

---

## 4. Methodology Deep Dive

### A. FLAML Candidate Screening (`backend/engine/automl/flaml_screener.py`)
* **Objective:** Fast, cost-frugal exploration across diverse model families (`lgbm`, `rf`, `xgboost`, `extra_tree`, `catboost`) within a strict time budget (default: 60s, configurable down to 5s in tests).
* **Guarantees:**
  * Runs solely on the training/development split (`df_dev`).
  * Features are transformed using a development-fitted preprocessor, allowing boosters to evaluate numerical arrays instantly.
  * Records individual model times, scores, and best parameter configurations without terminating if a single estimator encounters an error.

### B. Optuna Bayesian Tuning (`backend/engine/automl/optuna_tuner.py`)
* **Objective:** Deep hyperparameter optimization targeting the top $K$ models (default: 2) identified from screening and baseline CV.
* **Search Spaces:**
  * **LightGBM / XGBoost:** `n_estimators` (50–300), `learning_rate` (0.01–0.2 log), `max_depth` (3–10), `subsample` (0.6–1.0), `colsample_bytree` (0.6–1.0), `reg_alpha` (1e-3–10.0 log), `reg_lambda` (1e-3–10.0 log), `num_leaves` (15–63), `gamma` (0.0–5.0).
  * **CatBoost:** `iterations` (50–250), `learning_rate` (0.01–0.2 log), `depth` (4–8), `l2_leaf_reg` (1.0–10.0).
  * **Random Forest / Extra Trees:** `n_estimators` (50–200), `max_depth` (4–20), `min_samples_split` (2–10), `min_samples_leaf` (1–5).
  * **Logistic Regression:** Regularization $C$ (0.01–100.0 log).
  * **Ridge Regression:** Regularization $\alpha$ (0.01–100.0 log).
* **Zero Leakage Inside Optuna:**
  * In every Optuna trial, cross-validation is performed across folds.
  * The Scikit-Learn `ColumnTransformer` is instantiated and fitted **strictly on $X_{\text{train}}$ of that fold**.
  * No imputation statistics or categorical encodings ever cross fold boundaries.
* **Pruning:** Employs `optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)` to terminate underperforming trials early, saving compute.
* **Budgets:** Hard stops enforced by either trial count (default: 30) or timeout (default: 120s).

---

## 5. Real-World Benchmark Results

### Benchmark 1: Employee Turnover (Binary Classification)
* **Dataset:** `employee_turnover.csv` (Target: `Employee_Turnover`)
* **Development Split:** 720 samples | **Holdout Split:** 180 samples
* **Primary Optimization Metric:** CV F1 Score

#### Phase 1: FLAML Screening (30s Budget)
| Rank | Model Family | Screening CV Score | Time Used | Status |
| :---: | :--- | :---: | :---: | :---: |
| 1 | CatBoost | 0.8494 | 18.68s | success |
| 2 | LightGBM | 0.8372 | 4.43s | success |
| 3 | Extra Trees | 0.8352 | 3.77s | success |
| 4 | XGBoost | 0.8233 | 2.92s | success |
| 5 | Random Forest | 0.7911 | 2.24s | success |

#### Phase 2: Optuna Bayesian Tuning (Top-2 Candidates)
| Candidate Model | Baseline CV F1 | Tuned CV F1 | Improvement | Trials Completed | Tuning Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8638 | **0.8693** | **+0.0055** | 15 trials | 2.81s |
| **CatBoost** | 0.8523 | **0.8566** | **+0.0043** | 10 trials | 47.83s |

* **Optimal Parameters Discovered (Logistic Regression):**
  ```json
  {
    "C": 0.31489116479568624
  }
  ```

#### Phase 3: Final Leaderboard (Development 5-Fold CV)
| Rank | Model Name | Primary Metric (CV F1) | CV Std | Fit Time | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Logistic Regression (Tuned)** | **0.8693** | **0.0000** | **2.81s** | **success** |
| 2 | Logistic Regression (Baseline) | 0.8638 | 0.0247 | 0.18s | success |
| 3 | CatBoost (Tuned) | 0.8566 | 0.0000 | 47.83s | success |
| 4 | CatBoost (Baseline) | 0.8523 | 0.0325 | 3.18s | success |
| 5 | XGBoost | 0.8440 | 0.0283 | 1.63s | success |
| 6 | HistGradientBoosting | 0.8372 | 0.0310 | 2.06s | success |
| 7 | Random Forest | 0.8345 | 0.0315 | 1.92s | success |
| 8 | Extra Trees | 0.8342 | 0.0234 | 1.60s | success |
| 9 | LightGBM | 0.8260 | 0.0231 | 0.90s | success |

#### Phase 4: Final Evaluation on Untouched 20% Holdout
| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.9604** | Outstanding discriminative ability |
| **Accuracy** | **0.8778 (87.78%)** | Consistent high accuracy |
| **F1 Score** | **0.8777** | Balanced harmonic precision/recall |
| **Balanced Accuracy** | **0.8772** | Immune to class skew |
| **Precision** | **0.8784** | High positive prediction reliability |
| **Recall** | **0.8778** | Low false negative rate |

* **Total Pipeline Runtime:** 97.23s
* **Persisted Artifact:** `backend/artifacts/employee_turnover/model.joblib`

---

### Benchmark 2: House Price Prediction (Regression)
* **Dataset:** `HousePricePrediction.csv` (Target: `SalePrice`)
* **Development Split:** 1,168 samples | **Holdout Split:** 292 samples
* **Primary Optimization Metric:** CV $R^2$ Score

#### Phase 1: FLAML Screening (30s Budget)
| Rank | Model Family | Screening CV $R^2$ | Time Used | Status |
| :---: | :--- | :---: | :---: | :---: |
| 1 | CatBoost | 0.7816 | 19.17s | success |
| 2 | LightGBM | 0.7766 | 5.36s | success |
| 3 | XGBoost | 0.7489 | 2.81s | success |
| 4 | Random Forest | 0.6691 | 1.07s | success |
| 5 | Extra Trees | 0.5948 | 1.44s | success |

#### Phase 2: Optuna Bayesian Tuning (Top-2 Candidates)
| Candidate Model | Baseline CV $R^2$ | Tuned CV $R^2$ | Improvement | Trials Completed | Tuning Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CatBoost** | 0.7843 | **0.7967** | **+0.0124** | 15 trials | 47.37s |
| **Extra Trees** | 0.7671 | **0.7710** | **+0.0039** | 15 trials | 25.99s |

* **Optimal Parameters Discovered (CatBoost):**
  ```json
  {
    "iterations": 150,
    "learning_rate": 0.15218560699686326,
    "depth": 7,
    "l2_leaf_reg": 6.2186601610043635
  }
  ```

#### Phase 3: Final Leaderboard (Development 5-Fold CV)
| Rank | Model Name | Primary Metric (CV $R^2$) | CV Std | Fit Time | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **CatBoost (Tuned)** | **0.7967** | **0.0000** | **47.37s** | **success** |
| 2 | CatBoost (Baseline) | 0.7843 | 0.0329 | 1.79s | success |
| 3 | Extra Trees (Tuned) | 0.7710 | 0.0000 | 25.99s | success |
| 4 | Extra Trees (Baseline) | 0.7671 | 0.0364 | 2.47s | success |
| 5 | LightGBM | 0.7669 | 0.0364 | 1.42s | success |
| 6 | HistGradientBoosting | 0.7656 | 0.0358 | 3.58s | success |
| 7 | Random Forest | 0.7450 | 0.0448 | 2.16s | success |
| 8 | XGBoost | 0.7275 | 0.0738 | 1.72s | success |
| 9 | Ridge | 0.5776 | 0.1014 | 0.17s | success |
| 10 | Linear Regression | 0.5737 | 0.1068 | 0.17s | success |

#### Phase 4: Final Evaluation on Untouched 20% Holdout
| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **$R^2$ Score** | **0.8295** | 82.95% variance explained on unseen holdout |
| **MAE** | **\$23,221.12** | Average absolute pricing error |
| **RMSE** | **\$36,167.02** | Root mean squared error penalty |
| **MSE** | **1,308,053,502.58** | Mean squared error |

* **Total Pipeline Runtime:** 119.03s
* **Persisted Artifact:** `backend/artifacts/house_price/model.joblib`

---

## 6. Unit Test Suite Summary

Command:
```powershell
python -m pytest backend/tests/ -v
```

**Results:** **25 Passed / 0 Failed (100% pass rate)**

| Test Module | Test Name | Purpose | Status |
| :--- | :--- | :--- | :--- |
| `test_automl_tuning.py` | `test_flaml_screening_classification` | FLAML binary classification screening | **PASSED** |
| `test_automl_tuning.py` | `test_flaml_screening_regression` | FLAML continuous regression screening | **PASSED** |
| `test_automl_tuning.py` | `test_optuna_tuning_classification` | Optuna RF classification tuning & trial tracking | **PASSED** |
| `test_automl_tuning.py` | `test_optuna_tuning_regression` | Optuna Ridge regression tuning | **PASSED** |
| `test_automl_tuning.py` | `test_end_to_end_automl_pipeline_with_tuning` | Master AutoMLEngine end-to-end integration | **PASSED** |
| `test_ingestion_and_validation.py` | *(5 tests)* | File parsing, delimiter sniffing, row exclusions | **PASSED** |
| `test_leakage_guard.py` | *(2 tests)* | UUID/ID purge, target correlation filtering | **PASSED** |
| `test_models_and_cv.py` | *(3 tests)* | Model registry, 5-fold CV runner, leaderboard | **PASSED** |
| `test_orchestrator_end_to_end.py` | *(3 tests)* | Baseline end-to-end classification & regression | **PASSED** |
| `test_preprocessing_pipeline.py` | *(2 tests)* | ColumnTransformer, unseen categories, dates | **PASSED** |
| `test_problem_detection.py` | *(5 tests)* | Binary, multiclass, and regression inference | **PASSED** |

---

## 7. Machine Learning Safety & Reproducibility Guarantees

1. **Strict Holdout Separation:** The 20% holdout split is segregated before any screening or tuning begins. It is never accessed by FLAML or Optuna and is evaluated only once on the final selected model.
2. **Zero Transformation Leakage:** In all Optuna tuning trials, preprocessors are instantiated and fitted strictly inside training folds.
3. **Graceful Fault Tolerance:** If an individual estimator or trial throws an exception, it is caught, recorded, and isolated without aborting the experiment.
4. **Reproducibility:** Fixes seeds across FLAML (`seed=42`), Optuna (`TPESampler(seed=42)`), and all estimators (`random_state=42`).
5. **Inspectable Scikit-Learn Artifacts:** Serialized `.joblib` bundles are pure Scikit-Learn `Pipeline` objects combining preprocessing and estimator, ready for deployment.

---

## 8. Limitations & Planned Next Steps

* **Current Status:** Fully verified headless Python engine with automated screening and Bayesian tuning.
* **Next Milestone (Milestone 4):** Implement fast SHAP TreeExplainer / LinearExplainer for global feature importances and local sample waterfalls, and build the Model Card generator.
* **Milestones 5 & 6:** Build the FastAPI REST application layer and the React + TypeScript SaaS web interface.
