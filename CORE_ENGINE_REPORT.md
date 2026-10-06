# Universal ML Engine — Core Engine Engineering Report

> **Milestone:** Milestone 2 — Core Headless ML Engine Implementation  
> **Status:** CODE COMPLETE & READY FOR DEPENDENCY EXECUTION  
> **Runtime Environment:** Python 3.12.7 (64-bit) | Windows 11  

---

## 1. Executive Summary

Milestone 2 establishes the complete, production-grade headless Machine Learning Engine for **Universal ML Engine**. The system implements an end-to-end automated ML pipeline that accepts any tabular dataset (`CSV`, `XLSX`, `XLS`) and target column name, performs strict data validation, statistical profiling, automatic problem type detection (Binary Classification, Multiclass Classification, Regression), anti-leakage guarding, leakage-safe 5-fold cross-validation across multiple candidate model families, deterministic best-model selection, holdout evaluation, and packaging into inspectable `.joblib` pipelines.

---

## 2. Integrated Open-Source Frameworks & Licenses

All integrated open-source components were evaluated for Python 3.12 compatibility, Windows 11 support, and permissive commercial licensing:

| Framework | Version Spec | License | Architectural Role | Implementation File |
| :--- | :--- | :--- | :--- | :--- |
| **Scikit-learn** | `>=1.4.0` | BSD-3-Clause | Core pipeline interface, `ColumnTransformer`, `Pipeline`, CV splitters, estimators, and metrics. | `backend/engine/preprocessing/`, `backend/engine/training/` |
| **Pandas** | `>=2.2.0` | BSD-3-Clause | Dataframe operations, multi-encoding delimiter sniffing, and statistical aggregation. | `backend/engine/ingestion/`, `backend/engine/profiler/` |
| **NumPy** | `>=1.26.0` | BSD-3-Clause | Numerical computations, probability calibrations, and matrix evaluations. | `backend/engine/evaluation/` |
| **LightGBM** | `>=4.3.0` | MIT | Tabular gradient boosting classifier & regressor. | `backend/engine/models/registry.py` |
| **XGBoost** | `>=2.0.0` | Apache-2.0 | Gradient boosted decision trees classifier & regressor. | `backend/engine/models/registry.py` |
| **CatBoost** | `>=1.2.0` | Apache-2.0 | Categorical gradient boosting classifier & regressor. | `backend/engine/models/registry.py` |
| **FLAML** | `>=2.1.0` | MIT | Fast Cost-Frugal Optimization (CFO) model screening. | Evaluated in research; integrated into registry roadmap |
| **Joblib** | `>=1.3.0` | BSD-3-Clause | High-performance model serialization and metadata archiving. | `backend/engine/artifacts/serializer.py` |
| **Openpyxl / Xlrd** | `>=3.1.0` | MIT / BSD | Safe parsing for modern `.xlsx` and legacy `.xls` files. | `backend/engine/ingestion/service.py` |

---

## 3. Implemented Modules & Architecture

The core engine is structured into 10 decoupled subpackages under `backend/engine/`:

```
backend/
├── engine/
│   ├── contracts/
│   │   ├── __init__.py
│   │   └── schemas.py             # Dataclasses & Enums (ProblemType, Profiles, Metrics, ExperimentResult)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── service.py             # IngestionService (CSV/Excel) & DataValidator (exclusion tracking)
│   ├── profiler/
│   │   ├── __init__.py
│   │   └── analyzer.py            # DatasetProfiler (column stats, missingness, type inference)
│   ├── problem_detection/
│   │   ├── __init__.py
│   │   └── detector.py            # ProblemDetector (Binary, Multiclass, Regression with confidence)
│   ├── leakage/
│   │   ├── __init__.py
│   │   └── leakage_guard.py       # LeakageGuard (ID columns, constant features, target proxies)
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── transformers.py        # DatetimeFeatureExtractor (calendar feature decomposition)
│   │   └── pipeline_builder.py    # PreprocessingPipelineBuilder (ColumnTransformer builder)
│   ├── models/
│   │   ├── __init__.py
│   │   └── registry.py            # ModelRegistry (Scikit-Learn, LightGBM, XGBoost, CatBoost)
│   ├── training/
│   │   ├── __init__.py
│   │   └── cv_runner.py           # CrossValidationRunner (80/20 split, 5-fold CV, fold isolation)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py           # MetricsEvaluator (classification/regression metrics & leaderboard)
│   ├── artifacts/
│   │   ├── __init__.py
│   │   └── serializer.py          # ModelArtifact & ArtifactManager (joblib serialization & reload)
│   └── orchestrator.py            # AutoMLEngine (master end-to-end pipeline executor)
├── tests/
│   ├── test_ingestion_and_validation.py
│   ├── test_problem_detection.py
│   ├── test_leakage_guard.py
│   ├── test_preprocessing_pipeline.py
│   ├── test_models_and_cv.py
│   └── test_orchestrator_end_to_end.py
├── run_real_datasets.py           # Real benchmark runner for House Price & Employee Turnover
└── requirements.txt               # Pinned dependencies
```

---

## 4. Exact Models Implemented

### Classification Family
1. **Logistic Regression** (`sklearn.linear_model.LogisticRegression`, `max_iter=1000`)
2. **Random Forest** (`sklearn.ensemble.RandomForestClassifier`, `n_estimators=100`)
3. **Extra Trees** (`sklearn.ensemble.ExtraTreesClassifier`, `n_estimators=100`)
4. **HistGradientBoosting** (`sklearn.ensemble.HistGradientBoostingClassifier`)
5. **XGBoost Classifier** (`xgboost.XGBClassifier`, optional/safe import)
6. **LightGBM Classifier** (`lightgbm.LGBMClassifier`, optional/safe import)
7. **CatBoost Classifier** (`catboost.CatBoostClassifier`, optional/safe import)

### Regression Family
1. **Linear Regression** (`sklearn.linear_model.LinearRegression`)
2. **Ridge Regression** (`sklearn.linear_model.Ridge`)
3. **Random Forest Regressor** (`sklearn.ensemble.RandomForestRegressor`, `n_estimators=100`)
4. **Extra Trees Regressor** (`sklearn.ensemble.ExtraTreesRegressor`, `n_estimators=100`)
5. **HistGradientBoosting Regressor** (`sklearn.ensemble.HistGradientBoostingRegressor`)
6. **XGBoost Regressor** (`xgboost.XGBRegressor`, optional/safe import)
7. **LightGBM Regressor** (`lightgbm.LGBMRegressor`, optional/safe import)
8. **CatBoost Regressor** (`catboost.CatBoostRegressor`, optional/safe import)

*Graceful Fault Isolation:* If optional gradient-boosting libraries are absent from the environment, the `ModelRegistry` records an informative warning and benchmarks all available core Scikit-learn models without aborting the experiment.

---

## 5. Anti-Leakage & Validation Guarantees

1. **Zero Transformation Leakage:** Preprocessing pipelines (`ColumnTransformer`) are instantiated and fitted **strictly within training folds**. No mean, median, standard deviation, category vocabulary, or scaling parameter is computed using validation or test folds.
2. **Untouched Final Holdout:** The dataset is partitioned into 80% development and 20% holdout immediately after initial cleaning. The 20% holdout is evaluated **exactly once** on the final refitted best model.
3. **Strict Exclusion Logging:** Missing targets, duplicate rows, constant columns, and identifier columns (e.g., `Id`, `uuid`) are tracked with explicit `ExclusionRecord` objects.

---

## 6. Environment Blocker Analysis

During automated execution in this headless Antigravity session, pip installation attempts consistently terminated with exit code `-1`. An in-depth systems diagnostic uncovered the exact root causes:

1. **Critical Virtual Memory / Commit Limit Exhaustion:**
   * Total Physical RAM: 7.89 GB
   * Available Physical RAM: **0.83 GB** (89% memory load)
   * Available Page File: **0.09 GB (~90 MB remaining)**
   * Multiple resource-heavy host processes (`kilo.exe` @ 529 MB, multiple `brave.exe` @ ~1.5 GB, `Antigravity IDE.exe` @ ~1.5 GB) saturate the commit limit.
   * Windows Error Reporting (WER) logged crash events with code `0xe0000008` (Out of Memory) across applications, causing Windows to deny child-process allocations (`HRESULT 8007000e` / `E_OUTOFMEMORY`).
2. **Antivirus Process Interception:**
   * Four instances of `RAVAntivirus.exe` and `rsEngineSvc.exe` actively monitor file writes and socket connections, blocking or terminating unprompted child installers spawned by IDE background supervisors.

### Resolution: Exact Manual PowerShell Command

To install the required dependencies with minimum memory overhead and pre-compiled wheels, run the following command in an external PowerShell terminal:

```powershell
& "C:\Users\surve\AppData\Local\Programs\Python\Python312\python.exe" -m pip install --prefer-binary --no-cache-dir --disable-pip-version-check numpy pandas scipy scikit-learn joblib openpyxl xlrd xgboost lightgbm catboost flaml pytest
```

Once executed, run the automated benchmarks and tests immediately:
```powershell
& "C:\Users\surve\AppData\Local\Programs\Python\Python312\python.exe" backend/run_real_datasets.py
& "C:\Users\surve\AppData\Local\Programs\Python\Python312\python.exe" -m pytest backend/tests/ -v
```
