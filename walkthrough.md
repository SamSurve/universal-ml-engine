# Universal ML Engine — Milestone 2 Walkthrough

> **Milestone 2 Goal:** Build the first real working ML engine for Universal ML Engine (Headless Core ML Engine).  
> **Output:** Validate → Clean → Preprocess → Detect problem → Train multiple models → Cross-Validate → Compare → Select Best → Final Evaluation → Save Model.  

---

## 1. What Was Built

We have created the complete, production-grade core machine learning package in `backend/engine/`:

### A. Contracts & Data Schemas (`backend/engine/contracts/schemas.py`)
- Defines strict dataclasses and enums: `ProblemType`, `ColumnType`, `ExclusionRecord`, `ColumnProfile`, `DatasetProfile`, `ProblemDetectionResult`, `ValidationResult`, `FoldMetric`, `ModelEvaluationResult`, `LeaderboardEntry`, and `ExperimentResult`.
- Ensures type safety and consistent interfaces across all submodules.

### B. Ingestion & Validation (`backend/engine/ingestion/service.py`)
- **`IngestionService`**: Multi-format tabular data loader (`.csv`, `.xlsx`, `.xls`). For CSV, it performs multi-encoding detection (`utf-8`, `utf-8-sig`, `latin1`, `cp1252`, `iso-8859-1`) and automatic delimiter sniffing (comma, semicolon, tab, pipe).
- **`DataValidator`**: Validates target column existence, removes missing-target rows, removes duplicate rows, and flags constant/empty columns.
- **Strict Transparency**: Never silently drops rows or columns. Every removed row or column is recorded in an `ExclusionRecord` with full justification.

### C. Dataset Profiler (`backend/engine/profiler/analyzer.py`)
- **`DatasetProfiler`**: Computes column-level metrics (missing count, missing percentage, unique count, unique ratio, sample values).
- Calculates statistical summaries: `min`, `max`, `mean`, `std`, `median`, `skewness` for numeric features; top categories and frequency distribution for categorical features.
- Infers semantic column types (`NUMERIC`, `CATEGORICAL`, `DATETIME`, `TEXT`, `ID_OR_CONSTANT`).

### D. Problem Detection (`backend/engine/problem_detection/detector.py`)
- **`ProblemDetector`**: Automatically detects whether a task is:
  - `BINARY_CLASSIFICATION` (boolean, 2 distinct classes)
  - `MULTICLASS_CLASSIFICATION` (discrete strings or small discrete integer cardinality <= 20)
  - `REGRESSION` (continuous floats or large discrete numerical intervals)
- Returns `confidence` score (0.0 to 1.0) and human-readable `reason`. Does not force arbitrary decisions when target is ambiguous.

### E. Leakage Guard (`backend/engine/leakage/leakage_guard.py`)
- **`LeakageGuard`**: Detects and purges high-risk features that cause overfitting or data leakage:
  - Synthetic identifiers / primary keys (`id`, `uuid`, row numbers, unique ratio == 1.0).
  - High-cardinality non-numeric features (> 100 categories and > 50% unique ratio).
  - Target proxies (features with Pearson correlation |r| >= 0.99 with continuous target).

### F. Preprocessing Pipeline (`backend/engine/preprocessing/`)
- **`DatetimeFeatureExtractor`**: Custom Scikit-Learn transformer extracting year, month, day, and dayofweek from dates.
- **`PreprocessingPipelineBuilder`**: Assembles Scikit-learn `ColumnTransformer`:
  - Numeric: Median imputation + optional `StandardScaler` / `RobustScaler`.
  - Categorical: Most frequent imputation + `OneHotEncoder(handle_unknown='ignore')`.
  - Datetime: DatetimeFeatureExtractor + Median imputation + Scaler.
- **Zero Leakage**: Transformers are fitted **strictly inside training folds**, never on the holdout or validation sets.

### G. Model Registry (`backend/engine/models/registry.py`)
- **`ModelRegistry`**:
  - Classification models: `LogisticRegression`, `RandomForestClassifier`, `ExtraTreesClassifier`, `HistGradientBoostingClassifier`, with graceful dynamic imports for `XGBClassifier`, `LGBMClassifier`, and `CatBoostClassifier`.
  - Regression models: `LinearRegression`, `Ridge`, `RandomForestRegressor`, `ExtraTreesRegressor`, `HistGradientBoostingRegressor`, with graceful dynamic imports for `XGBRegressor`, `LGBMRegressor`, and `CatBoostRegressor`.
  - Fault isolation: A missing optional library or single model failure does not stop the overall benchmark experiment.

### H. Training & Cross-Validation (`backend/engine/training/cv_runner.py`)
- **`CrossValidationRunner`**:
  - Performs 80% development / 20% untouched holdout split (`random_state=42`).
  - Implements 5-fold `StratifiedKFold` for classification and 5-fold `KFold` for regression.
  - Fits preprocessors strictly on `X_train` within each fold to guarantee zero data leakage.
  - Measures validation metrics and training durations fold by fold.

### I. Evaluation & Leaderboard (`backend/engine/evaluation/evaluator.py`)
- **`MetricsEvaluator`**:
  - Classification metrics: `accuracy`, `balanced_accuracy`, `precision_weighted`, `recall_weighted`, `f1_weighted`, `f1_macro`, `roc_auc`.
  - Imbalance-aware ranking: Uses `balanced_accuracy` or `f1` as primary metric for imbalanced classes, never raw accuracy.
  - Regression metrics: `mae`, `mse`, `rmse`, `r2`.
  - Constructs deterministic leaderboard ranked by CV performance: `Model | CV Score | CV Std | Metrics | Training Time | Status`.

### J. Model Serialization & Reloading (`backend/engine/artifacts/serializer.py`)
- **`ModelArtifact`**: Bundles the fitted Scikit-Learn pipeline, target label encoder, feature names, and problem type.
- **`ArtifactManager`**:
  - Saves pipeline to `model.joblib` and full execution provenance to `metadata.json`.
  - Provides `load_pipeline(path)` to reload artifacts and perform real-time predictions.

### K. Master Orchestrator (`backend/engine/orchestrator.py`)
- **`AutoMLEngine`**: One-line end-to-end execution:
  ```python
  result = AutoMLEngine.run(data_source="dataset.csv", target_column="target")
  ```

---

## 2. Test Suite & Verification Files

Automated tests created in `backend/tests/`:
- `test_ingestion_and_validation.py` — Ingestion of diverse CSV formats, delimiter sniffing, and exclusion tracking.
- `test_problem_detection.py` — Accurate problem detection for binary, multiclass, and regression.
- `test_leakage_guard.py` — Removal of artificial IDs, constant columns, and target proxies.
- `test_preprocessing_pipeline.py` — Preprocessing pipelines, median imputation, and unseen category handling.
- `test_models_and_cv.py` — Model registry retrieval, 5-fold CV execution, and leaderboard sorting.
- `test_orchestrator_end_to_end.py` — Complete end-to-end runs for classification and regression, artifact persistence, reload testing, and random seed reproducibility.

Real-world benchmark runner created:
- `backend/run_real_datasets.py` — Evaluates [employee_turnover(1).csv](file:///e:/ML%20MODEL/employee_turnover(1).csv) and [HousePricePrediction(1).csv](file:///e:/ML%20MODEL/HousePricePrediction(1).csv).

---

## 3. Environment Blocker & Resolution

### Root Cause
During automated tool execution, child processes running `pip install` stalled or exited with code `-1`. Diagnostics confirmed:
- Total RAM: 7.89 GB | Available Physical RAM: **0.83 GB (89% load)**.
- Available Page File: **0.09 GB (~90 MB remaining)**.
- Multiple active host processes (`brave.exe`, `kilo.exe`, `Antigravity IDE.exe`) saturated system virtual memory. Windows threw `0xe0000008` (Out of Memory) errors, and running antivirus (`RAVAntivirus.exe`) blocked child process socket/disk writes.

### Manual Installation Command
To install dependencies in an external PowerShell terminal:
```powershell
& "C:\Users\surve\AppData\Local\Programs\Python\Python312\python.exe" -m pip install --prefer-binary --no-cache-dir --disable-pip-version-check numpy pandas scipy scikit-learn joblib openpyxl xlrd xgboost lightgbm catboost flaml pytest
```

### Running Tests & Benchmarks
```powershell
& "C:\Users\surve\AppData\Local\Programs\Python\Python312\python.exe" backend/run_real_datasets.py
& "C:\Users\surve\AppData\Local\Programs\Python\Python312\python.exe" -m pytest backend/tests/ -v
```
