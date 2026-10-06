# Universal ML Engine

A production-oriented AutoML platform engineered for automated end-to-end tabular machine learning.

Universal ML Engine automatically validates raw tabular data (CSV, XLSX, XLS), profiles schema and missingness, detects problem types (Binary Classification, Multiclass Classification, Regression), enforces strict anti-leakage preprocessing, screens model candidates with **FLAML**, benchmarks algorithms across 5-fold cross-validation, tunes top candidates with **Optuna**, selects the best model deterministically, evaluates on an untouched holdout set, and packages production-ready pipeline artifacts.

---

## Architecture Overview

```
Dataset (CSV/XLSX/XLS) + Target Column
               │
               ▼
   [1] Ingestion Service (Encoding & Delimiter Sniffing)
               │
               ▼
   [2] Data Validator (Exclusion & Drop Tracking)
               │
               ▼
   [3] Dataset Profiler (Summary Stats, Missingness, Types)
               │
               ▼
   [4] Problem Detector (Binary vs Multiclass vs Regression)
               │
               ▼
   [5] Leakage Guard (ID Columns, Constant Features, Target Proxies)
               │
               ▼
   [6] 80% Dev / 20% Untouched Holdout Split
               │
               ▼
   [7] FLAML Rapid Candidate Screening (Default: 60s budget on Dev data)
       ├── Evaluates LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees
       └── Identifies high-performing model families under time constraints
               │
               ▼
   [8] Leakage-Safe 5-Fold Cross-Validation (Preprocessors fit inside folds)
       ├── Scikit-learn Baselines (LogisticRegression, Ridge, RF, ExtraTrees, HistGB)
       └── Gradient Boosters (LightGBM, XGBoost, CatBoost)
               │
               ▼
   [9] Optuna Bayesian Hyperparameter Optimization (TPE + MedianPruner)
       ├── Tunes top K candidates across 5-fold CV
       └── Evaluates full preprocessing + estimator pipelines per trial
               │
               ▼
  [10] Leaderboard & Deterministic Model Selection (Dev CV Metrics)
               │
               ▼
  [11] Dev Refit & Final Holdout Evaluation (Untouched 20%)
               │
               ▼
  [12] Serialized Artifact (`model.joblib` + `metadata.json`)
```

---

## Core Principles & Guarantees

1. **Zero Data Leakage by Design:** All feature imputers, encoders, and scalers are strictly fitted inside training folds. No transformation ever peeks across validation or test boundaries.
2. **Untouched Final Holdout:** Final holdout data (20%) is strictly isolated before screening, baseline cross-validation, or Optuna tuning begins. It is evaluated exactly once after final model selection to guarantee unbiased out-of-sample metrics.
3. **Transparent Exclusion Tracking:** No data is silently discarded. Every removed row (missing target, duplicates) or dropped column (constant, ID, proxy) is explicitly logged as an `ExclusionRecord` with full justification.
4. **Bayesian Tuning with Early Pruning:** Uses Optuna's Tree-structured Parzen Estimator (TPE) with `MedianPruner` across cross-validation folds, stopping unpromising parameter trials early to optimize compute efficiency.
5. **Inspectable Scikit-Learn Pipeline:** Best models are saved as standard `Pipeline` objects combining preprocessing and estimator, ready for deployment without proprietary runtime dependencies.

---

## Python API Usage

```python
from backend.engine.orchestrator import AutoMLEngine

engine = AutoMLEngine()

result = engine.run(
    file_path="data/employee_turnover.csv",
    target_column="turnover",
    output_dir="artifacts/turnover_run",
    random_state=42,
    n_splits=5,
    # AutoML Screening & Tuning Options
    enable_screening=True,         # Fast FLAML candidate screening
    screening_time_budget=60,      # FLAML screening budget in seconds
    enable_tuning=True,            # Optuna Bayesian hyperparameter tuning
    top_k_to_tune=2,               # Number of top candidates to tune
    tuning_trials=30,              # Maximum Optuna trials per candidate
    tuning_time_budget=120,        # Maximum tuning duration per candidate
)

print(f"Problem Type: {result.problem_detection.problem_type.value}")
print(f"Best Model: {result.best_model_name}")
print(f"CV Score: {result.leaderboard[0].cv_score_mean:.4f}")
print(f"Holdout Metrics: {result.holdout_metrics}")
print(f"Saved Artifact: {result.artifact_path}")
```

---

## Installation & Setup

### Environment Requirements
- Python 3.12 (64-bit)
- Windows 11 / Linux / macOS

### Install Dependencies
```powershell
& "C:\Users\surve\AppData\Local\Programs\Python\Python312\python.exe" -m pip install --prefer-binary numpy pandas scipy scikit-learn joblib openpyxl xlrd xgboost lightgbm catboost flaml optuna pytest
```

---

## Running Benchmarks on Real Datasets

```powershell
& "C:\Users\surve\AppData\Local\Programs\Python\Python312\python.exe" backend/run_real_datasets.py
```

### Running Test Suite
```powershell
& "C:\Users\surve\AppData\Local\Programs\Python\Python312\python.exe" -m pytest backend/tests/ -v
```

---

## Third-Party Open-Source Components & Licenses

| Library | Version | License | Architectural Role |
| :--- | :--- | :--- | :--- |
| **Scikit-learn** | `>=1.4.0` | BSD-3-Clause | Core ML interface, `ColumnTransformer`, `Pipeline`, CV splitters, baselines, and metrics. |
| **FLAML** | `>=2.1.0` | MIT | Fast Cost-Frugal Optimization (CFO) model screening. |
| **Optuna** | `>=4.0.0` | MIT | Bayesian Hyperparameter Optimization (TPE sampler + MedianPruner). |
| **LightGBM** | `>=4.3.0` | MIT | High-performance tree-based gradient boosting. |
| **XGBoost** | `>=2.0.0` | Apache-2.0 | High-performance gradient boosted decision trees. |
| **CatBoost** | `>=1.2.0` | Apache-2.0 | Categorical-aware gradient boosting. |
| **Pandas** | `>=2.2.0` | BSD-3-Clause | Tabular ingestion, dataframe manipulation, series statistics. |
| **NumPy** | `>=1.26.0` | BSD-3-Clause | Vectorized numerical computing and matrix operations. |
| **Joblib** | `>=1.3.0` | BSD-3-Clause | Production pipeline serialization and disk persistence. |
| **Openpyxl / Xlrd** | `>=3.1.0` | MIT / BSD | Safe Excel `.xlsx` and legacy `.xls` spreadsheet parsing. |
