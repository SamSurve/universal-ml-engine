# Universal ML Engine

A production-oriented AutoML platform engineered for automated end-to-end tabular machine learning.

Universal ML Engine automatically validates raw tabular data (CSV, XLSX, XLS), profiles schema and missingness, detects problem types (Binary Classification, Multiclass Classification, Regression), enforces strict anti-leakage preprocessing, benchmarks multiple machine learning algorithms across stratified/standard cross-validation, selects the best model deterministically based on CV performance, evaluates on an untouched holdout set, and packages production-ready pipeline artifacts.

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
   [7] Leakage-Safe 5-Fold Cross-Validation (Preprocessors fit inside folds)
       ├── Scikit-learn Baselines (LogisticRegression, Ridge, RF, ExtraTrees, HistGB)
       ├── Gradient Boosters (LightGBM, XGBoost, CatBoost)
       └── Fast AutoML Screening (FLAML)
               │
               ▼
   [8] Leaderboard & Deterministic Model Selection (CV Metrics)
               │
               ▼
   [9] Dev Refit & Final Holdout Evaluation
               │
               ▼
  [10] Serialized Artifact (`model.joblib` + `metadata.json`)
```

---

## Core Principles

1. **Zero Data Leakage by Design:** All feature imputers, encoders, and scalers are strictly fitted inside training folds. No transformation ever peeks across validation or test boundaries.
2. **Transparent Exclusion Tracking:** No data is silently discarded. Every removed row (missing target, duplicates) or dropped column (constant, ID, proxy) is explicitly logged as an `ExclusionRecord` with full justification.
3. **Imbalance-Aware Evaluation:** Imbalanced classification tasks are ranked using `balanced_accuracy` or `f1_weighted`, never misleading raw accuracy.
4. **Inspectable Scikit-Learn Pipeline:** Best models are saved as standard `Pipeline` objects combining preprocessing and estimator, ready for deployment without proprietary runtime dependencies.

---

## Installation & Setup

### Environment Requirements
- Python 3.12 (64-bit)
- Windows 11 / Linux

### Install Dependencies
```powershell
& "C:\Users\surve\AppData\Local\Programs\Python\Python312\python.exe" -m pip install --prefer-binary numpy pandas scipy scikit-learn joblib openpyxl xlrd xgboost lightgbm catboost flaml pytest
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
| **Pandas** | `>=2.2.0` | BSD-3-Clause | Tabular ingestion, dataframe manipulation, series statistics. |
| **NumPy** | `>=1.26.0` | BSD-3-Clause | Vectorized numerical computing and matrix operations. |
| **LightGBM** | `>=4.3.0` | MIT | High-performance tree-based gradient boosting. |
| **XGBoost** | `>=2.0.0` | Apache-2.0 | High-performance gradient boosted decision trees. |
| **CatBoost** | `>=1.2.0` | Apache-2.0 | Categorical-aware gradient boosting. |
| **FLAML** | `>=2.1.0` | MIT | Fast Cost-Frugal Optimization (CFO) model screening. |
| **Joblib** | `>=1.3.0` | BSD-3-Clause | Production pipeline serialization and disk persistence. |
| **Openpyxl / Xlrd** | `>=3.1.0` | MIT / BSD | Safe Excel `.xlsx` and legacy `.xls` spreadsheet parsing. |
