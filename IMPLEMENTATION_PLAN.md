# Universal ML Engine — Phased Implementation Plan

> **Project:** Universal ML Engine (Production-Oriented SaaS AutoML Platform)  
> **Milestone:** Milestone 1 — Implementation Roadmap & Engineering Specifications  
> **Target Environment:** Python 3.12 | Windows 11 / Linux | FastAPI | React + TypeScript  

---

## 1. Project Directory Structure

The repository will be organized into a clean, modular structure with strict separation between the headless ML core (`backend/engine/`), the web API layer (`backend/app/`), and the client frontend (`frontend/`):

```
e:/ML MODEL/
├── .gitignore
├── README.md
├── REPO_RESEARCH.md                   # Milestone 1 Research Report
├── ARCHITECTURE.md                    # Milestone 1 System Architecture
├── IMPLEMENTATION_PLAN.md             # Milestone 1 Phased Execution Plan
├── backend/
│   ├── pyproject.toml                 # Dependencies, tool configs, ruff, pytest
│   ├── requirements.txt               # Pinned Python dependencies
│   ├── app/                           # FastAPI Application Layer
│   │   ├── __init__.py
│   │   ├── main.py                    # App entrypoint & CORS middleware
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── deps.py                # Dependency injection
│   │   │   ├── v1/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── router.py          # Unified v1 router
│   │   │   │   ├── datasets.py        # Upload, list, profile endpoints
│   │   │   │   ├── jobs.py            # Start AutoML, cancel, status endpoints
│   │   │   │   ├── models.py          # Leaderboard, details, download endpoints
│   │   │   │   ├── predict.py         # Real-time single & batch prediction
│   │   │   │   ├── explain.py         # Global & local SHAP explanations
│   │   │   │   └── ws.py              # WebSocket / SSE real-time progress
│   │   ├── core/
│   │   │   ├── config.py              # App settings, paths, env vars
│   │   │   └── events.py              # App startup/shutdown event handlers
│   │   ├── db/
│   │   │   ├── session.py             # SQLite connection / SQLAlchemy setup
│   │   │   └── models.py              # Database tables (Dataset, Job, ModelRecord)
│   │   └── schemas/
│   │       ├── dataset.py             # Pydantic schemas for datasets & profiles
│   │       ├── job.py                 # Pydantic schemas for AutoML jobs & configs
│   │       ├── model.py               # Pydantic schemas for leaderboard & metrics
│   │       ├── predict.py             # Pydantic schemas for inference requests/responses
│   │       └── explain.py             # Pydantic schemas for SHAP values
│   ├── engine/                        # Headless Core Machine Learning Package
│   │   ├── __init__.py
│   │   ├── orchestrator.py            # AutoML State Machine & DAG Executor
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   └── service.py             # Safe file loader (CSV, XLSX, XLS)
│   │   ├── profiler/
│   │   │   ├── __init__.py
│   │   │   ├── analyzer.py            # Summary statistics & missingness
│   │   │   ├── task_detector.py       # Classification vs. Regression detector
│   │   │   └── leakage_guard.py       # Mutual info & correlation leakage detector
│   │   ├── preprocessing/
│   │   │   ├── __init__.py
│   │   │   ├── pipeline_builder.py    # Scikit-learn ColumnTransformer builder
│   │   │   └── custom_transformers.py # Datetime, frequency & target encoders
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── registry.py            # Model factory & search spaces
│   │   │   └── wrappers.py            # LightGBM, XGBoost, Scikit-learn wrappers
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   └── cv_runner.py           # Stratified K-Fold / K-Fold runner & OOF
│   │   ├── tuning/
│   │   │   ├── __init__.py
│   │   │   ├── flaml_screener.py      # Rapid baseline screening via FLAML
│   │   │   └── optuna_tuner.py        # Deep Bayesian HPO via Optuna
│   │   ├── evaluation/
│   │   │   ├── __init__.py
│   │   │   ├── classification.py      # ROC-AUC, F1, PR curves, Confusion Matrix
│   │   │   ├── regression.py          # RMSE, MAE, R², Residuals
│   │   │   └── threshold_optimizer.py # Youden's J / F1 threshold calibration
│   │   ├── explainability/
│   │   │   ├── __init__.py
│   │   │   └── shap_explainer.py      # TreeExplainer & JSON attribution serializing
│   │   ├── prediction/
│   │   │   ├── __init__.py
│   │   │   └── service.py             # Batch & real-time inference validator
│   │   └── reporting/
│   │       ├── __init__.py
│   │       └── model_card.py          # Automated Model Card & package bundler
│   ├── storage/                       # Local artifact storage (gitignored)
│   │   ├── datasets/
│   │   ├── models/
│   │   └── reports/
│   └── tests/
│       ├── test_ingestion.py
│       ├── test_profiler.py
│       ├── test_leakage_guard.py
│       ├── test_preprocessing.py
│       ├── test_models.py
│       ├── test_training_cv.py
│       ├── test_tuning.py
│       ├── test_evaluation.py
│       ├── test_explainability.py
│       ├── test_prediction.py
│       └── test_api.py
└── frontend/                          # React + TypeScript SaaS Application
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    ├── index.html
    ├── src/
    │   ├── main.tsx
    │   ├── App.tsx
    │   ├── api/                       # API client (Axios / Fetch) & WebSocket client
    │   ├── components/
    │   │   ├── common/                # Buttons, Modals, Badges, Loaders, Alerts
    │   │   ├── layout/                # Sidebar, Navbar, Page Containers
    │   │   ├── upload/                # Drag-and-drop file uploader & sheet selector
    │   │   ├── profiling/             # EDA summary cards, distributions, leakage warnings
    │   │   ├── wizard/                # AutoML configuration wizard (mode, time budget)
    │   │   ├── monitor/               # Real-time training progress, trial log stream
    │   │   ├── leaderboard/           # Interactive model comparison matrix
    │   │   ├── evaluation/            # Confusion matrix, ROC/PR curves, Residual plots
    │   │   ├── explainability/        # SHAP beeswarm chart, interactive waterfall explorer
    │   │   ├── predict/               # Single-record playground & batch CSV tester
    │   │   └── report/                # Model Card viewer & export button
    │   ├── hooks/                     # Custom React hooks (useJobProgress, useDataset)
    │   ├── types/                     # TypeScript interfaces matching Pydantic schemas
    │   └── styles/                    # Global Tailwind CSS and design tokens
```

---

## 2. Milestone Execution Roadmap

```
  Milestone 1          Milestone 2          Milestone 3          Milestone 4
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Research &   │────►│ Headless ML  │────►│ AutoML Tuning│────►│ Explain &    │
│ Architecture │     │ Core Engine  │     │ & Selection  │     │ Persistence  │
│ (COMPLETED)  │     │ (Python)     │     │ (FLAML/Optuna│     │ (SHAP/Joblib)│
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                       │
  Milestone 7          Milestone 6          Milestone 5                │
┌──────────────┐     ┌──────────────┐     ┌──────────────┐             │
│ Polish &     │◄────│ React + TS   │◄────│ FastAPI &    │◄────────────┘
│ Verification │     │ Web SaaS UI  │     │ Realtime API │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

### Milestone 2: Core Headless Machine Learning Engine
* **Goal:** Build and verify the foundational pure-Python ML engine without web or UI overhead.
* **Deliverables:**
  1. `DataIngestionService`: Robust CSV/XLSX/XLS loader with encoding detection, sanitization, and error handling.
  2. `DatasetProfiler`: Column typing, missingness, cardinality, task detection (binary vs. multiclass vs. regression), and automated target leakage detection.
  3. `PreprocessingPipelineBuilder`: Strictly isolated Scikit-Learn `ColumnTransformer` builder (imputation, scaling, one-hot, datetime, target encoding).
  4. `ModelRegistry`: Wrappers for LightGBM, XGBoost, Random Forest, Extra Trees, and Linear models.
  5. `CrossValidationRunner`: Stratified K-Fold / K-Fold out-of-fold executor with zero leakage.
  6. `EvaluationEngine`: Full metric calculation suite (ROC-AUC, PR-AUC, F1, Log Loss, RMSE, MAE, R², Confusion Matrix, Curves).
* **Mandatory QA Gate:** 100% passing unit tests covering classification, regression, missing values, extreme imbalance, and anti-leakage verification.

---

### Milestone 3: AutoML Tuning & Model Selection
* **Goal:** Implement the multi-tiered AutoML search and selection engine.
* **Deliverables:**
  1. `FLAMLScreener`: Fast multi-model screening under strict time budgets (30–60s) using CFO.
  2. `OptunaDeepTuner`: Deep Bayesian hyperparameter tuning of top-2 candidate models with `MedianPruner` and step-by-step progress callbacks.
  3. `ThresholdOptimizer`: Optimal classification decision threshold calibration (Youden's J statistic / F1 maximization).
  4. `Leaderboard`: Multi-metric model ranking with latency and complexity tie-breakers.
* **Mandatory QA Gate:** End-to-end benchmark run on standard datasets (e.g., Titanic classification, California Housing regression) completing within target time budget.

---

### Milestone 4: Explainability, Persistence & Prediction
* **Goal:** Deliver explainable predictions and production artifact packaging.
* **Deliverables:**
  1. `ExplanationEngine`: Fast SHAP TreeExplainer and LinearExplainer with JSON-serializable outputs (global feature importance, local sample waterfall breakdowns).
  2. `PredictionService`: Single-row and batch-file inference engine with schema validation and probability calibration.
  3. `ModelCardGenerator`: Automated Model Card assembly and export package bundler (`model.joblib`, `metadata.json`, `sample_inference.py`).
* **Mandatory QA Gate:** Verify that SHAP values sum up to $f(x) - E[f(x)]$ within float precision and that serialized `.joblib` models produce identical predictions in a fresh Python process.

---

### Milestone 5: FastAPI Backend & Asynchronous Worker
* **Goal:** Expose the headless ML engine through a high-performance REST and WebSocket API.
* **Deliverables:**
  1. FastAPI application structure with CORS, error middleware, and Pydantic v2 schemas.
  2. Endpoints for dataset upload, profiling, job creation, cancellation, model leaderboard, predictions, explanations, and model downloads.
  3. SQLite database integration (`datasets`, `jobs`, `models`) with background job task execution.
  4. Real-time WebSocket / SSE endpoint streaming live training progress percentages, current steps, and Optuna trial logs.
* **Mandatory QA Gate:** Automated API test suite via `httpx.AsyncClient` verifying all endpoints, error codes, and concurrent job handling.

---

### Milestone 6: SaaS Web Frontend (React + TypeScript)
* **Goal:** Build an intuitive, responsive SaaS web application.
* **Deliverables:**
  1. Modern UI built with Vite, React 18+, TypeScript, Tailwind CSS, and Lucide icons.
  2. **Dataset Studio:** Drag-and-drop file upload with live profiling cards, missing value bars, distribution histograms, and target leakage warning alerts.
  3. **AutoML Wizard:** Simple mode selection ("Fast 60s Screening" vs. "Deep 5m Optimization"), target selector, and primary metric picker.
  4. **Live Training Monitor:** Real-time progress bar, animated pipeline stage stepper, and live Optuna trial score curve.
  5. **Model Leaderboard & Diagnostics:** Interactive model comparison table, confusion matrix heatmap, ROC and PR curves, and regression residual scatter plots.
  6. **Explainability Explorer:** Global SHAP importance bar chart and interactive local prediction waterfall breakdown.
  7. **Prediction & Export Studio:** Live form-based single prediction playground, batch CSV prediction uploader, and one-click Model Card / ZIP download.
* **Mandatory QA Gate:** Zero browser console errors, responsive layout tested down to 360px mobile width, accessible form elements, and key screen desktop/mobile screenshots.

---

### Milestone 7: Hardening, E2E Verification & Polish
* **Goal:** Final production validation, documentation, and packaging.
* **Deliverables:**
  1. Complete system audit on Windows 11 with Python 3.12.
  2. End-to-end regression testing on diverse test datasets:
     * Small binary classification (e.g. Titanic / Heart Disease)
     * Multiclass classification (e.g. Iris / Wine)
     * Continuous regression with outliers (e.g. Housing / Concrete)
     * Imbalanced dataset (e.g. Credit Fraud simulation)
  3. Comprehensive `Walkthrough` artifact with screenshots and operational guide.

---

## 3. Data Contracts & Pydantic Schema Specifications

### 3.1 Dataset Profile Report Schema (`schemas/dataset.py`)
```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ColumnProfile(BaseModel):
    name: str
    inferred_type: str  # "numeric", "categorical", "datetime", "id", "constant"
    missing_count: int
    missing_percentage: float
    unique_count: int
    is_target_leak_risk: bool
    leak_reason: Optional[str] = None
    sample_values: List[Any]
    stats: Dict[str, Any]  # mean, median, min, max, std, or top categories

class DatasetProfileReport(BaseModel):
    dataset_id: str
    filename: str
    total_rows: int
    total_columns: int
    target_column: Optional[str] = None
    detected_task: Optional[str] = None  # "binary", "multiclass", "regression"
    class_balance: Optional[Dict[str, float]] = None
    columns: List[ColumnProfile]
    leakage_warnings: List[str]
    quality_alerts: List[str]
```

### 3.2 AutoML Job Configuration & Progress Schema (`schemas/job.py`)
```python
class AutoMLMode(str, Enum):
    FAST_SCREENING = "fast_screening"     # 30-60s budget, FLAML CFO
    DEEP_OPTIMIZATION = "deep_optimization"# 2-5m budget, Optuna Bayesian HPO

class JobCreateRequest(BaseModel):
    dataset_id: str
    target_column: str
    mode: AutoMLMode = AutoMLMode.FAST_SCREENING
    time_budget_seconds: int = Field(default=60, ge=15, le=600)
    primary_metric: Optional[str] = None  # Auto-selected if None
    cv_folds: int = Field(default=5, ge=3, le=10)

class JobProgressEvent(BaseModel):
    job_id: str
    status: str  # "PROFILING", "SCREENING", "TUNING", "EVALUATING", etc.
    progress_percent: float
    current_model: Optional[str] = None
    current_trial: Optional[int] = None
    best_score: Optional[float] = None
    elapsed_seconds: float
    log_message: str
```

### 3.3 Evaluation & Leaderboard Schema (`schemas/model.py`)
```python
class ModelSummary(BaseModel):
    model_id: str
    algorithm: str  # "LightGBM", "XGBoost", "RandomForest", "LogisticRegression"
    is_champion: bool
    validation_score: float
    primary_metric: str
    all_metrics: Dict[str, float]  # ROC-AUC, F1, Accuracy, LogLoss, or RMSE, MAE, R²
    fit_time_seconds: float
    latency_ms_per_row: float
    hyperparameters: Dict[str, Any]

class LeaderboardResponse(BaseModel):
    job_id: str
    task_type: str
    primary_metric: str
    models: List[ModelSummary]
```

### 3.4 SHAP Explanation Schema (`schemas/explain.py`)
```python
class GlobalFeatureImportance(BaseModel):
    feature_name: str
    mean_abs_shap: float
    rank: int

class LocalWaterfallStep(BaseModel):
    feature_name: str
    feature_value: Any
    shap_value: float
    cumulative_value: float

class LocalExplanation(BaseModel):
    prediction_value: float
    expected_value: float  # Base value / average prediction
    waterfall_steps: List[LocalWaterfallStep]
```

---

## 4. Edge Cases, Failure Modes & Safeguards

| Category | Potential Failure Mode | Built-in Architectural Safeguard |
| :--- | :--- | :--- |
| **High Missingness** | Column has >90% missing values | Profiler flags column; Preprocessor automatically drops columns with $>90\%$ missingness. |
| **High Cardinality** | User column like `user_id` or `city` with 10,000 unique strings | Leakage/cardinality detector flags ID columns for drop; high-cardinality categoricals use Target/Frequency encoding rather than One-Hot to prevent memory explosion. |
| **Extreme Imbalance** | Fraud detection with 0.1% positive class | Preprocessor enforces `StratifiedKFold`; Evaluation automatically rejects Accuracy and enforces PR-AUC / Macro-F1; calculates cost-optimal threshold. |
| **Single-Class Fold** | Rare target class missing from a fold | `StratifiedKFold` ensures exact class ratio distribution across all validation splits; error guard catches rare categories. |
| **Target Leakage** | Column `churn_date` perfectly predicts `churn` ($r = 1.0$) | `LeakageGuard` calculates Pearson/Spearman $r$ and Mutual Information; warns user in UI and automatically isolates leaking feature. |
| **Timeout / Runaway Job** | Optuna or tree training hangs on complex data | `time_budget_seconds` is passed to workers; Optuna `timeout` parameter and trial time limits enforce hard wall-clock cutoffs. |
| **Inference Drift** | New prediction input has unseen categorical strings or missing columns | `OneHotEncoder(handle_unknown='ignore')`; `SimpleImputer` replaces unseen missing inputs with training fold medians. |
| **Windows Multiprocessing** | `joblib` or Optuna parallel workers failing with `PicklingError` | Enforce thread-based parallelism (`n_jobs=-1` inside LightGBM/XGBoost, single-process asyncio loop for orchestrator) avoiding Windows fork-process crashes. |

---

## 5. Verification & Acceptance Criteria

Every milestone must fulfill the following verification gates before proceeding:

1. **Deterministic Anti-Leakage Audit:**
   * Run synthetic dataset with artificially inserted leaking column $\rightarrow$ Leakage guard must detect and flag it.
   * Verify that `ColumnTransformer` fitted on Fold 0 does not alter or access test set stats.
2. **Speed & Resource Guardrails:**
   * Fast Screening Mode must complete all baseline models within $\le 60$ seconds on 50,000 rows.
   * Peak RAM usage must not exceed 2 GB on standard tabular datasets.
3. **Reproducibility & Serialization Integrity:**
   * Export `.joblib` pipeline $\rightarrow$ reload in a separate Python interpreter $\rightarrow$ assert `np.allclose(original_preds, reloaded_preds)`.
4. **UI/UX Production Quality:**
   * Zero uncaught JavaScript errors in browser developer console.
   * Responsive layout functional at 360px mobile width (no horizontal overflow, accessible touch targets).
   * Key screens captured at desktop and mobile widths.
