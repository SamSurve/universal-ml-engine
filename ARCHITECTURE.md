# Universal ML Engine — System Architecture Specification

> **System:** Universal ML Engine (Production SaaS AutoML Platform)  
> **Milestone:** Milestone 1 — Architecture & Technical Specifications  
> **Status:** APPROVED FOR IMPLEMENTATION  

---

## 1. Architectural Vision & Principles

The **Universal ML Engine** is a production-oriented, Python-first AutoML SaaS platform. It democratizes advanced machine learning for domain experts and developers by transforming raw tabular data (CSV, XLSX, XLS) and a selected target column into fully validated, production-ready, explainable machine learning pipelines.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 END-TO-END DATAFLOW                                    │
│                                                                                        │
│   Raw File Upload    Automated Profiling     Anti-Leakage Preprocessing   Multi-Model  │
│  ┌────────────────┐  ┌───────────────────┐  ┌───────────────────────────┐ ┌──────────┐ │
│  │ CSV / XLSX/ XLS│─►│ Task Detection    │─►│ Train/Test Split (80/20)  │─► Screening│─┐
│  │ Target Column  │  │ Leakage Guard     │  │ Pipeline Encapsulation    │ │ (FLAML)  │ │
│  └────────────────┘  │ Type Inference    │  │ Impute, Encode, Scale     │ └──────────┘ │
│                      └───────────────────┘  └───────────────────────────┘       │      │
│                                                                                 ▼      │
│     API & Export       Explainability           Evaluation & Leaderboard    Deep Tuning│
│  ┌────────────────┐  ┌───────────────────┐  ┌───────────────────────────┐ ┌──────────┐ │
│  │ FastAPI REST   │◄─│ SHAP Global/Local │◄─│ Out-of-Fold (OOF) Metrics │◄─ (Optuna) │ │
│  │ Realtime Preds │  │ Waterfall Breakdown│ │ ROC-AUC / F1 / RMSE / R²  │ │ Bayesian │ │
│  │ Model Card JSON│  │ Feature Attribs   │  │ Calibration & Threshold   │ └──────────┘ │
│  └────────────────┘  └───────────────────┘  └───────────────────────────┘              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Core Design Principles
1. **Clean Separation of Concerns:** The core machine learning engine is completely headless and decoupled from the web framework. It can run as an isolated Python package, a CLI tool, a background worker, or inside a FastAPI service.
2. **Zero Data Leakage by Design:** All feature transformations, imputations, encodings, and scalers are fitted strictly inside training folds. No transformation ever peeks across validation or test boundaries.
3. **No Black-Box Lock-in:** The engine constructs inspectable, standard Scikit-Learn pipelines and models. Users can download standard `.joblib` / `.onnx` artifacts and run them anywhere without our proprietary server.
4. **Fast First Value (FFV):** In Fast Mode, users receive an initial high-performing baseline model within 30–60 seconds, followed by progressive deep optimization.
5. **Radical Explainability:** Every prediction must be explainable down to the individual feature contribution (via SHAP) with automated human-readable risk notes.

---

## 2. High-Level System Architecture

The platform follows a layered, service-oriented modular monolith architecture:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            PRESENTATION LAYER                                    │
│   React 18+ | TypeScript | Vite | Tailwind CSS | Recharts / Plotly.js            │
│   - Guided Ingestion Wizard           - Interactive Leaderboard & ROC/PR Curves  │
│   - Live Training Monitor (SSE/WS)    - SHAP Global & Local Waterfall Explorer   │
│   - Single & Batch Prediction Studio  - Model Card & Compliance Export (PDF/JSON)│
└────────────────────────┬─────────────────────────────────▲───────────────────────┘
                         │ HTTPS / REST                    │ WebSocket / SSE Events
┌────────────────────────▼─────────────────────────────────┴───────────────────────┐
│                            API & APPLICATION LAYER                               │
│   FastAPI Gateway | Pydantic v2 | Python 3.12                                    │
│   - Endpoints: /datasets, /jobs, /models, /predict, /explain, /export            │
│   - Job State Machine & Asynchronous Task Executor (BackgroundTasks / AsyncIO)   │
│   - SQLite Job Metadata & Trial History Store                                    │
│   - Storage Engine (Local File / S3-compatible Artifact Store)                   │
└────────────────────────┬─────────────────────────────────▲───────────────────────┘
                         │ Direct Python In-Process Calls  │ Progress Callbacks
┌────────────────────────▼─────────────────────────────────┴───────────────────────┐
│                           CORE ML ENGINE LAYER (HEADLESS)                        │
│  ┌───────────────────────┐ ┌──────────────────────┐ ┌──────────────────────────┐ │
│  │ Data Ingestion        │ │ Dataset Profiler     │ │ Preprocessing Engine     │ │
│  │ - CSV/XLSX Parser     │ │ - Statistical Summary│ │ - Scikit-Learn Pipeline  │ │
│  │ - Encoding & Delimiters││ - Leakage Guard      │ │ - Type-Specific Encoders │ │
│  │ - Schema Verification │ │ - Task Detector      │ │ - Strict Fold Isolation  │ │
│  └───────────────────────┘ └──────────────────────┘ └──────────────────────────┘ │
│  ┌───────────────────────┐ ┌──────────────────────┐ ┌──────────────────────────┐ │
│  │ Fast Screening Engine │ │ Deep Tuning Engine   │ │ Evaluation & Selection   │ │
│  │ - FLAML Rapid Budget  │ │ - Optuna TPE Sampler │ │ - Out-of-Fold Metrics    │ │
│  │ - Baseline Ranking    │ │ - Pruners (Hyperband)│ │ - Multi-Metric Pareto    │ │
│  │ - LightGBM/XGB/RF     │ │ - Real-time Progress │ │ - Threshold Tuning       │ │
│  └───────────────────────┘ └──────────────────────┘ └──────────────────────────┘ │
│  ┌───────────────────────┐ ┌──────────────────────┐ ┌──────────────────────────┐ │
│  │ Explainability Engine │ │ Prediction Engine    │ │ Reporting Engine         │ │
│  │ - SHAP TreeExplainer  │ │ - Batch & Single API │ │ - Model Card Generator   │ │
│  │ - Local Waterfalls    │ │ - Schema Validation  │ │ - Provenance Tracking    │ │
│  │ - Global Summary JSON │ │ - Probability Calib. │ │ - Markdown / JSON Export │ │
│  └───────────────────────┘ └──────────────────────┘ └──────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack & Role Specifications

### 3.1 Retained Core Dependencies

| Library | Exact Role in Architecture |
| :--- | :--- |
| **Python 3.12** | Core programming language runtime. Leverages improved exception groups, performance speedups, and modern typing. |
| **FastAPI** | Asynchronous REST API framework. Provides automatic OpenAPI docs, high-throughput endpoints, and async streaming. |
| **Pydantic v2** | Strict schema validation for incoming datasets, training configs, prediction payloads, and API contracts. |
| **Pandas & NumPy** | In-memory tabular processing, exploratory data profiling, vectorized feature transformations, and matrix computations. |
| **Scikit-learn** | Core ML standard. Powers baseline estimators, unified `Pipeline` and `ColumnTransformer` execution, cross-validation splitters (`StratifiedKFold`, `KFold`), and official evaluation metrics. |
| **LightGBM & XGBoost** | Primary high-performance tabular gradient boosting engines. Selected for exceptional training velocity, low memory usage, and native handling of diverse numerical and categorical distributions. |
| **FLAML** | Primary rapid AutoML screening engine. Executes multi-model exploration under strict wall-clock time constraints (30–120 seconds) via Cost-Frugal Optimization (CFO). |
| **Optuna** | Surgical hyperparameter tuning engine. Executes deep Bayesian optimization (TPE) on top candidate models with early pruning (`MedianPruner`) and event callbacks for frontend streaming. |
| **SHAP** | Explainability engine. Utilizes `TreeExplainer` for fast exact Shapley values on tree models and `KernelExplainer` fallback for linear models. Produces structured attribution vectors serialized directly to JSON. |
| **Joblib / Cloudpickle**| Production model serialization. Persists complete end-to-end pipelines (preprocessing + model) for deterministic inference. |
| **SQLite (via aiosqlite / SQLAlchemy)** | Lightweight, zero-config relational store for dataset metadata, job statuses, leaderboard history, and Optuna study trials. |
| **React + TypeScript + Vite** | High-performance, type-safe SaaS frontend. |

### 3.2 Explicitly Excluded Libraries & Justification

1. **Auto-sklearn:** **REJECTED.** Has zero native Windows support due to Unix POSIX dependencies (`pynisher`, `fork`, `signal.SIGALRM`). Fragile dependencies and lagging Python 3.12 support disqualify it from a cross-platform SaaS product.
2. **AutoGluon:** **EXCLUDED FROM CORE RUNTIME.** Requires a massive 4–8 GB download, installs heavyweight PyTorch/Ray dependencies, and demands 8–16 GB RAM minimum. Its monolithic design obscures internal transformation steps and makes real-time SaaS progress streaming difficult. (Reserved as an optional async worker tier for enterprise deployments).
3. **MLJAR-Supervised:** **EXCLUDED AS DIRECT RUNTIME DEPENDENCY.** Wrapping MLJAR whole reduces our platform to a rigid wrapper around MLJAR's proprietary folder outputs. Instead, we natively implement its best user-facing features (automated baseline, tiered search, comprehensive model card reporting) directly within our own custom orchestration layer.

---

## 4. Custom Orchestration Layer Architecture

The core of Universal ML Engine is its **Custom Orchestration Engine (`AutoMLOrchestrator`)**. It operates as a deterministic, asynchronous state machine:

```
               ┌──────────┐
               │  QUEUED  │
               └────┬─────┘
                    │ Validate file & schema
               ┌────▼─────┐
               │ PROFILING│
               └────┬─────┘
                    │ Detect task & target leaks
             ┌──────▼──────┐
             │PREPROCESSING│
             └──────┬──────┘
                    │ Fit-transform train fold only
               ┌────▼─────┐
          ┌───►│SCREENING │ (FLAML Fast Phase)
          │    └────┬─────┘
          │         │ Top K Models identified
          │    ┌────▼─────┐
          │    │  TUNING  │ (Optuna Bayesian Phase)
          │    └────┬─────┘
          │         │ Optimal hyperparameters
          │    ┌────▼─────┐
          │    │EVALUATING│ (Out-of-Fold & Test Set)
          │    └────┬─────┘
          │         │ Compute SHAP values
          │    ┌────▼─────┐
          │    │EXPLAINING│
          │    └────┬─────┘
          │         │ Package model card & artifacts
          │    ┌────▼─────┐
          │    │COMPLETED │
          │    └──────────┘
          │
          └─── If error occurs at any stage ──► [ FAILED ]
```

### Orchestrator Component Architecture
```python
class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PROFILING = "PROFILING"
    PREPROCESSING = "PREPROCESSING"
    SCREENING = "SCREENING"
    TUNING = "TUNING"
    EVALUATING = "EVALUATING"
    EXPLAINING = "EXPLAINING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class OrchestrationCallback(Protocol):
    async def on_step_start(self, job_id: str, step: JobStatus, progress: float) -> None: ...
    async def on_trial_completed(self, job_id: str, trial_data: dict) -> None: ...
    async def on_step_complete(self, job_id: str, step: JobStatus, artifacts: dict) -> None: ...
    async def on_failure(self, job_id: str, error_message: str) -> None: ...
```

### Event Streaming & Cancellation
* **Real-time WebSockets / SSE:** The orchestrator emits fine-grained progress events (`current_step`, `progress_percent`, `elapsed_seconds`, `best_score`, `trial_log`) consumed by the React UI.
* **Cooperative Cancellation:** Every long-running stage checks a cancellation token (`threading.Event` / `asyncio.Event`) after each fold or trial to terminate compute immediately if requested by the user.

---

## 5. Module Boundaries & Class Specifications

The system is structured into strictly bounded packages under `backend/engine/`:

```
backend/
├── app/                        # Web & API Layer
│   ├── api/                    # Route handlers (datasets, jobs, models, etc.)
│   ├── core/                   # Config, security, database session
│   ├── models/                 # SQLAlchemy DB entities
│   └── schemas/                # Pydantic request/response schemas
├── engine/                     # Headless ML Core (Pure Python)
│   ├── ingestion/              # Data parsing and validation
│   ├── profiler/               # EDA, data health, task detection
│   ├── preprocessing/          # Leak-free Scikit-Learn transformers
│   ├── models/                 # Model registry & algorithm wrappers
│   ├── training/               # CV runners and fold executors
│   ├── tuning/                 # FLAML screening & Optuna tuning
│   ├── evaluation/             # Metrics, confusion matrix, ROC/PR curves
│   ├── explainability/         # SHAP TreeExplainer & JSON serializers
│   ├── prediction/             # Inference engine & validation
│   └── reporting/              # Model Card and artifact packager
└── tests/                      # Unit, integration, and anti-leakage tests
```

---

### Module 1: Data Ingestion (`engine/ingestion/`)
* **Purpose:** Safely parse CSV, TSV, XLSX, and XLS files into memory with strict size and memory guards.
* **Responsibilities:**
  * Delimiter sniffing and encoding detection (`utf-8`, `latin-1`).
  * Chunked loading with maximum row limits (e.g., 500,000 rows for SaaS tier).
  * Column sanitization (strip whitespace, normalize casing, replace invalid characters).
* **Key Class:** `DataIngestionService`
* **Input Schema:** Raw file bytes + metadata.
* **Output Schema:** In-memory sanitized `pd.DataFrame` + column data types dictionary.

---

### Module 2: Dataset Profiling & Task Detection (`engine/profiler/`)
* **Purpose:** Generate statistical summary, detect task type, and scan for data leakage.
* **Responsibilities:**
  * Missing value percentages per column.
  * Cardinality analysis (numeric, categorical, high-cardinality ID, datetime, constant, text).
  * Target column validation and automated task detection:
    * Categorical / boolean / integer with $\le 10$ unique values $\rightarrow$ **Binary or Multiclass Classification**.
    * Continuous numeric with $> 10$ unique values $\rightarrow$ **Regression**.
  * Class balance ratio computation for classification.
  * Skewness, Kurtosis, and outlier detection for regression.
  * **Target Leakage Detection:** Flag features with near-perfect correlation ($|r| > 0.98$), high mutual information ($I(X; Y) \approx H(Y)$), or ID-like columns (cardinality ratio $\approx 1.0$).
* **Key Class:** `DatasetProfiler`
* **Input:** `df: pd.DataFrame`, `target_column: str`
* **Output:** `DatasetProfileReport` (Pydantic model containing column summaries, task type, data quality alerts, leakage warnings).

---

### Module 3: Preprocessing Engine (`engine/preprocessing/`)
* **Purpose:** Build deterministic, leak-free Scikit-Learn pipelines.
* **Responsibilities:**
  * Separate columns into feature sets: `numeric_cols`, `categorical_low_cardinality`, `categorical_high_cardinality`, `datetime_cols`, `drop_cols`.
  * Construct a unified `ColumnTransformer`:
    * Numeric pipeline: `SimpleImputer(strategy='median')` $\rightarrow$ `RobustScaler()` (or `StandardScaler()`).
    * Low-cardinality categorical pipeline: `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
    * High-cardinality categorical pipeline: Target Encoding or Frequency Encoding fitted strictly on train folds.
    * Datetime pipeline: Custom transformer extracting `year`, `month`, `day`, `dayofweek`, `hour`.
  * Return an untouched Scikit-Learn `Pipeline` ready for fold-by-fold execution.
* **Key Class:** `PreprocessingPipelineBuilder`
* **Input:** `profile: DatasetProfileReport`, `user_overrides: dict`
* **Output:** `sklearn.pipeline.Pipeline`

---

### Module 4: Model Registry (`engine/models/`)
* **Purpose:** Maintain standardized wrappers and search spaces for candidate algorithms.
* **Supported Model Families:**
  1. `LightGBM` (`LGBMClassifier`, `LGBMRegressor`): High-speed gradient boosting.
  2. `XGBoost` (`XGBClassifier`, `XGBRegressor`): Robust gradient boosting with regularizers.
  3. `RandomForest` (`RandomForestClassifier`, `RandomForestRegressor`): Ensembling baseline.
  4. `ExtraTrees` (`ExtraTreesClassifier`, `ExtraTreesRegressor`): Fast randomized tree ensemble.
  5. `Linear` (`LogisticRegression`, `Ridge`, `Lasso`, `ElasticNet`): Interpretable baselines.
  6. `Dummy / Baseline` (`DummyClassifier`, `DummyRegressor`): Quality floor baseline.
* **Key Class:** `ModelRegistry` (provides default hyperparameter bounds and initialization factories).

---

### Module 5: Training & CV Runner (`engine/training/`)
* **Purpose:** Execute out-of-fold cross-validation with zero data leakage.
* **Responsibilities:**
  * Determine CV strategy:
    * `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` for classification.
    * `KFold(n_splits=5, shuffle=True, random_state=42)` for regression.
    * `TimeSeriesSplit` if datetime index is designated.
  * Fit full pipeline (preprocessing + estimator) strictly on each training fold.
  * Generate and accumulate Out-Of-Fold (OOF) predictions across all validation folds.
  * Evaluate fold-level metrics and aggregate standard deviations.
* **Key Class:** `CrossValidationRunner`
* **Input:** `pipeline: Pipeline`, `X: pd.DataFrame`, `y: pd.Series`, `cv_config: CVConfig`
* **Output:** `CVResult` (OOF predictions, fold scores, mean score, std dev, fit duration).

---

### Module 6: Tuning Engine (`engine/tuning/`)
* **Purpose:** Multi-tiered hyperparameter optimization.
* **Two-Tier Tuning Strategy:**
  1. **Tier 1 (Fast Screening via FLAML):** Tests all candidate model families with default CFO optimization within a fixed time budget (e.g., 30–60s) to prune weak families and identify the top 2–3 algorithms.
  2. **Tier 2 (Deep Tuning via Optuna):** Takes the top candidate models from Tier 1 and runs a focused Bayesian TPE search over critical hyperparameters (learning rate, num leaves, tree depth, regularization, subsampling) with `MedianPruner` to abort low-performing trials.
* **Key Classes:** `FLAMLScreener`, `OptunaDeepTuner`
* **Input:** `X_train`, `y_train`, `task_type`, `time_budget`, `metric`
* **Output:** `TuningResult` (best hyperparameters, full trial history, validation score curve).

---

### Module 7: Evaluation & Model Selection (`engine/evaluation/`)
* **Purpose:** Comprehensive metric evaluation, calibration, and leaderboard compilation.
* **Metric Suites:**
  * **Classification:**
    * Primary: `ROC-AUC` (balanced binary), `PR-AUC` / `Average Precision` (imbalanced binary), `Macro-F1` (multiclass), `Balanced Accuracy`.
    * Diagnostics: `Log Loss`, `Precision`, `Recall`, `Specificity`, `Brier Score`.
    * Visualizations: Confusion Matrix data, ROC curve coordinates, Precision-Recall curve coordinates, Calibration curve coordinates.
    * Automated Threshold Optimization: Evaluates decision thresholds from 0.01 to 0.99 to find optimal F1 or Youden's J threshold.
  * **Regression:**
    * Primary: `RMSE`, `MAE`, `R²`, `MAPE`, `Median Absolute Error`.
    * Diagnostics: Residual distribution, Q-Q plot data, Predicted vs. Actual scatter coordinates.
* **Leaderboard Assembly:** Compiles all trained models into a ranked leaderboard based on validation score, inference latency, training time, and model size.
* **Key Class:** `EvaluationEngine`
* **Input:** `y_true`, `y_pred`, `y_prob`, `task_type`
* **Output:** `EvaluationReport`

---

### Module 8: Explainability Engine (`engine/explainability/`)
* **Purpose:** Compute rigorous global and local SHAP attributions.
* **Responsibilities:**
  * Select appropriate explainer:
    * Tree models $\rightarrow$ `shap.TreeExplainer(model)` (exact, ultra-fast).
    * Linear models $\rightarrow$ `shap.LinearExplainer(model, X_background)`.
    * General fallback $\rightarrow$ `shap.sample(X, 100)` + `shap.KernelExplainer`.
  * Compute Global Feature Importance: Mean absolute SHAP values across representative sample (max 500 rows).
  * Compute Local Attributions: Waterfall breakdown for specific prediction rows ($f(x) = E[f(x)] + \sum \phi_i$).
  * Serialize values to clean JSON: Feature names, base values, SHAP values, and raw feature values.
* **Key Class:** `ExplanationEngine`
* **Input:** `trained_pipeline`, `X_sample`, `row_index`
* **Output:** `GlobalExplanationJSON`, `LocalExplanationJSON`

---

### Module 9: Prediction Engine (`engine/prediction/`)
* **Purpose:** Fast, safe inference on single rows or batch CSVs.
* **Responsibilities:**
  * Validate incoming payload schema against the training feature schema (missing columns, type mismatches).
  * Execute `pipeline.predict()` and `pipeline.predict_proba()` (with applied calibrated threshold if classification).
  * Attach confidence scores and risk flags.
* **Key Class:** `PredictionService`
* **Input:** `model_id: str`, `input_data: list[dict] | pd.DataFrame`
* **Output:** `PredictionResponse` (predictions, probabilities, confidence level).

---

### Module 10: Reporting & Packaging (`engine/reporting/`)
* **Purpose:** Generate production model cards and export bundles.
* **Responsibilities:**
  * Assemble comprehensive **Model Card** (Dataset summary, model parameters, cross-validation metrics, confusion matrix, SHAP global top features, limitations, and provenance).
  * Package deployable artifact bundle:
    * `model.joblib` (Full Scikit-Learn Pipeline)
    * `metadata.json` (Schema, target, task type, metric scores)
    * `model_card.json` / `model_card.md`
    * `sample_inference.py` (Ready-to-run standalone inference script)
* **Key Class:** `ModelCardGenerator`
* **Output:** Downloadable ZIP bundle / JSON payload.

---

## 6. Anti-Leakage Shield & Score Integrity

Data leakage is the single greatest cause of AutoML failure in production. The Universal ML Engine enforces a **6-Layer Anti-Leakage Shield**:

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      6-LAYER ANTI-LEAKAGE SHIELD                       │
 ├────────────────────────────────────────────────────────────────────────┤
 │ 1. Clean Test Holdout Set (20%)                                        │
 │    - Split immediately after ingestion.                                │
 │    - Stored untouched until final model validation.                    │
 ├────────────────────────────────────────────────────────────────────────┤
 │ 2. Strict Fold Encapsulation                                           │
 │    - ColumnTransformer fit strictly inside CV training folds.          │
 │    - Scaler means/stds and imputation medians never see val/test data.│
 ├────────────────────────────────────────────────────────────────────────┤
 │ 3. Automated Target Leakage Detection                                  │
 │    - Mutual Information & Pearson check flags columns with r > 0.98.  │
 │    - Detects ID columns, duplicate target columns, and future dates.   │
 ├────────────────────────────────────────────────────────────────────────┤
 │ 4. Out-of-Fold (OOF) Score Integrity                                   │
 │    - Validation scores computed strictly on out-of-fold predictions.   │
 │    - In-sample training scores are NEVER displayed as validation.      │
 ├────────────────────────────────────────────────────────────────────────┤
 │ 5. Class Imbalance Metric Safeguards                                   │
 │    - For minority class < 20%, system blocks default Accuracy metric.  │
 │    - Enforces PR-AUC, Macro-F1, or Balanced Accuracy.                  │
 ├────────────────────────────────────────────────────────────────────────┤
 │ 6. Optimal Decision Threshold Tuning                                   │
 │    - Calculates optimal threshold using Youden's J statistic or F1.    │
 │    - Prevents misleading classification caused by static 0.5 threshold │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Model Selection Framework

```
                       ┌──────────────────────────────┐
                       │     Model Selection Flow     │
                       └──────────────┬───────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
     [ Classification Task ]                       [ Regression Task ]
               │                                             │
      Minority Class < 20%?                         Target Highly Skewed?
      ┌────────┴────────┐                           ┌────────┴────────┐
     Yes                No                         Yes                No
      │                  │                          │                  │
   [PR-AUC]          [ROC-AUC]                  [MAE / Huber]      [RMSE / R²]
   [Macro-F1]       (Multiclass: Macro-F1)      (Log1p Candidate)
      │                  │                          │                  │
      └────────┬─────────┘                          └────────┬─────────┘
               ▼                                             ▼
 ┌───────────────────────────┐                 ┌───────────────────────────┐
 │   FLAML Quick Screening   │                 │   FLAML Quick Screening   │
 │   (LGBM, XGB, RF, Linear) │                 │   (LGBM, XGB, RF, Ridge)  │
 └─────────────┬─────────────┘                 └─────────────┬─────────────┘
               │ Select Top 2 Candidates                     │ Select Top 2 Candidates
 ┌─────────────▼─────────────┐                 ┌─────────────▼─────────────┐
 │    Optuna Deep Tuning     │                 │    Optuna Deep Tuning     │
 │   (Bayesian TPE Search)   │                 │   (Bayesian TPE Search)   │
 └─────────────┬─────────────┘                 └─────────────┬─────────────┘
               │                                             │
 ┌─────────────▼─────────────┐                 ┌─────────────▼─────────────┐
 │ OOF Evaluation & Ranking  │                 │ OOF Evaluation & Ranking  │
 │ Threshold Optimization    │                 │ Residual Analysis         │
 └─────────────┬─────────────┘                 └─────────────┬─────────────┘
               ▼                                             ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │       Champion Selection (Primary Score + Latency Tie-Breaker)          │
 └─────────────────────────────────────────────────────────────────────────┘
```

### Tie-Breaker Decision Logic
When two models achieve validation scores within a statistical margin of error ($\Delta < 0.005$):
1. **Inference Latency:** The faster model (lower millisecond per prediction) is preferred.
2. **Model Complexity:** A simpler model (e.g., LightGBM with fewer trees or Scikit-Learn Random Forest) is preferred over an over-parameterized model.
3. **Model Size:** The smaller artifact on disk is prioritized.

---

## 8. Milestone Roadmap: What to Build Now vs. Later

### Current Milestone (Milestone 1 — Complete)
* Research report comparing 6 open-source projects (`REPO_RESEARCH.md`).
* Production architecture specification (`ARCHITECTURE.md`).
* Implementation plan with detailed file tree and testing strategy (`IMPLEMENTATION_PLAN.md`).

### Milestone 2: Core Headless ML Engine
* Pure Python package (`engine/`).
* Data ingestion, profiler, leak-free preprocessing pipeline.
* Model registry (Scikit-Learn, LightGBM, XGBoost) and CV runner.
* Full evaluation metric suite and automated test verification.

### Milestone 3: AutoML Tuning & Selection
* FLAML rapid screening integration (30–60s budget).
* Optuna deep Bayesian tuning with early pruning.
* Model leaderboard and champion selection logic.

### Milestone 4: Explainability & Model Persistence
* SHAP TreeExplainer integration and JSON attribution serializers.
* Pipeline serialization (`.joblib`) and single/batch prediction service.
* Automated Model Card generator.

### Milestone 5: FastAPI Backend & Asynchronous Worker
* Async REST API endpoints (`/datasets`, `/jobs`, `/models`, `/predict`).
* WebSocket / SSE progress event streaming.
* SQLite job database and task state management.

### Milestone 6: SaaS Web Frontend (React + TypeScript)
* Modern, responsive UI (Vite, Tailwind, Recharts).
* Dataset upload & interactive profiling dashboard.
* Live training monitor with progress bars and trial curves.
* Interactive model leaderboard, ROC/PR curves, and SHAP waterfall viewer.
* Real-time prediction playground and Model Card export.

### Milestone 7: End-to-End Verification & Polish
* Comprehensive cross-platform testing (Windows 11 / Linux).
* Zero console error audit, mobile 360px responsive design audit.
* Desktop and mobile screenshot verification.
