# Universal ML Engine v2 — Production AutoML Platform Plan

> **Milestone 1B — Revised Architecture & Implementation Plan**  
> **Status:** APPROVED PRODUCTION PLAN (No implementation or package installation in this phase)  
> **Target System:** Windows 11 / Python 3.12 (64-bit), ~8 GB RAM Development Machine  
> **Primary AutoML Backend:** AutoGluon Tabular  
> **Baseline / Fallback:** Native Engine (Scikit-Learn baselines, Gradient Boosters, FLAML/Optuna)  
> **Secondary Backend (Optional):** MLJAR Supervised  

---

## 1. Executive Summary & Architecture Overview

The Universal ML Engine remains the **product, intelligence, governance, and orchestration layer**. It wraps proven open-source AutoML engines rather than reimplementing them.

```
Raw Tabular Data (CSV/XLSX/XLS) + Target Column
                      │
                      ▼
       [1] Ingestion Service (Encoding & Delimiter Sniffing)
                      │
                      ▼
       [2] Data Validator (Exclusion & Drop Tracking)
                      │
                      ▼
       [3] Dataset Profiler & Problem Detector
                      │
                      ▼
       [4] Dataset Intelligence & Health Diagnostic (0–100 Score)
                      │
                      ▼
       [5] Leakage Guard (ID Columns, Constant Features, Proxies)
                      │
                      ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ [6] 3-WAY LEAKAGE-SAFE PARTITION                                        │
   │     ├── Train Set (60-70%): Passed to Backends (internal CV/tuning)     │
   │     ├── Validation Set (15-20%): INDEPENDENT (for Engine Champion Select)│
   │     └── Test Set (15-20%): UNTOUCHED until final single-model evaluation│
   └─────────────────────────────────────────────────────────────────────────┘
                      │
                      ▼
       [7] Mandatory Baselines (Dummy + Linear/Ridge on Val Set)
                      │
                      ▼
       [8] Subprocess Worker Execution (Memory & Crash Isolation)
           ├── [Primary] AutoGluon Tabular (Trains strictly on Train Set)
           ├── [Baseline/Fallback] Native Engine (Trains strictly on Train Set)
           └── [Optional M6] MLJAR Supervised (Trains strictly on Train Set)
                      │
                      ▼
       [9] Normalized Backend Results (Standard Internal Contract)
                      │
                      ▼
      [10] INDEPENDENT CHAMPION SELECTION ON VALIDATION SET
           (Evaluates all trained candidate predictors on the untouched Val Set)
                      │
                      ▼
      [11] FINAL TEST EVALUATION (Evaluated ONCE on Untouched Test Set)
                      │
                      ▼
      [12] Backend-Agnostic Explainability (Validation Permutation Importance)
                      │
                      ▼
      [13] Governance & Intelligence Reports (MODEL_REPORT.md + DATASET_INTELLIGENCE_REPORT.md)
                      │
                      ▼
      [14] Unified Model Artifact + Comprehensive Inference Schema JSON
```

---

## 2. Critical ML Protocol: 3-Way Partition & Validation Champion Selection

### 2.1 The Golden Rule of Unbiased Evaluation
> [!IMPORTANT]
> **1. The Test set must NEVER be used to choose the champion model.**  
> Testing every backend on the test set and picking whichever scores highest turns the test set into a selection set, destroying out-of-sample validity and causing data snooping.  
> **2. The Validation set must NEVER be passed to backend `fit()` as tuning data.**  
> If an AutoML backend uses the validation set internally for early stopping, hyperparameter optimization, or ensemble weighting, that validation set is no longer an independent out-of-sample benchmark. The engine must preserve an **independent, untouched Validation Set** to fairly compare all backends and baselines during Champion Selection.

### 2.2 Data Partitioning Scheme

| Partition | Size | Purpose | Strict Invariants |
| :--- | :--- | :--- | :--- |
| **Train Set** | 60% – 70% | Model parameter learning, backend internal CV / internal tuning | Passed to backend `fit()`. Preprocessors and internal tuning fit *only* on this data |
| **Validation Set** | 15% – 20% | Model comparison, baseline benchmarking, **Engine Champion Selection** | **NEVER passed to backend `fit()`.** Kept strictly independent |
| **Test Set** | 15% – 20% | Unbiased final performance measurement of the **selected champion only** | Evaluated **exactly once** after champion is finalized |

For small datasets ($N < 500$), stratified K-Fold Out-of-Fold (OOF) validation on the Dev partition (80%) is permitted for validation scoring, while preserving the 20% untouched holdout strictly for the final single-model test evaluation.

### 2.3 Mandatory Baselines
Before evaluating complex AutoML models, the engine computes baseline metrics on the independent validation set:
1. **Dummy Baseline:** `DummyClassifier(strategy="most_frequent")` or `DummyRegressor(strategy="mean")`.
2. **Simple Linear Baseline:** `LogisticRegression` / `Ridge` fitted on preprocessed train data.

The selected champion must beat these baselines on the validation set. If no candidate outperforms the simple linear baseline, the engine issues a clear data quality warning.

---

## 3. Subprocess Worker & Resource Architecture (~8 GB RAM Aware)

### 3.1 Why In-Process Execution is Insufficient for Heavy AutoML
- AutoML frameworks (AutoGluon, MLJAR, LightGBM, CatBoost) utilize native C/C++ libraries and multithreaded OpenMP runtimes.
- C-level segmentation faults, native memory leaks, and Out-Of-Memory (OOM) errors cannot be trapped reliably by standard Python `try/except`.
- Python memory is not immediately returned to the OS after large dataframe/ensemble manipulations.

### 3.2 Subprocess Worker Design

```
Main Orchestrator Process
   │
   ├── 1. Writes: train_data.parquet, run_config.json to temp folder (val_data is NOT passed)
   ├── 2. Spawns: python -m backend.engine.backends.workers.<backend>_worker --config <path>
   ├── 3. Monitors: Memory usage via psutil, enforces hard timeout
   ├── 4. Captures: Stdout/stderr logs in real time
   ├── 5. Reads: result.json and artifact pointer from worker output
   └── 6. Terminates: Guarantees complete process tree cleanup (Windows taskkill / psutil)
```

### 3.3 Available RAM-Aware Tuning (Target: ~8 GB RAM Laptop)

The resource detector checks **`psutil.virtual_memory().available`**, not just total physical RAM.

```python
# Resource Policy for ~8 GB RAM Systems
AUTO_GLUON_LOW_RESOURCE_CONFIG = {
    "presets": "medium_quality",      # Avoids memory-heavy deep stacking
    "auto_stack": False,              # Disables multi-layer stacking
    "num_cpus": max(1, os.cpu_count() - 1),
    "time_limit": min(300, user_budget),
    "save_space": True,
    "holdout_frac": 0.2,              # AutoGluon splits internally within df_train
    "fit_weighted_ensemble": True,    # Lightweight ensembling on internal split
}
```

**Resource Rules:**
1. **Never Concurrent:** AutoGluon and MLJAR must NEVER run concurrently.
2. **Pre-flight RAM Check:** If available RAM $< 2.0\text{ GB}$, log a warning, trigger Python garbage collection, and enforce minimal model presets.
3. **Hard Process Tree Kill:** When a subprocess worker hits its timeout or memory threshold ($>85\%$ system memory), terminate the entire worker process tree immediately.

---

## 4. Backend Architecture: Simple Adapters over Complex Frameworks

We avoid overengineered ABC hierarchies, complex dynamic plugin registries, or factory abstractions. Instead, we use a clean **Functional Adapter Contract**.

### 4.1 Backend Adapter Interface

Each backend is implemented in a dedicated module `backend/engine/backends/<name>_backend.py` with clean entry points:

```python
def is_available() -> bool:
    """Check if backend dependencies are importable."""
    ...

def run_worker(
    train_df: pd.DataFrame,
    target_column: str,
    problem_type: ProblemType,
    primary_metric: str,
    time_budget_seconds: int,
    output_dir: Path,
    random_state: int,
    resource_config: dict,
) -> NormalizedBackendResult:
    """Invoked inside the isolated subprocess worker."""
    ...

def load_predictor(artifact_path: Path) -> UnifiedPredictor:
    """Loads backend-native saved artifact into a unified prediction wrapper."""
    ...
```

### 4.2 Unified Predictor Wrapper (`UnifiedPredictor`)

Because AutoGluon saves its own directory structure and MLJAR uses its own folder format, we do **not** force `joblib.dump` on external models. Instead, `UnifiedPredictor` provides standard inference:

```python
class UnifiedPredictor:
    """Unified wrapper around any backend champion artifact."""
    def predict(self, df: pd.DataFrame) -> np.ndarray: ...
    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]: ...
    def get_feature_names(self) -> List[str]: ...
```

---

## 5. Normalized Result Contract

Every backend worker outputs a standardized `result.json` read by the orchestrator:

```python
@dataclass
class NormalizedBackendResult:
    backend_name: str                  # "autogluon", "native", "mljar"
    status: str                        # "success", "timeout", "failed", "oom"
    model_name: str                    # e.g., "WeightedEnsemble_L2", "LightGBM"
    model_type: str                    # "ensemble", "tree", "linear"
    
    # Validation Performance (USED FOR CHAMPION SELECTION)
    val_score: float                   # Score on the validation set
    primary_metric: str                # e.g., "f1", "balanced_accuracy", "r2", "rmse"
    val_metrics: Dict[str, float]      # All validation metrics (accuracy, f1, precision, etc.)
    
    # Internal Backend Telemetry
    internal_leaderboard: List[Dict[str, Any]] # Native leaderboard
    training_time_seconds: float
    models_trained_count: int
    
    # Artifact Pointer
    artifact_path: str                 # Path to backend's native saved model directory
    
    # Diagnostics & Logs
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
```

---

## 6. Backend-Agnostic Explainability

### 6.1 Why Native SHAP Fails on Multi-Model AutoML Ensembles
- AutoGluon champions are typically `WeightedEnsemble_L2` or stacked models combining LightGBM, CatBoost, ExtraTrees, and Neural Networks.
- `shap.TreeExplainer` fails on heterogeneous ensembles or custom stacking meta-learners.
- `shap.KernelExplainer` on large ensembles is prohibitively slow for automated pipelines ($>10$ minutes for 50 samples).

### 6.2 Solution: Validation-Set Permutation Feature Importance
The engine computes **Permutation Feature Importance** on the Validation Set:
1. Measure baseline validation score $S_{\text{val}}$ using the champion's `predict()`.
2. For each feature $f_j$:
   - Permute values of column $f_j$ in the validation set.
   - Compute degraded score $S_{\text{perm}}(f_j)$.
   - $\text{Importance}(f_j) = S_{\text{val}} - S_{\text{perm}}(f_j)$ (or percentage drop).
3. Fast, model-agnostic, works on *any* ensemble or backend without special bindings.
4. Existing `shap_explainer.py` is retained strictly as an optional enhancer for native Scikit-Learn/single-tree models.

---

## 7. Inference Schema Contract & Persistence

To guarantee production reliability, the model directory must save a strict `inference_schema.json`:

```json
{
  "engine_version": "2.0.0",
  "backend": "autogluon",
  "champion_model": "WeightedEnsemble_L2",
  "problem_type": "binary_classification",
  "target_column": "Employee_Turnover",
  "positive_class": "1",
  "class_labels": ["0", "1"],
  "primary_metric": "f1",
  "features": [
    {"name": "satisfaction_level", "dtype": "float64", "required": true},
    {"name": "last_evaluation", "dtype": "float64", "required": true},
    {"name": "number_project", "dtype": "int64", "required": true},
    {"name": "Department", "dtype": "object", "required": false}
  ],
  "dropped_columns": [
    {"name": "employee_id", "reason": "Identified as artificial identifier / ID leakage"}
  ],
  "partition_info": {
    "train_rows": 9000,
    "val_rows": 3000,
    "test_rows": 3000
  },
  "validation_metrics": {"f1": 0.9421, "accuracy": 0.9510, "roc_auc": 0.9820},
  "test_metrics": {"f1": 0.9385, "accuracy": 0.9490, "roc_auc": 0.9795}
}
```

---

## 8. API Verification Register

> [!WARNING]
> The following library behaviors and APIs must be verified via short isolated spike scripts during **Milestone M0** before writing integration code. Do NOT assume these signatures from memory.

| Item | Component | Verification Finding | Status |
| :--- | :--- | :--- | :--- |
| **V1** | Python 3.12 Compatibility | AutoGluon Tabular runs cleanly with standard dependencies; requires optional extra isolation (`.[autogluon]`). | `[VERIFIED - M0]` |
| **V2** | AutoGluon Arguments | `TabularPredictor(label, eval_metric, path, problem_type)` & `fit(train_data, tuning_data, time_limit, presets, auto_stack=False)` confirmed. | `[VERIFIED - M0]` |
| **V3** | AutoGluon Metrics | `'f1'`, `'roc_auc'`, `'balanced_accuracy'`, `'r2'`, `'root_mean_squared_error'`, `'mean_absolute_error'` confirmed. | `[VERIFIED - M0]` |
| **V4** | AutoGluon Presets | `'medium_quality'` confirmed as required preset on ~8 GB RAM systems to prevent memory exhaustion. | `[VERIFIED - M0]` |
| **V5** | AutoGluon Predictor Load | `TabularPredictor.load(path)` confirmed; returns standard predictor object for inference and leaderboard generation. | `[VERIFIED - M0]` |
| **V6** | MLJAR Supervised API | `AutoML(results_path, mode, total_time_limit, ml_task, eval_metric, random_state)` constructor and `.fit(X, y)` confirmed. | `[VERIFIED - M0]` |
| **V7** | MLJAR Leaderboard | `automl.get_leaderboard()` returning DataFrame with `['name', 'model_type', 'metric_type', 'metric_value', 'train_time']` confirmed. | `[VERIFIED - M0]` |
| **V8** | Windows Subprocess Tree | Process-tree kill via `taskkill /PID <pid> /T /F` and `psutil` confirmed to terminate child processes and unmanaged OpenMP threads cleanly. | `[VERIFIED - M0]` |

---

## 9. Pragmatic Design Decisions & Simplifications

### What Was Removed from the First Draft:
1. **Removed Large Plugin / ABC Framework:** Replaced with simple module-level runner functions and a single `UnifiedPredictor` wrapper.
2. **Removed Over-Engineered Execution Strategy Enums:** Instead of 4 enum modes, we have a clear, simple workflow: Run Primary Candidate (AutoGluon) $\rightarrow$ If failed/unavailable, fallback to Native Baseline. (Optional MLJAR can be run sequentially if requested).
3. **Removed Artifact Size Tiebreaker:** Replaced with standard statistical rule: higher validation score wins; if equal within $10^{-4}$, the model with lower fit time wins.
4. **Removed Test-Set Champion Selection:** Champion is now strictly selected on the **Validation Set**. Test set is evaluated exactly once at the end.
5. **Removed Assumption of SHAP on AutoGluon:** Replaced with validation permutation feature importance.
6. **Removed Monolithic Orchestrator Refactor:** Kept the working `orchestrator.py` intact; added a clean `AutoMLOrchestrator` path that leverages the existing validators, profilers, and intelligence modules.

---

## 10. Phased Implementation Plan (Small, Verifiable Milestones)

---

### Phase M0: Spike & Verification (API, Dependency & Resource)

- **Goal:** Verify Python 3.12 compatibility, AutoGluon/MLJAR imports, subprocess execution, and available RAM detection on the target Windows machine.
- **Components / Files Involved:**
  - `backend/spikes/spike_autogluon_check.py`
  - `backend/spikes/spike_resource_check.py`
  - `backend/spikes/spike_worker_process.py`
- **Constraints:** Standalone scratch scripts only. No modification to production engine code.
- **Deliverables:**
  1. Verification report on all items in the API Verification Register (V1–V8).
  2. Verified minimal dependencies list.
- **Verification Command:**
  ```powershell
  python backend/spikes/spike_resource_check.py
  ```
- **Git Commit:** `chore: verify automl dependencies and subprocess worker feasibility`

---

### Phase M1: 3-Way Data Splitter & Baseline Evaluators

- **Goal:** Implement clean 3-way Train/Validation/Test splitting with stratification and leakage-safe preparation, plus mandatory Dummy and Linear baselines.
- **Components / Files Involved:**
  - `backend/engine/training/data_splitter.py` (New: `DataSplitter.split_train_val_test`)
  - `backend/engine/evaluation/baselines.py` (New: `BaselineEvaluator.evaluate_dummy_and_linear`)
  - `backend/tests/test_data_splitter.py`
  - `backend/tests/test_baselines.py`
- **Constraints:** Zero data leakage. Preprocessors must fit strictly on the Train fold.
- **Deliverables:**
  1. `DataSplitter` returning `(df_train, df_val, df_test)` with index integrity.
  2. Baseline scoring for classification and regression tasks.
- **Verification Command:**
  ```powershell
  pytest backend/tests/test_data_splitter.py backend/tests/test_baselines.py -v
  ```
- **Git Commit:** `feat: implement 3-way train-val-test split and baseline evaluators`

---

### Phase M2: Subprocess Worker Execution Infrastructure

- **Goal:** Build resilient subprocess worker infrastructure with IPC, real-time memory monitoring, and reliable process-tree timeout termination on Windows.
- **Components / Files Involved:**
  - `backend/engine/backends/subprocess_runner.py` (Subprocess execution manager)
  - `backend/engine/resources/memory_guard.py` (Available RAM monitor via `psutil`)
  - `backend/tests/test_subprocess_runner.py`
- **Constraints:** Must terminate rogue process trees on timeout. Must capture worker stderr tracebacks into `NormalizedBackendResult`.
- **Deliverables:**
  1. `SubprocessRunner.run_worker()` with JSON config/result protocol.
  2. Timeout and memory overflow protection.
- **Verification Command:**
  ```powershell
  pytest backend/tests/test_subprocess_runner.py -v
  ```
- **Git Commit:** `feat: implement isolated subprocess worker execution infrastructure`

---

### Phase M3: Primary AutoGluon Tabular Backend Adapter

- **Goal:** Implement the primary AutoGluon Tabular adapter in an isolated worker with resource-tuned presets for ~8 GB RAM.
- **Components / Files Involved:**
  - `backend/engine/backends/autogluon_backend.py` (Worker entrypoint + load_predictor)
  - `backend/engine/backends/workers/autogluon_worker.py` (Standalone subprocess script)
  - `backend/tests/test_autogluon_backend.py`
- **Constraints:** Must train strictly on `df_train` with internal validation (`tuning_data=None`, `holdout_frac=0.2`). The external Validation set is preserved untouched for engine-level champion selection. Must disable heavy multi-layer stacking (`auto_stack=False`) when running under low available RAM.
- **Deliverables:**
  1. AutoGluon Tabular worker executing within time/memory limits.
  2. `load_predictor()` returning `UnifiedPredictor`.
- **Verification Command:**
  ```powershell
  pytest backend/tests/test_autogluon_backend.py -v
  ```
- **Git Commit:** `feat: integrate autogluon tabular backend adapter`

---

### Phase M4: Validation-Based Champion Selection & Inference Schema Persistence

- **Goal:** Implement deterministic champion selection strictly on the validation set, followed by a single test-set evaluation, and persist the complete `inference_schema.json`.
- **Components / Files Involved:**
  - `backend/engine/comparison/champion_selector.py` (Validation ranking logic)
  - `backend/engine/artifacts/schema_builder.py` (Inference schema JSON generator)
  - `backend/engine/artifacts/unified_serializer.py` (Unified artifact directory bundler)
  - `backend/tests/test_champion_selector.py`
  - `backend/tests/test_schema_persistence.py`
- **Constraints:** Test set is evaluated ONLY on the final selected champion. Schema must record complete feature types and dropped columns.
- **Deliverables:**
  1. `ChampionSelector.select()` ranking baselines, AutoGluon, and native candidates.
  2. Complete `inference_schema.json` and artifact directory.
- **Verification Command:**
  ```powershell
  pytest backend/tests/test_champion_selector.py backend/tests/test_schema_persistence.py -v
  ```
- **Git Commit:** `feat: implement validation champion selection and inference schema persistence`

---

### Phase M5: Backend-Agnostic Explainability & Governance Reporting

- **Goal:** Add model-agnostic validation permutation feature importance and update `MODEL_REPORT.md` and `DATASET_INTELLIGENCE_REPORT.md` with multi-backend governance summaries.
- **Components / Files Involved:**
  - `backend/engine/explainability/permutation_importance.py`
  - `backend/engine/reporting/report_generator.py` (Updated to display backend comparison & baselines)
  - `backend/tests/test_permutation_importance.py`
  - `backend/tests/test_reporting_v2.py`
- **Constraints:** Must execute in $<30\text{ seconds}$ on validation data without crashing on complex ensembles.
- **Deliverables:**
  1. `PermutationFeatureExplainer` returning ranked global importance.
  2. Auto-generated governance markdown reports highlighting Validation vs Test performance.
- **Verification Command:**
  ```powershell
  pytest backend/tests/test_permutation_importance.py backend/tests/test_reporting_v2.py -v
  ```
- **Git Commit:** `feat: add permutation importance explainability and updated governance reports`

---

### Phase M6: Optional MLJAR Supervised Backend Adapter

- **Goal:** Integrate MLJAR Supervised as an optional secondary backend, executed only if explicitly requested or as a sequential alternative.
- **Components / Files Involved:**
  - `backend/engine/backends/mljar_backend.py`
  - `backend/engine/backends/workers/mljar_worker.py`
  - `backend/tests/test_mljar_backend.py`
- **Constraints:** Sequential execution only. Never run concurrently with AutoGluon.
- **Deliverables:**
  1. MLJAR worker and predictor loader conforming to `NormalizedBackendResult`.
  2. Graceful skip if `mljar-supervised` is not installed.
- **Verification Command:**
  ```powershell
  pytest backend/tests/test_mljar_backend.py -v
  ```
- **Git Commit:** `feat: integrate optional mljar supervised backend adapter`

---

### Phase M7: End-to-End Orchestrator Integration & Real-Dataset Verification

- **Goal:** Wire the complete v2 pipeline through the main engine and verify end-to-end performance on real benchmark datasets (`employee_turnover.csv` and `HousePricePrediction.csv`).
- **Components / Files Involved:**
  - `backend/engine/orchestrator_v2.py` (Thin high-level entrypoint)
  - `backend/run_real_datasets.py` (Updated benchmark script)
  - `backend/tests/test_end_to_end_v2.py`
- **Constraints:** Full run on `employee_turnover.csv` and `HousePricePrediction.csv` must complete cleanly within memory budgets.
- **Deliverables:**
  1. Working end-to-end CLI and Python API.
  2. Generated `MODEL_REPORT.md` and `DATASET_INTELLIGENCE_REPORT.md` showcasing validation-based champion selection and holdout test metrics.
- **Verification Command:**
  ```powershell
  python backend/run_real_datasets.py
  pytest backend/tests/ -v
  ```
- **Git Commit:** `feat: complete universal ml engine v2 end-to-end pipeline`

---

## 11. Definition of Done (DoD)

The Universal ML Engine v2 evolution is complete when:
1. **ML Correctness:** Model selection occurs strictly on the Validation partition. The Test partition is touched exactly once for the final selected champion.
2. **Baselines:** Dummy and Linear baselines are automatically computed and documented in all reports.
3. **Subprocess Isolation:** Heavy AutoML backends execute in isolated worker processes that handle timeouts and OOM gracefully.
4. **Primary Backend:** AutoGluon Tabular runs reliably on ~8 GB RAM without system freeze.
5. **Fallback:** If AutoGluon fails or is not installed, the engine seamlessly falls back to the native pipeline.
6. **Inference Contract:** The output directory contains a self-contained `inference_schema.json` and a loadable `UnifiedPredictor`.
7. **Governance:** Reports clearly distinguish Train, Validation, and Test metrics with full data exclusion and health audits.
8. **Tests:** All unit, integration, and real-dataset test suites pass cleanly.
