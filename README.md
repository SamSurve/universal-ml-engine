# Universal ML Engine

A production-oriented AutoML platform engineered for automated end-to-end tabular machine learning, dataset quality intelligence, and explainable AI governance.

Universal ML Engine automatically validates raw tabular data (CSV, XLSX, XLS), profiles schema and missingness, audits dataset health (0–100 diagnostic score), detects problem types (Binary Classification, Multiclass Classification, Regression), enforces strict zero-leakage preprocessing, screens candidate models with **FLAML**, benchmarks algorithms across 5-fold cross-validation, tunes top candidates with **Optuna**, selects the best model deterministically, evaluates on an untouched holdout set, computes global and local feature explanations with **SHAP**, and auto-generates comprehensive model governance and dataset intelligence markdown reports.

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
   [4] Dataset Intelligence & Health Diagnostic (0–100 Health Score)
       ├── Completeness, Uniqueness, Feature Quality, Target Integrity
       └── Risk Auditing (Multicollinearity, High Cardinality, ID Leakage)
               │
               ▼
   [5] Problem Detector (Binary vs Multiclass vs Regression)
               │
               ▼
   [6] Leakage Guard (ID Columns, Constant Features, Target Proxies)
               │
               ▼
   [7] 80% Dev / 20% Untouched Holdout Split
               │
               ▼
   [8] FLAML Rapid Candidate Screening (Default: 60s budget on Dev data)
       ├── Evaluates LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees
       └── Identifies high-performing model families under time constraints
               │
               ▼
   [9] Leakage-Safe 5-Fold Cross-Validation (Preprocessors fit inside folds)
       ├── Scikit-learn Baselines (LogisticRegression, Ridge, RF, ExtraTrees, HistGB)
       └── Gradient Boosters (LightGBM, XGBoost, CatBoost)
               │
               ▼
  [10] Optuna Bayesian Hyperparameter Optimization (TPE + MedianPruner)
       ├── Tunes top K candidates across 5-fold CV
       └── Evaluates full preprocessing + estimator pipelines per trial
               │
               ▼
  [11] Leaderboard & Deterministic Model Selection (Dev CV Metrics)
               │
               ▼
  [12] Dev Refit & Final Holdout Evaluation (Untouched 20%)
               │
               ▼
  [13] Explainable AI Layer (SHAP: Tree, Linear & Kernel Explainers)
       ├── Global Feature Attribution (One-hot mapped to parent features)
       └── Local Per-Prediction Explanations (Base values & directional drivers)
               │
               ▼
  [14] Decision Summary & Auto-Generated Governance Reports
       ├── MODEL_REPORT.md (Complete executive & technical governance report)
       └── DATASET_INTELLIGENCE_REPORT.md (Data health audit report)
               │
               ▼
  [15] Serialized Artifact (`model.joblib` + `metadata.json`)
```

---

## Core Principles & Guarantees

1. **Zero Data Leakage by Design:** All feature imputers, encoders, and scalers are strictly fitted inside training folds. No transformation ever peeks across validation or test boundaries.
2. **Untouched Final Holdout:** Final holdout data (20%) is strictly isolated before screening, baseline cross-validation, or Optuna tuning begins. It is evaluated exactly once after final model selection to guarantee unbiased out-of-sample metrics.
3. **Transparent Exclusion Tracking:** No data is silently discarded. Every removed row (missing target, duplicates) or dropped column (constant, ID, proxy) is explicitly logged as an `ExclusionRecord` with full justification.
4. **Actionable Dataset Health Score:** Diagnostic health score (0–100) evaluates Completeness, Uniqueness, Feature Quality, and Target Integrity with deterministic deductions and clean signals.
5. **Model-Agnostic & Pipeline-Aware SHAP Explanations:** Employs appropriate SHAP explainers (`TreeExplainer`, `LinearExplainer`, `KernelExplainer`) on fitted pipelines, cleans one-hot encoding prefixes, and aggregates parent feature importance.
6. **Automated Documentation & Governance:** Every experiment run generates both a technical `MODEL_REPORT.md` and a data audit `DATASET_INTELLIGENCE_REPORT.md` with complete provenance.

---

## Python API Usage

```python
from backend.engine.orchestrator import AutoMLEngine

engine = AutoMLEngine()

result = engine.run(
    data_source="employee_turnover.csv",
    target_column="Employee_Turnover",
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
    # Intelligence & Explainability Options (Milestone 5)
    enable_intelligence=True,      # Data quality audit and health score
    enable_explainability=True,    # SHAP global and local feature attribution
    generate_reports=True,         # Auto-generate MODEL_REPORT.md & DATASET_INTELLIGENCE_REPORT.md
)

print(f"Problem Type: {result.problem_detection.problem_type.value}")
print(f"Health Score: {result.dataset_intelligence.health_score.overall_score}/100 ({result.dataset_intelligence.health_score.grade})")
print(f"Best Model: {result.best_model_name}")
print(f"CV Score: {result.leaderboard[0].cv_score_mean:.4f}")
print(f"Holdout Metrics: {result.holdout_metrics}")
print(f"SHAP Explainer: {result.explainability.explainer_type}")
print(f"Model Report: {result.model_report_path}")
print(f"Saved Artifact: {result.artifact_path}")
```

---

## Web Dashboard & User Interface

Universal ML Engine includes a clean, high-contrast, lightweight web dashboard for dataset inspection, AutoML orchestration, validation vs final test metrics comparison, diagnostic visualizations, and real-time model inference.

### Quick Start (Fresh Clone Setup)

#### 1. Clone the Repository
```bash
git clone https://github.com/SamSurve/universal-ml-engine.git
cd universal-ml-engine
```

#### 2. Environment Requirements
* **Python Version:** `Python 3.12` (or `3.11`) 64-bit
* **Operating System:** Windows 10/11, macOS, or Linux

#### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```
*(On Windows, you can also double-click `setup_windows.bat`)*

#### 4. (Optional) AutoGluon Worker Environment
If you wish to use the dedicated AutoGluon tabular booster worker in an isolated virtual environment:
```powershell
python -m venv .venvs\autogluon
.\.venvs\autogluon\Scripts\pip install autogluon.tabular
```
*Note: If `.venvs\autogluon` is omitted, the engine automatically uses standard high-performance boosters (XGBoost, LightGBM, CatBoost, Scikit-learn baselines) and falls back gracefully to `sys.executable`.*

#### 5. Launch the Web Interface
```bash
python run_ui.py --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## Running Benchmarks on Real Datasets

```bash
python backend/run_real_datasets.py
```

### Running Automated Test Suite
```bash
pytest backend/tests/ -v
```

---

## Third-Party Open-Source Components & Licenses

| Library | Version | License | Architectural Role |
| :--- | :--- | :--- | :--- |
| **Scikit-learn** | `>=1.4.0` | BSD-3-Clause | Core ML interface, `ColumnTransformer`, `Pipeline`, CV splitters, baselines, and metrics. |
| **FLAML** | `>=2.1.0` | MIT | Fast Cost-Frugal Optimization (CFO) model screening. |
| **Optuna** | `>=4.0.0` | MIT | Bayesian Hyperparameter Optimization (TPE sampler + MedianPruner). |
| **SHAP** | `>=0.46.0` | MIT | Model-agnostic and tree-based Shapley value explainability (Tree, Linear, and Kernel explainers). |
| **LightGBM** | `>=4.3.0` | MIT | High-performance tree-based gradient boosting. |
| **XGBoost** | `>=2.0.0` | Apache-2.0 | High-performance gradient boosted decision trees. |
| **CatBoost** | `>=1.2.0` | Apache-2.0 | Categorical-aware gradient boosting. |
| **Pandas** | `>=2.2.0` | BSD-3-Clause | Tabular ingestion, dataframe manipulation, series statistics. |
| **NumPy** | `>=1.26.0` | BSD-3-Clause | Vectorized numerical computing and matrix operations. |
| **Joblib** | `>=1.3.0` | BSD-3-Clause | Production pipeline serialization and disk persistence. |
| **Openpyxl / Xlrd** | `>=3.1.0` | MIT / BSD | Safe Excel `.xlsx` and legacy `.xls` spreadsheet parsing. |
