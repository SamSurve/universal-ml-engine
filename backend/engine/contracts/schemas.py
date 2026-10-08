from enum import Enum
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field


class ProblemType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    ID_OR_CONSTANT = "id_or_constant"
    UNKNOWN = "unknown"


class QualitySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ExclusionRecord:
    column_name: str
    reason: str
    action: str = "dropped"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ColumnProfile:
    name: str
    detected_type: ColumnType
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_ratio: float
    is_constant: bool
    is_target: bool = False
    sample_values: List[Any] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetProfile:
    row_count: int
    column_count: int
    column_profiles: Dict[str, ColumnProfile]
    duplicate_rows_count: int
    missing_cells_count: int
    memory_usage_mb: float


@dataclass
class ProblemDetectionResult:
    problem_type: ProblemType
    confidence: float
    reason: str
    target_name: str
    unique_target_values: int
    class_distribution: Optional[Dict[str, int]] = None


@dataclass
class ValidationResult:
    is_valid: bool
    cleaned_rows: int
    initial_rows: int
    exclusions: List[ExclusionRecord]
    warnings: List[str]
    errors: List[str]


@dataclass
class FoldMetric:
    fold: int
    train_score: float
    val_score: float
    metrics: Dict[str, float]
    fit_time_seconds: float


@dataclass
class ModelEvaluationResult:
    model_name: str
    problem_type: ProblemType
    primary_metric: str
    mean_cv_score: float
    std_cv_score: float
    fold_metrics: List[FoldMetric]
    aggregated_metrics: Dict[str, float]
    total_fit_time: float
    status: str = "success"
    error_message: Optional[str] = None


@dataclass
class LeaderboardEntry:
    rank: int
    model_name: str
    cv_score_mean: float
    cv_score_std: float
    primary_metric: str
    summary_metrics: Dict[str, float]
    fit_time_seconds: float
    status: str


@dataclass
class ScreeningCandidate:
    model_name: str
    cv_score: float
    fit_time_seconds: float
    rank: int
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    error_message: Optional[str] = None


@dataclass
class TunedCandidate:
    model_name: str
    baseline_cv_score: float
    tuned_cv_score: float
    best_params: Dict[str, Any]
    n_trials: int
    tuning_time_seconds: float
    improvement: float
    cv_score_std: float = 0.0
    status: str = "success"
    error_message: Optional[str] = None


# --- Milestone 5: Dataset Intelligence & Health Contracts ---

@dataclass
class QualityWarning:
    warning_id: str
    severity: QualitySeverity
    affected_columns: List[str]
    reason: str
    recommended_action: str


@dataclass
class HealthCategoryScore:
    category: str
    score: float
    weight: float
    description: str
    deductions: List[str] = field(default_factory=list)


@dataclass
class DatasetHealthScore:
    overall_score: float
    grade: str
    categories: Dict[str, HealthCategoryScore]
    major_warnings: List[QualityWarning]
    clean_signals: List[str]
    interpretation: str


@dataclass
class CorrelatedPair:
    feature_a: str
    feature_b: str
    correlation: float


@dataclass
class DatasetIntelligenceResult:
    dataset_shape: tuple
    target_column: str
    problem_type: ProblemType
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    text_columns: List[str]
    missing_percentages: Dict[str, float]
    duplicate_rows_count: int
    constant_columns: List[str]
    near_constant_columns: List[str]
    high_cardinality_columns: List[str]
    class_imbalance: Optional[Dict[str, Any]]
    suspicious_id_columns: List[str]
    target_leakage_risks: List[str]
    correlated_numeric_pairs: List[CorrelatedPair]
    special_handling_columns: Dict[str, str]
    warnings: List[QualityWarning]
    health_score: DatasetHealthScore


# --- Milestone 5: Explainability & SHAP Contracts ---

@dataclass
class FeatureImportanceEntry:
    feature_name: str
    raw_feature_name: str
    importance_score: float
    relative_importance_pct: float
    rank: int


@dataclass
class PredictionContribution:
    feature_name: str
    raw_feature_name: str
    feature_value: Any
    shap_value: float
    direction: str


@dataclass
class PredictionExplanation:
    sample_index: int
    predicted_value: Any
    predicted_probability: Optional[Dict[str, float]]
    base_value: float
    contributions: List[PredictionContribution]
    top_positive_features: List[str]
    top_negative_features: List[str]


@dataclass
class ExplainabilityResult:
    status: str
    explainer_type: str
    global_importance: List[FeatureImportanceEntry]
    raw_feature_importance: List[FeatureImportanceEntry]
    example_explanations: List[PredictionExplanation]
    warnings: List[str] = field(default_factory=list)
    transformed_feature_names: List[str] = field(default_factory=list)
    summary_text: str = ""


# --- Milestone 5: Decision Summary & Reporting Contracts ---

@dataclass
class ModelDecisionSummary:
    problem_type: str
    target_column: str
    primary_metric: str
    models_screened_count: int
    models_considered: List[str]
    selected_model_name: str
    selection_rationale: str
    best_cv_score: float
    cv_score_std: float
    holdout_metrics: Dict[str, float]
    tuned_hyperparameters: Optional[Dict[str, Any]]
    training_runtime_seconds: float
    tuning_runtime_seconds: float
    total_runtime_seconds: float
    dataset_risks: List[str]
    model_limitations: List[str]
    explainability_available: bool
    top_features_summary: List[str]


@dataclass
class ExperimentResult:
    experiment_id: str
    dataset_name: str
    target_column: str
    problem_detection: ProblemDetectionResult
    dataset_profile: DatasetProfile
    validation_result: ValidationResult
    dev_shape: tuple
    holdout_shape: tuple
    leaderboard: List[LeaderboardEntry]
    best_model_name: str
    best_model_cv_score: float
    holdout_metrics: Dict[str, float]
    artifact_path: Optional[str] = None
    metadata_path: Optional[str] = None
    random_state: int = 42
    screened_candidates: List[ScreeningCandidate] = field(default_factory=list)
    tuned_candidates: List[TunedCandidate] = field(default_factory=list)
    tuned_hyperparameters: Optional[Dict[str, Any]] = None
    screening_time_seconds: float = 0.0
    tuning_time_seconds: float = 0.0
    total_runtime_seconds: float = 0.0
    # Milestone 5 extensions:
    dataset_intelligence: Optional[DatasetIntelligenceResult] = None
    explainability: Optional[ExplainabilityResult] = None
    decision_summary: Optional[ModelDecisionSummary] = None
    intelligence_report_path: Optional[str] = None
    model_report_path: Optional[str] = None


# --- Milestone 1 (v2): 3-Way Partition & Baseline Contracts ---

@dataclass
class PartitionInfo:
    train_rows: int
    val_rows: int
    test_rows: int
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    train_sha256: str
    val_sha256: str
    test_sha256: str
    stratified: bool
    random_state: int
    target_column: str
    problem_type: ProblemType


@dataclass
class BaselineModelResult:
    model_name: str
    problem_type: ProblemType
    primary_metric: str
    val_score: float
    val_metrics: Dict[str, float]
    fit_time_seconds: float
    status: str = "success"
    error_message: Optional[str] = None


@dataclass
class BaselineSuiteResult:
    problem_type: ProblemType
    primary_metric: str
    dummy_baseline: BaselineModelResult
    linear_baseline: BaselineModelResult
    best_baseline_name: str
    best_baseline_score: float
    all_baselines: List[BaselineModelResult]


# --- Milestone 2 (v2): Subprocess Worker Infrastructure Contracts ---

class WorkerStatus(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    MEMORY_EXCEEDED = "memory_exceeded"
    FAILED = "failed"
    CRASHED = "crashed"


@dataclass
class WorkerJobSpec:
    job_id: str
    worker_module: str
    job_type: str
    time_limit_seconds: float = 300.0
    memory_limit_mb: Optional[float] = None
    python_executable: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerResult:
    job_id: str
    status: WorkerStatus
    exit_code: Optional[int]
    runtime_seconds: float
    peak_memory_mb: float
    stdout: str = ""
    stderr: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class AutoGluonBackendResult:
    backend_name: str
    status: WorkerStatus
    model_best: Optional[str]
    problem_type: ProblemType
    eval_metric: str
    artifact_path: str
    fit_time_seconds: float
    peak_memory_mb: float
    class_labels: Optional[List[Any]] = None
    leaderboard: List[Dict[str, Any]] = field(default_factory=list)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    autogluon_version: Optional[str] = None
    error_message: Optional[str] = None


