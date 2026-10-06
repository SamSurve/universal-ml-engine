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
    status: str = "success"
    error_message: Optional[str] = None


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
