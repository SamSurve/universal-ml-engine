import pytest
from pathlib import Path
import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import (
    ProblemType,
    LeaderboardEntry,
    ValidationResult,
    DatasetProfile,
    ProblemDetectionResult,
    ExperimentResult,
)
from backend.engine.intelligence.data_intelligence import DatasetIntelligenceAnalyzer
from backend.engine.reporting.decision_summary import DecisionSummaryBuilder
from backend.engine.reporting.report_generator import ReportGenerator


def test_decision_summary_and_report_generation(tmp_path):
    # Setup mock ExperimentResult
    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "target": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })

    intel = DatasetIntelligenceAnalyzer.analyze(
        df=df, target_column="target", problem_type=ProblemType.REGRESSION
    )

    leaderboard = [
        LeaderboardEntry(
            rank=1,
            model_name="RandomForestRegressor",
            cv_score_mean=0.85,
            cv_score_std=0.04,
            primary_metric="r2",
            summary_metrics={"r2": 0.85},
            fit_time_seconds=1.2,
            status="success",
        ),
        LeaderboardEntry(
            rank=2,
            model_name="Ridge",
            cv_score_mean=0.81,
            cv_score_std=0.03,
            primary_metric="r2",
            summary_metrics={"r2": 0.81},
            fit_time_seconds=0.1,
            status="success",
        ),
    ]

    summary = DecisionSummaryBuilder.build_summary(
        dataset_name="mock_data.csv",
        target_column="target",
        problem_type=ProblemType.REGRESSION,
        primary_metric="r2",
        leaderboard=leaderboard,
        best_model_name="RandomForestRegressor",
        best_cv_score=0.85,
        cv_score_std=0.04,
        holdout_metrics={"r2": 0.88, "rmse": 4.5},
        tuned_hyperparameters={"n_estimators": 100},
        training_runtime_seconds=1.3,
        tuning_runtime_seconds=0.0,
        total_runtime_seconds=2.5,
        dataset_intelligence=intel,
        explainability=None,
    )

    assert summary.selected_model_name == "RandomForestRegressor"
    assert "0.85" in summary.selection_rationale
    assert len(summary.model_limitations) > 0

    exp_res = ExperimentResult(
        experiment_id="test_exp_123",
        dataset_name="mock_data.csv",
        target_column="target",
        problem_detection=ProblemDetectionResult(
            problem_type=ProblemType.REGRESSION,
            confidence=1.0,
            reason="Continuous values",
            target_name="target",
            unique_target_values=6,
        ),
        dataset_profile=DatasetProfile(
            row_count=6,
            column_count=2,
            column_profiles={},
            duplicate_rows_count=0,
            missing_cells_count=0,
            memory_usage_mb=0.01,
        ),
        validation_result=ValidationResult(
            is_valid=True,
            cleaned_rows=6,
            initial_rows=6,
            exclusions=[],
            warnings=[],
            errors=[],
        ),
        dev_shape=(5, 2),
        holdout_shape=(1, 2),
        leaderboard=leaderboard,
        best_model_name="RandomForestRegressor",
        best_model_cv_score=0.85,
        holdout_metrics={"r2": 0.88, "rmse": 4.5},
        dataset_intelligence=intel,
        decision_summary=summary,
    )

    model_rep_path = tmp_path / "MODEL_REPORT.md"
    intel_rep_path = tmp_path / "DATASET_INTELLIGENCE_REPORT.md"

    ReportGenerator.generate_model_report(exp_res, model_rep_path)
    ReportGenerator.generate_intelligence_report(exp_res, intel_rep_path)

    assert model_rep_path.exists()
    assert intel_rep_path.exists()

    model_rep_content = model_rep_path.read_text(encoding="utf-8")
    assert "Executive Summary" in model_rep_content
    assert "Problem Detection" in model_rep_content
    assert "Cross-Validation Results" in model_rep_content
    assert "Final Holdout Evaluation" in model_rep_content
    assert "Model Limitations" in model_rep_content

    intel_rep_content = intel_rep_path.read_text(encoding="utf-8")
    assert "Dataset Intelligence & Health Report" in intel_rep_content
    assert "Health Diagnostic" in intel_rep_content
