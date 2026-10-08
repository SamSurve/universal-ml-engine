import os
import uuid
import time
import logging
from typing import Union, Optional, Dict, Any, Tuple, List
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from backend.engine.contracts.schemas import (
    ProblemType,
    ExperimentResult,
    ValidationResult,
    DatasetProfile,
    ProblemDetectionResult,
    LeaderboardEntry,
    ScreeningCandidate,
    TunedCandidate,
    DatasetIntelligenceResult,
    ExplainabilityResult,
    ModelDecisionSummary,
)
from backend.engine.ingestion.service import IngestionService, DataValidator
from backend.engine.profiler.analyzer import DatasetProfiler
from backend.engine.problem_detection.detector import ProblemDetector
from backend.engine.leakage.leakage_guard import LeakageGuard
from backend.engine.intelligence.data_intelligence import DatasetIntelligenceAnalyzer
from backend.engine.preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from backend.engine.models.registry import ModelRegistry
from backend.engine.training.cv_runner import CrossValidationRunner
from backend.engine.evaluation.evaluator import MetricsEvaluator
from backend.engine.artifacts.serializer import ModelArtifact, ArtifactManager
from backend.engine.automl.flaml_screener import FLAMLScreener
from backend.engine.automl.optuna_tuner import OptunaTuner
from backend.engine.explainability.shap_explainer import ModelExplainer
from backend.engine.reporting.decision_summary import DecisionSummaryBuilder
from backend.engine.reporting.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class AutoMLEngine:
    """
    Unified Orchestrator for the Universal ML Engine.
    Executes the complete end-to-end pipeline:
    Validate -> Clean -> Profiling -> Dataset Intelligence & Health Score
    -> Detect Problem -> Leakage Guard -> 80/20 Dev/Holdout Split
    -> FLAML Screening -> Multi-Model CV Baseline -> Optuna Bayesian Tuning
    -> Leaderboard Ranking -> Best Model Selection -> Dev Refit
    -> Single Holdout Evaluation -> SHAP Explainability -> Decision Summary
    -> Markdown Reports -> Serialization
    """

    @classmethod
    def run(
        cls,
        data_source: Union[str, Path, pd.DataFrame],
        target_column: str,
        output_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        random_state: int = 42,
        n_splits: int = 5,
        enable_screening: bool = True,
        screening_time_budget: int = 60,
        screening_top_k: int = 3,
        enable_tuning: bool = True,
        top_k_to_tune: int = 2,
        tuning_trials: int = 30,
        tuning_time_budget: int = 120,
        enable_intelligence: bool = True,
        enable_explainability: bool = True,
        generate_reports: bool = True,
    ) -> ExperimentResult:
        experiment_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # 1. Ingestion
        if isinstance(data_source, (str, Path)):
            dataset_name = dataset_name or Path(data_source).name
            df_raw = IngestionService.load_dataset(data_source)
        elif isinstance(data_source, pd.DataFrame):
            dataset_name = dataset_name or "in_memory_dataframe"
            df_raw = data_source.copy()
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")

        # 2. Validation & Initial Cleaning (record all exclusions)
        df_cleaned, validation_result = DataValidator.validate_and_clean(
            df_raw, target_column=target_column
        )
        if not validation_result.is_valid:
            raise ValueError(f"Data validation failed: {validation_result.errors}")

        # 3. Initial Profiling
        initial_profile = DatasetProfiler.profile_dataset(df_cleaned, target_column=target_column)

        # 4. Problem Detection
        problem_detection = ProblemDetector.detect(df_cleaned, target_column=target_column)
        if problem_detection.problem_type == ProblemType.UNKNOWN:
            raise ValueError(
                f"Could not determine problem type for target '{target_column}': {problem_detection.reason}"
            )

        # 5. Dataset Intelligence & Health Score (Milestone 5)
        dataset_intelligence: Optional[DatasetIntelligenceResult] = None
        if enable_intelligence:
            try:
                dataset_intelligence = DatasetIntelligenceAnalyzer.analyze(
                    df=df_cleaned,
                    target_column=target_column,
                    problem_type=problem_detection.problem_type,
                    column_profiles=initial_profile.column_profiles,
                )
            except Exception as e:
                logger.warning(f"Dataset intelligence analysis failed: {e}")

        # 6. Leakage Guard & High-Risk Feature Filtering
        df_safe, leakage_exclusions = LeakageGuard.audit_and_filter(
            df_cleaned,
            target_column=target_column,
            problem_type=problem_detection.problem_type,
        )
        validation_result.exclusions.extend(leakage_exclusions)

        # Update profile with safe features
        final_profile = DatasetProfiler.profile_dataset(df_safe, target_column=target_column)

        # 7. Train/Test Split (80% Dev / 20% Untouched Holdout)
        df_dev, df_holdout = CrossValidationRunner.split_dev_holdout(
            df_safe,
            target_column=target_column,
            problem_type=problem_detection.problem_type,
            holdout_ratio=0.20,
            random_state=random_state,
        )

        # Determine primary metric for ranking
        imbalance_ratio = 1.0
        if problem_detection.class_distribution and len(problem_detection.class_distribution) >= 2:
            counts = list(problem_detection.class_distribution.values())
            imbalance_ratio = max(counts) / min(counts)

        primary_metric = MetricsEvaluator.get_primary_metric_name(
            problem_detection.problem_type, class_imbalance_ratio=imbalance_ratio
        )
        higher_is_better = primary_metric in [
            "f1", "macro_f1", "r2", "accuracy", "balanced_accuracy", "roc_auc"
        ]

        # 8. FLAML Fast Candidate Screening
        screened_candidates: List[ScreeningCandidate] = []
        screening_start = time.time()
        if enable_screening:
            try:
                screened_candidates = FLAMLScreener.screen(
                    df_dev=df_dev,
                    target_column=target_column,
                    problem_type=problem_detection.problem_type,
                    column_profiles=final_profile.column_profiles,
                    time_budget_seconds=screening_time_budget,
                    n_splits=n_splits,
                    random_state=random_state,
                )
            except Exception as e:
                logger.warning(f"FLAML screening skipped due to exception: {e}")
        screening_time = time.time() - screening_start

        # 9. Baseline Cross-Validation on Development Data (Strict Zero-Leakage)
        all_candidate_models = ModelRegistry.get_models(
            problem_type=problem_detection.problem_type,
            random_state=random_state,
        )
        if not all_candidate_models:
            raise RuntimeError(f"No candidate models available for problem type {problem_detection.problem_type}.")

        # Select Top-K candidate models based on FLAML screening ranking
        successful_screened = [c for c in screened_candidates if c.status == "success"]
        if enable_screening and successful_screened and screening_top_k > 0:
            top_screened_names = [c.model_name for c in successful_screened[:screening_top_k]]
            candidate_models = {
                name: model for name, model in all_candidate_models.items()
                if name in top_screened_names
            }
            if not candidate_models:
                logger.warning("No FLAML screened candidates matched registry models; falling back to all models.")
                candidate_models = all_candidate_models
            else:
                logger.info(
                    f"FLAML screening selected top {len(candidate_models)} candidates for CV: {list(candidate_models.keys())}"
                )
        else:
            candidate_models = all_candidate_models

        cv_results, label_encoder = CrossValidationRunner.run_cv(
            df_dev=df_dev,
            target_column=target_column,
            problem_type=problem_detection.problem_type,
            models=candidate_models,
            column_profiles=final_profile.column_profiles,
            n_splits=n_splits,
            random_state=random_state,
        )

        # 10. Initial Leaderboard Ranking
        leaderboard = MetricsEvaluator.build_leaderboard(cv_results, primary_metric=primary_metric)
        if not leaderboard or not any(entry.status == "success" for entry in leaderboard):
            raise RuntimeError("All candidate models failed during baseline cross-validation.")

        # 11. Optuna Bayesian Hyperparameter Tuning on Top Candidates
        tuned_candidates: List[TunedCandidate] = []
        tuned_pipelines: Dict[str, Pipeline] = {}
        tuning_start = time.time()

        if enable_tuning:
            # Select top-K models from successful baseline entries
            candidates_to_tune = [
                entry.model_name for entry in leaderboard if entry.status == "success"
            ][:top_k_to_tune]

            for model_name in candidates_to_tune:
                baseline_entry = next((e for e in leaderboard if e.model_name == model_name), None)
                baseline_score = baseline_entry.cv_score_mean if baseline_entry else 0.0

                try:
                    tuned_cand, tuned_pipe = OptunaTuner.tune_model(
                        model_name=model_name,
                        df_dev=df_dev,
                        target_column=target_column,
                        problem_type=problem_detection.problem_type,
                        column_profiles=final_profile.column_profiles,
                        baseline_cv_score=baseline_score,
                        n_trials=tuning_trials,
                        timeout_seconds=tuning_time_budget,
                        n_splits=n_splits,
                        random_state=random_state,
                    )
                    tuned_candidates.append(tuned_cand)

                    if tuned_cand.status == "success" and tuned_cand.improvement > 0:
                        tuned_pipelines[model_name] = tuned_pipe
                        # Add tuned model entry to leaderboard
                        leaderboard.append(
                            LeaderboardEntry(
                                rank=0,
                                model_name=f"{model_name} (Tuned)",
                                cv_score_mean=tuned_cand.tuned_cv_score,
                                cv_score_std=getattr(tuned_cand, "cv_score_std", 0.0),
                                primary_metric=primary_metric,
                                summary_metrics={primary_metric: tuned_cand.tuned_cv_score},
                                fit_time_seconds=tuned_cand.tuning_time_seconds,
                                status="success",
                            )
                        )
                except Exception as e:
                    logger.warning(f"Optuna tuning for '{model_name}' failed: {e}")
                    tuned_candidates.append(
                        TunedCandidate(
                            model_name=model_name,
                            baseline_cv_score=baseline_score,
                            tuned_cv_score=baseline_score,
                            best_params={},
                            n_trials=0,
                            tuning_time_seconds=0.0,
                            improvement=0.0,
                            status="failed",
                            error_message=str(e),
                        )
                    )

        tuning_time = time.time() - tuning_start

        # Re-sort leaderboard with tuned models included
        leaderboard.sort(
            key=lambda e: (e.status == "success", e.cv_score_mean if higher_is_better else -e.cv_score_mean),
            reverse=True,
        )
        for rank_idx, entry in enumerate(leaderboard):
            entry.rank = rank_idx + 1

        best_entry = leaderboard[0]
        best_model_name = best_entry.model_name

        # 12. Final Model Pipeline Preparation
        X_dev = df_dev.drop(columns=[target_column])
        y_dev = df_dev[target_column]
        if label_encoder is not None:
            y_dev_encoded = label_encoder.transform(y_dev.astype(str))
        else:
            y_dev_encoded = y_dev.to_numpy(dtype=float)

        num_cols, cat_cols, date_cols = PreprocessingPipelineBuilder.identify_feature_types(
            X_dev, final_profile.column_profiles
        )

        tuned_hyperparameters = None
        if "(Tuned)" in best_model_name:
            raw_model_name = best_model_name.replace(" (Tuned)", "")
            final_pipeline = tuned_pipelines[raw_model_name]
            matched_tuned = next((tc for tc in tuned_candidates if tc.model_name == raw_model_name), None)
            if matched_tuned:
                tuned_hyperparameters = matched_tuned.best_params
        else:
            # Baseline model won: refit baseline estimator on 100% development data
            best_model_instance = candidate_models[best_model_name]
            scale_features = "Linear" in best_model_name or "Ridge" in best_model_name or "Logistic" in best_model_name
            refit_preprocessor = PreprocessingPipelineBuilder.build_preprocessor(
                num_cols, cat_cols, date_cols, scale_numeric=scale_features
            )
            final_pipeline = Pipeline([
                ("preprocessor", refit_preprocessor),
                ("model", clone(best_model_instance)),
            ])
            final_pipeline.fit(X_dev, y_dev_encoded)

        # 13. Final Evaluation on Untouched 20% Holdout Data
        X_holdout = df_holdout.drop(columns=[target_column])
        y_holdout = df_holdout[target_column]
        if label_encoder is not None:
            known_classes = set(label_encoder.classes_)
            holdout_classes = set(y_holdout.astype(str))
            unseen = holdout_classes - known_classes
            if unseen:
                raise ValueError(
                    f"Target label validation error: Holdout set contains unseen class(es) {sorted(list(unseen))}. "
                    f"Known classes from training: {sorted(list(known_classes))}."
                )
            y_holdout_encoded = label_encoder.transform(y_holdout.astype(str))
        else:
            y_holdout_encoded = y_holdout.to_numpy(dtype=float)

        y_holdout_pred = final_pipeline.predict(X_holdout)

        if problem_detection.problem_type in [
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ]:
            y_holdout_prob = None
            if hasattr(final_pipeline, "predict_proba"):
                try:
                    y_holdout_prob = final_pipeline.predict_proba(X_holdout)
                except Exception:
                    y_holdout_prob = None
            holdout_metrics = MetricsEvaluator.evaluate_classification(
                y_holdout_encoded,
                y_holdout_pred,
                y_holdout_prob,
                is_binary=(problem_detection.problem_type == ProblemType.BINARY_CLASSIFICATION),
            )
        else:
            holdout_metrics = MetricsEvaluator.evaluate_regression(y_holdout_encoded, y_holdout_pred)

        # 14. Explainable AI Layer (Milestone 5)
        explainability_result: Optional[ExplainabilityResult] = None
        if enable_explainability:
            try:
                explainability_result = ModelExplainer.explain_model(
                    pipeline=final_pipeline,
                    df_dev=df_dev,
                    target_column=target_column,
                    problem_type=problem_detection.problem_type,
                    model_name=best_model_name,
                    df_eval=df_holdout,
                    max_background_samples=50,
                    max_eval_samples=50,
                    random_state=random_state,
                )
            except Exception as e:
                logger.warning(f"Model explainability computation failed: {e}")

        # 15. Model Decision Summary (Milestone 5)
        decision_summary = DecisionSummaryBuilder.build_summary(
            dataset_name=dataset_name,
            target_column=target_column,
            problem_type=problem_detection.problem_type,
            primary_metric=primary_metric,
            leaderboard=leaderboard,
            best_model_name=best_model_name,
            best_cv_score=best_entry.cv_score_mean,
            cv_score_std=best_entry.cv_score_std,
            holdout_metrics=holdout_metrics,
            tuned_hyperparameters=tuned_hyperparameters,
            training_runtime_seconds=sum(e.fit_time_seconds for e in leaderboard),
            tuning_runtime_seconds=tuning_time,
            total_runtime_seconds=round(time.time() - start_time, 2),
            dataset_intelligence=dataset_intelligence,
            explainability=explainability_result,
        )

        # 16. Model Artifact Serialization & Metadata
        artifact = ModelArtifact(
            pipeline=final_pipeline,
            problem_type=problem_detection.problem_type,
            target_column=target_column,
            label_encoder=label_encoder,
            feature_names=list(X_dev.columns),
        )

        total_runtime = round(time.time() - start_time, 2)
        run_output_dir = output_dir or f"artifacts/run_{experiment_id}"

        metadata = {
            "experiment_id": experiment_id,
            "dataset_name": dataset_name,
            "target_column": target_column,
            "problem_type": problem_detection.problem_type.value,
            "confidence": problem_detection.confidence,
            "reason": problem_detection.reason,
            "best_model_name": best_model_name,
            "primary_metric": primary_metric,
            "best_cv_score_mean": best_entry.cv_score_mean,
            "best_cv_score_std": best_entry.cv_score_std,
            "tuned_hyperparameters": tuned_hyperparameters,
            "holdout_metrics": holdout_metrics,
            "dev_samples": len(df_dev),
            "holdout_samples": len(df_holdout),
            "features_used": list(X_dev.columns),
            "exclusions_count": len(validation_result.exclusions),
            "exclusions": [
                {"column": ex.column_name, "reason": ex.reason, "action": ex.action}
                for ex in validation_result.exclusions
            ],
            "health_score": dataset_intelligence.health_score.overall_score if dataset_intelligence else None,
            "health_grade": dataset_intelligence.health_score.grade if dataset_intelligence else None,
            "screening": {
                "enabled": enable_screening,
                "time_budget_seconds": screening_time_budget,
                "duration_seconds": round(screening_time, 2),
                "candidates": [
                    {
                        "model": sc.model_name,
                        "cv_score": sc.cv_score,
                        "rank": sc.rank,
                        "fit_time": sc.fit_time_seconds,
                        "status": sc.status,
                    }
                    for sc in screened_candidates
                ],
            },
            "tuning": {
                "enabled": enable_tuning,
                "trials_budget": tuning_trials,
                "time_budget_seconds": tuning_time_budget,
                "duration_seconds": round(tuning_time, 2),
                "candidates": [
                    {
                        "model": tc.model_name,
                        "baseline_score": tc.baseline_cv_score,
                        "tuned_score": tc.tuned_cv_score,
                        "improvement": tc.improvement,
                        "n_trials": tc.n_trials,
                        "best_params": tc.best_params,
                        "duration": tc.tuning_time_seconds,
                        "status": tc.status,
                    }
                    for tc in tuned_candidates
                ],
            },
            "leaderboard": [
                {
                    "rank": entry.rank,
                    "model": entry.model_name,
                    "cv_score": entry.cv_score_mean,
                    "cv_std": entry.cv_score_std,
                    "metrics": entry.summary_metrics,
                    "fit_time_seconds": entry.fit_time_seconds,
                    "status": entry.status,
                }
                for entry in leaderboard
            ],
            "explainability": {
                "status": explainability_result.status if explainability_result else None,
                "explainer_type": explainability_result.explainer_type if explainability_result else None,
                "top_features": [
                    {"feature": e.feature_name, "importance": e.importance_score, "pct": e.relative_importance_pct}
                    for e in (explainability_result.raw_feature_importance[:5] if explainability_result else [])
                ],
            },
            "total_runtime_seconds": total_runtime,
            "random_state": random_state,
        }

        saved_model_path, saved_meta_path = ArtifactManager.save_pipeline(
            artifact=artifact,
            metadata=metadata,
            output_dir=run_output_dir,
        )

        # 17. Experiment Result Construction
        exp_result = ExperimentResult(
            experiment_id=experiment_id,
            dataset_name=dataset_name,
            target_column=target_column,
            problem_detection=problem_detection,
            dataset_profile=final_profile,
            validation_result=validation_result,
            dev_shape=df_dev.shape,
            holdout_shape=df_holdout.shape,
            leaderboard=leaderboard,
            best_model_name=best_model_name,
            best_model_cv_score=best_entry.cv_score_mean,
            holdout_metrics=holdout_metrics,
            artifact_path=saved_model_path,
            metadata_path=saved_meta_path,
            random_state=random_state,
            screened_candidates=screened_candidates,
            tuned_candidates=tuned_candidates,
            tuned_hyperparameters=tuned_hyperparameters,
            screening_time_seconds=round(screening_time, 2),
            tuning_time_seconds=round(tuning_time, 2),
            total_runtime_seconds=total_runtime,
            dataset_intelligence=dataset_intelligence,
            explainability=explainability_result,
            decision_summary=decision_summary,
        )

        # 18. Auto-Generated Reports (Milestone 5)
        if generate_reports:
            out_p = Path(run_output_dir)
            model_rep_path = out_p / "MODEL_REPORT.md"
            intel_rep_path = out_p / "DATASET_INTELLIGENCE_REPORT.md"

            try:
                ReportGenerator.generate_model_report(exp_result, model_rep_path)
                exp_result.model_report_path = str(model_rep_path)
            except Exception as e:
                logger.warning(f"Failed to generate model report: {e}")

            if dataset_intelligence:
                try:
                    ReportGenerator.generate_intelligence_report(exp_result, intel_rep_path)
                    exp_result.intelligence_report_path = str(intel_rep_path)
                except Exception as e:
                    logger.warning(f"Failed to generate dataset intelligence report: {e}")

        return exp_result
