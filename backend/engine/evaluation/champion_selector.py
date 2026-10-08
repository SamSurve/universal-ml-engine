import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import (
    ProblemType,
    AutoGluonBackendResult,
    BaselineSuiteResult,
    BaselineModelResult,
    ChampionCandidate,
    ChampionSelectionResult,
)
from backend.engine.evaluation.evaluator import MetricsEvaluator

logger = logging.getLogger(__name__)


class ChampionSelector:
    """
    Evaluates candidate models strictly on the external VALIDATION partition and selects exactly ONE champion.
    
    Guarantees:
    - Zero exposure to TEST partition: TEST must never be passed to selection.
    - Explicit, task-appropriate primary metric with directional comparison (higher vs lower is better).
    - Comprehensive record of all candidate validation metrics and reasons for selection.
    """

    @classmethod
    def is_higher_better(cls, metric_name: str) -> bool:
        """Determines whether a larger metric score indicates better performance."""
        m = str(metric_name).lower()
        return m not in ["mae", "mse", "rmse", "mean_absolute_error", "mean_squared_error", "root_mean_squared_error"]

    @classmethod
    def candidate_from_autogluon(
        cls,
        ag_result: AutoGluonBackendResult,
        df_val: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType,
        expected_class_labels: Optional[List[Any]] = None,
    ) -> ChampionCandidate:
        """
        Creates a ChampionCandidate from an AutoGluonBackendResult by evaluating on external VALIDATION.
        """
        if df_val.empty:
            raise ValueError("Validation DataFrame cannot be empty.")

        from backend.engine.backends.autogluon_backend import AutoGluonBackend

        val_metrics = AutoGluonBackend.evaluate(

            df_val=df_val,
            target_column=target_column,
            artifact_path=ag_result.artifact_path,
            problem_type=problem_type,
            expected_class_labels=expected_class_labels or ag_result.class_labels,
        )

        model_name = ag_result.model_best or "AutoGluon_Best"
        return ChampionCandidate(
            candidate_id=f"autogluon_{model_name}",
            backend_name="autogluon",
            model_name=model_name,
            problem_type=problem_type,
            val_metrics=val_metrics,
            primary_metric=ag_result.eval_metric or "default",
            primary_val_score=0.0,  # Will be set during selection based on chosen primary_metric
            artifact_path=ag_result.artifact_path,
            fit_time_seconds=ag_result.fit_time_seconds,
            class_labels=expected_class_labels or ag_result.class_labels,
            extra_metadata={
                "autogluon_version": ag_result.autogluon_version,
                "peak_memory_mb": ag_result.peak_memory_mb,
                "leaderboard": ag_result.leaderboard,
            },
        )

    @classmethod
    def candidate_from_baseline_result(
        cls,
        baseline_result: BaselineModelResult,
        problem_type: ProblemType,
        class_labels: Optional[List[Any]] = None,
        model_instance: Optional[Any] = None,
    ) -> ChampionCandidate:
        """
        Creates a ChampionCandidate from a BaselineModelResult.
        """
        resolved_instance = model_instance if model_instance is not None else baseline_result.model_instance
        return ChampionCandidate(
            candidate_id=f"baseline_{baseline_result.model_name.lower().replace(' ', '_')}",
            backend_name="baseline",
            model_name=baseline_result.model_name,
            problem_type=problem_type,
            val_metrics=dict(baseline_result.val_metrics),
            primary_metric=baseline_result.primary_metric,
            primary_val_score=baseline_result.val_score,
            model_instance=resolved_instance,
            fit_time_seconds=baseline_result.fit_time_seconds,
            class_labels=class_labels,
            extra_metadata={"status": baseline_result.status},
        )

    @classmethod
    def candidates_from_baseline_suite(
        cls,
        suite_result: BaselineSuiteResult,
        problem_type: ProblemType,
        class_labels: Optional[List[Any]] = None,
    ) -> List[ChampionCandidate]:
        """
        Extracts candidates from a full BaselineSuiteResult.
        """
        candidates: List[ChampionCandidate] = []
        for b in suite_result.all_baselines:
            if b.status == "success":
                candidates.append(
                    cls.candidate_from_baseline_result(
                        baseline_result=b,
                        problem_type=problem_type,
                        class_labels=class_labels,
                    )
                )
        return candidates

    @classmethod
    def select_champion(
        cls,
        candidates: List[ChampionCandidate],
        problem_type: ProblemType,
        primary_metric: Optional[str] = None,
        class_imbalance_ratio: float = 1.0,
    ) -> ChampionSelectionResult:
        """
        Selects exactly one champion from evaluated candidates using external VALIDATION metrics.

        Parameters:
            candidates: List of ChampionCandidate objects.
            problem_type: ProblemType (binary, multiclass, regression).
            primary_metric: Explicit metric for selection (e.g., 'f1', 'r2', 'balanced_accuracy').
            class_imbalance_ratio: Ratio of majority to minority class (for metric selection fallback).

        Returns:
            ChampionSelectionResult containing the single champion and complete audit trail.
        """
        start_time = time.time()
        if not candidates:
            raise ValueError("Cannot select champion from an empty candidate list.")

        # Determine task primary metric if not explicitly specified
        metric = primary_metric or MetricsEvaluator.get_primary_metric_name(
            problem_type=problem_type,
            class_imbalance_ratio=class_imbalance_ratio,
        )
        metric = metric.lower()
        higher_better = cls.is_higher_better(metric)

        # Update primary_val_score on all candidates for the active primary_metric
        valid_candidates: List[ChampionCandidate] = []
        comparison_dict: Dict[str, Dict[str, float]] = {}

        for cand in candidates:
            cand.primary_metric = metric
            score = cand.val_metrics.get(metric)
            if score is None:
                logger.warning(
                    f"Candidate '{cand.model_name}' ({cand.backend_name}) has no validation metric for '{metric}'. "
                    f"Available metrics: {list(cand.val_metrics.keys())}"
                )
                score = -float("inf") if higher_better else float("inf")
            cand.primary_val_score = float(score)
            valid_candidates.append(cand)
            comparison_dict[cand.candidate_id] = {
                "backend": cand.backend_name,
                "model_name": cand.model_name,
                "primary_metric": metric,
                "primary_score": cand.primary_val_score,
                **cand.val_metrics,
            }

        # Deterministic sorting
        # Primary key: score (descending if higher is better, ascending if lower is better)
        # Secondary key: faster fit_time_seconds
        # Tertiary key: candidate_id for stability
        def sort_key(c: ChampionCandidate):
            score_val = c.primary_val_score if higher_better else -c.primary_val_score
            # Fallback for NaN / inf
            if np.isnan(score_val):
                score_val = -float("inf")
            return (score_val, -c.fit_time_seconds, c.candidate_id)

        valid_candidates.sort(key=sort_key, reverse=True)
        champion = valid_candidates[0]

        # Build human-readable rationale
        other_candidates = valid_candidates[1:]
        if other_candidates:
            runner_up = other_candidates[0]
            diff = abs(champion.primary_val_score - runner_up.primary_val_score)
            rationale = (
                f"Champion '{champion.model_name}' (backend: {champion.backend_name}) selected with "
                f"best validation {metric}={champion.primary_val_score:.4f} on external validation set. "
                f"Outperformed runner-up '{runner_up.model_name}' ({runner_up.backend_name}, "
                f"{metric}={runner_up.primary_val_score:.4f}) by {diff:.4f} ({len(valid_candidates)} total candidates evaluated)."
            )
        else:
            rationale = (
                f"Champion '{champion.model_name}' (backend: {champion.backend_name}) selected with "
                f"validation {metric}={champion.primary_val_score:.4f} on external validation set (single candidate evaluated)."
            )

        logger.info(f"[ChampionSelector] {rationale}")
        selection_time = time.time() - start_time

        return ChampionSelectionResult(
            champion=champion,
            problem_type=problem_type,
            primary_metric=metric,
            selection_rationale=rationale,
            all_candidates=valid_candidates,
            validation_comparison=comparison_dict,
            selection_time_seconds=selection_time,
        )
