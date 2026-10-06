from typing import Dict, Any, List, Optional
from backend.engine.contracts.schemas import (
    ProblemType,
    LeaderboardEntry,
    DatasetIntelligenceResult,
    ExplainabilityResult,
    ModelDecisionSummary,
)


class DecisionSummaryBuilder:
    """
    Builds a structured model decision summary, clearly distinguishing
    development cross-validation performance from final holdout evaluation.
    """

    @classmethod
    def get_model_limitations(cls, model_name: str, problem_type: ProblemType) -> List[str]:
        limitations = []
        name_lower = model_name.lower()

        if "logistic" in name_lower or "linear" in name_lower or "ridge" in name_lower:
            limitations.append(
                "Linear decision boundary: Assumes linear relationships between features and target; "
                "cannot capture complex non-linear interactions without polynomial expansion."
            )
            limitations.append(
                "Sensitive to feature scaling: Preprocessing scaling (StandardScaler) is mandatory to prevent "
                "features with large numeric scales from dominating regularization penalties."
            )
        elif "catboost" in name_lower:
            limitations.append(
                "Tree extrapolation limitation: Oblivious decision trees cannot extrapolate beyond minimum/maximum "
                "feature values observed in training data."
            )
            limitations.append(
                "Inference overhead: Categorical hashing and deep ensembles require marginally higher single-record latency "
                "compared to simple linear models."
            )
        elif "lightgbm" in name_lower or "xgb" in name_lower:
            limitations.append(
                "Tree extrapolation boundary: Gradient boosted decision trees predict constant values outside "
                "the feature range seen in training data."
            )
            limitations.append(
                "Hyperparameter sensitivity: Tree boosters can overfit on noisy datasets if tree depth, subsampling, "
                "or minimum child weights are inadequately regularized."
            )
        elif "forest" in name_lower or "extra" in name_lower:
            limitations.append(
                "Memory and ensemble size: Random Forest stores hundreds of unpruned trees in memory, resulting in larger "
                "serialized artifact sizes compared to compact linear models."
            )
            limitations.append(
                "Extrapolation limit: Ensembles of orthogonal trees cannot model diagonal trends beyond observed sample boundaries."
            )
        else:
            limitations.append(
                "Statistical assumption: Performance depends upon representative training distribution matching future production inputs."
            )

        if problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            limitations.append(
                "Probability calibration: Raw output probabilities should be monitored in production; extreme class imbalances "
                "may require threshold tuning for specific recall/precision targets."
            )

        return limitations

    @classmethod
    def build_summary(
        cls,
        dataset_name: str,
        target_column: str,
        problem_type: ProblemType,
        primary_metric: str,
        leaderboard: List[LeaderboardEntry],
        best_model_name: str,
        best_cv_score: float,
        cv_score_std: float,
        holdout_metrics: Dict[str, float],
        tuned_hyperparameters: Optional[Dict[str, Any]],
        training_runtime_seconds: float,
        tuning_runtime_seconds: float,
        total_runtime_seconds: float,
        dataset_intelligence: Optional[DatasetIntelligenceResult] = None,
        explainability: Optional[ExplainabilityResult] = None,
    ) -> ModelDecisionSummary:
        # Determine selection rationale
        models_considered = [entry.model_name for entry in leaderboard]
        second_best = next((e for e in leaderboard if e.model_name != best_model_name and e.status == "success"), None)

        if second_best:
            diff = best_cv_score - second_best.cv_score_mean
            rationale = (
                f"Selected '{best_model_name}' as the champion model because it achieved the highest mean cross-validation "
                f"{primary_metric} ({best_cv_score:.4f} +/- {cv_score_std:.4f}) on 5-fold development data, "
                f"outperforming the runner-up '{second_best.model_name}' ({second_best.cv_score_mean:.4f}) by {diff:+.4f} points."
            )
        else:
            rationale = (
                f"Selected '{best_model_name}' based on superior 5-fold development cross-validation "
                f"{primary_metric} ({best_cv_score:.4f} +/- {cv_score_std:.4f})."
            )

        # Dataset risks
        dataset_risks: List[str] = []
        if dataset_intelligence:
            for w in dataset_intelligence.warnings:
                if w.severity.value in ("critical", "high", "medium"):
                    dataset_risks.append(f"[{w.severity.value.upper()}] {w.reason}")
        if not dataset_risks:
            dataset_risks.append("No critical dataset quality risks identified.")

        # Model limitations
        model_limitations = cls.get_model_limitations(best_model_name, problem_type)

        # Explainability features summary
        top_features_summary: List[str] = []
        explainability_available = False
        if explainability and explainability.status in ("success", "fallback"):
            explainability_available = True
            top_features_summary = [
                f"{e.feature_name} (Rank #{e.rank}, {e.relative_importance_pct}%)"
                for e in explainability.raw_feature_importance[:5]
            ]

        return ModelDecisionSummary(
            problem_type=problem_type.value,
            target_column=target_column,
            primary_metric=primary_metric,
            models_screened_count=len(models_considered),
            models_considered=models_considered,
            selected_model_name=best_model_name,
            selection_rationale=rationale,
            best_cv_score=round(best_cv_score, 4),
            cv_score_std=round(cv_score_std, 4),
            holdout_metrics=holdout_metrics,
            tuned_hyperparameters=tuned_hyperparameters,
            training_runtime_seconds=round(training_runtime_seconds, 2),
            tuning_runtime_seconds=round(tuning_runtime_seconds, 2),
            total_runtime_seconds=round(total_runtime_seconds, 2),
            dataset_risks=dataset_risks,
            model_limitations=model_limitations,
            explainability_available=explainability_available,
            top_features_summary=top_features_summary,
        )
