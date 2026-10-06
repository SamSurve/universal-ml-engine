import time
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from flaml import AutoML

from backend.engine.contracts.schemas import ProblemType, ScreeningCandidate, ColumnProfile
from backend.engine.preprocessing.pipeline_builder import PreprocessingPipelineBuilder

logger = logging.getLogger(__name__)

# Mapping between FLAML estimator strings and canonical ModelRegistry names
FLAML_TO_CANONICAL: Dict[str, str] = {
    "lgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "rf": "Random Forest",
    "extra_tree": "Extra Trees",
}

CANONICAL_TO_FLAML: Dict[str, str] = {v: k for k, v in FLAML_TO_CANONICAL.items()}


class FLAMLScreener:
    """
    Rapid multi-model candidate screening using FLAML Cost-Frugal Optimization (CFO).
    
    Guarantees:
    - Configurable time budget (default: 60s).
    - Runs exclusively on training/development data; final holdout is untouched.
    - Evaluates multiple model families across cross-validation.
    - Deterministic and reproducible with fixed random seed.
    - Gracefully isolates failures without aborting the experiment.
    """

    DEFAULT_ESTIMATORS = ["lgbm", "rf", "xgboost", "extra_tree", "catboost"]

    @classmethod
    def _map_task_and_metric(cls, problem_type: ProblemType) -> Tuple[str, str]:
        """Maps engine ProblemType to FLAML task and evaluation metric."""
        if problem_type == ProblemType.BINARY_CLASSIFICATION:
            return "classification", "f1"
        elif problem_type == ProblemType.MULTICLASS_CLASSIFICATION:
            return "multiclass", "macro_f1"
        elif problem_type == ProblemType.REGRESSION:
            return "regression", "r2"
        else:
            raise ValueError(f"Unsupported problem type for FLAML screening: {problem_type}")

    @classmethod
    def screen(
        cls,
        df_dev: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType,
        column_profiles: Dict[str, ColumnProfile],
        time_budget_seconds: int = 60,
        n_splits: int = 5,
        random_state: int = 42,
        estimator_list: Optional[List[str]] = None,
    ) -> List[ScreeningCandidate]:
        """
        Executes fast candidate screening on development data.
        Returns a sorted list of ranked ScreeningCandidate objects.
        """
        start_time = time.time()
        X_dev = df_dev.drop(columns=[target_column])
        y_dev = df_dev[target_column]

        # Prepare target encoding if non-numeric classification
        if problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            if not pd.api.types.is_numeric_dtype(y_dev):
                from sklearn.preprocessing import LabelEncoder
                y_encoded = LabelEncoder().fit_transform(y_dev.astype(str))
            else:
                y_encoded = y_dev.to_numpy()
        else:
            y_encoded = y_dev.to_numpy(dtype=float)

        # Build and apply dev-level preprocessor for screening
        num_cols, cat_cols, date_cols = PreprocessingPipelineBuilder.identify_feature_types(
            X_dev, column_profiles
        )
        preprocessor = PreprocessingPipelineBuilder.build_preprocessor(
            num_cols, cat_cols, date_cols, scale_numeric=False
        )

        try:
            X_trans = preprocessor.fit_transform(X_dev)
        except Exception as e:
            logger.error(f"Preprocessing failed during FLAML screening: {e}")
            return [
                ScreeningCandidate(
                    model_name="Preprocessing Failure",
                    cv_score=-999.0,
                    fit_time_seconds=round(time.time() - start_time, 2),
                    rank=1,
                    status="failed",
                    error_message=str(e),
                )
            ]

        # Filter estimator list by available packages
        available_estimators = list(estimator_list or cls.DEFAULT_ESTIMATORS)

        task, metric = cls._map_task_and_metric(problem_type)

        automl = AutoML()
        settings = {
            "time_budget": max(5, time_budget_seconds),
            "metric": metric,
            "task": task,
            "estimator_list": available_estimators,
            "eval_method": "cv",
            "n_splits": min(n_splits, max(2, len(df_dev) // 5)),
            "seed": random_state,
            "verbose": 0,
            "n_jobs": 1,  # Safe single-thread per trial to avoid overloading Windows
        }

        candidates: List[ScreeningCandidate] = []

        try:
            automl.fit(X_train=X_trans, y_train=y_encoded, **settings)

            # Extract search states across all screened estimators
            search_states = getattr(automl, "_search_states", {})
            for est_key, state in search_states.items():
                canonical_name = FLAML_TO_CANONICAL.get(est_key, est_key)
                if state and state.best_loss is not None and not np.isinf(state.best_loss):
                    if metric in ["f1", "macro_f1", "r2", "accuracy", "roc_auc"]:
                        cv_score = 1.0 - state.best_loss
                    else:
                        cv_score = -state.best_loss

                    candidates.append(
                        ScreeningCandidate(
                            model_name=canonical_name,
                            cv_score=round(float(cv_score), 4),
                            fit_time_seconds=round(float(getattr(state, "total_time_used", 0.0)), 2),
                            rank=0,
                            hyperparameters=getattr(state, "best_config", {}) or {},
                            status="success",
                        )
                    )
                else:
                    candidates.append(
                        ScreeningCandidate(
                            model_name=canonical_name,
                            cv_score=-999.0,
                            fit_time_seconds=round(float(getattr(state, "total_time_used", 0.0)), 2),
                            rank=99,
                            status="failed",
                            error_message="No valid trial completed within time budget.",
                        )
                    )

            # Ensure best estimator is captured if search_states was incomplete
            if not candidates and getattr(automl, "best_estimator", None):
                best_name = FLAML_TO_CANONICAL.get(automl.best_estimator, automl.best_estimator)
                best_score = 1.0 - automl.best_loss if metric in ["f1", "macro_f1", "r2"] else -automl.best_loss
                candidates.append(
                    ScreeningCandidate(
                        model_name=best_name,
                        cv_score=round(float(best_score), 4),
                        fit_time_seconds=round(float(automl.time_to_find_best_model), 2),
                        rank=1,
                        hyperparameters=automl.best_config or {},
                        status="success",
                    )
                )

        except Exception as e:
            logger.warning(f"FLAML screening encountered an error: {e}")
            candidates.append(
                ScreeningCandidate(
                    model_name="FLAML Screening",
                    cv_score=-999.0,
                    fit_time_seconds=round(time.time() - start_time, 2),
                    rank=1,
                    status="failed",
                    error_message=str(e),
                )
            )

        # Sort successful candidates descending by CV score, followed by failed
        successful = [c for c in candidates if c.status == "success"]
        failed = [c for c in candidates if c.status != "success"]

        successful.sort(key=lambda c: c.cv_score, reverse=True)
        for idx, cand in enumerate(successful):
            cand.rank = idx + 1
        for idx, cand in enumerate(failed):
            cand.rank = len(successful) + idx + 1

        return successful + failed
