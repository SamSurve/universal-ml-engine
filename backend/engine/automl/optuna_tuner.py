import time
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.base import clone
import optuna

from backend.engine.contracts.schemas import (
    ProblemType,
    TunedCandidate,
    ColumnProfile,
)
from backend.engine.preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from backend.engine.evaluation.evaluator import MetricsEvaluator

# Reduce Optuna log verbosity to warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


class OptunaTuner:
    """
    Bayesian Hyperparameter Optimization for top candidate models using Optuna.

    Key Guarantees:
    - Fixed random seed via TPESampler for reproducible trials.
    - Trial and time budget enforcement (stops when either limit reached).
    - Leakage-safe 5-fold CV: Preprocessor fitted strictly on training fold.
    - Final holdout remains 100% untouched.
    - MedianPruner for early stopping of underperforming trials.
    - Gracefully isolates trial errors without crashing the experiment.
    """

    @classmethod
    def _create_model_instance(
        cls,
        model_name: str,
        params: Dict[str, Any],
        problem_type: ProblemType,
        random_state: int,
    ):
        """Instantiates a model estimator configured with hyperparameter choices."""
        is_classification = problem_type in [
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ]

        if model_name == "LightGBM":
            import lightgbm as lgb
            if is_classification:
                return lgb.LGBMClassifier(random_state=random_state, verbosity=-1, n_jobs=1, **params)
            return lgb.LGBMRegressor(random_state=random_state, verbosity=-1, n_jobs=1, **params)

        elif model_name == "XGBoost":
            import xgboost as xgb
            eval_metric = "logloss" if problem_type == ProblemType.BINARY_CLASSIFICATION else ("mlogloss" if is_classification else "rmse")
            if is_classification:
                return xgb.XGBClassifier(random_state=random_state, verbosity=0, n_jobs=1, eval_metric=eval_metric, **params)
            return xgb.XGBRegressor(random_state=random_state, verbosity=0, n_jobs=1, eval_metric="rmse", **params)

        elif model_name == "CatBoost":
            import catboost as cb
            if is_classification:
                return cb.CatBoostClassifier(random_seed=random_state, verbose=False, thread_count=1, **params)
            return cb.CatBoostRegressor(random_seed=random_state, verbose=False, thread_count=1, **params)

        elif model_name == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if is_classification:
                return RandomForestClassifier(random_state=random_state, n_jobs=1, **params)
            return RandomForestRegressor(random_state=random_state, n_jobs=1, **params)

        elif model_name == "Extra Trees":
            from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
            if is_classification:
                return ExtraTreesClassifier(random_state=random_state, n_jobs=1, **params)
            return ExtraTreesRegressor(random_state=random_state, n_jobs=1, **params)

        elif model_name == "HistGradientBoosting":
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
            if is_classification:
                return HistGradientBoostingClassifier(random_state=random_state, **params)
            return HistGradientBoostingRegressor(random_state=random_state, **params)

        elif model_name == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(random_state=random_state, max_iter=1000, **params)

        elif model_name == "Ridge":
            from sklearn.linear_model import Ridge
            return Ridge(random_state=random_state, **params)

        elif model_name == "Linear Regression":
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**params)

        else:
            raise ValueError(f"Unsupported model for Optuna tuning: {model_name}")

    @classmethod
    def _sample_params(cls, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """Defines concise, high-impact search spaces for each model family."""
        params: Dict[str, Any] = {}

        if model_name in ["LightGBM", "XGBoost"]:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300, step=50)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
            params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
            params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
            params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)
            params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
            if model_name == "LightGBM":
                params["num_leaves"] = trial.suggest_int("num_leaves", 15, 63)
            elif model_name == "XGBoost":
                params["gamma"] = trial.suggest_float("gamma", 0.0, 5.0)

        elif model_name == "CatBoost":
            params["iterations"] = trial.suggest_int("iterations", 50, 250, step=50)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            params["depth"] = trial.suggest_int("depth", 4, 8)
            params["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1.0, 10.0)

        elif model_name in ["Random Forest", "Extra Trees"]:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 200, step=50)
            params["max_depth"] = trial.suggest_int("max_depth", 4, 20)
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
            params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 5)

        elif model_name == "HistGradientBoosting":
            params["max_iter"] = trial.suggest_int("max_iter", 50, 200, step=50)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
            params["l2_regularization"] = trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True)

        elif model_name == "Logistic Regression":
            params["C"] = trial.suggest_float("C", 0.01, 100.0, log=True)

        elif model_name == "Ridge":
            params["alpha"] = trial.suggest_float("alpha", 0.01, 100.0, log=True)

        return params

    @classmethod
    def tune_model(
        cls,
        model_name: str,
        df_dev: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType,
        column_profiles: Dict[str, ColumnProfile],
        baseline_cv_score: float = 0.0,
        n_trials: int = 30,
        timeout_seconds: int = 120,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> Tuple[TunedCandidate, Pipeline]:
        """
        Tunes a specific model family using Optuna.
        Returns a TunedCandidate and the refitted best Pipeline.
        """
        start_time = time.time()
        X_dev = df_dev.drop(columns=[target_column])
        y_dev = df_dev[target_column]

        is_classification = problem_type in [
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ]
        is_binary = problem_type == ProblemType.BINARY_CLASSIFICATION

        # Target encoding for classification if needed
        label_encoder = None
        if is_classification:
            if not pd.api.types.is_numeric_dtype(y_dev):
                from sklearn.preprocessing import LabelEncoder
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y_dev.astype(str))
            else:
                y_encoded = y_dev.to_numpy()
        else:
            y_encoded = y_dev.to_numpy(dtype=float)

        num_cols, cat_cols, date_cols = PreprocessingPipelineBuilder.identify_feature_types(
            X_dev, column_profiles
        )
        scale_features = "Linear" in model_name or "Ridge" in model_name or "Logistic" in model_name

        imbalance_ratio = 1.0
        if is_classification:
            counts = pd.Series(y_encoded).value_counts()
            if len(counts) >= 2:
                imbalance_ratio = float(counts.max() / counts.min())

        primary_metric = MetricsEvaluator.get_primary_metric_name(problem_type, imbalance_ratio)
        direction = "maximize" if primary_metric in ["f1", "macro_f1", "r2", "accuracy", "balanced_accuracy", "roc_auc"] else "minimize"

        # Configure cross-validation splitter
        if is_classification:
            min_class_count = int(pd.Series(y_encoded).value_counts().min())
            effective_splits = min(n_splits, max(2, min_class_count))
            splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
        else:
            effective_splits = min(n_splits, max(2, len(df_dev) // 2))
            splitter = KFold(n_splits=effective_splits, shuffle=True, random_state=random_state)

        def objective(trial: optuna.Trial) -> float:
            try:
                params = cls._sample_params(trial, model_name)
                estimator = cls._create_model_instance(model_name, params, problem_type, random_state)
            except Exception as e:
                logger.warning(f"Error sampling parameters for {model_name}: {e}")
                raise optuna.TrialPruned()

            fold_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_dev, y_encoded)):
                X_tr, X_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
                y_tr, y_val = y_encoded[train_idx], y_encoded[val_idx]

                # Preprocessor strictly fit on training fold
                preprocessor = PreprocessingPipelineBuilder.build_preprocessor(
                    num_cols, cat_cols, date_cols, scale_numeric=scale_features
                )
                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", clone(estimator)),
                ])

                try:
                    pipeline.fit(X_tr, y_tr)
                    y_pred = pipeline.predict(X_val)

                    if is_classification:
                        y_prob = None
                        if hasattr(pipeline, "predict_proba"):
                            try:
                                y_prob = pipeline.predict_proba(X_val)
                            except Exception:
                                y_prob = None
                        metrics = MetricsEvaluator.evaluate_classification(y_val, y_pred, y_prob, is_binary=is_binary)
                    else:
                        metrics = MetricsEvaluator.evaluate_regression(y_val, y_pred)

                    score = metrics[primary_metric]
                    fold_scores.append(score)

                    # Report intermediate progress for pruning
                    trial.report(score, step=fold_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                except optuna.TrialPruned:
                    raise
                except Exception as e:
                    logger.debug(f"Trial failed on fold {fold_idx}: {e}")
                    raise optuna.TrialPruned()

            return float(np.mean(fold_scores))

        # Setup study with fixed sampler and median pruner
        sampler = optuna.samplers.TPESampler(seed=random_state)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

        actual_trials = 0
        try:
            study.optimize(
                objective,
                n_trials=max(1, n_trials),
                timeout=max(5, timeout_seconds),
                catch=(Exception,),
            )
            actual_trials = len(study.trials)
        except Exception as e:
            logger.error(f"Optuna optimization error for {model_name}: {e}")

        tuning_time = time.time() - start_time

        # Extract best parameters
        if study.best_trials:
            best_params = study.best_params
            best_score = round(float(study.best_value), 4)
            improvement = round(best_score - baseline_cv_score, 4) if direction == "maximize" else round(baseline_cv_score - best_score, 4)
            status = "success"
            err = None
        else:
            best_params = {}
            best_score = baseline_cv_score
            improvement = 0.0
            status = "failed"
            err = "No successful trials completed."

        # Build and refit final best pipeline on 100% of dev data
        refit_preprocessor = PreprocessingPipelineBuilder.build_preprocessor(
            num_cols, cat_cols, date_cols, scale_numeric=scale_features
        )
        best_estimator = cls._create_model_instance(model_name, best_params, problem_type, random_state)
        best_pipeline = Pipeline([
            ("preprocessor", refit_preprocessor),
            ("model", best_estimator),
        ])
        best_pipeline.fit(X_dev, y_encoded)

        candidate = TunedCandidate(
            model_name=model_name,
            baseline_cv_score=round(baseline_cv_score, 4),
            tuned_cv_score=best_score,
            best_params=best_params,
            n_trials=actual_trials,
            tuning_time_seconds=round(tuning_time, 2),
            improvement=improvement,
            status=status,
            error_message=err,
        )

        return candidate, best_pipeline
