import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier, is_regressor

from backend.engine.contracts.schemas import (
    ProblemType,
    FeatureImportanceEntry,
    PredictionContribution,
    PredictionExplanation,
    ExplainabilityResult,
)

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None


class ModelExplainer:
    """
    Production-grade Explainable AI service using SHAP.
    Explains Scikit-Learn pipelines, handles feature transformations,
    computes global feature importances, and produces per-prediction local explanations.
    """

    @classmethod
    def _clean_feature_name(cls, col_name: str) -> Tuple[str, str]:
        """
        Cleans Scikit-Learn column transformer prefixes (e.g., 'num__Age' -> 'Age',
        'cat__Department_sales' -> 'Department_sales').
        Returns (cleaned_name, raw_parent_name).
        """
        cleaned = col_name
        for prefix in ["num__", "cat__", "date__", "remainder__"]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break

        # Extract raw parent name for one-hot encoded features
        # e.g., 'Department_sales' -> 'Department' if contains underscore
        # We also check common one-hot delimiter patterns
        parts = cleaned.split("_")
        raw_parent = parts[0] if len(parts) > 1 else cleaned
        return cleaned, raw_parent

    @classmethod
    def _extract_pipeline_components(
        cls, pipeline: Union[Pipeline, Any], X_sample: pd.DataFrame
    ) -> Tuple[Any, Any, List[str], List[str]]:
        """
        Extracts the preprocessor and estimator from a Pipeline.
        Transforms X_sample to obtain transformed feature names.
        """
        if isinstance(pipeline, Pipeline):
            preprocessor = pipeline.named_steps.get("preprocessor", None)
            estimator = pipeline.named_steps.get("model", pipeline.steps[-1][1])
        else:
            preprocessor = None
            estimator = pipeline

        transformed_names = []
        raw_parents = []

        if preprocessor is not None:
            try:
                feature_names_out = preprocessor.get_feature_names_out()
                for name in feature_names_out:
                    clean_name, raw_name = cls._clean_feature_name(str(name))
                    transformed_names.append(clean_name)
                    raw_parents.append(raw_name)
            except Exception:
                pass

        if not transformed_names:
            if hasattr(X_sample, "columns"):
                transformed_names = list(X_sample.columns)
                raw_parents = list(X_sample.columns)
            else:
                n_feats = X_sample.shape[1] if hasattr(X_sample, "shape") else 1
                transformed_names = [f"feature_{i}" for i in range(n_feats)]
                raw_parents = transformed_names.copy()

        return preprocessor, estimator, transformed_names, raw_parents

    @classmethod
    def _transform_data(cls, preprocessor: Any, X: pd.DataFrame) -> np.ndarray:
        """Transforms input features into a dense 2D numpy array."""
        if preprocessor is not None:
            X_trans = preprocessor.transform(X)
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()
            return np.asarray(X_trans, dtype=float)
        else:
            if hasattr(X, "to_numpy"):
                return np.asarray(X.to_numpy(), dtype=float)
            return np.asarray(X, dtype=float)

    @classmethod
    def _select_shap_explainer(
        cls, estimator: Any, background_data: np.ndarray, model_name: str
    ) -> Tuple[Any, str]:
        """Selects the most suitable SHAP explainer for the estimator."""
        estimator_type_str = type(estimator).__name__.lower()

        is_tree = any(
            tree_keyword in estimator_type_str
            for tree_keyword in [
                "forest", "trees", "gradientboosting", "lgbm", "xgb", "catboost"
            ]
        )
        is_linear = any(
            lin_keyword in estimator_type_str
            for lin_keyword in ["linear", "ridge", "logistic"]
        )

        if is_tree:
            try:
                explainer = shap.TreeExplainer(estimator)
                return explainer, "shap_tree"
            except Exception as e:
                logger.debug(f"TreeExplainer failed ({e}), falling back to shap.Explainer")

        if is_linear:
            try:
                explainer = shap.LinearExplainer(estimator, background_data)
                return explainer, "shap_linear"
            except Exception as e:
                logger.debug(f"LinearExplainer failed ({e}), falling back to shap.Explainer")

        # Fallback to model-agnostic explainer
        try:
            explainer = shap.Explainer(estimator, background_data)
            return explainer, "shap_explainer"
        except Exception:
            # Last-resort sample-based KernelExplainer
            predict_fn = estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator.predict
            explainer = shap.KernelExplainer(predict_fn, background_data[:20])
            return explainer, "shap_kernel"

    @classmethod
    def explain_model(
        cls,
        pipeline: Any,
        df_dev: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType,
        model_name: str = "model",
        df_eval: Optional[pd.DataFrame] = None,
        max_background_samples: int = 50,
        max_eval_samples: int = 100,
        random_state: int = 42,
    ) -> ExplainabilityResult:
        """
        Computes SHAP global feature importances and local predictions explanations.
        Never touches final holdout labels; uses development data as background.
        """
        warnings: List[str] = []

        if not SHAP_AVAILABLE:
            warnings.append("SHAP library is not installed. Returning native feature importance fallback.")
            return cls._native_fallback(pipeline, df_dev, target_column, model_name, warnings)

        X_dev = df_dev.drop(columns=[target_column], errors="ignore")
        preprocessor, estimator, transformed_names, raw_parents = cls._extract_pipeline_components(
            pipeline, X_dev
        )

        # 1. Prepare Background Data (strictly sampled from Dev data)
        sample_size = min(len(X_dev), max_background_samples)
        bg_sample = X_dev.sample(n=sample_size, random_state=random_state)
        try:
            bg_matrix = cls._transform_data(preprocessor, bg_sample)
        except Exception as e:
            warnings.append(f"Preprocessing transform failed on background data: {e}")
            return cls._native_fallback(pipeline, df_dev, target_column, model_name, warnings)

        # 2. Select & Initialize SHAP Explainer
        try:
            explainer, explainer_type = cls._select_shap_explainer(estimator, bg_matrix, model_name)
        except Exception as e:
            warnings.append(f"Failed to initialize SHAP explainer: {e}")
            return cls._native_fallback(pipeline, df_dev, target_column, model_name, warnings)

        # 3. Compute SHAP Values on Evaluation Sample (from Dev or provided Eval inputs)
        eval_source = df_eval if df_eval is not None else df_dev
        X_eval = eval_source.drop(columns=[target_column], errors="ignore")
        eval_size = min(len(X_eval), max_eval_samples)
        eval_subset = X_eval.iloc[:eval_size]

        try:
            eval_matrix = cls._transform_data(preprocessor, eval_subset)
            raw_shap_values = explainer(eval_matrix) if callable(explainer) and not hasattr(explainer, "shap_values") else explainer.shap_values(eval_matrix)
        except Exception as e:
            # Fallback to .shap_values or callable
            try:
                raw_shap_values = explainer.shap_values(eval_matrix)
            except Exception as e2:
                warnings.append(f"SHAP value computation failed: {e2}")
                return cls._native_fallback(pipeline, df_dev, target_column, model_name, warnings)

        # 4. Standardize SHAP Values Array
        shap_values_arr = cls._standardize_shap_values(raw_shap_values, problem_type)
        if shap_values_arr is None or shap_values_arr.shape[1] != len(transformed_names):
            # Align lengths if mismatch
            n_features = shap_values_arr.shape[1] if shap_values_arr is not None else len(transformed_names)
            transformed_names = [f"feat_{i}" for i in range(n_features)]
            raw_parents = transformed_names.copy()

        # 5. Global Feature Importance (Mean |SHAP| across evaluated instances)
        mean_abs_shap = np.mean(np.abs(shap_values_arr), axis=0)
        total_importance = float(np.sum(mean_abs_shap))
        if total_importance == 0:
            total_importance = 1e-9

        # Transformed feature ranking
        global_importance: List[FeatureImportanceEntry] = []
        sorted_indices = np.argsort(-mean_abs_shap)

        for rank, idx in enumerate(sorted_indices, start=1):
            feat_name = transformed_names[idx] if idx < len(transformed_names) else f"feature_{idx}"
            parent_name = raw_parents[idx] if idx < len(raw_parents) else feat_name
            score = float(mean_abs_shap[idx])
            rel_pct = round((score / total_importance) * 100.0, 2)

            global_importance.append(
                FeatureImportanceEntry(
                    feature_name=feat_name,
                    raw_feature_name=parent_name,
                    importance_score=round(score, 5),
                    relative_importance_pct=rel_pct,
                    rank=rank,
                )
            )

        # Aggregate parent/raw feature ranking (grouping one-hot columns)
        parent_importance_map: Dict[str, float] = {}
        for entry in global_importance:
            p_name = entry.raw_feature_name
            parent_importance_map[p_name] = parent_importance_map.get(p_name, 0.0) + entry.importance_score

        sorted_parents = sorted(parent_importance_map.items(), key=lambda kv: kv[1], reverse=True)
        total_parent_imp = sum(score for _, score in sorted_parents) or 1e-9

        raw_feature_importance: List[FeatureImportanceEntry] = []
        for rank, (p_name, score) in enumerate(sorted_parents, start=1):
            rel_pct = round((score / total_parent_imp) * 100.0, 2)
            raw_feature_importance.append(
                FeatureImportanceEntry(
                    feature_name=p_name,
                    raw_feature_name=p_name,
                    importance_score=round(score, 5),
                    relative_importance_pct=rel_pct,
                    rank=rank,
                )
            )

        # 6. Extract Base Value
        base_val = cls._extract_base_value(explainer, raw_shap_values, problem_type)

        # 7. Local Per-Prediction Explanations (First 2-3 samples)
        example_explanations: List[PredictionExplanation] = []
        n_examples = min(len(eval_subset), 3)

        for i in range(n_examples):
            sample_shap = shap_values_arr[i]
            sample_feats = eval_matrix[i]

            pred_val = None
            pred_prob = None
            try:
                single_input = eval_subset.iloc[[i]]
                if hasattr(pipeline, "predict"):
                    pred_raw = pipeline.predict(single_input)
                    pred_val = pred_raw[0].item() if hasattr(pred_raw[0], "item") else pred_raw[0]
                if hasattr(pipeline, "predict_proba"):
                    probs = pipeline.predict_proba(single_input)[0]
                    pred_prob = {f"class_{c}": round(float(p), 4) for c, p in enumerate(probs)}
            except Exception:
                pass

            contributions: List[PredictionContribution] = []
            for j, s_val in enumerate(sample_shap):
                val = float(s_val)
                f_name = transformed_names[j] if j < len(transformed_names) else f"feature_{j}"
                p_name = raw_parents[j] if j < len(raw_parents) else f_name
                raw_val = float(sample_feats[j]) if j < len(sample_feats) else 0.0

                contributions.append(
                    PredictionContribution(
                        feature_name=f_name,
                        raw_feature_name=p_name,
                        feature_value=round(raw_val, 4),
                        shap_value=round(val, 5),
                        direction="positive" if val >= 0 else "negative",
                    )
                )

            # Sort by absolute SHAP magnitude for this prediction
            contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
            top_pos = [c.feature_name for c in contributions if c.shap_value > 0][:3]
            top_neg = [c.feature_name for c in contributions if c.shap_value < 0][:3]

            example_explanations.append(
                PredictionExplanation(
                    sample_index=i,
                    predicted_value=pred_val,
                    predicted_probability=pred_prob,
                    base_value=round(base_val, 5),
                    contributions=contributions[:10],  # Top 10 driving features
                    top_positive_features=top_pos,
                    top_negative_features=top_neg,
                )
            )

        top_3_names = [e.feature_name for e in global_importance[:3]]
        summary_text = (
            f"Model explanations successfully computed via {explainer_type}. "
            f"The primary global drivers are: {', '.join(top_3_names)}."
        )

        return ExplainabilityResult(
            status="success",
            explainer_type=explainer_type,
            global_importance=global_importance,
            raw_feature_importance=raw_feature_importance,
            example_explanations=example_explanations,
            warnings=warnings,
            transformed_feature_names=transformed_names,
            summary_text=summary_text,
        )

    @classmethod
    def _standardize_shap_values(
        cls, raw_shap_values: Any, problem_type: ProblemType
    ) -> Optional[np.ndarray]:
        """Converts raw SHAP outputs (lists, Explanation objects, 3D arrays) into 2D ndarray."""
        if hasattr(raw_shap_values, "values"):
            # shap.Explanation object
            vals = raw_shap_values.values
        else:
            vals = raw_shap_values

        if isinstance(vals, list):
            # Binary or Multiclass list of arrays
            if len(vals) == 2:
                # Binary classification: take positive class
                return np.asarray(vals[1], dtype=float)
            elif len(vals) > 0:
                # Multiclass: take mean absolute across classes or class 0
                return np.asarray(vals[0], dtype=float)

        arr = np.asarray(vals, dtype=float)
        if arr.ndim == 3:
            # (n_samples, n_features, n_classes)
            if arr.shape[2] == 2:
                return arr[:, :, 1]
            return np.mean(arr, axis=2)
        elif arr.ndim == 2:
            return arr
        return None

    @classmethod
    def _extract_base_value(
        cls, explainer: Any, raw_shap_values: Any, problem_type: ProblemType
    ) -> float:
        """Safely extracts expected / base value from explainer."""
        try:
            if hasattr(raw_shap_values, "base_values"):
                bv = raw_shap_values.base_values
                if hasattr(bv, "__len__") and len(bv) > 0:
                    val = bv[0]
                    if hasattr(val, "__len__") and len(val) > 1:
                        return float(val[1])
                    return float(val)
                return float(bv)

            if hasattr(explainer, "expected_value"):
                ev = explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    if len(ev) == 2:
                        return float(ev[1])
                    return float(ev[0])
                return float(ev)
        except Exception:
            pass
        return 0.0

    @classmethod
    def _native_fallback(
        cls,
        pipeline: Any,
        df_dev: pd.DataFrame,
        target_column: str,
        model_name: str,
        warnings: List[str],
    ) -> ExplainabilityResult:
        """Native feature importance fallback when SHAP is unavailable or encounters errors."""
        X_dev = df_dev.drop(columns=[target_column], errors="ignore")
        preprocessor, estimator, transformed_names, raw_parents = cls._extract_pipeline_components(
            pipeline, X_dev
        )

        importances = None
        method = "model_native"

        if hasattr(estimator, "feature_importances_"):
            importances = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_, dtype=float)
            importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

        if importances is None or len(importances) != len(transformed_names):
            importances = np.ones(len(transformed_names)) / max(1, len(transformed_names))
            method = "uniform_fallback"

        total_imp = float(np.sum(importances)) or 1e-9
        global_importance: List[FeatureImportanceEntry] = []
        sorted_indices = np.argsort(-importances)

        for rank, idx in enumerate(sorted_indices, start=1):
            f_name = transformed_names[idx] if idx < len(transformed_names) else f"feature_{idx}"
            p_name = raw_parents[idx] if idx < len(raw_parents) else f_name
            val = float(importances[idx])
            rel_pct = round((val / total_imp) * 100.0, 2)

            global_importance.append(
                FeatureImportanceEntry(
                    feature_name=f_name,
                    raw_feature_name=p_name,
                    importance_score=round(val, 5),
                    relative_importance_pct=rel_pct,
                    rank=rank,
                )
            )

        top_3 = [e.feature_name for e in global_importance[:3]]
        summary_text = f"Native model importance generated ({method}). Top features: {', '.join(top_3)}."

        return ExplainabilityResult(
            status="fallback",
            explainer_type=method,
            global_importance=global_importance,
            raw_feature_importance=global_importance,
            example_explanations=[],
            warnings=warnings,
            transformed_feature_names=transformed_names,
            summary_text=summary_text,
        )
