import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

try:
    import autogluon
    import autogluon.tabular as ag_tabular
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False


def _map_problem_type(problem_type_str: str) -> str:
    """Maps engine ProblemType strings to AutoGluon's problem_type."""
    pt = str(problem_type_str).lower().replace("problemtype.", "")
    if "binary" in pt:
        return "binary"
    elif "multiclass" in pt:
        return "multiclass"
    elif "regression" in pt:
        return "regression"
    raise ValueError(f"Unsupported problem_type for AutoGluon: {problem_type_str}")


def _map_eval_metric(problem_type_ag: str, metric_str: Optional[str]) -> str:
    """Maps engine metric strings to AutoGluon supported evaluation metrics."""
    if not metric_str:
        if problem_type_ag == "binary":
            return "f1"
        elif problem_type_ag == "multiclass":
            return "f1_macro"
        elif problem_type_ag == "regression":
            return "r2"
        return "accuracy"

    m = str(metric_str).lower()
    if m in ["f1", "f1_score"]:
        return "f1" if problem_type_ag == "binary" else "f1_macro"
    elif m in ["f1_macro", "macro_f1"]:
        return "f1_macro"
    elif m in ["balanced_accuracy", "balanced_acc"]:
        return "balanced_accuracy"
    elif m in ["roc_auc", "auc"]:
        return "roc_auc"
    elif m in ["accuracy", "acc"]:
        return "accuracy"
    elif m in ["r2", "r2_score"]:
        return "r2"
    elif m in ["rmse", "root_mean_squared_error"]:
        return "root_mean_squared_error"
    elif m in ["mae", "mean_absolute_error"]:
        return "mean_absolute_error"
    elif m in ["mse", "mean_squared_error"]:
        return "mean_squared_error"
    return m


def handle_fit(params: Dict[str, Any], result_path: Path) -> None:
    """Executes AutoGluon Tabular training on isolated TRAIN data partition."""
    train_parquet_path = params.get("train_parquet_path")
    target_column = params.get("target_column")
    problem_type_raw = params.get("problem_type")
    eval_metric_raw = params.get("eval_metric")
    artifact_dir = params.get("artifact_dir")
    time_limit_seconds = float(params.get("time_limit_seconds", 60.0))
    presets = params.get("presets", "medium_quality")
    auto_stack = bool(params.get("auto_stack", False))
    num_cpus = params.get("num_cpus") or max(1, os.cpu_count() - 1)
    num_gpus = int(params.get("num_gpus", 0))
    save_space = bool(params.get("save_space", True))

    if not train_parquet_path or not Path(train_parquet_path).exists():
        raise FileNotFoundError(f"Training dataset parquet file not found: {train_parquet_path}")

    if not target_column:
        raise ValueError("target_column parameter is required for AutoGluon training.")

    if not artifact_dir:
        raise ValueError("artifact_dir parameter is required for AutoGluon model persistence.")

    # 1. Load TRAIN dataset
    df_train = pd.read_parquet(train_parquet_path)
    if target_column not in df_train.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in training dataset. Available columns: {list(df_train.columns)}"
        )

    # 2. Map Problem Type and Metric
    ag_problem_type = _map_problem_type(problem_type_raw)
    ag_eval_metric = _map_eval_metric(ag_problem_type, eval_metric_raw)

    # 3. Ensure clean artifact directory
    artifact_path = Path(artifact_dir).resolve()
    artifact_path.mkdir(parents=True, exist_ok=True)

    print(
        f"[AutoGluonWorker] Starting fit: problem_type={ag_problem_type}, metric={ag_eval_metric}, "
        f"rows={len(df_train)}, cols={len(df_train.columns)}, time_limit={time_limit_seconds}s, "
        f"presets={presets}, auto_stack={auto_stack}, num_cpus={num_cpus}, num_gpus={num_gpus}"
    )

    # 4. Instantiate TabularPredictor
    predictor = TabularPredictor(
        label=target_column,
        problem_type=ag_problem_type,
        eval_metric=ag_eval_metric,
        path=str(artifact_path),
        verbosity=2,
    )

    # 5. Fit TabularPredictor on TRAIN ONLY
    predictor.fit(
        train_data=df_train,
        tuning_data=None,  # CRITICAL: AutoGluon holds out internal tuning partition within df_train
        time_limit=max(5, int(time_limit_seconds)),
        presets=presets,
        auto_stack=auto_stack,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        save_space=save_space,
    )

    # 6. Extract Leaderboard & Metadata
    model_best = predictor.model_best
    class_labels = getattr(predictor, "class_labels", None)
    if class_labels is not None and hasattr(class_labels, "tolist"):
        class_labels = class_labels.tolist()
    elif class_labels is not None:
        class_labels = list(class_labels)

    lb_df = predictor.leaderboard(silent=True)
    leaderboard_records = lb_df.to_dict(orient="records") if lb_df is not None else []

    print(f"[AutoGluonWorker] Training completed successfully. Best model: {model_best}")

    result_payload = {
        "status": "success",
        "payload": {
            "backend": "autogluon",
            "model_best": model_best,
            "problem_type": ag_problem_type,
            "eval_metric": ag_eval_metric,
            "class_labels": class_labels,
            "artifact_dir": str(artifact_path),
            "leaderboard": leaderboard_records,
            "autogluon_version": getattr(ag_tabular, "__version__", getattr(autogluon, "__version__", "1.6.3")),
        },
        "error_message": None,
    }
    result_path.write_text(json.dumps(result_payload, indent=2, default=str), encoding="utf-8")


def handle_predict(params: Dict[str, Any], result_path: Path) -> None:
    """Executes AutoGluon Tabular inference from saved native artifact."""
    artifact_dir = params.get("artifact_dir")
    input_parquet_path = params.get("input_parquet_path")
    problem_type_raw = params.get("problem_type")
    expected_class_labels = params.get("expected_class_labels")
    output_parquet_path = params.get("output_parquet_path")

    if not artifact_dir or not Path(artifact_dir).exists():
        raise FileNotFoundError(f"AutoGluon artifact directory not found at: {artifact_dir}")

    if not input_parquet_path or not Path(input_parquet_path).exists():
        raise FileNotFoundError(f"Input prediction dataset not found at: {input_parquet_path}")

    # 1. Load Predictor
    predictor = TabularPredictor.load(path=str(Path(artifact_dir).resolve()), verbosity=0)

    # 2. Load Input Data
    df_input = pd.read_parquet(input_parquet_path)

    # 3. Generate Predictions
    preds = predictor.predict(df_input)
    preds_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)

    # 4. Generate & Normalize Probabilities (Classification only)
    ag_problem_type = _map_problem_type(problem_type_raw or predictor.problem_type)
    probs_list = None
    class_labels = getattr(predictor, "class_labels", None)
    if class_labels is not None and hasattr(class_labels, "tolist"):
        class_labels = class_labels.tolist()
    elif class_labels is not None:
        class_labels = list(class_labels)

    if ag_problem_type in ["binary", "multiclass"]:
        probs_df = predictor.predict_proba(df_input)
        if isinstance(probs_df, pd.DataFrame):
            # Normalize column ordering if expected_class_labels provided
            if expected_class_labels:
                for lbl in expected_class_labels:
                    if lbl not in probs_df.columns:
                        probs_df[lbl] = 0.0
                probs_df = probs_df[expected_class_labels]
            probs_matrix = probs_df.to_numpy()
        elif isinstance(probs_df, pd.Series):
            probs_matrix = probs_df.to_numpy()
            if probs_matrix.ndim == 1:
                probs_matrix = np.column_stack([1.0 - probs_matrix, probs_matrix])
        else:
            probs_matrix = np.asarray(probs_df)

        # Validate probabilities (no NaNs, finite)
        if np.isnan(probs_matrix).any():
            raise ValueError("AutoGluon predict_proba produced NaN values.")

        probs_list = probs_matrix.tolist()

    # 5. Optionally save to output parquet
    if output_parquet_path:
        out_path = Path(output_parquet_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame({"prediction": preds_list})
        if probs_list is not None:
            for idx, col_name in enumerate(expected_class_labels or class_labels or []):
                df_out[f"prob_{col_name}"] = [row[idx] for row in probs_list]
        df_out.to_parquet(out_path, index=False)

    print(f"[AutoGluonWorker] Inference completed on {len(df_input)} rows.")

    result_payload = {
        "status": "success",
        "payload": {
            "predictions": preds_list,
            "probabilities": probs_list,
            "class_labels": class_labels,
            "num_samples": len(df_input),
            "output_parquet_path": output_parquet_path,
        },
        "error_message": None,
    }
    result_path.write_text(json.dumps(result_payload, indent=2, default=str), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="AutoGluon Tabular Subprocess Worker")
    parser.add_argument("--job-spec", required=True, help="Path to input job_spec.json")
    parser.add_argument("--result-path", required=True, help="Path to output result.json")

    args = parser.parse_args()
    spec_path = Path(args.job_spec)
    result_path = Path(args.result_path)

    if not spec_path.exists():
        err_msg = f"Job spec file does not exist: {spec_path}"
        print(f"FATAL: {err_msg}", file=sys.stderr)
        result_payload = {"status": "failed", "payload": {}, "error_message": err_msg}
        result_path.write_text(json.dumps(result_payload), encoding="utf-8")
        sys.exit(1)

    try:
        spec_data = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as e:
        err_msg = f"Failed to parse job spec JSON: {e}"
        print(f"FATAL: {err_msg}", file=sys.stderr)
        result_payload = {"status": "failed", "payload": {}, "error_message": err_msg}
        result_path.write_text(json.dumps(result_payload), encoding="utf-8")
        sys.exit(1)

    if not AUTOGLUON_AVAILABLE:
        err_msg = "autogluon.tabular is not installed in this Python environment."
        print(f"FATAL: {err_msg}", file=sys.stderr)
        result_payload = {"status": "failed", "payload": {}, "error_message": err_msg}
        result_path.write_text(json.dumps(result_payload), encoding="utf-8")
        sys.exit(0)

    job_type = spec_data.get("job_type", "fit")
    params = spec_data.get("params", {})

    try:
        if job_type == "fit":
            handle_fit(params, result_path)
        elif job_type == "predict":
            handle_predict(params, result_path)
        else:
            raise ValueError(f"Unknown job_type for AutoGluon worker: '{job_type}'")
        sys.exit(0)

    except Exception as e:
        tb = traceback.format_exc()
        err_msg = f"AutoGluon worker failed during {job_type}: {e}\n{tb}"
        print(f"ERROR: {err_msg}", file=sys.stderr)
        result_payload = {
            "status": "failed",
            "payload": {},
            "error_message": f"{type(e).__name__}: {str(e)}",
            "traceback": tb,
        }
        result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
        sys.exit(0)


if __name__ == "__main__":
    main()
