import sys
import time
import os
import json
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import ProblemType
from backend.engine.training.m4_orchestrator import M4PipelineRunner
from backend.engine.inference.unified_predictor import UnifiedPredictor


def run_m4_verification():
    print("=" * 80)
    print("MILESTONE M4 — CHAMPION SELECTION & FINAL TEST EVALUATION VERIFICATION")
    print("=" * 80)

    summary_results = {}

    # -------------------------------------------------------------------------
    # 1. Classification Benchmark: Employee Turnover
    # -------------------------------------------------------------------------
    turnover_path = project_root / "employee_turnover.csv"
    if not turnover_path.exists():
        turnover_path = project_root / "employee_turnover(1).csv"

    print(f"\n[1/2] RUNNING REAL CLASSIFICATION BENCHMARK: {turnover_path.name}")
    print("-" * 70)

    clf_output_dir = project_root / "backend" / "artifacts" / "m4_champion_classification"
    clf_stage_dir = project_root / "scratch" / "stage_m4_clf"

    start_clf = time.time()
    clf_result = M4PipelineRunner.run(
        data_source=turnover_path,
        target_column="Employee_Turnover",
        output_dir=clf_output_dir,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        eval_metric="f1",
        time_limit_seconds=30.0,
        presets="medium_quality",
        include_baselines=True,
        random_state=42,
        stage_dir=clf_stage_dir,
    )
    clf_duration = time.time() - start_clf

    clf_champion = clf_result["champion_selection"].champion
    clf_test_eval = clf_result["final_test_evaluation"]
    clf_manifest = clf_result["manifest"]
    clf_predictor = clf_result["predictor"]

    print(f"\nCLASSIFICATION RESULTS:")
    print(f"  Execution Time:        {clf_duration:.2f}s")
    print(f"  Selected Champion:     {clf_champion.model_name} (Backend: {clf_champion.backend_name})")
    print(f"  Champion Candidate ID: {clf_champion.candidate_id}")
    print(f"  Selection Primary Metric: {clf_champion.primary_metric} = {clf_champion.primary_val_score:.4f}")
    print(f"  Selection Rationale:   {clf_result['champion_selection'].selection_rationale}")
    print(f"\n  All Candidates Validation Comparison:")
    for cand in clf_result["champion_selection"].all_candidates:
        print(f"    - {cand.candidate_id:<25} ({cand.backend_name:<10}): {cand.primary_metric}={cand.primary_val_score:.4f}, time={cand.fit_time_seconds:.2f}s")

    print(f"\n  Validation Metrics (Champion):")
    for k, v in clf_champion.val_metrics.items():
        if isinstance(v, (int, float)):
            print(f"    - {k:<20}: {v:.4f}")

    print(f"\n  FINAL TEST METRICS (Untouched Test Partition):")
    for k, v in clf_test_eval.test_metrics.items():
        if isinstance(v, (int, float)):
            print(f"    - {k:<20}: {v:.4f}")

    # Verify Reload & Inference on fresh holdout sample
    df_raw_clf = pd.read_csv(turnover_path)
    test_sample_clf = df_raw_clf.sample(n=5, random_state=123).drop(columns=["Employee_Turnover"])
    sample_preds_clf = clf_predictor.predict(test_sample_clf)
    sample_probs_clf = clf_predictor.predict_proba(test_sample_clf)

    print(f"\n  Inference Verification (Sample 5 rows):")
    print(f"    Predictions: {sample_preds_clf}")
    print(f"    Probabilities shape: {sample_probs_clf.shape}")
    print(f"    Manifest Path: {clf_output_dir / 'champion_manifest.json'}")
    assert (clf_output_dir / "champion_manifest.json").exists(), "Manifest missing!"

    summary_results["classification"] = {
        "dataset": turnover_path.name,
        "champion_model": clf_champion.model_name,
        "champion_backend": clf_champion.backend_name,
        "primary_metric": clf_champion.primary_metric,
        "val_score": clf_champion.primary_val_score,
        "val_metrics": clf_champion.val_metrics,
        "test_metrics": clf_test_eval.test_metrics,
        "selection_rationale": clf_result["champion_selection"].selection_rationale,
        "manifest_path": str(clf_output_dir / "champion_manifest.json"),
    }

    # -------------------------------------------------------------------------
    # 2. Regression Benchmark: House Price Prediction
    # -------------------------------------------------------------------------
    housing_path = project_root / "HousePricePrediction.csv"
    print(f"\n\n[2/2] RUNNING REAL REGRESSION BENCHMARK: {housing_path.name}")
    print("-" * 70)

    reg_output_dir = project_root / "backend" / "artifacts" / "m4_champion_regression"
    reg_stage_dir = project_root / "scratch" / "stage_m4_reg"

    start_reg = time.time()
    reg_result = M4PipelineRunner.run(
        data_source=housing_path,
        target_column="SalePrice",
        output_dir=reg_output_dir,
        problem_type=ProblemType.REGRESSION,
        eval_metric="r2",
        time_limit_seconds=30.0,
        presets="medium_quality",
        include_baselines=True,
        random_state=42,
        stage_dir=reg_stage_dir,
    )
    reg_duration = time.time() - start_reg

    reg_champion = reg_result["champion_selection"].champion
    reg_test_eval = reg_result["final_test_evaluation"]
    reg_manifest = reg_result["manifest"]
    reg_predictor = reg_result["predictor"]

    print(f"\nREGRESSION RESULTS:")
    print(f"  Execution Time:        {reg_duration:.2f}s")
    print(f"  Selected Champion:     {reg_champion.model_name} (Backend: {reg_champion.backend_name})")
    print(f"  Champion Candidate ID: {reg_champion.candidate_id}")
    print(f"  Selection Primary Metric: {reg_champion.primary_metric} = {reg_champion.primary_val_score:.4f}")
    print(f"  Selection Rationale:   {reg_result['champion_selection'].selection_rationale}")
    print(f"\n  All Candidates Validation Comparison:")
    for cand in reg_result["champion_selection"].all_candidates:
        print(f"    - {cand.candidate_id:<25} ({cand.backend_name:<10}): {cand.primary_metric}={cand.primary_val_score:.4f}, time={cand.fit_time_seconds:.2f}s")

    print(f"\n  Validation Metrics (Champion):")
    for k, v in reg_champion.val_metrics.items():
        if isinstance(v, (int, float)):
            print(f"    - {k:<20}: {v:.4f}")

    print(f"\n  FINAL TEST METRICS (Untouched Test Partition):")
    for k, v in reg_test_eval.test_metrics.items():
        if isinstance(v, (int, float)):
            print(f"    - {k:<20}: {v:.4f}")

    # Verify Reload & Inference on fresh holdout sample
    df_raw_reg = pd.read_csv(housing_path)
    test_sample_reg = df_raw_reg.sample(n=5, random_state=123).drop(columns=["SalePrice"])
    sample_preds_reg = reg_predictor.predict(test_sample_reg)

    print(f"\n  Inference Verification (Sample 5 rows):")
    print(f"    Predictions: {sample_preds_reg}")
    print(f"    Manifest Path: {reg_output_dir / 'champion_manifest.json'}")
    assert (reg_output_dir / "champion_manifest.json").exists(), "Manifest missing!"

    summary_results["regression"] = {
        "dataset": housing_path.name,
        "champion_model": reg_champion.model_name,
        "champion_backend": reg_champion.backend_name,
        "primary_metric": reg_champion.primary_metric,
        "val_score": reg_champion.primary_val_score,
        "val_metrics": reg_champion.val_metrics,
        "test_metrics": reg_test_eval.test_metrics,
        "selection_rationale": reg_result["champion_selection"].selection_rationale,
        "manifest_path": str(reg_output_dir / "champion_manifest.json"),
    }

    # Write results summary JSON
    results_json_path = project_root / "m4_verification_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"M4 VERIFICATION COMPLETED SUCCESSFULLY. Results saved to {results_json_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_m4_verification()
