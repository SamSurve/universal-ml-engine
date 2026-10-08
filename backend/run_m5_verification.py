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
from backend.engine.training.m5_orchestrator import M5PipelineRunner


def run_m5_verification():
    print("=" * 80)
    print("MILESTONE M5 — EXPLAINABILITY & AUTOMATED VISUAL REPORTS VERIFICATION")
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

    clf_output_dir = project_root / "backend" / "artifacts" / "m5_verified_classification"
    clf_stage_dir = project_root / "scratch" / "stage_m5_clf"

    start_clf = time.time()
    clf_result = M5PipelineRunner.run(
        data_source=turnover_path,
        target_column="Employee_Turnover",
        output_dir=clf_output_dir,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        eval_metric="f1",
        time_limit_seconds=30.0,
        presets="medium_quality",
        include_baselines=True,
        n_importance_repeats=5,
        random_state=42,
        stage_dir=clf_stage_dir,
        dataset_name="employee_turnover",
    )
    clf_duration = time.time() - start_clf

    clf_champ = clf_result["champion_selection"].champion
    clf_test = clf_result["final_test_evaluation"]
    clf_imp = clf_result["importance_result"]
    clf_plots = clf_result["plot_paths"]
    clf_rep = clf_result["report_artifacts"]

    print(f"\nCLASSIFICATION RESULTS:")
    print(f"  Duration:            {clf_duration:.2f}s")
    print(f"  Selected Champion:   {clf_champ.model_name} ({clf_champ.backend_name})")
    print(f"  Validation Score:    {clf_champ.primary_metric} = {clf_champ.primary_val_score:.4f}")
    print(f"  Final Test Metrics:  {clf_test.test_metrics}")
    print(f"\n  Top 5 Permutation Features (External Validation):")
    for e in clf_imp.importance_entries[:5]:
        print(f"    - Rank {e.rank}: {e.feature_name:<20} | Score Delta: {e.importance_score:+.4f} | Rel: {e.relative_importance_pct:.1f}%")

    print(f"\n  Generated Plot Artifacts:")
    for name, p in clf_plots.items():
        assert Path(p).exists(), f"Missing plot: {p}"
        print(f"    - {name}: {p} ({Path(p).stat().st_size:,} bytes)")

    print(f"\n  Generated Reports:")
    assert Path(clf_rep.report_json_path).exists()
    assert Path(clf_rep.report_md_path).exists()
    print(f"    - JSON Report: {clf_rep.report_json_path} ({Path(clf_rep.report_json_path).stat().st_size:,} bytes)")
    print(f"    - Markdown Report: {clf_rep.report_md_path} ({Path(clf_rep.report_md_path).stat().st_size:,} bytes)")

    summary_results["classification"] = {
        "dataset": turnover_path.name,
        "champion_model": clf_champ.model_name,
        "champion_backend": clf_champ.backend_name,
        "primary_metric": clf_champ.primary_metric,
        "val_score": clf_champ.primary_val_score,
        "test_metrics": clf_test.test_metrics,
        "top_features": [(e.feature_name, e.importance_score, e.relative_importance_pct) for e in clf_imp.importance_entries[:5]],
        "plots": clf_plots,
        "report_json": clf_rep.report_json_path,
        "report_md": clf_rep.report_md_path,
    }

    # -------------------------------------------------------------------------
    # 2. Regression Benchmark: House Price Prediction
    # -------------------------------------------------------------------------
    housing_path = project_root / "HousePricePrediction.csv"
    print(f"\n\n[2/2] RUNNING REAL REGRESSION BENCHMARK: {housing_path.name}")
    print("-" * 70)

    reg_output_dir = project_root / "backend" / "artifacts" / "m5_verified_regression"
    reg_stage_dir = project_root / "scratch" / "stage_m5_reg"

    start_reg = time.time()
    reg_result = M5PipelineRunner.run(
        data_source=housing_path,
        target_column="SalePrice",
        output_dir=reg_output_dir,
        problem_type=ProblemType.REGRESSION,
        eval_metric="r2",
        time_limit_seconds=30.0,
        presets="medium_quality",
        include_baselines=True,
        n_importance_repeats=5,
        random_state=42,
        stage_dir=reg_stage_dir,
        dataset_name="HousePricePrediction",
    )
    reg_duration = time.time() - start_reg

    reg_champ = reg_result["champion_selection"].champion
    reg_test = reg_result["final_test_evaluation"]
    reg_imp = reg_result["importance_result"]
    reg_plots = reg_result["plot_paths"]
    reg_rep = reg_result["report_artifacts"]

    print(f"\nREGRESSION RESULTS:")
    print(f"  Duration:            {reg_duration:.2f}s")
    print(f"  Selected Champion:   {reg_champ.model_name} ({reg_champ.backend_name})")
    print(f"  Validation Score:    {reg_champ.primary_metric} = {reg_champ.primary_val_score:.4f}")
    print(f"  Final Test Metrics:  {reg_test.test_metrics}")
    print(f"\n  Top 5 Permutation Features (External Validation):")
    for e in reg_imp.importance_entries[:5]:
        print(f"    - Rank {e.rank}: {e.feature_name:<20} | Score Delta: {e.importance_score:+.4f} | Rel: {e.relative_importance_pct:.1f}%")

    print(f"\n  Generated Plot Artifacts:")
    for name, p in reg_plots.items():
        assert Path(p).exists(), f"Missing plot: {p}"
        print(f"    - {name}: {p} ({Path(p).stat().st_size:,} bytes)")

    print(f"\n  Generated Reports:")
    assert Path(reg_rep.report_json_path).exists()
    assert Path(reg_rep.report_md_path).exists()
    print(f"    - JSON Report: {reg_rep.report_json_path} ({Path(reg_rep.report_json_path).stat().st_size:,} bytes)")
    print(f"    - Markdown Report: {reg_rep.report_md_path} ({Path(reg_rep.report_md_path).stat().st_size:,} bytes)")

    summary_results["regression"] = {
        "dataset": housing_path.name,
        "champion_model": reg_champ.model_name,
        "champion_backend": reg_champ.backend_name,
        "primary_metric": reg_champ.primary_metric,
        "val_score": reg_champ.primary_val_score,
        "test_metrics": reg_test.test_metrics,
        "top_features": [(e.feature_name, e.importance_score, e.relative_importance_pct) for e in reg_imp.importance_entries[:5]],
        "plots": reg_plots,
        "report_json": reg_rep.report_json_path,
        "report_md": reg_rep.report_md_path,
    }

    print("\n" + "=" * 80)
    print("M5 VERIFICATION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_m5_verification()
