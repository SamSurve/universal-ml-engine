import sys
import time
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import ProblemType, WorkerStatus
from backend.engine.ingestion.service import IngestionService, DataValidator
from backend.engine.training.data_splitter import DataSplitter
from backend.engine.problem_detection.detector import ProblemDetector
from backend.engine.backends.autogluon_backend import AutoGluonBackend


def run_m3_verification():
    print("=" * 80)
    print("MILESTONE M3 — AUTOGLUON BACKEND END-TO-END VERIFICATION")
    print("=" * 80)

    results_summary = {}

    # -------------------------------------------------------------
    # 1. Dataset: Employee Turnover (Binary Classification)
    # -------------------------------------------------------------
    turnover_path = project_root / "employee_turnover.csv"
    if not turnover_path.exists():
        turnover_path = project_root / "employee_turnover(1).csv"

    print(f"\n[1/2] Processing Classification Benchmark: {turnover_path.name}")
    print("-" * 60)

    df_turnover_raw = IngestionService.load_dataset(turnover_path)
    df_turnover_clean, _ = DataValidator.validate_and_clean(df_turnover_raw, "Employee_Turnover")
    pd_turnover = ProblemDetector.detect(df_turnover_clean, "Employee_Turnover")

    print(f"Problem Type: {pd_turnover.problem_type.value} (Confidence: {pd_turnover.confidence:.2f})")
    print(f"Cleaned Dataset Size: {len(df_turnover_clean)} rows, {len(df_turnover_clean.columns)} columns")

    train_to, val_to, test_to, pinfo_to = DataSplitter.split_train_val_test(
        df=df_turnover_clean,
        target_column="Employee_Turnover",
        problem_type=pd_turnover.problem_type,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        random_state=42,
    )

    print(f"Data Partitions (60/20/20):")
    print(f"  TRAIN: {len(train_to)} rows | SHA-256: {pinfo_to.train_sha256[:16]}...")
    print(f"  VAL:   {len(val_to)} rows | SHA-256: {pinfo_to.val_sha256[:16]}...")
    print(f"  TEST:  {len(test_to)} rows | SHA-256: {pinfo_to.test_sha256[:16]}... (STRICTLY PROHIBITED FROM FIT)")

    art_dir_to = project_root / "backend" / "artifacts" / "autogluon_employee_turnover"
    stage_dir_to = project_root / "scratch" / "stage_ag_turnover"

    start_fit_to = time.time()
    fit_res_to = AutoGluonBackend.fit(
        df_train=train_to,
        target_column="Employee_Turnover",
        problem_type=pd_turnover.problem_type,
        eval_metric="f1",
        time_limit_seconds=30.0,
        presets="medium_quality",
        auto_stack=False,
        output_dir=art_dir_to,
        working_dir=stage_dir_to,
    )
    fit_time_to = time.time() - start_fit_to

    assert fit_res_to.status == WorkerStatus.SUCCESS, f"AutoGluon classification fit failed: {fit_res_to.error_message}"
    print(f"\nAutoGluon Fit Completed:")
    print(f"  Status: {fit_res_to.status.value}")
    print(f"  Best Model: {fit_res_to.model_best}")
    print(f"  Fit Wall Time: {fit_time_to:.2f}s (Worker Reported: {fit_res_to.fit_time_seconds:.2f}s)")
    print(f"  Peak Process Memory: {fit_res_to.peak_memory_mb:.2f} MB")
    print(f"  Artifact Path: {fit_res_to.artifact_path}")
    print(f"  AutoGluon Version: {fit_res_to.autogluon_version}")

    # Top models on internal leaderboard
    print(f"\nInternal Holdout Leaderboard (Top 5 Models):")
    for row in fit_res_to.leaderboard[:5]:
        print(f"  - {row.get('model', 'unknown'):<28}: score_val={row.get('score_val', 0.0):.4f}, fit_time={row.get('fit_time', 0.0):.2f}s")

    # Evaluate on independent VALIDATION set
    expected_classes = sorted(df_turnover_clean["Employee_Turnover"].unique().tolist())
    val_metrics_to = AutoGluonBackend.evaluate(
        df_val=val_to,
        target_column="Employee_Turnover",
        artifact_path=art_dir_to,
        problem_type=pd_turnover.problem_type,
        expected_class_labels=expected_classes,
    )
    print(f"\nIndependent Validation Metrics:")
    for k, v in sorted(val_metrics_to.items()):
        print(f"  {k:<20}: {v:.4f}")

    # Persistence / Model Reload Verification
    print(f"\nReloading AutoGluon Artifact from {art_dir_to}...")
    preds_initial_to, probs_initial_to = AutoGluonBackend.predict(
        df_input=val_to.drop(columns=["Employee_Turnover"]),
        artifact_path=art_dir_to,
        problem_type=pd_turnover.problem_type,
        expected_class_labels=expected_classes,
    )
    preds_reload_to, probs_reload_to = AutoGluonBackend.predict(
        df_input=val_to.drop(columns=["Employee_Turnover"]),
        artifact_path=art_dir_to,
        problem_type=pd_turnover.problem_type,
        expected_class_labels=expected_classes,
    )

    np.testing.assert_array_equal(preds_initial_to, preds_reload_to)
    np.testing.assert_allclose(probs_initial_to, probs_reload_to, atol=1e-6)
    print("  Artifact Reload & Prediction Equality: VERIFIED (100% identical predictions)")

    results_summary["classification"] = {
        "dataset": turnover_path.name,
        "rows_train": len(train_to),
        "rows_val": len(val_to),
        "best_model": fit_res_to.model_best,
        "fit_time_seconds": fit_res_to.fit_time_seconds,
        "peak_memory_mb": fit_res_to.peak_memory_mb,
        "val_metrics": val_metrics_to,
        "artifact_path": str(art_dir_to),
        "reload_verified": True,
    }

    # -------------------------------------------------------------
    # 2. Dataset: House Price Prediction (Regression)
    # -------------------------------------------------------------
    house_path = project_root / "HousePricePrediction.csv"
    if not house_path.exists():
        house_path = project_root / "HousePricePrediction(1).csv"

    print(f"\n\n[2/2] Processing Regression Benchmark: {house_path.name}")
    print("-" * 60)

    df_house_raw = IngestionService.load_dataset(house_path)
    df_house_clean, _ = DataValidator.validate_and_clean(df_house_raw, "SalePrice")
    pd_house = ProblemDetector.detect(df_house_clean, "SalePrice")

    print(f"Problem Type: {pd_house.problem_type.value} (Confidence: {pd_house.confidence:.2f})")
    print(f"Cleaned Dataset Size: {len(df_house_clean)} rows, {len(df_house_clean.columns)} columns")

    train_hp, val_hp, test_hp, pinfo_hp = DataSplitter.split_train_val_test(
        df=df_house_clean,
        target_column="SalePrice",
        problem_type=pd_house.problem_type,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    print(f"Data Partitions (70/15/15):")
    print(f"  TRAIN: {len(train_hp)} rows | SHA-256: {pinfo_hp.train_sha256[:16]}...")
    print(f"  VAL:   {len(val_hp)} rows | SHA-256: {pinfo_hp.val_sha256[:16]}...")
    print(f"  TEST:  {len(test_hp)} rows | SHA-256: {pinfo_hp.test_sha256[:16]}... (STRICTLY PROHIBITED FROM FIT)")

    art_dir_hp = project_root / "backend" / "artifacts" / "autogluon_house_prices"
    stage_dir_hp = project_root / "scratch" / "stage_ag_house"

    start_fit_hp = time.time()
    fit_res_hp = AutoGluonBackend.fit(
        df_train=train_hp,
        target_column="SalePrice",
        problem_type=pd_house.problem_type,
        eval_metric="r2",
        time_limit_seconds=30.0,
        presets="medium_quality",
        auto_stack=False,
        output_dir=art_dir_hp,
        working_dir=stage_dir_hp,
    )
    fit_time_hp = time.time() - start_fit_hp

    assert fit_res_hp.status == WorkerStatus.SUCCESS, f"AutoGluon regression fit failed: {fit_res_hp.error_message}"
    print(f"\nAutoGluon Fit Completed:")
    print(f"  Status: {fit_res_hp.status.value}")
    print(f"  Best Model: {fit_res_hp.model_best}")
    print(f"  Fit Wall Time: {fit_time_hp:.2f}s (Worker Reported: {fit_res_hp.fit_time_seconds:.2f}s)")
    print(f"  Peak Process Memory: {fit_res_hp.peak_memory_mb:.2f} MB")
    print(f"  Artifact Path: {fit_res_hp.artifact_path}")
    print(f"  AutoGluon Version: {fit_res_hp.autogluon_version}")

    # Top models on internal leaderboard
    print(f"\nInternal Holdout Leaderboard (Top 5 Models):")
    for row in fit_res_hp.leaderboard[:5]:
        print(f"  - {row.get('model', 'unknown'):<28}: score_val={row.get('score_val', 0.0):.4f}, fit_time={row.get('fit_time', 0.0):.2f}s")

    # Evaluate on independent VALIDATION set
    val_metrics_hp = AutoGluonBackend.evaluate(
        df_val=val_hp,
        target_column="SalePrice",
        artifact_path=art_dir_hp,
        problem_type=pd_house.problem_type,
    )
    print(f"\nIndependent Validation Metrics:")
    for k, v in sorted(val_metrics_hp.items()):
        print(f"  {k:<20}: {v:.4f}")

    # Persistence / Model Reload Verification
    print(f"\nReloading AutoGluon Artifact from {art_dir_hp}...")
    preds_initial_hp, _ = AutoGluonBackend.predict(
        df_input=val_hp.drop(columns=["SalePrice"]),
        artifact_path=art_dir_hp,
        problem_type=pd_house.problem_type,
    )
    preds_reload_hp, _ = AutoGluonBackend.predict(
        df_input=val_hp.drop(columns=["SalePrice"]),
        artifact_path=art_dir_hp,
        problem_type=pd_house.problem_type,
    )

    np.testing.assert_array_equal(preds_initial_hp, preds_reload_hp)
    print("  Artifact Reload & Prediction Equality: VERIFIED (100% identical predictions)")

    results_summary["regression"] = {
        "dataset": house_path.name,
        "rows_train": len(train_hp),
        "rows_val": len(val_hp),
        "best_model": fit_res_hp.model_best,
        "fit_time_seconds": fit_res_hp.fit_time_seconds,
        "peak_memory_mb": fit_res_hp.peak_memory_mb,
        "val_metrics": val_metrics_hp,
        "artifact_path": str(art_dir_hp),
        "reload_verified": True,
    }

    print("\n" + "=" * 80)
    print("ALL M3 REAL DATASET BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return results_summary


if __name__ == "__main__":
    run_m3_verification()
