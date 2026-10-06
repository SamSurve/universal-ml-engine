import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.engine.orchestrator import AutoMLEngine


def run_benchmarks():
    print("=" * 80)
    print("UNIVERSAL ML ENGINE — REAL DATASET BENCHMARK EXECUTION")
    print("=" * 80)

    # 1. Employee Turnover Dataset (Classification)
    turnover_path = project_root / "employee_turnover.csv"
    if not turnover_path.exists():
        turnover_path = project_root / "employee_turnover(1).csv"

    print(f"\n[1/2] Running Classification Benchmark on: {turnover_path.name}")
    print("Target: Employee_Turnover")
    print("-" * 60)

    turnover_result = AutoMLEngine.run(
        data_source=turnover_path,
        target_column="Employee_Turnover",
        output_dir=str(project_root / "backend" / "artifacts" / "employee_turnover"),
        random_state=42,
        n_splits=5,
    )

    print(f"Detected Problem: {turnover_result.problem_detection.problem_type.value} "
          f"(Confidence: {turnover_result.problem_detection.confidence:.2f})")
    print(f"Exclusions Recorded: {len(turnover_result.validation_result.exclusions)}")
    print(f"Dev Shape: {turnover_result.dev_shape}, Holdout Shape: {turnover_result.holdout_shape}")
    print("\nLeaderboard:")
    for entry in turnover_result.leaderboard:
        print(f"  Rank {entry.rank}: {entry.model_name:<25} | "
              f"CV {entry.primary_metric}: {entry.cv_score_mean:.4f} +/- {entry.cv_score_std:.4f} | "
              f"Fit Time: {entry.fit_time_seconds:.2f}s | Status: {entry.status}")
    print(f"\nBest Model Selected: {turnover_result.best_model_name}")
    print(f"Holdout Evaluation Metrics: {json.dumps(turnover_result.holdout_metrics, indent=2)}")
    print(f"Saved Artifact: {turnover_result.artifact_path}")

    # 2. House Price Prediction Dataset (Regression)
    house_path = project_root / "HousePricePrediction.csv"
    if not house_path.exists():
        house_path = project_root / "HousePricePrediction(1).csv"

    print(f"\n[2/2] Running Regression Benchmark on: {house_path.name}")
    print("Target: SalePrice")
    print("-" * 60)

    house_result = AutoMLEngine.run(
        data_source=house_path,
        target_column="SalePrice",
        output_dir=str(project_root / "backend" / "artifacts" / "house_price"),
        random_state=42,
        n_splits=5,
    )

    print(f"Detected Problem: {house_result.problem_detection.problem_type.value} "
          f"(Confidence: {house_result.problem_detection.confidence:.2f})")
    print(f"Exclusions Recorded: {len(house_result.validation_result.exclusions)}")
    print(f"Dev Shape: {house_result.dev_shape}, Holdout Shape: {house_result.holdout_shape}")
    print("\nLeaderboard:")
    for entry in house_result.leaderboard:
        print(f"  Rank {entry.rank}: {entry.model_name:<25} | "
              f"CV {entry.primary_metric}: {entry.cv_score_mean:.4f} +/- {entry.cv_score_std:.4f} | "
              f"Fit Time: {entry.fit_time_seconds:.2f}s | Status: {entry.status}")
    print(f"\nBest Model Selected: {house_result.best_model_name}")
    print(f"Holdout Evaluation Metrics: {json.dumps(house_result.holdout_metrics, indent=2)}")
    print(f"Saved Artifact: {house_result.artifact_path}")

    print("\n" + "=" * 80)
    print("ALL REAL DATASET BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmarks()
