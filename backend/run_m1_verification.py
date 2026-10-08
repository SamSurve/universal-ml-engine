import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import ProblemType
from backend.engine.ingestion.service import IngestionService, DataValidator
from backend.engine.training.data_splitter import DataSplitter
from backend.engine.evaluation.baselines import BaselineEvaluator
from backend.engine.problem_detection.detector import ProblemDetector


def run_verification():
    print("=" * 80)
    print("MILESTONE M1 — 3-WAY SPLIT & BASELINE VERIFICATION")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Dataset: Employee Turnover (Binary Classification)
    # -------------------------------------------------------------
    turnover_path = project_root / "employee_turnover.csv"
    if not turnover_path.exists():
        turnover_path = project_root / "employee_turnover(1).csv"

    print(f"\n[1/2] Processing Classification Dataset: {turnover_path.name}")
    print("-" * 60)

    df_turnover_raw = IngestionService.load_dataset(turnover_path)
    df_turnover_clean, val_res_turnover = DataValidator.validate_and_clean(
        df_turnover_raw, "Employee_Turnover"
    )

    pd_turnover = ProblemDetector.detect(df_turnover_clean, "Employee_Turnover")
    print(f"Detected Problem: {pd_turnover.problem_type.value} (Confidence: {pd_turnover.confidence:.2f})")
    print(f"Total Cleaned Rows: {len(df_turnover_clean)}")

    train_to, val_to, test_to, pinfo_to = DataSplitter.split_train_val_test(
        df=df_turnover_clean,
        target_column="Employee_Turnover",
        problem_type=pd_turnover.problem_type,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        random_state=42,
    )

    print(f"\nPartitioning Summary (60/20/20):")
    print(f"  Train Set: {len(train_to)} rows ({len(train_to)/len(df_turnover_clean)*100:.1f}%) | SHA-256: {pinfo_to.train_sha256[:16]}...")
    print(f"  Val Set:   {len(val_to)} rows ({len(val_to)/len(df_turnover_clean)*100:.1f}%) | SHA-256: {pinfo_to.val_sha256[:16]}...")
    print(f"  Test Set:  {len(test_to)} rows ({len(test_to)/len(df_turnover_clean)*100:.1f}%) | SHA-256: {pinfo_to.test_sha256[:16]}...")
    print(f"  Stratified: {pinfo_to.stratified}")

    # Check Disjointness
    set_tr = set(pinfo_to.train_indices)
    set_v = set(pinfo_to.val_indices)
    set_te = set(pinfo_to.test_indices)
    assert len(set_tr & set_v) == 0, "Train and Val overlap!"
    assert len(set_tr & set_te) == 0, "Train and Test overlap!"
    assert len(set_v & set_te) == 0, "Val and Test overlap!"
    print("  Disjointness Check: PASSED (Zero row overlap across all 3 partitions)")

    # Baseline Evaluation on Validation Set
    baseline_res_to = BaselineEvaluator.evaluate_baselines(
        df_train=train_to,
        df_val=val_to,
        target_column="Employee_Turnover",
        problem_type=pd_turnover.problem_type,
        random_state=42,
    )

    print(f"\nBaseline Performance on Independent Validation Set:")
    print(f"  Primary Metric: {baseline_res_to.primary_metric}")
    for b in baseline_res_to.all_baselines:
        print(f"  - {b.model_name:<32}: Val Score={b.val_score:.4f} (Fit Time: {b.fit_time_seconds:.4f}s)")
        print(f"    Metrics: {b.val_metrics}")
    print(f"  Best Baseline: {baseline_res_to.best_baseline_name} ({baseline_res_to.best_baseline_score:.4f})")

    # -------------------------------------------------------------
    # 2. Dataset: House Price Prediction (Regression)
    # -------------------------------------------------------------
    house_path = project_root / "HousePricePrediction.csv"
    if not house_path.exists():
        house_path = project_root / "HousePricePrediction(1).csv"

    print(f"\n[2/2] Processing Regression Dataset: {house_path.name}")
    print("-" * 60)

    df_house_raw = IngestionService.load_dataset(house_path)
    df_house_clean, val_res_house = DataValidator.validate_and_clean(
        df_house_raw, "SalePrice"
    )

    pd_house = ProblemDetector.detect(df_house_clean, "SalePrice")
    print(f"Detected Problem: {pd_house.problem_type.value} (Confidence: {pd_house.confidence:.2f})")
    print(f"Total Cleaned Rows: {len(df_house_clean)}")

    train_hp, val_hp, test_hp, pinfo_hp = DataSplitter.split_train_val_test(
        df=df_house_clean,
        target_column="SalePrice",
        problem_type=pd_house.problem_type,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    print(f"\nPartitioning Summary (70/15/15):")
    print(f"  Train Set: {len(train_hp)} rows ({len(train_hp)/len(df_house_clean)*100:.1f}%) | SHA-256: {pinfo_hp.train_sha256[:16]}...")
    print(f"  Val Set:   {len(val_hp)} rows ({len(val_hp)/len(df_house_clean)*100:.1f}%) | SHA-256: {pinfo_hp.val_sha256[:16]}...")
    print(f"  Test Set:  {len(test_hp)} rows ({len(test_hp)/len(df_house_clean)*100:.1f}%) | SHA-256: {pinfo_hp.test_sha256[:16]}...")
    print(f"  Stratified: {pinfo_hp.stratified}")

    # Check Disjointness
    set_tr_hp = set(pinfo_hp.train_indices)
    set_v_hp = set(pinfo_hp.val_indices)
    set_te_hp = set(pinfo_hp.test_indices)
    assert len(set_tr_hp & set_v_hp) == 0, "Train and Val overlap!"
    assert len(set_tr_hp & set_te_hp) == 0, "Train and Test overlap!"
    assert len(set_v_hp & set_te_hp) == 0, "Val and Test overlap!"
    print("  Disjointness Check: PASSED (Zero row overlap across all 3 partitions)")

    # Baseline Evaluation on Validation Set
    baseline_res_hp = BaselineEvaluator.evaluate_baselines(
        df_train=train_hp,
        df_val=val_hp,
        target_column="SalePrice",
        problem_type=pd_house.problem_type,
        random_state=42,
    )

    print(f"\nBaseline Performance on Independent Validation Set:")
    print(f"  Primary Metric: {baseline_res_hp.primary_metric}")
    for b in baseline_res_hp.all_baselines:
        print(f"  - {b.model_name:<32}: Val Score={b.val_score:.4f} (Fit Time: {b.fit_time_seconds:.4f}s)")
        print(f"    Metrics: {b.val_metrics}")
    print(f"  Best Baseline: {baseline_res_hp.best_baseline_name} ({baseline_res_hp.best_baseline_score:.4f})")

    print("\n" + "=" * 80)
    print("MILESTONE M1 VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
