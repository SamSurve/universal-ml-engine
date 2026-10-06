import os
import sys
import json
import shutil
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.engine.orchestrator import AutoMLEngine


def run_benchmarks():
    print("=" * 80)
    print("UNIVERSAL ML ENGINE — INTELLIGENCE, EXPLAINABILITY & REPORTING BENCHMARKS")
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
        enable_screening=True,
        screening_time_budget=30,
        enable_tuning=True,
        top_k_to_tune=2,
        tuning_trials=15,
        tuning_time_budget=45,
        enable_intelligence=True,
        enable_explainability=True,
        generate_reports=True,
    )

    print(f"Detected Problem: {turnover_result.problem_detection.problem_type.value} "
          f"(Confidence: {turnover_result.problem_detection.confidence:.2f})")
    print(f"Exclusions Recorded: {len(turnover_result.validation_result.exclusions)}")
    print(f"Dev Shape: {turnover_result.dev_shape}, Holdout Shape: {turnover_result.holdout_shape}")

    # Dataset Intelligence summary
    if turnover_result.dataset_intelligence:
        hs = turnover_result.dataset_intelligence.health_score
        print(f"\nDataset Health Score: {hs.overall_score}/100 ({hs.grade})")
        print(f"Health Interpretation: {hs.interpretation}")
        print(f"Quality Warnings Count: {len(turnover_result.dataset_intelligence.warnings)}")
        if hs.clean_signals:
            print("Clean Signals:")
            for s in hs.clean_signals[:3]:
                print(f"  + {s}")

    if turnover_result.screened_candidates:
        print("\nFLAML Screened Candidates:")
        for sc in turnover_result.screened_candidates:
            print(f"  Rank {sc.rank}: {sc.model_name:<20} | CV Score: {sc.cv_score:.4f} | Time: {sc.fit_time_seconds:.2f}s | Status: {sc.status}")

    if turnover_result.tuned_candidates:
        print("\nOptuna Tuned Candidates:")
        for tc in turnover_result.tuned_candidates:
            print(f"  Model: {tc.model_name:<20} | Baseline: {tc.baseline_cv_score:.4f} -> Tuned: {tc.tuned_cv_score:.4f} "
                  f"(Imp: {tc.improvement:+.4f}) | Trials: {tc.n_trials} | Time: {tc.tuning_time_seconds:.2f}s")

    print("\nFinal Leaderboard:")
    for entry in turnover_result.leaderboard:
        print(f"  Rank {entry.rank}: {entry.model_name:<25} | "
              f"CV {entry.primary_metric}: {entry.cv_score_mean:.4f} +/- {entry.cv_score_std:.4f} | "
              f"Fit Time: {entry.fit_time_seconds:.2f}s | Status: {entry.status}")

    print(f"\nBest Model Selected: {turnover_result.best_model_name}")
    if turnover_result.tuned_hyperparameters:
        print(f"Tuned Hyperparameters: {json.dumps(turnover_result.tuned_hyperparameters, indent=2)}")
    print(f"Holdout Evaluation Metrics: {json.dumps(turnover_result.holdout_metrics, indent=2)}")

    # Explainability summary
    if turnover_result.explainability and turnover_result.explainability.status in ("success", "fallback"):
        expl = turnover_result.explainability
        print(f"\nSHAP Explainability ({expl.explainer_type}):")
        print(f"  Summary: {expl.summary_text}")
        print("  Top Global Features:")
        for feat in expl.raw_feature_importance[:5]:
            print(f"    Rank #{feat.rank}: {feat.feature_name:<20} | Score: {feat.importance_score:.4f} ({feat.relative_importance_pct:.1f}%)")
        if expl.example_explanations:
            ex = expl.example_explanations[0]
            print(f"  Example Sample #1 Prediction: {ex.predicted_value} (Base Value: {ex.base_value:.4f})")
            print(f"    Top Positive: {', '.join(ex.top_positive_features)}")
            print(f"    Top Negative: {', '.join(ex.top_negative_features)}")

    print(f"\nTotal Runtime: {turnover_result.total_runtime_seconds:.2f}s")
    print(f"Saved Artifact: {turnover_result.artifact_path}")
    print(f"Generated Reports: {turnover_result.model_report_path}")

    # Copy turnover reports to root as primary showcase reports
    if turnover_result.model_report_path and Path(turnover_result.model_report_path).exists():
        shutil.copyfile(turnover_result.model_report_path, project_root / "MODEL_REPORT.md")
    if turnover_result.intelligence_report_path and Path(turnover_result.intelligence_report_path).exists():
        shutil.copyfile(turnover_result.intelligence_report_path, project_root / "DATASET_INTELLIGENCE_REPORT.md")

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
        enable_screening=True,
        screening_time_budget=30,
        enable_tuning=True,
        top_k_to_tune=2,
        tuning_trials=15,
        tuning_time_budget=45,
        enable_intelligence=True,
        enable_explainability=True,
        generate_reports=True,
    )

    print(f"Detected Problem: {house_result.problem_detection.problem_type.value} "
          f"(Confidence: {house_result.problem_detection.confidence:.2f})")
    print(f"Exclusions Recorded: {len(house_result.validation_result.exclusions)}")
    print(f"Dev Shape: {house_result.dev_shape}, Holdout Shape: {house_result.holdout_shape}")

    # Dataset Intelligence summary
    if house_result.dataset_intelligence:
        hs = house_result.dataset_intelligence.health_score
        print(f"\nDataset Health Score: {hs.overall_score}/100 ({hs.grade})")
        print(f"Health Interpretation: {hs.interpretation}")
        print(f"Quality Warnings Count: {len(house_result.dataset_intelligence.warnings)}")
        if hs.clean_signals:
            print("Clean Signals:")
            for s in hs.clean_signals[:3]:
                print(f"  + {s}")

    if house_result.screened_candidates:
        print("\nFLAML Screened Candidates:")
        for sc in house_result.screened_candidates:
            print(f"  Rank {sc.rank}: {sc.model_name:<20} | CV Score: {sc.cv_score:.4f} | Time: {sc.fit_time_seconds:.2f}s | Status: {sc.status}")

    if house_result.tuned_candidates:
        print("\nOptuna Tuned Candidates:")
        for tc in house_result.tuned_candidates:
            print(f"  Model: {tc.model_name:<20} | Baseline: {tc.baseline_cv_score:.4f} -> Tuned: {tc.tuned_cv_score:.4f} "
                  f"(Imp: {tc.improvement:+.4f}) | Trials: {tc.n_trials} | Time: {tc.tuning_time_seconds:.2f}s")

    print("\nFinal Leaderboard:")
    for entry in house_result.leaderboard:
        print(f"  Rank {entry.rank}: {entry.model_name:<25} | "
              f"CV {entry.primary_metric}: {entry.cv_score_mean:.4f} +/- {entry.cv_score_std:.4f} | "
              f"Fit Time: {entry.fit_time_seconds:.2f}s | Status: {entry.status}")

    print(f"\nBest Model Selected: {house_result.best_model_name}")
    if house_result.tuned_hyperparameters:
        print(f"Tuned Hyperparameters: {json.dumps(house_result.tuned_hyperparameters, indent=2)}")
    print(f"Holdout Evaluation Metrics: {json.dumps(house_result.holdout_metrics, indent=2)}")

    # Explainability summary
    if house_result.explainability and house_result.explainability.status in ("success", "fallback"):
        expl = house_result.explainability
        print(f"\nSHAP Explainability ({expl.explainer_type}):")
        print(f"  Summary: {expl.summary_text}")
        print("  Top Global Features:")
        for feat in expl.raw_feature_importance[:5]:
            print(f"    Rank #{feat.rank}: {feat.feature_name:<20} | Score: {feat.importance_score:.4f} ({feat.relative_importance_pct:.1f}%)")
        if expl.example_explanations:
            ex = expl.example_explanations[0]
            print(f"  Example Sample #1 Prediction: {ex.predicted_value} (Base Value: {ex.base_value:.4f})")
            print(f"    Top Positive: {', '.join(ex.top_positive_features)}")
            print(f"    Top Negative: {', '.join(ex.top_negative_features)}")

    print(f"\nTotal Runtime: {house_result.total_runtime_seconds:.2f}s")
    print(f"Saved Artifact: {house_result.artifact_path}")
    print(f"Generated Reports: {house_result.model_report_path}")

    print("\n" + "=" * 80)
    print("ALL REAL DATASET BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmarks()
