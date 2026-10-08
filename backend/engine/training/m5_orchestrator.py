import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import (
    ProblemType,
    PermutationImportanceResult,
    M5ReportArtifacts,
)
from backend.engine.training.m4_orchestrator import M4PipelineRunner
from backend.engine.explainability.permutation_explainer import PermutationExplainer
from backend.engine.reporting.visualizer import M5Visualizer
from backend.engine.reporting.m5_report_generator import M5ReportGenerator

logger = logging.getLogger(__name__)


class M5PipelineRunner:
    """
    End-to-End M5 Orchestration:
    Runs complete M4 pipeline (Train -> Val Selection -> Untouched Test Eval -> Persist -> Reload)
    + M5 lightweight explainability (Permutation Importance strictly on VALIDATION partition)
    + M5 automated diagnostic visualizations (Confusion matrix / Actual vs Predicted / Residuals / Importance)
    + M5 unified machine-readable JSON & human-readable Markdown reports.
    """

    @classmethod
    def run(
        cls,
        data_source: Union[str, Path, pd.DataFrame],
        target_column: str,
        output_dir: Union[str, Path],
        problem_type: Optional[ProblemType] = None,
        eval_metric: Optional[str] = None,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
        time_limit_seconds: float = 30.0,
        presets: str = "medium_quality",
        auto_stack: bool = False,
        include_baselines: bool = True,
        n_importance_repeats: int = 5,
        random_state: int = 42,
        stage_dir: Optional[Union[str, Path]] = None,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes the complete M5 pipeline with explainability and visual reports.
        """
        start_time = time.time()
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Run Complete M4 Pipeline
        m4_res = M4PipelineRunner.run(
            data_source=data_source,
            target_column=target_column,
            output_dir=out_path,
            problem_type=problem_type,
            eval_metric=eval_metric,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            time_limit_seconds=time_limit_seconds,
            presets=presets,
            auto_stack=auto_stack,
            include_baselines=include_baselines,
            random_state=random_state,
            stage_dir=stage_dir,
        )

        df_val = m4_res["df_val"]
        df_test = m4_res["df_test"]
        resolved_p_type = m4_res["problem_type"]
        predictor = m4_res["predictor"]

        if dataset_name is None:
            if isinstance(data_source, (str, Path)):
                dataset_name = Path(data_source).name
            else:
                dataset_name = "dataset"

        # 2. Permutation Importance strictly on VALIDATION partition
        importance_result = PermutationExplainer.compute_importance(
            predictor=predictor,
            df_val=df_val,
            target_column=target_column,
            problem_type=resolved_p_type,
            primary_metric=eval_metric or m4_res["champion_selection"].primary_metric,
            n_repeats=n_importance_repeats,
            random_state=random_state,
        )

        # 3. Generate Visual Diagnostics
        plots_dir = out_path / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        y_true_test = df_test[target_column].to_numpy()
        X_test = df_test.drop(columns=[target_column])
        y_pred_test = predictor.predict(X_test)

        is_classification = resolved_p_type in [
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ]

        if is_classification:
            plot_paths = M5Visualizer.generate_classification_plots(
                y_true_test=y_true_test,
                y_pred_test=y_pred_test,
                importance_entries=importance_result.importance_entries,
                output_dir=plots_dir,
                class_labels=m4_res["manifest"].class_labels,
            )
        else:
            plot_paths = M5Visualizer.generate_regression_plots(
                y_true_test=y_true_test,
                y_pred_test=y_pred_test,
                importance_entries=importance_result.importance_entries,
                output_dir=plots_dir,
            )

        # 4. Generate Machine & Human-Readable Reports
        report_artifacts = M5ReportGenerator.generate_report(
            manifest=m4_res["manifest"],
            champion_selection=m4_res["champion_selection"],
            test_evaluation=m4_res["final_test_evaluation"],
            importance_result=importance_result,
            plot_paths=plot_paths,
            output_dir=out_path,
            dataset_name=dataset_name,
            partition_info=m4_res["partition_info"],
        )

        total_runtime = time.time() - start_time
        logger.info(
            f"[M5PipelineRunner] Complete M5 pipeline finished in {total_runtime:.2f}s. "
            f"Report: {report_artifacts.report_md_path}"
        )

        return {
            **m4_res,
            "importance_result": importance_result,
            "plot_paths": plot_paths,
            "report_artifacts": report_artifacts,
            "m5_total_runtime_seconds": total_runtime,
        }
