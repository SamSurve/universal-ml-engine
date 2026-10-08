import os
import uuid
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import (
    ProblemType,
    PartitionInfo,
    AutoGluonBackendResult,
    BaselineSuiteResult,
    ChampionCandidate,
    ChampionSelectionResult,
    FinalTestEvaluationResult,
    PersistedChampionManifest,
)
from backend.engine.ingestion.service import IngestionService, DataValidator
from backend.engine.problem_detection.detector import ProblemDetector
from backend.engine.training.data_splitter import DataSplitter
from backend.engine.evaluation.baselines import BaselineEvaluator
from backend.engine.backends.autogluon_backend import AutoGluonBackend
from backend.engine.evaluation.champion_selector import ChampionSelector
from backend.engine.evaluation.final_evaluator import FinalTestEvaluator
from backend.engine.artifacts.champion_store import ChampionStore
from backend.engine.inference.unified_predictor import UnifiedPredictor

logger = logging.getLogger(__name__)


class M4PipelineRunner:
    """
    End-to-End M4 Orchestration:
    1. Ingestion & Validation
    2. 3-Way Partitioning (Train / Validation / Test)
    3. Fit Candidates on TRAIN only (AutoGluon + Baselines)
    4. External VALIDATION Champion Selection
    5. Untouched TEST Evaluation (Single-Pass)
    6. Native Champion Persistence & Manifest
    7. Unified Inference Verification
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
        random_state: int = 42,
        stage_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the complete M4 pipeline.
        """
        start_time = time.time()
        run_id = uuid.uuid4().hex[:8]

        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        work_stage = Path(stage_dir) if stage_dir else Path(os.environ.get("TEMP", ".")) / f"m4_run_{run_id}"
        work_stage.mkdir(parents=True, exist_ok=True)

        # 1. Ingestion
        if isinstance(data_source, (str, Path)):
            df_raw = IngestionService.load_dataset(data_source)
        elif isinstance(data_source, pd.DataFrame):
            df_raw = data_source.copy()
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")

        # 2. Data Validation & Cleaning
        df_clean, val_res = DataValidator.validate_and_clean(df_raw, target_column=target_column)
        if not val_res.is_valid:
            raise ValueError(f"Dataset validation failed: {val_res.errors}")

        # 3. Problem Detection
        if problem_type is None or problem_type == ProblemType.UNKNOWN:
            pd_res = ProblemDetector.detect(df_clean, target_column=target_column)
            resolved_problem_type = pd_res.problem_type
        else:
            resolved_problem_type = problem_type

        # 4. 3-Way Leakage-Safe Data Split
        df_train, df_val, df_test, partition_info = DataSplitter.split_train_val_test(
            df=df_clean,
            target_column=target_column,
            problem_type=resolved_problem_type,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )

        expected_classes: Optional[List[Any]] = None
        if resolved_problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            expected_classes = sorted(df_clean[target_column].unique().tolist())

        candidates: List[ChampionCandidate] = []

        # 5. Fit Baselines on TRAIN, Evaluate on VALIDATION
        if include_baselines:
            try:
                baseline_suite = BaselineEvaluator.evaluate_baselines(
                    df_train=df_train,
                    df_val=df_val,
                    target_column=target_column,
                    problem_type=resolved_problem_type,
                    random_state=random_state,
                )
                baseline_candidates = ChampionSelector.candidates_from_baseline_suite(
                    suite_result=baseline_suite,
                    problem_type=resolved_problem_type,
                    class_labels=expected_classes,
                )
                candidates.extend(baseline_candidates)
            except Exception as e:
                logger.warning(f"Baseline evaluation encountered an issue: {e}")

        # 6. Fit AutoGluon on TRAIN ONLY
        ag_model_dir = work_stage / "ag_fit_raw"
        ag_stage_dir = work_stage / "ag_ipc_work"

        ag_result: AutoGluonBackendResult = AutoGluonBackend.fit(
            df_train=df_train,
            target_column=target_column,
            problem_type=resolved_problem_type,
            eval_metric=eval_metric,
            time_limit_seconds=time_limit_seconds,
            presets=presets,
            auto_stack=auto_stack,
            output_dir=ag_model_dir,
            working_dir=ag_stage_dir,
        )

        if ag_result.status.value != "success":
            raise RuntimeError(f"AutoGluon training failed: {ag_result.error_message}")

        # 7. Evaluate AutoGluon on external VALIDATION and build candidate
        ag_candidate = ChampionSelector.candidate_from_autogluon(
            ag_result=ag_result,
            df_val=df_val,
            target_column=target_column,
            problem_type=resolved_problem_type,
            expected_class_labels=expected_classes,
        )
        candidates.append(ag_candidate)

        # 8. Champion Selection on External VALIDATION
        selection_result = ChampionSelector.select_champion(
            candidates=candidates,
            problem_type=resolved_problem_type,
            primary_metric=eval_metric,
        )

        # 9. Final Test Evaluation (Untouched TEST partition, single pass)
        test_eval_result = FinalTestEvaluator.evaluate_champion(
            champion=selection_result.champion,
            df_test=df_test,
            target_column=target_column,
            problem_type=resolved_problem_type,
            expected_class_labels=expected_classes,
            test_sha256=partition_info.test_sha256,
        )

        # 10. Persist Champion & Manifest
        feature_cols = [c for c in df_train.columns if c != target_column]
        feature_types = {c: str(df_train[c].dtype) for c in feature_cols}

        manifest_path, manifest = ChampionStore.save_champion(
            champion_selection=selection_result,
            test_evaluation=test_eval_result,
            output_dir=out_path,
            feature_names=feature_cols,
            feature_types=feature_types,
            target_column=target_column,
            problem_type=resolved_problem_type,
            partition_info=partition_info,
            random_state=random_state,
        )

        # 11. Verify UnifiedPredictor Reload & Inference
        predictor = UnifiedPredictor.load(out_path)
        sample_features = df_test.drop(columns=[target_column]).iloc[:min(5, len(df_test))]
        sample_preds = predictor.predict(sample_features)
        assert len(sample_preds) == len(sample_features), "Reloaded predictor failed sanity check."

        total_runtime = time.time() - start_time
        logger.info(
            f"[M4PipelineRunner] Completed run in {total_runtime:.2f}s. "
            f"Champion: '{manifest.model_name}' ({manifest.backend_name}) | "
            f"Val {selection_result.primary_metric}: {selection_result.champion.primary_val_score:.4f} | "
            f"Test metrics: {test_eval_result.test_metrics}"
        )

        return {
            "run_id": run_id,
            "problem_type": resolved_problem_type,
            "partition_info": partition_info,
            "champion_selection": selection_result,
            "final_test_evaluation": test_eval_result,
            "manifest": manifest,
            "manifest_path": manifest_path,
            "predictor": predictor,
            "df_train": df_train,
            "df_val": df_val,
            "df_test": df_test,
            "total_runtime_seconds": total_runtime,
        }
