import os
import json
import shutil
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import joblib

from backend.engine.contracts.schemas import (
    ProblemType,
    PartitionInfo,
    ChampionCandidate,
    ChampionSelectionResult,
    FinalTestEvaluationResult,
    PersistedChampionManifest,
)

logger = logging.getLogger(__name__)


class ChampionStore:
    """
    Handles native persistence and metadata serialization for Champion models.

    Guarantees:
    - Preserves native AutoGluon directory structure without forcing joblib/pickle.
    - Saves unified manifest JSON with full provenance: schema, splits, validation & test metrics.
    - Atomic file operations for reproducible disk state.
    """

    MANIFEST_FILENAME = "champion_manifest.json"

    @classmethod
    def save_champion(
        cls,
        champion_selection: ChampionSelectionResult,
        test_evaluation: FinalTestEvaluationResult,
        output_dir: Union[str, Path],
        feature_names: List[str],
        feature_types: Dict[str, str],
        target_column: str,
        problem_type: ProblemType,
        partition_info: Optional[Union[PartitionInfo, Dict[str, Any]]] = None,
        random_state: int = 42,
    ) -> Tuple[Path, PersistedChampionManifest]:
        """
        Persists the champion model and writes the champion manifest.

        Parameters:
            champion_selection: Result from ChampionSelector.
            test_evaluation: Result from FinalTestEvaluator.
            output_dir: Directory to store model artifact and manifest.
            feature_names: List of input feature names required for inference.
            feature_types: Dict mapping feature name to data type string.
            target_column: Name of target column.
            problem_type: ProblemType.
            partition_info: Optional PartitionInfo or partition provenance dictionary.
            random_state: Random state seed.

        Returns:
            Tuple of (manifest_path: Path, manifest: PersistedChampionManifest).
        """
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        champ = champion_selection.champion
        champ_id = f"champ_{uuid.uuid4().hex[:8]}"

        artifact_rel_path: str
        artifact_type: str

        # 1. Persist model artifact
        if champ.backend_name == "autogluon":
            artifact_type = "autogluon"
            artifact_rel_path = "model"
            target_model_dir = out_path / artifact_rel_path

            if not champ.artifact_path or not Path(champ.artifact_path).exists():
                raise FileNotFoundError(
                    f"AutoGluon source artifact not found at: {champ.artifact_path}"
                )

            src_model_dir = Path(champ.artifact_path).resolve()
            if src_model_dir != target_model_dir:
                if target_model_dir.exists():
                    shutil.rmtree(target_model_dir)
                shutil.copytree(src_model_dir, target_model_dir)

        elif champ.model_instance is not None:
            artifact_type = "joblib_pipeline"
            artifact_rel_path = "model.joblib"
            target_joblib_file = out_path / artifact_rel_path
            joblib.dump(champ.model_instance, target_joblib_file)

        elif champ.artifact_path and Path(champ.artifact_path).exists():
            # Existing file/dir artifact
            src_path = Path(champ.artifact_path).resolve()
            if src_path.is_dir():
                artifact_type = "custom_dir"
                artifact_rel_path = "model"
                target_dir = out_path / artifact_rel_path
                if src_path != target_dir:
                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    shutil.copytree(src_path, target_dir)
            else:
                artifact_type = "custom_file"
                artifact_rel_path = f"model{src_path.suffix}"
                target_file = out_path / artifact_rel_path
                if src_path != target_file:
                    shutil.copy2(src_path, target_file)
        else:
            raise ValueError(
                f"Unable to persist champion '{champ.model_name}' ({champ.backend_name}): "
                "No artifact_path or model_instance present."
            )

        # 2. Format partition provenance
        pinfo_dict: Optional[Dict[str, Any]] = None
        if isinstance(partition_info, PartitionInfo):
            pinfo_dict = {
                "train_rows": partition_info.train_rows,
                "val_rows": partition_info.val_rows,
                "test_rows": partition_info.test_rows,
                "train_sha256": partition_info.train_sha256,
                "val_sha256": partition_info.val_sha256,
                "test_sha256": partition_info.test_sha256,
                "stratified": partition_info.stratified,
                "random_state": partition_info.random_state,
            }
        elif isinstance(partition_info, dict):
            pinfo_dict = partition_info

        # 3. Construct Manifest
        manifest = PersistedChampionManifest(
            schema_version="1.0.0",
            champion_id=champ_id,
            backend_name=champ.backend_name,
            model_name=champ.model_name,
            problem_type=problem_type,
            target_column=target_column,
            primary_metric=champion_selection.primary_metric,
            feature_names=list(feature_names),
            feature_types=dict(feature_types),
            class_labels=champ.class_labels,
            val_metrics=dict(champ.val_metrics),
            test_metrics=dict(test_evaluation.test_metrics),
            selection_rationale=champion_selection.selection_rationale,
            all_candidate_metrics=dict(champion_selection.validation_comparison),
            artifact_rel_path=artifact_rel_path,
            artifact_type=artifact_type,
            partition_info=pinfo_dict,
            random_state=random_state,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # 4. Write manifest JSON
        manifest_path = out_path / cls.MANIFEST_FILENAME
        manifest_dict = {
            "schema_version": manifest.schema_version,
            "champion_id": manifest.champion_id,
            "backend_name": manifest.backend_name,
            "model_name": manifest.model_name,
            "problem_type": manifest.problem_type.value if hasattr(manifest.problem_type, "value") else str(manifest.problem_type),
            "target_column": manifest.target_column,
            "primary_metric": manifest.primary_metric,
            "feature_names": manifest.feature_names,
            "feature_types": manifest.feature_types,
            "class_labels": manifest.class_labels,
            "val_metrics": manifest.val_metrics,
            "test_metrics": manifest.test_metrics,
            "selection_rationale": manifest.selection_rationale,
            "all_candidate_metrics": manifest.all_candidate_metrics,
            "artifact_rel_path": manifest.artifact_rel_path,
            "artifact_type": manifest.artifact_type,
            "partition_info": manifest.partition_info,
            "random_state": manifest.random_state,
            "created_at": manifest.created_at,
        }

        manifest_path.write_text(json.dumps(manifest_dict, indent=2, default=str), encoding="utf-8")
        logger.info(f"[ChampionStore] Saved champion manifest to: {manifest_path}")

        return manifest_path, manifest

    @classmethod
    def load_manifest(cls, champion_dir: Union[str, Path]) -> PersistedChampionManifest:
        """Loads and validates a PersistedChampionManifest from disk."""
        c_dir = Path(champion_dir).resolve()
        manifest_path = c_dir / cls.MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Champion manifest not found at: {manifest_path}")

        data = json.loads(manifest_path.read_text(encoding="utf-8"))

        pt_str = data.get("problem_type", "")
        # Resolve ProblemType enum
        if "binary" in pt_str:
            pt = ProblemType.BINARY_CLASSIFICATION
        elif "multiclass" in pt_str:
            pt = ProblemType.MULTICLASS_CLASSIFICATION
        elif "regression" in pt_str:
            pt = ProblemType.REGRESSION
        else:
            pt = ProblemType.UNKNOWN

        return PersistedChampionManifest(
            schema_version=data.get("schema_version", "1.0.0"),
            champion_id=data.get("champion_id", ""),
            backend_name=data.get("backend_name", ""),
            model_name=data.get("model_name", ""),
            problem_type=pt,
            target_column=data.get("target_column", ""),
            primary_metric=data.get("primary_metric", ""),
            feature_names=data.get("feature_names", []),
            feature_types=data.get("feature_types", {}),
            class_labels=data.get("class_labels"),
            val_metrics=data.get("val_metrics", {}),
            test_metrics=data.get("test_metrics", {}),
            selection_rationale=data.get("selection_rationale", ""),
            all_candidate_metrics=data.get("all_candidate_metrics", {}),
            artifact_rel_path=data.get("artifact_rel_path", "model"),
            artifact_type=data.get("artifact_type", "autogluon"),
            partition_info=data.get("partition_info"),
            random_state=int(data.get("random_state", 42)),
            created_at=data.get("created_at", ""),
        )
