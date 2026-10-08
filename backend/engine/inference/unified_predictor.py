import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib

from backend.engine.contracts.schemas import (
    ProblemType,
    PersistedChampionManifest,
)
from backend.engine.artifacts.champion_store import ChampionStore

logger = logging.getLogger(__name__)



class UnifiedPredictor:
    """
    Unified Inference Interface for reloaded Champion models.
    Supports native AutoGluon artifacts and serialized pipelines with strict schema validation.
    """

    def __init__(
        self,
        manifest: PersistedChampionManifest,
        champion_dir: Path,
        pipeline_instance: Optional[Any] = None,
    ):
        self.manifest = manifest
        self.champion_dir = Path(champion_dir).resolve()
        self.artifact_path = (self.champion_dir / manifest.artifact_rel_path).resolve()
        self._pipeline_instance = pipeline_instance

    @classmethod
    def load(cls, champion_dir: Union[str, Path]) -> "UnifiedPredictor":
        """
        Loads a persisted champion model from its directory.

        Parameters:
            champion_dir: Path to directory containing champion_manifest.json and model artifact.

        Returns:
            Instantiated UnifiedPredictor ready for inference.
        """
        c_dir = Path(champion_dir).resolve()
        manifest = ChampionStore.load_manifest(c_dir)
        art_path = (c_dir / manifest.artifact_rel_path).resolve()

        if not art_path.exists():
            raise FileNotFoundError(
                f"Model artifact referenced in manifest not found at: {art_path}"
            )

        pipeline_inst = None
        if manifest.artifact_type == "joblib_pipeline":
            pipeline_inst = joblib.load(art_path)

        logger.info(
            f"[UnifiedPredictor] Loaded champion '{manifest.model_name}' ({manifest.backend_name}) "
            f"for {manifest.problem_type.value} from {c_dir}"
        )

        return cls(
            manifest=manifest,
            champion_dir=c_dir,
            pipeline_instance=pipeline_inst,
        )

    def _validate_and_prepare_input(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Validates that required feature columns exist and aligns the DataFrame.
        """
        if not isinstance(df_input, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df_input)}.")

        if df_input.empty:
            raise ValueError("Input DataFrame is empty.")

        expected_features = self.manifest.feature_names
        missing_features = [f for f in expected_features if f not in df_input.columns]

        if missing_features:
            raise ValueError(
                f"Input DataFrame is missing required feature column(s): {missing_features}. "
                f"Expected features: {expected_features}"
            )

        # Subset and align exact feature column order
        return df_input[expected_features].copy()

    def predict(self, df_input: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for input data.

        Parameters:
            df_input: pandas DataFrame containing required feature columns.

        Returns:
            1D numpy ndarray of predictions.
        """
        df_features = self._validate_and_prepare_input(df_input)

        if self.manifest.backend_name == "autogluon":
            from backend.engine.backends.autogluon_backend import AutoGluonBackend

            preds, _ = AutoGluonBackend.predict(
                df_input=df_features,
                artifact_path=self.artifact_path,
                problem_type=self.manifest.problem_type,
                expected_class_labels=self.manifest.class_labels,
            )
            return preds

        elif self._pipeline_instance is not None:
            raw_preds = self._pipeline_instance.predict(df_features)
            return np.asarray(raw_preds)

        else:
            raise RuntimeError(
                f"Unsupported artifact backend '{self.manifest.backend_name}' or missing model instance."
            )

    def predict_proba(self, df_input: pd.DataFrame) -> np.ndarray:
        """
        Generates predicted class probabilities for classification tasks.

        Parameters:
            df_input: pandas DataFrame containing required feature columns.

        Returns:
            2D numpy ndarray of probabilities with shape (n_samples, n_classes).
        """
        if self.manifest.problem_type not in [
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ]:
            raise ValueError(
                f"predict_proba is only supported for classification tasks. Current task is: {self.manifest.problem_type.value}"
            )

        df_features = self._validate_and_prepare_input(df_input)

        if self.manifest.backend_name == "autogluon":
            from backend.engine.backends.autogluon_backend import AutoGluonBackend

            _, probs = AutoGluonBackend.predict(
                df_input=df_features,
                artifact_path=self.artifact_path,
                problem_type=self.manifest.problem_type,
                expected_class_labels=self.manifest.class_labels,
            )

            if probs is None:
                raise RuntimeError("AutoGluon model did not return predicted probabilities.")
            return probs

        elif self._pipeline_instance is not None:
            if hasattr(self._pipeline_instance, "predict_proba"):
                probs = self._pipeline_instance.predict_proba(df_features)
                return np.asarray(probs)
            else:
                raise AttributeError(
                    f"Underlying model '{type(self._pipeline_instance).__name__}' does not support predict_proba."
                )

        else:
            raise RuntimeError(
                f"Unsupported artifact backend '{self.manifest.backend_name}' or missing model instance."
            )

    @property
    def problem_type(self) -> ProblemType:
        return self.manifest.problem_type

    @property
    def target_column(self) -> str:
        return self.manifest.target_column

    @property
    def class_labels(self) -> Optional[List[Any]]:
        return self.manifest.class_labels

    @property
    def feature_names(self) -> List[str]:
        return list(self.manifest.feature_names)
