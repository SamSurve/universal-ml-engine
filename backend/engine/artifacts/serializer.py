import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from backend.engine.contracts.schemas import ProblemType


class ModelArtifact:
    """Wrapper that holds the refitted end-to-end Pipeline along with target encoding."""

    def __init__(
        self,
        pipeline: Pipeline,
        problem_type: ProblemType,
        target_column: str,
        label_encoder: Optional[LabelEncoder] = None,
        feature_names: Optional[list] = None,
    ):
        self.pipeline = pipeline
        self.problem_type = problem_type
        self.target_column = target_column
        self.label_encoder = label_encoder
        self.feature_names = feature_names or []

    def predict(self, X: pd.DataFrame):
        """Generates real-time predictions on unseen tabular data."""
        preds = self.pipeline.predict(X)
        if self.label_encoder is not None:
            try:
                return self.label_encoder.inverse_transform(preds)
            except Exception as e:
                raise ValueError(f"Target decoding error: Unable to map predicted classes to original labels: {e}")
        return preds

    def predict_proba(self, X: pd.DataFrame):
        """Generates class probabilities if model supports predict_proba."""
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X)
        raise AttributeError(f"Underlying model '{self.pipeline.named_steps['model']}' does not support predict_proba.")


class ArtifactManager:
    """Handles serialization, versioning, metadata saving, and reloading of model pipelines."""

    @classmethod
    def save_pipeline(
        cls,
        artifact: ModelArtifact,
        metadata: Dict[str, Any],
        output_dir: str = "artifacts",
        model_filename: str = "model.joblib",
        metadata_filename: str = "metadata.json",
    ) -> Tuple[str, str]:
        dir_path = Path(output_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        model_path = dir_path / model_filename
        meta_path = dir_path / metadata_filename

        # Save pipeline via joblib
        joblib.dump(artifact, model_path)

        # Save metadata via JSON
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        return str(model_path.resolve()), str(meta_path.resolve())

    @classmethod
    def load_pipeline(cls, model_path: str) -> ModelArtifact:
        """Reloads a saved ModelArtifact from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {path}")

        artifact = joblib.load(path)
        if not isinstance(artifact, ModelArtifact):
            raise TypeError(f"Loaded object is of type {type(artifact)}, expected ModelArtifact.")

        return artifact
