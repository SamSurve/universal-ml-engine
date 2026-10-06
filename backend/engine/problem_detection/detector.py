from typing import Optional, Dict
import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import ProblemType, ProblemDetectionResult


class ProblemDetector:
    """
    Infers whether a given tabular prediction task is:
    - Binary Classification
    - Multiclass Classification
    - Regression

    Provides confidence score, justification, and does not force arbitrary decisions when ambiguous.
    """

    MAX_MULTICLASS_UNIQUE_CARDINALITY = 20

    @classmethod
    def detect(cls, df: pd.DataFrame, target_column: str) -> ProblemDetectionResult:
        if target_column not in df.columns:
            return ProblemDetectionResult(
                problem_type=ProblemType.UNKNOWN,
                confidence=0.0,
                reason=f"Target column '{target_column}' is missing from the dataset.",
                target_name=target_column,
                unique_target_values=0,
            )

        target_series = df[target_column].dropna()
        if target_series.empty:
            return ProblemDetectionResult(
                problem_type=ProblemType.UNKNOWN,
                confidence=0.0,
                reason=f"Target column '{target_column}' contains only missing (null) values.",
                target_name=target_column,
                unique_target_values=0,
            )

        unique_vals = target_series.unique()
        num_unique = len(unique_vals)
        n_samples = len(target_series)

        # 1. Check boolean / binary
        if pd.api.types.is_bool_dtype(target_series):
            dist = {str(k): int(v) for k, v in target_series.value_counts().items()}
            return ProblemDetectionResult(
                problem_type=ProblemType.BINARY_CLASSIFICATION,
                confidence=1.0,
                reason="Target column is boolean type with 2 classes.",
                target_name=target_column,
                unique_target_values=num_unique,
                class_distribution=dist,
            )

        if num_unique == 2:
            dist = {str(k): int(v) for k, v in target_series.value_counts().items()}
            return ProblemDetectionResult(
                problem_type=ProblemType.BINARY_CLASSIFICATION,
                confidence=1.0,
                reason=f"Target column has exactly 2 distinct values ({list(dist.keys())}).",
                target_name=target_column,
                unique_target_values=2,
                class_distribution=dist,
            )

        # 2. Check string / categorical / object
        if (
            pd.api.types.is_object_dtype(target_series)
            or pd.api.types.is_string_dtype(target_series)
            or isinstance(target_series.dtype, pd.CategoricalDtype)
        ):
            if num_unique <= cls.MAX_MULTICLASS_UNIQUE_CARDINALITY:
                dist = {str(k): int(v) for k, v in target_series.value_counts().items()}
                return ProblemDetectionResult(
                    problem_type=ProblemType.MULTICLASS_CLASSIFICATION,
                    confidence=0.98,
                    reason=(
                        f"Target column is non-numeric with {num_unique} discrete classes "
                        f"(<= {cls.MAX_MULTICLASS_UNIQUE_CARDINALITY})."
                    ),
                    target_name=target_column,
                    unique_target_values=num_unique,
                    class_distribution=dist,
                )
            else:
                return ProblemDetectionResult(
                    problem_type=ProblemType.UNKNOWN,
                    confidence=0.30,
                    reason=(
                        f"Target column is non-numeric but contains {num_unique} distinct values, "
                        "which is unusually high for classification (possible raw text, ID, or high-cardinality label)."
                    ),
                    target_name=target_column,
                    unique_target_values=num_unique,
                )

        # 3. Numeric target
        if pd.api.types.is_numeric_dtype(target_series):
            # Check if values are floating point with continuous distribution
            is_float = pd.api.types.is_float_dtype(target_series)
            has_decimals = bool(np.any(target_series.to_numpy() % 1 != 0)) if is_float else False

            if has_decimals or (num_unique > cls.MAX_MULTICLASS_UNIQUE_CARDINALITY):
                return ProblemDetectionResult(
                    problem_type=ProblemType.REGRESSION,
                    confidence=0.95,
                    reason=(
                        f"Target column is numeric with {num_unique} unique continuous values "
                        f"(float: {is_float}, has fractional values: {has_decimals})."
                    ),
                    target_name=target_column,
                    unique_target_values=num_unique,
                )

            # Integer target with small number of unique values (e.g. 3 to 20)
            unique_ratio = num_unique / n_samples
            if num_unique <= 10 or (num_unique <= cls.MAX_MULTICLASS_UNIQUE_CARDINALITY and unique_ratio < 0.05):
                dist = {str(k): int(v) for k, v in target_series.value_counts().items()}
                return ProblemDetectionResult(
                    problem_type=ProblemType.MULTICLASS_CLASSIFICATION,
                    confidence=0.88,
                    reason=(
                        f"Target column is integer-encoded with {num_unique} distinct classes "
                        f"(constituting < 5% unique ratio of dataset size)."
                    ),
                    target_name=target_column,
                    unique_target_values=num_unique,
                    class_distribution=dist,
                )
            else:
                return ProblemDetectionResult(
                    problem_type=ProblemType.REGRESSION,
                    confidence=0.80,
                    reason=(
                        f"Target column is integer-valued with {num_unique} discrete levels, "
                        "representing an interval or count regression target."
                    ),
                    target_name=target_column,
                    unique_target_values=num_unique,
                )

        return ProblemDetectionResult(
            problem_type=ProblemType.UNKNOWN,
            confidence=0.0,
            reason=f"Ambiguous or unrecognized target column type '{target_series.dtype}'.",
            target_name=target_column,
            unique_target_values=num_unique,
        )
