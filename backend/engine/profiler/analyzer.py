import re
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import (
    ColumnType,
    ColumnProfile,
    DatasetProfile,
)


class DatasetProfiler:
    """Computes comprehensive statistical profiles and infers semantic data types."""

    @classmethod
    def infer_column_type(cls, series: pd.Series, col_name: str) -> ColumnType:
        """Infers the semantic type of a single column."""
        non_null = series.dropna()
        if non_null.empty:
            return ColumnType.UNKNOWN

        unique_count = non_null.nunique()
        total_count = len(non_null)

        if unique_count <= 1:
            return ColumnType.ID_OR_CONSTANT

        # Check datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnType.DATETIME

        # Check if object/string column can be parsed as datetime
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            sample = non_null.head(50).astype(str)
            # Heuristic regex check for common date patterns (YYYY-MM-DD, DD/MM/YYYY, etc.)
            date_pattern = re.compile(
                r"^(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})"
            )
            matches = sum(1 for v in sample if date_pattern.match(v.strip()))
            if matches / len(sample) > 0.7:
                try:
                    pd.to_datetime(sample, errors="raise")
                    return ColumnType.DATETIME
                except Exception:
                    pass

        # Check numeric
        if pd.api.types.is_numeric_dtype(series):
            # Check if this numeric column is an index/ID
            is_id_name = bool(re.search(r"(^id$|_id$|^id_|_id_|uuid|guid|^index$)", col_name.lower()))
            if is_id_name and unique_count == total_count:
                return ColumnType.ID_OR_CONSTANT
            return ColumnType.NUMERIC

        # String / Object / Categorical
        if isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(series):
            return ColumnType.CATEGORICAL

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # Check for high cardinality text vs categorical
            avg_word_count = non_null.head(100).astype(str).apply(lambda x: len(x.split())).mean()
            if avg_word_count > 5 and unique_count / total_count > 0.5:
                return ColumnType.TEXT

            is_id_name = bool(re.search(r"(^id$|_id$|id_|index|uuid|guid)", col_name.lower()))
            if is_id_name and unique_count == total_count:
                return ColumnType.ID_OR_CONSTANT

            return ColumnType.CATEGORICAL

        return ColumnType.UNKNOWN

    @classmethod
    def profile_column(
        cls, series: pd.Series, col_name: str, is_target: bool = False
    ) -> ColumnProfile:
        total = len(series)
        missing_count = int(series.isna().sum())
        missing_pct = float(missing_count / total) if total > 0 else 0.0

        non_null = series.dropna()
        unique_count = int(non_null.nunique())
        unique_ratio = float(unique_count / len(non_null)) if len(non_null) > 0 else 0.0
        is_constant = unique_count <= 1

        detected_type = cls.infer_column_type(series, col_name)

        sample_vals = [
            v.item() if hasattr(v, "item") else v
            for v in non_null.head(5).tolist()
        ]

        stats: Dict[str, Any] = {}
        if pd.api.types.is_numeric_dtype(series) and not non_null.empty:
            stats = {
                "min": float(non_null.min()),
                "max": float(non_null.max()),
                "mean": float(non_null.mean()),
                "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
                "median": float(non_null.median()),
                "skewness": float(non_null.skew()) if len(non_null) > 2 else 0.0,
            }
        elif detected_type == ColumnType.CATEGORICAL and not non_null.empty:
            value_counts = non_null.value_counts().head(5).to_dict()
            stats = {
                "top_categories": {str(k): int(v) for k, v in value_counts.items()},
                "num_unique": unique_count,
            }

        return ColumnProfile(
            name=col_name,
            detected_type=detected_type,
            missing_count=missing_count,
            missing_percentage=round(missing_pct, 4),
            unique_count=unique_count,
            unique_ratio=round(unique_ratio, 4),
            is_constant=is_constant,
            is_target=is_target,
            sample_values=sample_vals,
            stats=stats,
        )

    @classmethod
    def profile_dataset(cls, df: pd.DataFrame, target_column: Optional[str] = None) -> DatasetProfile:
        profiles: Dict[str, ColumnProfile] = {}
        for col in df.columns:
            profiles[col] = cls.profile_column(
                df[col], col_name=col, is_target=(col == target_column)
            )

        row_count = len(df)
        col_count = len(df.columns)
        dup_count = int(df.duplicated().sum())
        missing_cells = int(df.isna().sum().sum())
        mem_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))

        return DatasetProfile(
            row_count=row_count,
            column_count=col_count,
            column_profiles=profiles,
            duplicate_rows_count=dup_count,
            missing_cells_count=missing_cells,
            memory_usage_mb=round(mem_mb, 2),
        )
