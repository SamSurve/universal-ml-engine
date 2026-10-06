import re
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import ExclusionRecord, ProblemType


class LeakageGuard:
    """
    Guards against data leakage, pseudo-identifiers, constant features,
    and features directly correlated/co-derived with the target variable.
    """

    ID_REGEX = re.compile(r"(^id$|_id$|^id_|_id_|uuid|guid|row_number|serial_no|^index$)", re.IGNORECASE)

    @classmethod
    def audit_and_filter(
        cls,
        df: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType,
        max_cardinality_ratio: float = 0.50,
        leakage_corr_threshold: float = 0.99,
    ) -> Tuple[pd.DataFrame, List[ExclusionRecord]]:
        exclusions: List[ExclusionRecord] = []
        df_filtered = df.copy()
        n_samples = len(df_filtered)

        cols_to_drop = []

        for col in df_filtered.columns:
            if col == target_column:
                continue

            series = df_filtered[col]
            non_null = series.dropna()

            if len(non_null) == 0:
                cols_to_drop.append(col)
                exclusions.append(
                    ExclusionRecord(
                        column_name=col,
                        reason="Column contains 100% missing values.",
                        action="dropped",
                    )
                )
                continue

            unique_count = non_null.nunique()
            unique_ratio = unique_count / len(non_null)

            # 1. Constant Column Check
            if unique_count <= 1:
                cols_to_drop.append(col)
                exclusions.append(
                    ExclusionRecord(
                        column_name=col,
                        reason=f"Constant column with {unique_count} distinct value(s).",
                        action="dropped",
                        metadata={"unique_count": unique_count},
                    )
                )
                continue

            # 2. Suspicious ID Column Check
            is_id_name = bool(cls.ID_REGEX.search(col))
            is_str = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
            is_integer = pd.api.types.is_integer_dtype(series)

            is_suspicious_id = False
            if is_id_name and unique_ratio > 0.8:
                is_suspicious_id = True
            elif is_str and unique_ratio == 1.0 and unique_count > 50:
                is_suspicious_id = True
            elif is_integer and unique_ratio == 1.0 and unique_count > 50:
                # Sequential or monotonic integer IDs
                diffs = series.sort_values().diff().dropna()
                if (diffs == 1).mean() > 0.8:
                    is_suspicious_id = True

            if is_suspicious_id:
                cols_to_drop.append(col)
                exclusions.append(
                    ExclusionRecord(
                        column_name=col,
                        reason="Identified as an artificial identifier/primary key column (risk of overfitting).",
                        action="dropped",
                        metadata={"unique_ratio": round(unique_ratio, 4), "matched_name": is_id_name},
                    )
                )
                continue

            # 3. High-cardinality non-numeric categorical check
            if (
                (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series))
                and unique_count > 100
                and unique_ratio > max_cardinality_ratio
            ):
                cols_to_drop.append(col)
                exclusions.append(
                    ExclusionRecord(
                        column_name=col,
                        reason=(
                            f"High-cardinality categorical feature ({unique_count} distinct values, "
                            f"{unique_ratio*100:.1f}% unique ratio) prone to severe overfitting."
                        ),
                        action="dropped",
                        metadata={"unique_count": unique_count, "unique_ratio": round(unique_ratio, 4)},
                    )
                )
                continue

            # 4. Target Leakage (High Pearson correlation for numeric features and numeric/regression target)
            if (
                pd.api.types.is_numeric_dtype(series)
                and pd.api.types.is_numeric_dtype(df_filtered[target_column])
                and problem_type == ProblemType.REGRESSION
            ):
                try:
                    corr = float(series.corr(df_filtered[target_column]))
                    if abs(corr) >= leakage_corr_threshold:
                        cols_to_drop.append(col)
                        exclusions.append(
                            ExclusionRecord(
                                column_name=col,
                                reason=(
                                    f"Extreme correlation with target (|r| = {abs(corr):.4f} >= {leakage_corr_threshold}), "
                                    "indicating direct data leakage or duplicate target representation."
                                ),
                                action="dropped",
                                metadata={"correlation": round(corr, 4)},
                            )
                        )
                        continue
                except Exception:
                    pass

        if cols_to_drop:
            df_filtered = df_filtered.drop(columns=cols_to_drop)

        return df_filtered, exclusions
