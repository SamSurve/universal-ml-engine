import re
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import (
    ProblemType,
    ColumnType,
    ColumnProfile,
    QualityWarning,
    QualitySeverity,
    CorrelatedPair,
    DatasetIntelligenceResult,
)
from backend.engine.intelligence.health_score import DatasetHealthCalculator


class DatasetIntelligenceAnalyzer:
    """
    Performs comprehensive, automated data quality analysis and risk detection
    across any tabular dataset before model training.
    """

    ID_REGEX = re.compile(
        r"(^id$|_id$|^id_|_id_|uuid|guid|row_number|serial_no|^index$)",
        re.IGNORECASE,
    )

    @classmethod
    def analyze(
        cls,
        df: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType,
        column_profiles: Optional[Dict[str, ColumnProfile]] = None,
    ) -> DatasetIntelligenceResult:
        n_rows, n_cols = df.shape
        warnings: List[QualityWarning] = []

        # 1. Column Type Categorization
        numeric_columns: List[str] = []
        categorical_columns: List[str] = []
        datetime_columns: List[str] = []
        text_columns: List[str] = []

        for col in df.columns:
            if col == target_column:
                continue
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                numeric_columns.append(col)
            elif pd.api.types.is_datetime64_any_dtype(series):
                datetime_columns.append(col)
            elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                # Distinguish text from categorical
                non_null = series.dropna()
                avg_words = non_null.head(50).astype(str).apply(lambda s: len(s.split())).mean() if len(non_null) > 0 else 0
                if avg_words > 5 and (non_null.nunique() / max(1, len(non_null))) > 0.5:
                    text_columns.append(col)
                else:
                    categorical_columns.append(col)
            else:
                categorical_columns.append(col)

        # 2. Missing Value Percentages & Counts
        missing_percentages: Dict[str, float] = {}
        for col in df.columns:
            cnt = int(df[col].isna().sum())
            pct = round(cnt / max(1, n_rows), 4)
            missing_percentages[col] = pct
            if pct > 0.20:
                severity = QualitySeverity.HIGH if pct > 0.50 else QualitySeverity.MEDIUM
                warnings.append(
                    QualityWarning(
                        warning_id=f"WARN_HIGH_MISSING_{col}",
                        severity=severity,
                        affected_columns=[col],
                        reason=f"Column '{col}' has {pct*100:.1f}% missing values ({cnt}/{n_rows} rows).",
                        recommended_action=(
                            "Impute using median/mode with an added missingness indicator indicator, "
                            "or consider dropping if >80% missing."
                        ),
                    )
                )

        # 3. Duplicate Rows Count
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            dup_ratio = dup_count / max(1, n_rows)
            severity = QualitySeverity.HIGH if dup_ratio > 0.10 else QualitySeverity.MEDIUM
            warnings.append(
                QualityWarning(
                    warning_id="WARN_DUPLICATE_ROWS",
                    severity=severity,
                    affected_columns=[],
                    reason=f"Dataset contains {dup_count} duplicate row(s) ({dup_ratio*100:.2f}% of total).",
                    recommended_action="Deduplicate identical rows to prevent artificial weighting and fold leakage.",
                )
            )

        # 4. Constant and Near-Constant Columns
        constant_columns: List[str] = []
        near_constant_columns: List[str] = []

        for col in df.columns:
            if col == target_column:
                continue
            non_null = df[col].dropna()
            if len(non_null) == 0 or non_null.nunique() <= 1:
                constant_columns.append(col)
                warnings.append(
                    QualityWarning(
                        warning_id=f"WARN_CONSTANT_{col}",
                        severity=QualitySeverity.HIGH,
                        affected_columns=[col],
                        reason=f"Column '{col}' has 0 or 1 unique value across all rows (zero variance).",
                        recommended_action="Drop column; constant features provide zero predictive signal.",
                    )
                )
            else:
                top_freq = non_null.value_counts(normalize=True).iloc[0]
                if top_freq >= 0.95 and non_null.nunique() > 1:
                    near_constant_columns.append(col)
                    warnings.append(
                        QualityWarning(
                            warning_id=f"WARN_NEAR_CONSTANT_{col}",
                            severity=QualitySeverity.LOW,
                            affected_columns=[col],
                            reason=f"Column '{col}' is near-constant: single value accounts for {top_freq*100:.1f}% of non-null entries.",
                            recommended_action="Monitor feature importance; consider dropping if tree models ignore it.",
                        )
                    )

        # 5. High Cardinality Categoricals
        high_cardinality_columns: List[str] = []
        for col in categorical_columns:
            non_null = df[col].dropna()
            n_uniq = non_null.nunique()
            ratio = n_uniq / max(1, len(non_null))
            if n_uniq > 50 or (n_uniq > 20 and ratio > 0.3):
                high_cardinality_columns.append(col)
                warnings.append(
                    QualityWarning(
                        warning_id=f"WARN_HIGH_CARDINALITY_{col}",
                        severity=QualitySeverity.MEDIUM,
                        affected_columns=[col],
                        reason=f"Categorical column '{col}' has high cardinality ({n_uniq} distinct values, {ratio*100:.1f}% unique ratio).",
                        recommended_action="Apply target encoding, frequency encoding, or group infrequent categories into an 'Other' bucket.",
                    )
                )

        # 6. Suspicious ID Columns
        suspicious_id_columns: List[str] = []
        for col in df.columns:
            if col == target_column:
                continue
            series = df[col].dropna()
            if len(series) == 0:
                continue
            unique_count = series.nunique()
            unique_ratio = unique_count / len(series)

            is_id_name = bool(cls.ID_REGEX.search(col))
            is_str = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
            is_integer = pd.api.types.is_integer_dtype(series)

            is_suspicious_id = False
            if is_id_name and unique_ratio > 0.7:
                is_suspicious_id = True
            elif is_str and unique_ratio == 1.0 and unique_count > 50:
                is_suspicious_id = True
            elif is_integer and unique_ratio == 1.0 and unique_count > 50:
                diffs = series.sort_values().diff().dropna()
                if (diffs == 1).mean() > 0.8:
                    is_suspicious_id = True

            if is_suspicious_id:
                suspicious_id_columns.append(col)
                warnings.append(
                    QualityWarning(
                        warning_id=f"WARN_SUSPICIOUS_ID_{col}",
                        severity=QualitySeverity.HIGH,
                        affected_columns=[col],
                        reason=f"Column '{col}' exhibits artificial identifier characteristics (ID pattern match or 100% unique key).",
                        recommended_action="Drop column; training on artificial keys causes memorization and severe test overfitting.",
                    )
                )

        # 7. Class Imbalance (Classification)
        class_imbalance: Optional[Dict[str, Any]] = None
        if problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            target_series = df[target_column].dropna()
            counts = target_series.value_counts().to_dict()
            if len(counts) >= 2:
                sorted_counts = sorted(counts.values())
                min_count = sorted_counts[0]
                max_count = sorted_counts[-1]
                ratio = round(max_count / max(1, min_count), 2)

                if ratio >= 10.0:
                    severity_label = "severe"
                    warn_sev = QualitySeverity.HIGH
                elif ratio >= 3.0:
                    severity_label = "moderate"
                    warn_sev = QualitySeverity.MEDIUM
                else:
                    severity_label = "none"
                    warn_sev = QualitySeverity.LOW

                class_imbalance = {
                    "distribution": {str(k): int(v) for k, v in counts.items()},
                    "imbalance_ratio": ratio,
                    "severity": severity_label,
                    "minority_class": str(min(counts, key=counts.get)),
                    "majority_class": str(max(counts, key=counts.get)),
                }

                if severity_label in ("moderate", "severe"):
                    warnings.append(
                        QualityWarning(
                            warning_id="WARN_CLASS_IMBALANCE",
                            severity=warn_sev,
                            affected_columns=[target_column],
                            reason=f"Target column exhibits {severity_label} class imbalance (ratio {ratio:.1f}:1 between majority and minority classes).",
                            recommended_action=(
                                "Use balanced metrics (Macro F1, Balanced Accuracy, ROC-AUC), "
                                "stratified cross-validation, and class-weighted estimators."
                            ),
                        )
                    )

        # 8. Target Leakage Candidates
        target_leakage_risks: List[str] = []
        if pd.api.types.is_numeric_dtype(df[target_column]):
            target_numeric = df[target_column].dropna()
            for col in numeric_columns:
                try:
                    corr = float(df[col].corr(df[target_column]))
                    if abs(corr) >= 0.95:
                        target_leakage_risks.append(col)
                        warnings.append(
                            QualityWarning(
                                warning_id=f"WARN_TARGET_LEAKAGE_{col}",
                                severity=QualitySeverity.CRITICAL,
                                affected_columns=[col],
                                reason=f"Feature '{col}' has extreme linear correlation with target (|r| = {abs(corr):.4f} >= 0.95).",
                                recommended_action="Drop feature; extreme correlation suggests target leakage, duplicate column, or post-event data.",
                            )
                        )
                except Exception:
                    pass

        # 9. Highly Correlated Feature Pairs (|r| >= 0.85)
        correlated_numeric_pairs: List[CorrelatedPair] = []
        if len(numeric_columns) >= 2:
            df_num = df[numeric_columns].dropna()
            if len(df_num) > 10:
                corr_matrix = df_num.corr().abs()
                visited = set()
                for i, col1 in enumerate(numeric_columns):
                    for j, col2 in enumerate(numeric_columns):
                        if i < j and (col1, col2) not in visited:
                            visited.add((col1, col2))
                            val = float(corr_matrix.loc[col1, col2])
                            if val >= 0.85:
                                correlated_numeric_pairs.append(
                                    CorrelatedPair(feature_a=col1, feature_b=col2, correlation=round(val, 4))
                                )
                                warnings.append(
                                    QualityWarning(
                                        warning_id=f"WARN_MULTICOLLINEARITY_{col1}_{col2}",
                                        severity=QualitySeverity.LOW,
                                        affected_columns=[col1, col2],
                                        reason=f"Features '{col1}' and '{col2}' are highly correlated (|r| = {val:.4f} >= 0.85).",
                                        recommended_action="Tree models handle collinearity well; linear models may benefit from Ridge (L2) regularization.",
                                    )
                                )

        # 10. Columns Requiring Special Handling
        special_handling_columns: Dict[str, str] = {}
        for col, pct in missing_percentages.items():
            if pct > 0.20 and col != target_column:
                special_handling_columns[col] = f"High missingness ({pct*100:.1f}% nulls): Requires robust imputation"

        for col in high_cardinality_columns:
            special_handling_columns[col] = "High-cardinality categorical: Requires frequency/target encoding or top-K pruning"

        for col in numeric_columns:
            non_null = df[col].dropna()
            if len(non_null) > 5:
                skew = float(non_null.skew())
                if abs(skew) > 3.0:
                    special_handling_columns[col] = f"High skewness ({skew:.2f}): Consider log/power transformation"

        # 11. Calculate Dataset Health Score
        health_score = DatasetHealthCalculator.calculate(
            total_rows=n_rows,
            total_cols=n_cols,
            duplicate_rows_count=dup_count,
            missing_percentages=missing_percentages,
            constant_columns=constant_columns,
            near_constant_columns=near_constant_columns,
            high_cardinality_columns=high_cardinality_columns,
            suspicious_id_columns=suspicious_id_columns,
            target_leakage_risks=target_leakage_risks,
            correlated_pairs_count=len(correlated_numeric_pairs),
            class_imbalance=class_imbalance,
            target_column=target_column,
            warnings=warnings,
        )

        return DatasetIntelligenceResult(
            dataset_shape=(n_rows, n_cols),
            target_column=target_column,
            problem_type=problem_type,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            text_columns=text_columns,
            missing_percentages=missing_percentages,
            duplicate_rows_count=dup_count,
            constant_columns=constant_columns,
            near_constant_columns=near_constant_columns,
            high_cardinality_columns=high_cardinality_columns,
            class_imbalance=class_imbalance,
            suspicious_id_columns=suspicious_id_columns,
            target_leakage_risks=target_leakage_risks,
            correlated_numeric_pairs=correlated_numeric_pairs,
            special_handling_columns=special_handling_columns,
            warnings=warnings,
            health_score=health_score,
        )
