import os
from pathlib import Path
from typing import Union, Optional, Tuple, List
import pandas as pd

from backend.engine.contracts.schemas import ExclusionRecord, ValidationResult


class DataIngestionError(Exception):
    """Raised when data ingestion fails."""
    pass


class IngestionService:
    """Handles dataset ingestion from CSV, XLSX, and XLS formats."""

    ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]

    @classmethod
    def load_dataset(
        cls, file_path_or_buffer: Union[str, Path], file_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Loads a tabular dataset from disk or buffer.

        Supported formats:
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        """
        path = Path(file_path_or_buffer) if isinstance(file_path_or_buffer, (str, Path)) else None

        if path and not path.exists():
            raise DataIngestionError(f"File not found: {path}")

        ext = path.suffix.lower() if path else (f".{file_type.lower()}" if file_type else "")

        if ext == ".csv":
            return cls._load_csv(path)
        elif ext in [".xlsx", ".xls"]:
            return cls._load_excel(path, ext)
        else:
            # Attempt CSV first, then Excel
            try:
                return cls._load_csv(path)
            except Exception:
                try:
                    return cls._load_excel(path, ext)
                except Exception as e:
                    raise DataIngestionError(
                        f"Unsupported or unparseable file format for '{path}'. "
                        f"Only CSV, XLSX, and XLS are supported. Root error: {e}"
                    )

    @classmethod
    def _load_csv(cls, path: Path) -> pd.DataFrame:
        last_error = None
        for encoding in cls.ENCODINGS_TO_TRY:
            try:
                # Use python engine with sep=None to sniff delimiters (comma, semicolon, tab, pipe)
                df = pd.read_csv(
                    path,
                    encoding=encoding,
                    sep=None,
                    engine="python",
                    on_bad_lines="skip",
                )
                if not df.empty and df.columns.size > 0:
                    return df
            except Exception as e:
                last_error = e
                continue

        # Fallback to standard comma read with error replacement
        try:
            return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
        except Exception as e:
            raise DataIngestionError(f"Failed to read CSV '{path}': {last_error or e}")

    @classmethod
    def _load_excel(cls, path: Path, ext: str) -> pd.DataFrame:
        try:
            if ext == ".xlsx":
                return pd.read_excel(path, engine="openpyxl")
            elif ext == ".xls":
                # Try xlrd first, fallback to default
                try:
                    return pd.read_excel(path, engine="xlrd")
                except Exception:
                    return pd.read_excel(path)
            else:
                return pd.read_excel(path)
        except Exception as e:
            raise DataIngestionError(f"Failed to read Excel file '{path}': {e}")


class DataValidator:
    """Validates raw tabular data and enforces strict tracking of all exclusions."""

    @classmethod
    def validate_and_clean(
        cls,
        df: pd.DataFrame,
        target_column: str,
        drop_duplicates: bool = True,
        max_missing_ratio_col: float = 0.90,
    ) -> Tuple[pd.DataFrame, ValidationResult]:
        """
        Validates the dataset against target presence, missing target rows,
        constant columns, all-null columns, and duplicates.

        Never silently drops data — all exclusions are recorded.
        """
        exclusions: List[ExclusionRecord] = []
        warnings: List[str] = []
        errors: List[str] = []

        initial_rows = len(df)
        df_clean = df.copy()

        # 1. Target column presence check
        if target_column not in df_clean.columns:
            errors.append(f"Target column '{target_column}' does not exist in dataset.")
            return df_clean, ValidationResult(
                is_valid=False,
                cleaned_rows=0,
                initial_rows=initial_rows,
                exclusions=exclusions,
                warnings=warnings,
                errors=errors,
            )

        # 2. Missing target rows
        missing_target_count = df_clean[target_column].isna().sum()
        if missing_target_count > 0:
            df_clean = df_clean.dropna(subset=[target_column])
            exclusions.append(
                ExclusionRecord(
                    column_name=target_column,
                    reason=f"Dropped {missing_target_count} rows with missing target values.",
                    action="dropped_rows",
                    metadata={"dropped_rows_count": int(missing_target_count)},
                )
            )
            warnings.append(f"Dropped {missing_target_count} rows where target was NaN.")

        if len(df_clean) == 0:
            errors.append(f"All rows in dataset had missing target values in '{target_column}'.")
            return df_clean, ValidationResult(
                is_valid=False,
                cleaned_rows=0,
                initial_rows=initial_rows,
                exclusions=exclusions,
                warnings=warnings,
                errors=errors,
            )

        # 3. Duplicate rows
        if drop_duplicates:
            dup_count = df_clean.duplicated().sum()
            if dup_count > 0:
                df_clean = df_clean.drop_duplicates()
                exclusions.append(
                    ExclusionRecord(
                        column_name="[ROWS]",
                        reason=f"Dropped {dup_count} duplicate rows.",
                        action="dropped_rows",
                        metadata={"duplicate_rows_count": int(dup_count)},
                    )
                )
                warnings.append(f"Removed {dup_count} duplicate rows.")

        # 4. Check each column (except target) for all-null, high-missingness, and constant values
        cols_to_drop = []
        for col in df_clean.columns:
            if col == target_column:
                continue

            series = df_clean[col]
            missing_ratio = series.isna().mean()
            non_null_series = series.dropna()
            unique_count = non_null_series.nunique()

            # Empty / All-null column
            if non_null_series.empty or unique_count == 0:
                cols_to_drop.append(col)
                exclusions.append(
                    ExclusionRecord(
                        column_name=col,
                        reason="Column contains 100% missing values (empty column).",
                        action="dropped_column",
                    )
                )
                warnings.append(f"Column '{col}' was excluded because it is empty.")
                continue

            # Constant column
            if unique_count == 1:
                cols_to_drop.append(col)
                val = non_null_series.iloc[0]
                exclusions.append(
                    ExclusionRecord(
                        column_name=col,
                        reason=f"Column is constant (single unique value: '{val}').",
                        action="dropped_column",
                        metadata={"constant_value": str(val)},
                    )
                )
                warnings.append(f"Column '{col}' was excluded because it contains a constant value.")
                continue

            # Excessive missingness (> 90%)
            if missing_ratio >= max_missing_ratio_col:
                cols_to_drop.append(col)
                exclusions.append(
                    ExclusionRecord(
                        column_name=col,
                        reason=f"Column contains excessive missing values ({missing_ratio*100:.1f}% missing).",
                        action="dropped_column",
                        metadata={"missing_ratio": float(missing_ratio)},
                    )
                )
                warnings.append(f"Column '{col}' was excluded due to excessive missingness ({missing_ratio*100:.1f}%).")
                continue

        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)

        # Check remaining features
        remaining_features = [c for c in df_clean.columns if c != target_column]
        if len(remaining_features) == 0:
            errors.append("No valid feature columns remain after data cleaning.")
            return df_clean, ValidationResult(
                is_valid=False,
                cleaned_rows=len(df_clean),
                initial_rows=initial_rows,
                exclusions=exclusions,
                warnings=warnings,
                errors=errors,
            )

        return df_clean, ValidationResult(
            is_valid=True,
            cleaned_rows=len(df_clean),
            initial_rows=initial_rows,
            exclusions=exclusions,
            warnings=warnings,
            errors=errors,
        )
