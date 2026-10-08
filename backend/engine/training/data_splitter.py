import hashlib
import logging
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.engine.contracts.schemas import ProblemType, PartitionInfo

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Executes leakage-safe, reproducible 3-way Train / Validation / Test data partitioning.

    Key Guarantees:
    - Train (60-70%): Passed to AutoML backends / baseline fitting.
    - Validation (15-20%): Kept independent for fair champion selection and baseline benchmarking.
    - Test (15-20%): Untouched until final single-model governance evaluation.
    - Stratified splitting for classification tasks when minimum class count >= 3.
    - Deterministic and reproducible with fixed random_state.
    - Zero row overlap: Pairwise disjoint partition index sets.
    - Cryptographic SHA-256 partition hashing for full data provenance.
    """

    @classmethod
    def _compute_dataframe_sha256(cls, df: pd.DataFrame) -> str:
        """Computes a deterministic SHA-256 hash representing the DataFrame content."""
        # Use pandas hash_pandas_object for fast, deterministic column & value hashing
        hashed_series = pd.util.hash_pandas_object(df, index=False)
        return hashlib.sha256(hashed_series.to_numpy().tobytes()).hexdigest()

    @classmethod
    def split_train_val_test(
        cls,
        df: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PartitionInfo]:
        """
        Splits a DataFrame into 3 disjoint subsets: Train, Validation, and Test.

        Parameters:
            df: Cleaned input DataFrame.
            target_column: Target column name.
            problem_type: Binary, Multiclass, or Regression task.
            train_ratio: Proportion for train split (default: 0.60).
            val_ratio: Proportion for validation split (default: 0.20).
            test_ratio: Proportion for test split (default: 0.20).
            random_state: Random seed for reproducibility.

        Returns:
            Tuple of (df_train, df_val, df_test, partition_info).
        """
        if df.empty:
            raise ValueError("Cannot split an empty DataFrame.")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' is missing from the dataset.")

        if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
            raise ValueError(
                f"Split ratios must be strictly positive. Got train={train_ratio}, val={val_ratio}, test={test_ratio}"
            )

        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-4):
            raise ValueError(
                f"Split ratios must sum to 1.0. Got train={train_ratio} + val={val_ratio} + test={test_ratio} = {total_ratio:.4f}"
            )

        n_samples = len(df)
        if n_samples < 3:
            raise ValueError(f"Dataset must have at least 3 samples for a 3-way split. Found {n_samples} rows.")

        # Ensure index is aligned for tracking original row IDs
        df_work = df.copy().reset_index(drop=True)
        original_indices = np.arange(n_samples)

        is_classification = problem_type in [
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ]

        # Determine stratification viability
        stratified = False
        target_series = df_work[target_column]

        if is_classification:
            class_counts = target_series.value_counts()
            min_class_count = int(class_counts.min()) if not class_counts.empty else 0

            # For a 3-way stratified split, each class needs at least 3 samples (1 per partition)
            if min_class_count >= 3:
                stratified = True
            else:
                logger.warning(
                    f"Classification target '{target_column}' has minority class count of {min_class_count} (< 3). "
                    "Falling back to non-stratified random split to avoid splitting error."
                )

        # Stage 1: Split into Train and (Val + Test)
        val_test_ratio = val_ratio + test_ratio
        stratify_stage1 = target_series if stratified else None

        train_idx, val_test_idx = train_test_split(
            original_indices,
            test_size=val_test_ratio,
            random_state=random_state,
            stratify=stratify_stage1,
        )

        # Stage 2: Split (Val + Test) into Val and Test
        # Relative proportion of val in (val + test)
        relative_val_ratio = val_ratio / val_test_ratio
        val_test_target = target_series.iloc[val_test_idx]

        stratify_stage2 = None
        if stratified:
            # Verify sub-sample has at least 2 samples per class for second stage
            sub_counts = val_test_target.value_counts()
            if int(sub_counts.min()) >= 2:
                stratify_stage2 = val_test_target
            else:
                stratify_stage2 = None

        val_rel_idx, test_rel_idx = train_test_split(
            np.arange(len(val_test_idx)),
            test_size=(1.0 - relative_val_ratio),
            random_state=random_state,
            stratify=stratify_stage2,
        )

        val_idx = val_test_idx[val_rel_idx]
        test_idx = val_test_idx[test_rel_idx]

        # --- Strict Index Integrity Verification ---
        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)

        if train_set & val_set:
            raise RuntimeError("Data split error: Overlap detected between Train and Validation sets.")
        if train_set & test_set:
            raise RuntimeError("Data split error: Overlap detected between Train and Test sets.")
        if val_set & test_set:
            raise RuntimeError("Data split error: Overlap detected between Validation and Test sets.")
        if len(train_set | val_set | test_set) != n_samples:
            raise RuntimeError("Data split error: Total partitioned rows do not match original sample count.")

        # Slice DataFrames
        df_train = df_work.iloc[train_idx].copy().reset_index(drop=True)
        df_val = df_work.iloc[val_idx].copy().reset_index(drop=True)
        df_test = df_work.iloc[test_idx].copy().reset_index(drop=True)

        # Compute Hashes
        train_sha256 = cls._compute_dataframe_sha256(df_train)
        val_sha256 = cls._compute_dataframe_sha256(df_val)
        test_sha256 = cls._compute_dataframe_sha256(df_test)

        partition_info = PartitionInfo(
            train_rows=len(df_train),
            val_rows=len(df_val),
            test_rows=len(df_test),
            train_indices=train_idx.tolist(),
            val_indices=val_idx.tolist(),
            test_indices=test_idx.tolist(),
            train_sha256=train_sha256,
            val_sha256=val_sha256,
            test_sha256=test_sha256,
            stratified=stratified,
            random_state=random_state,
            target_column=target_column,
            problem_type=problem_type,
        )

        logger.info(
            f"3-Way Split complete: Train={len(df_train)} rows ({train_ratio*100:.1f}%), "
            f"Val={len(df_val)} rows ({val_ratio*100:.1f}%), "
            f"Test={len(df_test)} rows ({test_ratio*100:.1f}%), "
            f"Stratified={stratified}"
        )

        return df_train, df_val, df_test, partition_info
