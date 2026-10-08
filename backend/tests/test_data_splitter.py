import pytest
import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import ProblemType, PartitionInfo
from backend.engine.training.data_splitter import DataSplitter


def test_3way_split_disjoint_indices():
    """Verify that Train, Validation, and Test partitions have zero row overlap and cover all data."""
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.choice(["cat_A", "cat_B", "cat_C"], size=n_samples),
        "target": np.random.choice(["class_0", "class_1", "class_2"], size=n_samples, p=[0.5, 0.3, 0.2]),
    })

    df_train, df_val, df_test, p_info = DataSplitter.split_train_val_test(
        df=df,
        target_column="target",
        problem_type=ProblemType.MULTICLASS_CLASSIFICATION,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        random_state=42,
    )

    assert len(df_train) == 60
    assert len(df_val) == 20
    assert len(df_test) == 20
    assert p_info.train_rows == 60
    assert p_info.val_rows == 20
    assert p_info.test_rows == 20

    # Verify pairwise disjointness
    train_set = set(p_info.train_indices)
    val_set = set(p_info.val_indices)
    test_set = set(p_info.test_indices)

    assert len(train_set & val_set) == 0, "Overlap found between Train and Validation indices"
    assert len(train_set & test_set) == 0, "Overlap found between Train and Test indices"
    assert len(val_set & test_set) == 0, "Overlap found between Validation and Test indices"
    assert len(train_set | val_set | test_set) == 100

    # Verify SHA-256 hashes
    assert len(p_info.train_sha256) == 64
    assert len(p_info.val_sha256) == 64
    assert len(p_info.test_sha256) == 64
    assert p_info.train_sha256 != p_info.val_sha256
    assert p_info.val_sha256 != p_info.test_sha256


def test_3way_split_reproducibility():
    """Verify that identical random_state generates identical partition indices and hashes."""
    df = pd.DataFrame({
        "num": np.arange(150),
        "cat": ["A", "B", "C"] * 50,
        "target": [0, 1] * 75,
    })

    _, _, _, p1 = DataSplitter.split_train_val_test(
        df, "target", ProblemType.BINARY_CLASSIFICATION, random_state=42
    )
    _, _, _, p2 = DataSplitter.split_train_val_test(
        df, "target", ProblemType.BINARY_CLASSIFICATION, random_state=42
    )
    _, _, _, p3 = DataSplitter.split_train_val_test(
        df, "target", ProblemType.BINARY_CLASSIFICATION, random_state=99
    )

    # p1 and p2 must be identical
    assert p1.train_indices == p2.train_indices
    assert p1.val_indices == p2.val_indices
    assert p1.test_indices == p2.test_indices
    assert p1.train_sha256 == p2.train_sha256
    assert p1.val_sha256 == p2.val_sha256
    assert p1.test_sha256 == p2.test_sha256

    # p3 with different seed must differ
    assert p1.train_indices != p3.train_indices


def test_3way_split_stratification_preserves_class_ratios():
    """Verify that class distributions are balanced across all three partitions."""
    np.random.seed(123)
    n = 300
    # Imbalanced classes: 70% class 0, 20% class 1, 10% class 2
    y = np.random.choice([0, 1, 2], size=n, p=[0.70, 0.20, 0.10])
    df = pd.DataFrame({"feat": np.random.randn(n), "target": y})

    df_train, df_val, df_test, p_info = DataSplitter.split_train_val_test(
        df, "target", ProblemType.MULTICLASS_CLASSIFICATION, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, random_state=42
    )

    assert p_info.stratified is True

    # Check class ratios in train, val, and test (should all be roughly 0.70, 0.20, 0.10)
    for part_name, part_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        props = part_df["target"].value_counts(normalize=True).to_dict()
        assert abs(props.get(0, 0.0) - 0.70) < 0.08, f"Class 0 ratio skewed in {part_name}"
        assert abs(props.get(1, 0.0) - 0.20) < 0.08, f"Class 1 ratio skewed in {part_name}"
        assert abs(props.get(2, 0.0) - 0.10) < 0.08, f"Class 2 ratio skewed in {part_name}"


def test_3way_split_regression():
    """Verify that regression continuous target splits without error and honors proportions."""
    n = 120
    df = pd.DataFrame({
        "x1": np.linspace(0, 10, n),
        "target": np.linspace(100, 500, n) + np.random.randn(n),
    })

    df_train, df_val, df_test, p_info = DataSplitter.split_train_val_test(
        df, "target", ProblemType.REGRESSION, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42
    )

    assert p_info.stratified is False
    assert len(df_train) == 84  # 70% of 120
    assert len(df_val) == 18    # 15% of 120
    assert len(df_test) == 18   # 15% of 120

    train_set = set(p_info.train_indices)
    val_set = set(p_info.val_indices)
    test_set = set(p_info.test_indices)
    assert len(train_set | val_set | test_set) == 120


def test_3way_split_insufficient_class_samples_fallback():
    """Verify that if a minority class has < 3 samples, split falls back gracefully to non-stratified without crash."""
    # Class 'rare' has only 2 samples
    df = pd.DataFrame({
        "feat": range(20),
        "target": ["common"] * 18 + ["rare"] * 2,
    })

    df_train, df_val, df_test, p_info = DataSplitter.split_train_val_test(
        df, "target", ProblemType.BINARY_CLASSIFICATION, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, random_state=42
    )

    assert p_info.stratified is False  # Fallback triggered
    assert len(df_train) + len(df_val) + len(df_test) == 20


def test_3way_split_invalid_ratios_and_inputs():
    """Verify that invalid split ratios or missing target column raise clear ValueErrors."""
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})

    # Ratios do not sum to 1.0
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        DataSplitter.split_train_val_test(
            df, "target", ProblemType.BINARY_CLASSIFICATION, train_ratio=0.5, val_ratio=0.2, test_ratio=0.1
        )

    # Negative ratio
    with pytest.raises(ValueError, match="strictly positive"):
        DataSplitter.split_train_val_test(
            df, "target", ProblemType.BINARY_CLASSIFICATION, train_ratio=-0.1, val_ratio=0.5, test_ratio=0.6
        )

    # Missing target column
    with pytest.raises(ValueError, match="missing from the dataset"):
        DataSplitter.split_train_val_test(
            df, "non_existent_col", ProblemType.BINARY_CLASSIFICATION
        )
