import pytest
import pandas as pd
import numpy as np

from backend.engine.contracts.schemas import ColumnType, ColumnProfile
from backend.engine.preprocessing.pipeline_builder import PreprocessingPipelineBuilder


def test_preprocessing_pipeline_handles_missing_and_categories():
    df = pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 4.0, 5.0],
        "cat1": ["A", "B", np.nan, "A", "C"],
        "date1": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],
    })

    profiles = {
        "num1": ColumnProfile("num1", ColumnType.NUMERIC, 1, 0.2, 4, 0.8, False),
        "cat1": ColumnProfile("cat1", ColumnType.CATEGORICAL, 1, 0.2, 3, 0.6, False),
        "date1": ColumnProfile("date1", ColumnType.DATETIME, 0, 0.0, 5, 1.0, False),
    }

    num_cols, cat_cols, date_cols = PreprocessingPipelineBuilder.identify_feature_types(df, profiles)
    assert num_cols == ["num1"]
    assert cat_cols == ["cat1"]
    assert date_cols == ["date1"]

    preprocessor = PreprocessingPipelineBuilder.build_preprocessor(
        num_cols, cat_cols, date_cols, scale_numeric=True
    )

    X_trans = preprocessor.fit_transform(df)
    assert X_trans is not None
    assert X_trans.shape[0] == 5
    # Zero NaNs after transformation
    assert not np.isnan(X_trans).any()


def test_preprocessing_handles_unseen_categories():
    df_train = pd.DataFrame({"cat": ["cat", "dog", "bird"]})
    df_test = pd.DataFrame({"cat": ["elephant", "dog", "unknown"]})

    preprocessor = PreprocessingPipelineBuilder.build_preprocessor(
        numeric_cols=[], categorical_cols=["cat"], datetime_cols=[]
    )

    X_train_trans = preprocessor.fit_transform(df_train)
    X_test_trans = preprocessor.transform(df_test)

    assert X_test_trans.shape[0] == 3
    assert X_test_trans.shape[1] == X_train_trans.shape[1]
    assert not np.isnan(X_test_trans).any()
