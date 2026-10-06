from typing import List, Dict, Tuple, Optional
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler

from backend.engine.contracts.schemas import ColumnType, ColumnProfile
from backend.engine.preprocessing.transformers import DatetimeFeatureExtractor


class PreprocessingPipelineBuilder:
    """
    Constructs leakage-safe Scikit-Learn ColumnTransformer and Pipeline objects.

    Guarantees:
    - Zero data leakage: Transformer is fitted solely on training folds.
    - Numeric features: Median imputation + optional StandardScaler.
    - Categorical features: Most frequent imputation + OneHotEncoder(handle_unknown='ignore').
    - Datetime features: Calendar decomposition via DatetimeFeatureExtractor.
    """

    @classmethod
    def identify_feature_types(
        cls, df_features: pd.DataFrame, column_profiles: Dict[str, ColumnProfile]
    ) -> Tuple[List[str], List[str], List[str]]:
        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        datetime_cols: List[str] = []

        for col in df_features.columns:
            profile = column_profiles.get(col)
            if profile and profile.detected_type == ColumnType.DATETIME:
                datetime_cols.append(col)
            elif profile and profile.detected_type == ColumnType.CATEGORICAL:
                categorical_cols.append(col)
            elif profile and profile.detected_type == ColumnType.NUMERIC:
                numeric_cols.append(col)
            else:
                # Heuristic fallback
                if pd.api.types.is_numeric_dtype(df_features[col]):
                    numeric_cols.append(col)
                elif pd.api.types.is_datetime64_any_dtype(df_features[col]):
                    datetime_cols.append(col)
                else:
                    categorical_cols.append(col)

        return numeric_cols, categorical_cols, datetime_cols

    @classmethod
    def build_preprocessor(
        cls,
        numeric_cols: List[str],
        categorical_cols: List[str],
        datetime_cols: List[str],
        scale_numeric: bool = True,
        robust_scaling: bool = False,
    ) -> ColumnTransformer:
        transformers = []

        # 1. Numeric pipeline
        if numeric_cols:
            num_steps = [("imputer", SimpleImputer(strategy="median"))]
            if scale_numeric:
                scaler = RobustScaler() if robust_scaling else StandardScaler()
                num_steps.append(("scaler", scaler))
            transformers.append(("numeric", Pipeline(num_steps), numeric_cols))

        # 2. Categorical pipeline
        if categorical_cols:
            cat_steps = [
                ("imputer", SimpleImputer(strategy="most_frequent", fill_value="missing")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
            transformers.append(("categorical", Pipeline(cat_steps), categorical_cols))

        # 3. Datetime pipeline
        if datetime_cols:
            date_steps = [
                ("extractor", DatetimeFeatureExtractor(date_columns=datetime_cols)),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
            transformers.append(("datetime", Pipeline(date_steps), datetime_cols))

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )
