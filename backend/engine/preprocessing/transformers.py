from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts informative calendar/time features (year, month, day, dayofweek, hour)
    from datetime or date-string columns.
    """

    def __init__(self, date_columns: Optional[List[str]] = None):
        self.date_columns = date_columns or []
        self.extracted_feature_names_: List[str] = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        output_features = []

        for col in X_df.columns:
            try:
                parsed_dt = pd.to_datetime(X_df[col], errors="coerce")
                year_feature = parsed_dt.dt.year.fillna(2000).to_numpy()
                month_feature = parsed_dt.dt.month.fillna(1).to_numpy()
                day_feature = parsed_dt.dt.day.fillna(1).to_numpy()
                dayofweek_feature = parsed_dt.dt.dayofweek.fillna(0).to_numpy()

                col_str = str(col)
                output_features.extend([
                    year_feature,
                    month_feature,
                    day_feature,
                    dayofweek_feature,
                ])
                if not self.extracted_feature_names_:
                    self.extracted_feature_names_.extend([
                        f"{col_str}_year",
                        f"{col_str}_month",
                        f"{col_str}_day",
                        f"{col_str}_dayofweek",
                    ])
            except Exception:
                # Fallback: fill with 0
                output_features.append(np.zeros(len(X_df)))

        if not output_features:
            return np.empty((len(X_df), 0))

        return np.column_stack(output_features)
