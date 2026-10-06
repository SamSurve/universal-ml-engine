import logging
from typing import Dict, List, Any, Optional, Type
from sklearn.base import BaseEstimator

# Core Scikit-learn estimators
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from backend.engine.contracts.schemas import ProblemType

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for classification and regression models.
    Supports Scikit-Learn, XGBoost, LightGBM, CatBoost, and FLAML.
    Safely tests and imports external gradient-boosting packages without crashing if absent.
    """

    @classmethod
    def get_models(
        cls, problem_type: ProblemType, random_state: int = 42
    ) -> Dict[str, BaseEstimator]:
        """
        Returns a dictionary of named estimator instances configured with reproducible random_state.
        Gracefully skips optional models (XGBoost, LightGBM, CatBoost) if not installed.
        """
        models: Dict[str, BaseEstimator] = {}

        if problem_type in [
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ]:
            # 1. Core Scikit-learn Classification Models
            models["Logistic Regression"] = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
            )
            models["Random Forest"] = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
            )
            models["Extra Trees"] = ExtraTreesClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
            )
            models["HistGradientBoosting"] = HistGradientBoostingClassifier(
                random_state=random_state,
            )

            # 2. XGBoost (Optional)
            try:
                import xgboost as xgb
                models["XGBoost"] = xgb.XGBClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    eval_metric="logloss" if problem_type == ProblemType.BINARY_CLASSIFICATION else "mlogloss",
                    verbosity=0,
                    n_jobs=-1,
                )
            except Exception as e:
                logger.info(f"XGBoost unavailable: {e}")

            # 3. LightGBM (Optional)
            try:
                import lightgbm as lgb
                models["LightGBM"] = lgb.LGBMClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    verbosity=-1,
                    n_jobs=-1,
                )
            except Exception as e:
                logger.info(f"LightGBM unavailable: {e}")

            # 4. CatBoost (Optional)
            try:
                import catboost as cb
                models["CatBoost"] = cb.CatBoostClassifier(
                    iterations=100,
                    random_seed=random_state,
                    verbose=False,
                    thread_count=-1,
                )
            except Exception as e:
                logger.info(f"CatBoost unavailable: {e}")

        elif problem_type == ProblemType.REGRESSION:
            # 1. Core Scikit-learn Regression Models
            models["Linear Regression"] = LinearRegression()
            models["Ridge"] = Ridge(random_state=random_state)
            models["Random Forest"] = RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
            )
            models["Extra Trees"] = ExtraTreesRegressor(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
            )
            models["HistGradientBoosting"] = HistGradientBoostingRegressor(
                random_state=random_state,
            )

            # 2. XGBoost (Optional)
            try:
                import xgboost as xgb
                models["XGBoost"] = xgb.XGBRegressor(
                    n_estimators=100,
                    random_state=random_state,
                    verbosity=0,
                    n_jobs=-1,
                )
            except Exception as e:
                logger.info(f"XGBoost unavailable: {e}")

            # 3. LightGBM (Optional)
            try:
                import lightgbm as lgb
                models["LightGBM"] = lgb.LGBMRegressor(
                    n_estimators=100,
                    random_state=random_state,
                    verbosity=-1,
                    n_jobs=-1,
                )
            except Exception as e:
                logger.info(f"LightGBM unavailable: {e}")

            # 4. CatBoost (Optional)
            try:
                import catboost as cb
                models["CatBoost"] = cb.CatBoostRegressor(
                    iterations=100,
                    random_seed=random_state,
                    verbose=False,
                    thread_count=-1,
                )
            except Exception as e:
                logger.info(f"CatBoost unavailable: {e}")

        return models

    @classmethod
    def check_environment_capabilities(cls) -> Dict[str, bool]:
        """Audits available third-party machine learning frameworks."""
        capabilities = {
            "scikit-learn": True,
            "xgboost": False,
            "lightgbm": False,
            "catboost": False,
            "flaml": False,
        }
        try:
            import xgboost
            capabilities["xgboost"] = True
        except ImportError:
            pass

        try:
            import lightgbm
            capabilities["lightgbm"] = True
        except ImportError:
            pass

        try:
            import catboost
            capabilities["catboost"] = True
        except ImportError:
            pass

        try:
            import flaml
            capabilities["flaml"] = True
        except ImportError:
            pass

        return capabilities
