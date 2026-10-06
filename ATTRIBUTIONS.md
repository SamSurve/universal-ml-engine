# Third-Party Open-Source Software & License Attributions

**Universal ML Engine** leverages multiple permissive, open-source machine learning and numerical computation libraries. We gratefully acknowledge the authors and maintainers of these projects.

---

## Active Core Dependencies

### 1. Scikit-learn
* **Repository / Homepage:** [https://github.com/scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)
* **License:** BSD-3-Clause
* **Role in Engine:** Foundational machine learning pipeline interface (`ColumnTransformer`, `Pipeline`), cross-validation splitters (`StratifiedKFold`, `KFold`), baseline estimators (`LogisticRegression`, `Ridge`, `LinearRegression`, `RandomForestClassifier`, `RandomForestRegressor`, `ExtraTreesClassifier`, `ExtraTreesRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`), and evaluation metrics.

### 2. FLAML
* **Repository / Homepage:** [https://github.com/microsoft/FLAML](https://github.com/microsoft/FLAML)
* **License:** MIT
* **Role in Engine:** Fast Cost-Frugal Optimization (CFO) multi-model screening across LightGBM, XGBoost, CatBoost, Random Forest, and Extra Trees under strict time budgets (30–60s).

### 3. Optuna
* **Repository / Homepage:** [https://github.com/optuna/optuna](https://github.com/optuna/optuna)
* **License:** MIT
* **Role in Engine:** Bayesian Hyperparameter Optimization engine utilizing Tree-structured Parzen Estimator (TPE) sampler and `MedianPruner` for fold-level cross-validated parameter tuning.

### 4. CatBoost
* **Repository / Homepage:** [https://github.com/catboost/catboost](https://github.com/catboost/catboost)
* **License:** Apache-2.0
* **Role in Engine:** Categorical-aware gradient boosted decision tree classifier (`CatBoostClassifier`) and regressor (`CatBoostRegressor`).

### 5. LightGBM
* **Repository / Homepage:** [https://github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM)
* **License:** MIT
* **Role in Engine:** High-performance, fast tabular gradient boosting classifier (`LGBMClassifier`) and regressor (`LGBMRegressor`).

### 6. XGBoost
* **Repository / Homepage:** [https://github.com/dmlc/xgboost](https://github.com/dmlc/xgboost)
* **License:** Apache-2.0
* **Role in Engine:** Extreme gradient boosted decision tree classifier (`XGBClassifier`) and regressor (`XGBRegressor`).

### 7. Pandas
* **Repository / Homepage:** [https://github.com/pandas-dev/pandas](https://github.com/pandas-dev/pandas)
* **License:** BSD-3-Clause
* **Role in Engine:** Tabular dataset ingestion, automatic delimiter sniffing, multi-encoding fallback, statistical profiling, and dataframe manipulation.

### 8. NumPy
* **Repository / Homepage:** [https://github.com/numpy/numpy](https://github.com/numpy/numpy)
* **License:** BSD-3-Clause
* **Role in Engine:** Vectorized mathematical operations, matrix operations, and metric calibrations.

### 9. Joblib
* **Repository / Homepage:** [https://github.com/joblib/joblib](https://github.com/joblib/joblib)
* **License:** BSD-3-Clause
* **Role in Engine:** High-performance serialization and disk persistence of fitted end-to-end model pipelines.

### 10. Openpyxl & Xlrd
* **Openpyxl:** [https://foss.heptapod.net/openpyxl/openpyxl](https://foss.heptapod.net/openpyxl/openpyxl) | License: MIT
* **Xlrd:** [https://github.com/python-excel/xlrd](https://github.com/python-excel/xlrd) | License: BSD-2-Clause
* **Role in Engine:** Safe ingestion of modern `.xlsx` workbooks and legacy `.xls` spreadsheets.

### 11. Pytest
* **Repository / Homepage:** [https://github.com/pytest-dev/pytest](https://github.com/pytest-dev/pytest)
* **License:** MIT
* **Role in Engine:** Automated unit and regression test suite execution.

---

## Planned Future Integrations

### 12. SHAP
* **Repository / Homepage:** [https://github.com/shap/shap](https://github.com/shap/shap)
* **License:** MIT
* **Role in Engine:** Model-agnostic and tree-based Shapley value explainability (waterfall plots and beeswarm charts).
