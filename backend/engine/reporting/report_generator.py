import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd

from backend.engine.contracts.schemas import (
    ExperimentResult,
    DatasetIntelligenceResult,
    ExplainabilityResult,
    ModelDecisionSummary,
)


class ReportGenerator:
    """
    Generates professional, comprehensive engineering reports:
    - MODEL_REPORT.md (End-to-End Governance & Performance Report)
    - DATASET_INTELLIGENCE_REPORT.md (Data Quality & Risk Audit Report)
    """

    @classmethod
    def generate_model_report(
        cls, result: ExperimentResult, output_path: Union[str, Path]
    ) -> Path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = []
        lines.append(f"# Universal ML Engine — Model Governance & Decision Report")
        lines.append("")
        lines.append(f"> **Dataset:** `{result.dataset_name}`  ")
        lines.append(f"> **Target Variable:** `{result.target_column}`  ")
        lines.append(f"> **Experiment ID:** `{result.experiment_id}`  ")
        lines.append(f"> **Champion Model:** `{result.best_model_name}`  ")
        lines.append(f"> **Generated Date:** October 6, 2026  ")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 1. Executive Summary
        lines.append("## 1. Executive Summary")
        lines.append("")
        lines.append(
            f"The Universal ML Engine automated pipeline processed `{result.dataset_name}` "
            f"({result.dev_shape[0] + result.holdout_shape[0]:,} rows, {result.dev_shape[1]:,} features). "
            f"The task was identified as **{result.problem_detection.problem_type.value.replace('_', ' ').title()}** "
            f"with {result.problem_detection.confidence*100:.1f}% confidence."
        )
        lines.append("")
        if result.decision_summary:
            lines.append(f"**Champion Selection:** {result.decision_summary.selection_rationale}")
        lines.append("")
        lines.append(
            f"* **Development 5-Fold CV Score:** `{result.best_model_cv_score:.4f}` ({result.problem_detection.problem_type.value})  \n"
            f"* **Untouched 20% Holdout Performance:** "
            + ", ".join(f"`{k}`: **{v:.4f}**" for k, v in result.holdout_metrics.items())
            + f"  \n* **Total End-to-End Execution Time:** `{result.total_runtime_seconds:.2f}s`"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

        # 2. Dataset Overview
        lines.append("## 2. Dataset Overview & Partitioning")
        lines.append("")
        lines.append(f"| Dimension | Count / Proportion | Details |")
        lines.append(f"| :--- | :--- | :--- |")
        lines.append(f"| **Raw Samples** | `{result.validation_result.initial_rows:,}` rows | Initial dataset size |")
        lines.append(f"| **Cleaned Samples** | `{result.validation_result.cleaned_rows:,}` rows | After removing missing targets / duplicate records |")
        lines.append(f"| **Development Split (80%)** | `{result.dev_shape[0]:,}` rows | Solely used for screening, CV, and tuning |")
        lines.append(f"| **Final Holdout Split (20%)** | `{result.holdout_shape[0]:,}` rows | Untouched benchmark partition |")
        lines.append(f"| **Total Features Retained** | `{result.dev_shape[1] - 1:,}` features | Cleaned predictive features |")
        lines.append(f"| **Exclusions Logged** | `{len(result.validation_result.exclusions):,}` items | Audit trail of dropped columns/rows |")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 3. Dataset Intelligence & Health Score
        intel = result.dataset_intelligence
        if intel:
            lines.append("## 3. Dataset Intelligence & Health Diagnostic")
            lines.append("")
            hs = intel.health_score
            lines.append(f"### Health Score: **{hs.overall_score}/100** — Grade: `{hs.grade}`")
            lines.append(f"> *{hs.interpretation}*")
            lines.append("")
            lines.append("#### Quality Category Breakdown")
            lines.append("| Category | Score | Weight | Assessment |")
            lines.append("| :--- | :--- | :--- | :--- |")
            for cat_name, cat_obj in hs.categories.items():
                deduct_str = "; ".join(cat_obj.deductions) if cat_obj.deductions else "No deductions"
                lines.append(f"| **{cat_name}** | `{cat_obj.score}/100` | `{int(cat_obj.weight*100)}%` | {deduct_str} |")
            lines.append("")

            if hs.clean_signals:
                lines.append("#### Verified Clean Signals")
                for sig in hs.clean_signals:
                    lines.append(f"- [x] {sig}")
                lines.append("")

            # 4. Data Quality Warnings
            lines.append("## 4. Data Quality Warnings & Risks")
            lines.append("")
            if intel.warnings:
                lines.append("| Warning ID | Severity | Affected Column(s) | Reason | Recommended Action |")
                lines.append("| :--- | :--- | :--- | :--- | :--- |")
                for w in intel.warnings:
                    cols = ", ".join(w.affected_columns) if w.affected_columns else "Dataset Level"
                    lines.append(f"| `{w.warning_id}` | **{w.severity.value.upper()}** | `{cols}` | {w.reason} | {w.recommended_action} |")
                lines.append("")
            else:
                lines.append("No data quality warnings recorded. Dataset meets all health thresholds.")
                lines.append("")
            lines.append("---")
            lines.append("")

        # 5. Problem Detection
        pd_res = result.problem_detection
        lines.append("## 5. Problem Detection")
        lines.append("")
        lines.append(f"* **Detected Problem Type:** `{pd_res.problem_type.value}`")
        lines.append(f"* **Detection Confidence:** `{pd_res.confidence:.2f}`")
        lines.append(f"* **Diagnostic Reasoning:** {pd_res.reason}")
        if pd_res.class_distribution:
            lines.append(f"* **Target Class Distribution:** `{json.dumps(pd_res.class_distribution)}`")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 6. Preprocessing Summary
        lines.append("## 6. Preprocessing Architecture & Anti-Leakage")
        lines.append("")
        lines.append("All transformations are strictly encapsulated inside Scikit-Learn `ColumnTransformer` pipelines:")
        lines.append("1. **Numerical Pipeline:** Median imputation followed by optional StandardScaler (applied to linear models).")
        lines.append("2. **Categorical Pipeline:** Mode (most frequent) imputation followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.")
        lines.append("3. **Zero-Leakage Enforcement:** Preprocessors are fitted **solely inside training folds** during CV and inside Optuna trials. No transformations peek across validation or holdout boundaries.")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 7. Models Screened
        lines.append("## 7. Fast Candidate Screening (FLAML)")
        lines.append("")
        if result.screened_candidates:
            lines.append(f"Evaluated multiple model families under strict time constraints (`{result.screening_time_seconds:.2f}s` elapsed):")
            lines.append("")
            lines.append("| Rank | Model Family | Development CV Score | Fit Time | Status |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for sc in result.screened_candidates:
                lines.append(f"| #{sc.rank} | **{sc.model_name}** | `{sc.cv_score:.4f}` | `{sc.fit_time_seconds:.2f}s` | `{sc.status}` |")
            lines.append("")
        else:
            lines.append("FLAML screening was bypassed or executed with direct baseline cross-validation.")
            lines.append("")
        lines.append("---")
        lines.append("")

        # 8. Hyperparameter Tuning (Optuna)
        lines.append("## 8. Bayesian Hyperparameter Optimization (Optuna)")
        lines.append("")
        if result.tuned_candidates:
            lines.append(f"Top candidate algorithms tuned using Tree-structured Parzen Estimator (`TPESampler`) with `MedianPruner` (`{result.tuning_time_seconds:.2f}s` elapsed):")
            lines.append("")
            lines.append("| Model Candidate | Baseline CV Score | Tuned CV Score | Delta | Trials | Status |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for tc in result.tuned_candidates:
                lines.append(
                    f"| **{tc.model_name}** | `{tc.baseline_cv_score:.4f}` | `{tc.tuned_cv_score:.4f}` | "
                    f"**{tc.improvement:+.4f}** | `{tc.n_trials}` | `{tc.status}` |"
                )
            lines.append("")
            if result.tuned_hyperparameters:
                lines.append("#### Optimal Tuned Hyperparameters")
                lines.append("```json")
                lines.append(json.dumps(result.tuned_hyperparameters, indent=2))
                lines.append("```")
                lines.append("")
        else:
            lines.append("Hyperparameter optimization was not triggered or baseline estimators were retained.")
            lines.append("")
        lines.append("---")
        lines.append("")

        # 9. Cross-Validation Results & Leaderboard
        lines.append("## 9. Development Cross-Validation Results")
        lines.append("")
        lines.append(f"> **Evaluation Scheme:** 5-Fold Cross-Validation on Development Data (80% Partition)  ")
        lines.append(f"> **Primary Metric:** `{result.leaderboard[0].primary_metric if result.leaderboard else 'score'}`")
        lines.append("")
        lines.append("| Rank | Model Name | Mean CV Score | CV Std Dev | Fit Time | Status |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for entry in result.leaderboard:
            champ_marker = " 🏆" if entry.rank == 1 else ""
            lines.append(
                f"| #{entry.rank} | **{entry.model_name}{champ_marker}** | `{entry.cv_score_mean:.4f}` | "
                f"`+/- {entry.cv_score_std:.4f}` | `{entry.fit_time_seconds:.2f}s` | `{entry.status}` |"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

        # 10. Final Model Selection
        lines.append("## 10. Final Champion Model Selection")
        lines.append("")
        lines.append(f"### Champion: **{result.best_model_name}**")
        if result.decision_summary:
            lines.append(f"**Selection Rationale:** {result.decision_summary.selection_rationale}")
        lines.append("")
        lines.append("The champion model was refitted on 100% of the development partition before proceeding to holdout verification.")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 11. Final Holdout Evaluation
        lines.append("## 11. Final Holdout Evaluation (Untouched 20%)")
        lines.append("")
        lines.append(
            "> [!IMPORTANT]\n"
            "> The final 20% holdout split was completely segregated prior to screening, cross-validation, and tuning. "
            "It was evaluated exactly once to verify out-of-sample generalization without data leakage."
        )
        lines.append("")
        lines.append("| Evaluation Metric | Holdout Value | Architectural Significance |")
        lines.append("| :--- | :--- | :--- |")
        for metric_name, val in result.holdout_metrics.items():
            lines.append(f"| **{metric_name.upper()}** | `{val:.4f}` | Unbiased generalization score on test partition |")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 12. Feature Importance / Explainability
        expl = result.explainability
        if expl and expl.status in ("success", "fallback"):
            lines.append("## 12. Explainable AI & Global Feature Importance (SHAP)")
            lines.append("")
            lines.append(f"* **Explainer Employed:** `{expl.explainer_type}`")
            lines.append(f"* **Diagnostic Summary:** {expl.summary_text}")
            lines.append("")
            lines.append("### Top Driving Features (Aggregated Raw Features)")
            lines.append("| Rank | Feature Name | Mean |SHAP| Value | Relative Contribution |")
            lines.append("| :--- | :--- | :--- | :--- |")
            for feat in expl.raw_feature_importance[:10]:
                lines.append(f"| #{feat.rank} | **{feat.feature_name}** | `{feat.importance_score:.4f}` | **{feat.relative_importance_pct:.1f}%** |")
            lines.append("")

            # 13. Example Prediction Explanation
            if expl.example_explanations:
                lines.append("## 13. Example Local Prediction Explanations")
                lines.append("")
                lines.append("Local feature attribution for sample predictions:")
                lines.append("")
                for ex in expl.example_explanations[:2]:
                    lines.append(f"### Sample #{ex.sample_index + 1}")
                    lines.append(f"* **Predicted Output:** `{ex.predicted_value}`")
                    if ex.predicted_probability:
                        lines.append(f"* **Prediction Probabilities:** `{json.dumps(ex.predicted_probability)}`")
                    lines.append(f"* **Base Expected Value (E[f(X)]):** `{ex.base_value:.4f}`")
                    lines.append(f"* **Top Positive Drivers:** {', '.join(ex.top_positive_features) or 'None'}")
                    lines.append(f"* **Top Negative Drivers:** {', '.join(ex.top_negative_features) or 'None'}")
                    lines.append("")
                    lines.append("| Feature | Feature Value | SHAP Value | Impact Direction |")
                    lines.append("| :--- | :--- | :--- | :--- |")
                    for cont in ex.contributions[:5]:
                        lines.append(f"| `{cont.feature_name}` | `{cont.feature_value}` | `{cont.shap_value:+.4f}` | `{cont.direction}` |")
                    lines.append("")
            lines.append("---")
            lines.append("")

        # 14. Model Limitations
        lines.append("## 14. Model Limitations & Operational Considerations")
        lines.append("")
        if result.decision_summary and result.decision_summary.model_limitations:
            for lim in result.decision_summary.model_limitations:
                lines.append(f"- **{lim}**")
        else:
            lines.append("- Performance depends upon runtime input distributions matching the development data profile.")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 15. Reproducibility & Runtime Information
        lines.append("## 15. Reproducibility & Provenance Information")
        lines.append("")
        lines.append(f"* **Random Seed:** `{result.random_state}`")
        lines.append(f"* **Cross-Validation Splits:** `5 folds`")
        lines.append(f"* **Screening Time:** `{result.screening_time_seconds:.2f}s`")
        lines.append(f"* **Tuning Time:** `{result.tuning_time_seconds:.2f}s`")
        lines.append(f"* **Total Runtime:** `{result.total_runtime_seconds:.2f}s`")
        lines.append(f"* **Serialized Model Pipeline:** `{result.artifact_path}`")
        lines.append(f"* **Serialized Run Provenance:** `{result.metadata_path}`")
        lines.append("")
        lines.append("---")
        lines.append("*Report automatically generated by Universal ML Engine.*")

        content = "\n".join(lines)
        output_file.write_text(content, encoding="utf-8")
        return output_file

    @classmethod
    def generate_intelligence_report(
        cls, result: ExperimentResult, output_path: Union[str, Path]
    ) -> Path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        intel = result.dataset_intelligence
        if not intel:
            raise ValueError("ExperimentResult does not contain dataset_intelligence data.")

        lines: List[str] = []
        lines.append("# Universal ML Engine — Dataset Intelligence & Health Report")
        lines.append("")
        lines.append(f"> **Dataset:** `{result.dataset_name}`  ")
        lines.append(f"> **Target Variable:** `{result.target_column}`  ")
        lines.append(f"> **Problem Type:** `{intel.problem_type.value}`  ")
        lines.append(f"> **Health Score:** **{intel.health_score.overall_score}/100** ({intel.health_score.grade})  ")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 1. Executive Health Diagnostic
        lines.append("## 1. Executive Health Diagnostic")
        lines.append("")
        lines.append(f"> *{intel.health_score.interpretation}*")
        lines.append("")
        lines.append("| Quality Dimension | Score | Weight | Assessment & Deductions |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for cat_name, cat in intel.health_score.categories.items():
            deductions_str = "<br>".join(cat.deductions) if cat.deductions else "Full Score (Zero deductions)"
            lines.append(f"| **{cat_name}** | `{cat.score}/100` | `{int(cat.weight*100)}%` | {deductions_str} |")
        lines.append("")

        if intel.health_score.clean_signals:
            lines.append("### Verified Clean Signals")
            for sig in intel.health_score.clean_signals:
                lines.append(f"- [x] {sig}")
            lines.append("")
        lines.append("---")
        lines.append("")

        # 2. Structural Profile
        lines.append("## 2. Structural & Semantic Column Profile")
        lines.append("")
        lines.append(f"* **Total Rows:** `{intel.dataset_shape[0]:,}`")
        lines.append(f"* **Total Columns:** `{intel.dataset_shape[1]:,}`")
        lines.append(f"* **Duplicate Rows:** `{intel.duplicate_rows_count:,}`")
        lines.append(f"* **Numerical Features ({len(intel.numeric_columns)}):** `{', '.join(intel.numeric_columns) or 'None'}`")
        lines.append(f"* **Categorical Features ({len(intel.categorical_columns)}):** `{', '.join(intel.categorical_columns) or 'None'}`")
        lines.append(f"* **Datetime Features ({len(intel.datetime_columns)}):** `{', '.join(intel.datetime_columns) or 'None'}`")
        lines.append(f"* **Free Text Features ({len(intel.text_columns)}):** `{', '.join(intel.text_columns) or 'None'}`")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 3. Risk & Leakage Audit
        lines.append("## 3. Risk, Leakage & Overfitting Audit")
        lines.append("")
        lines.append(f"* **Constant Features (Zero Variance):** `{', '.join(intel.constant_columns) or 'None detected'}`")
        lines.append(f"* **Near-Constant Features (>95% single value):** `{', '.join(intel.near_constant_columns) or 'None detected'}`")
        lines.append(f"* **Artificial Identifier Columns (ID Leakage):** `{', '.join(intel.suspicious_id_columns) or 'None detected'}`")
        lines.append(f"* **High-Cardinality Categoricals (>50 categories):** `{', '.join(intel.high_cardinality_columns) or 'None detected'}`")
        lines.append(f"* **Direct Target Leakage Risks (|r| >= 0.95):** `{', '.join(intel.target_leakage_risks) or 'None detected'}`")
        lines.append("")

        if intel.class_imbalance:
            lines.append("### Class Distribution & Imbalance")
            lines.append(f"* **Distribution:** `{json.dumps(intel.class_imbalance['distribution'])}`")
            lines.append(f"* **Imbalance Ratio:** `{intel.class_imbalance['imbalance_ratio']}:1`")
            lines.append(f"* **Imbalance Severity:** `{intel.class_imbalance['severity'].upper()}`")
            lines.append("")

        if intel.correlated_numeric_pairs:
            lines.append("### Multicollinearity Audit (|r| >= 0.85)")
            lines.append("| Feature A | Feature B | Pearson Correlation | Risk Assessment |")
            lines.append("| :--- | :--- | :--- | :--- |")
            for pair in intel.correlated_numeric_pairs:
                lines.append(f"| `{pair.feature_a}` | `{pair.feature_b}` | `{pair.correlation:.4f}` | High mutual redundancy |")
            lines.append("")
        lines.append("---")
        lines.append("")

        # 4. Warnings and Recommended Actions
        lines.append("## 4. Comprehensive Quality Warnings & Recommendations")
        lines.append("")
        if intel.warnings:
            lines.append("| ID | Severity | Scope | Finding | Recommended Remedy |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for w in intel.warnings:
                scope = ", ".join(w.affected_columns) if w.affected_columns else "Dataset"
                lines.append(f"| `{w.warning_id}` | **{w.severity.value.upper()}** | `{scope}` | {w.reason} | {w.recommended_action} |")
            lines.append("")
        else:
            lines.append("Zero warnings detected.")
            lines.append("")

        lines.append("---")
        lines.append("*Report automatically generated by Universal ML Engine.*")

        content = "\n".join(lines)
        output_file.write_text(content, encoding="utf-8")
        return output_file
