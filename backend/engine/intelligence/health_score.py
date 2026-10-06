from typing import Dict, Any, List, Optional
from backend.engine.contracts.schemas import (
    QualityWarning,
    QualitySeverity,
    HealthCategoryScore,
    DatasetHealthScore,
)


class DatasetHealthCalculator:
    """
    Computes an interpretable, deterministic engineering diagnostic health score (0-100)
    for a tabular dataset based on measurable data quality signals.
    """

    @classmethod
    def calculate(
        cls,
        total_rows: int,
        total_cols: int,
        duplicate_rows_count: int,
        missing_percentages: Dict[str, float],
        constant_columns: List[str],
        near_constant_columns: List[str],
        high_cardinality_columns: List[str],
        suspicious_id_columns: List[str],
        target_leakage_risks: List[str],
        correlated_pairs_count: int,
        class_imbalance: Optional[Dict[str, Any]],
        target_column: str,
        warnings: List[QualityWarning],
    ) -> DatasetHealthScore:
        clean_signals: List[str] = []

        # ---------------------------------------------------------
        # 1. Completeness Score (Weight: 25%)
        # ---------------------------------------------------------
        completeness_deductions = []
        completeness_score = 100.0

        target_missing = missing_percentages.get(target_column, 0.0)
        if target_missing > 0.0:
            deduct = min(30.0, target_missing * 100.0)
            completeness_score -= deduct
            completeness_deductions.append(f"Target column '{target_column}' has {target_missing*100:.1f}% missing values (-{deduct:.1f} pts)")

        feature_missings = [pct for col, pct in missing_percentages.items() if col != target_column]
        avg_missing = sum(feature_missings) / len(feature_missings) if feature_missings else 0.0

        if avg_missing > 0.0:
            deduct = min(25.0, avg_missing * 50.0)
            completeness_score -= deduct
            completeness_deductions.append(f"Average feature missingness is {avg_missing*100:.1f}% (-{deduct:.1f} pts)")
        else:
            clean_signals.append("Zero missing values detected across feature matrix.")

        severe_missing_cols = [col for col, pct in missing_percentages.items() if pct > 0.20 and col != target_column]
        if severe_missing_cols:
            deduct = min(20.0, len(severe_missing_cols) * 5.0)
            completeness_score -= deduct
            completeness_deductions.append(f"{len(severe_missing_cols)} feature(s) have >20% missing values (-{deduct:.1f} pts)")

        crit_missing_cols = [col for col, pct in missing_percentages.items() if pct > 0.50 and col != target_column]
        if crit_missing_cols:
            deduct = min(20.0, len(crit_missing_cols) * 10.0)
            completeness_score -= deduct
            completeness_deductions.append(f"{len(crit_missing_cols)} feature(s) have >50% missing values (-{deduct:.1f} pts)")

        completeness_score = max(0.0, min(100.0, completeness_score))
        cat_completeness = HealthCategoryScore(
            category="Completeness",
            score=round(completeness_score, 1),
            weight=0.25,
            description="Measures data completeness, null density, and target availability.",
            deductions=completeness_deductions,
        )

        # ---------------------------------------------------------
        # 2. Uniqueness & Structural Integrity (Weight: 25%)
        # ---------------------------------------------------------
        uniqueness_deductions = []
        uniqueness_score = 100.0

        if duplicate_rows_count > 0:
            dup_ratio = duplicate_rows_count / max(1, total_rows)
            deduct = min(30.0, max(5.0, dup_ratio * 100.0))
            uniqueness_score -= deduct
            uniqueness_deductions.append(f"{duplicate_rows_count} duplicate row(s) found ({dup_ratio*100:.1f}%) (-{deduct:.1f} pts)")
        else:
            clean_signals.append("Zero duplicate rows found in dataset.")

        if constant_columns:
            deduct = min(25.0, len(constant_columns) * 10.0)
            uniqueness_score -= deduct
            uniqueness_deductions.append(f"{len(constant_columns)} constant column(s) detected (-{deduct:.1f} pts)")
        else:
            clean_signals.append("No constant features detected.")

        if near_constant_columns:
            deduct = min(15.0, len(near_constant_columns) * 5.0)
            uniqueness_score -= deduct
            uniqueness_deductions.append(f"{len(near_constant_columns)} near-constant column(s) (>95% single value) (-{deduct:.1f} pts)")

        if suspicious_id_columns:
            deduct = min(20.0, len(suspicious_id_columns) * 5.0)
            uniqueness_score -= deduct
            uniqueness_deductions.append(f"{len(suspicious_id_columns)} artificial ID / index column(s) detected (-{deduct:.1f} pts)")
        else:
            clean_signals.append("No artificial identifier or surrogate key columns found.")

        uniqueness_score = max(0.0, min(100.0, uniqueness_score))
        cat_uniqueness = HealthCategoryScore(
            category="Uniqueness & Structure",
            score=round(uniqueness_score, 1),
            weight=0.25,
            description="Evaluates row uniqueness, constant features, and presence of ID columns.",
            deductions=uniqueness_deductions,
        )

        # ---------------------------------------------------------
        # 3. Feature Quality & Independence (Weight: 25%)
        # ---------------------------------------------------------
        feature_deductions = []
        feature_score = 100.0

        if high_cardinality_columns:
            deduct = min(25.0, len(high_cardinality_columns) * 5.0)
            feature_score -= deduct
            feature_deductions.append(f"{len(high_cardinality_columns)} high-cardinality categorical feature(s) (-{deduct:.1f} pts)")
        else:
            clean_signals.append("Categorical features exhibit manageable cardinality.")

        if correlated_pairs_count > 0:
            deduct = min(20.0, correlated_pairs_count * 4.0)
            feature_score -= deduct
            feature_deductions.append(f"{correlated_pairs_count} highly correlated feature pair(s) (|r| >= 0.85) (-{deduct:.1f} pts)")
        else:
            clean_signals.append("No severe multicollinearity (|r| >= 0.85) detected between features.")

        feature_score = max(0.0, min(100.0, feature_score))
        cat_feature_quality = HealthCategoryScore(
            category="Feature Quality",
            score=round(feature_score, 1),
            weight=0.25,
            description="Evaluates feature cardinality, multicollinearity, and numerical stability.",
            deductions=feature_deductions,
        )

        # ---------------------------------------------------------
        # 4. Target & Label Integrity (Weight: 25%)
        # ---------------------------------------------------------
        target_deductions = []
        target_score = 100.0

        if target_leakage_risks:
            deduct = min(40.0, len(target_leakage_risks) * 20.0)
            target_score -= deduct
            target_deductions.append(f"{len(target_leakage_risks)} feature(s) flagged with severe target leakage risk (-{deduct:.1f} pts)")
        else:
            clean_signals.append("Target variable exhibits zero direct leakage risks.")

        if class_imbalance:
            ratio = class_imbalance.get("imbalance_ratio", 1.0)
            severity = class_imbalance.get("severity", "none")
            if severity == "severe":
                target_score -= 20.0
                target_deductions.append(f"Severe class imbalance detected ({ratio:.1f}:1 ratio) (-20.0 pts)")
            elif severity == "moderate":
                target_score -= 10.0
                target_deductions.append(f"Moderate class imbalance detected ({ratio:.1f}:1 ratio) (-10.0 pts)")
            else:
                clean_signals.append("Target classes are well-balanced.")
        else:
            clean_signals.append("Target distribution is valid.")

        target_score = max(0.0, min(100.0, target_score))
        cat_target_integrity = HealthCategoryScore(
            category="Target Integrity",
            score=round(target_score, 1),
            weight=0.25,
            description="Evaluates target leakage risks, class balance, and distribution validity.",
            deductions=target_deductions,
        )

        # ---------------------------------------------------------
        # Overall Weighted Health Score & Grade
        # ---------------------------------------------------------
        categories = {
            "Completeness": cat_completeness,
            "Uniqueness & Structure": cat_uniqueness,
            "Feature Quality": cat_feature_quality,
            "Target Integrity": cat_target_integrity,
        }

        overall_score = sum(cat.score * cat.weight for cat in categories.values())
        overall_score = round(max(0.0, min(100.0, overall_score)), 1)

        if overall_score >= 90.0:
            grade = "Excellent"
            interpretation = (
                f"Dataset health is Excellent ({overall_score}/100). The dataset exhibits strong completeness, "
                "clean structural integrity, and no critical leakage or imbalance risks."
            )
        elif overall_score >= 75.0:
            grade = "Good"
            interpretation = (
                f"Dataset health is Good ({overall_score}/100). The data is production-viable with minor quality "
                "warnings that are automatically mitigated by preprocessing pipelines."
            )
        elif overall_score >= 60.0:
            grade = "Fair"
            interpretation = (
                f"Dataset health is Fair ({overall_score}/100). Noticeable data quality concerns detected "
                "(such as class imbalance, missingness, or multicollinearity). Proceed with caution."
            )
        else:
            grade = "Needs Attention"
            interpretation = (
                f"Dataset health Needs Attention ({overall_score}/100). Substantial data risks detected "
                "that may impair generalization if unaddressed."
            )

        major_warnings = [
            w for w in warnings if w.severity in (QualitySeverity.HIGH, QualitySeverity.CRITICAL)
        ]

        return DatasetHealthScore(
            overall_score=overall_score,
            grade=grade,
            categories=categories,
            major_warnings=major_warnings,
            clean_signals=clean_signals,
            interpretation=interpretation,
        )
