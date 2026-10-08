import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from backend.engine.contracts.schemas import (
    ProblemType,
    PartitionInfo,
    ChampionSelectionResult,
    FinalTestEvaluationResult,
    PersistedChampionManifest,
    PermutationImportanceResult,
    M5ReportArtifacts,
)

logger = logging.getLogger(__name__)


class M5ReportGenerator:
    """
    Generates unified machine-readable (JSON) and human-readable (Markdown) reports
    documenting dataset profiling, champion selection, validation vs test evaluation,
    permutation feature importance, and diagnostic visual plots.
    """

    REPORT_JSON_NAME = "m5_report.json"
    REPORT_MD_NAME = "m5_report.md"

    @classmethod
    def generate_report(
        cls,
        manifest: PersistedChampionManifest,
        champion_selection: ChampionSelectionResult,
        test_evaluation: FinalTestEvaluationResult,
        importance_result: PermutationImportanceResult,
        plot_paths: Dict[str, str],
        output_dir: Union[str, Path],
        dataset_name: str = "dataset",
        total_rows: Optional[int] = None,
        partition_info: Optional[Union[PartitionInfo, Dict[str, Any]]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> M5ReportArtifacts:
        """
        Creates and saves m5_report.json and m5_report.md in output_dir.
        """
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()

        # Resolve partition details
        pinfo: Dict[str, Any] = {}
        if isinstance(partition_info, PartitionInfo):
            pinfo = {
                "train_rows": partition_info.train_rows,
                "val_rows": partition_info.val_rows,
                "test_rows": partition_info.test_rows,
                "train_sha256": partition_info.train_sha256,
                "val_sha256": partition_info.val_sha256,
                "test_sha256": partition_info.test_sha256,
                "stratified": partition_info.stratified,
                "random_state": partition_info.random_state,
            }
        elif isinstance(partition_info, dict):
            pinfo = partition_info
        elif manifest.partition_info:
            pinfo = manifest.partition_info

        train_rows = pinfo.get("train_rows", 0)
        val_rows = pinfo.get("val_rows", 0)
        test_rows = pinfo.get("test_rows", test_evaluation.test_rows)
        calc_total_rows = total_rows or (train_rows + val_rows + test_rows)

        # -------------------------------------------------------------
        # 1. Build Machine-Readable JSON Payload
        # -------------------------------------------------------------
        report_data = {
            "schema_version": "1.0.0",
            "report_type": "M5_EXPLAINABILITY_AND_EVALUATION",
            "generated_at": timestamp,
            "dataset_overview": {
                "dataset_name": dataset_name,
                "total_rows": calc_total_rows,
                "target_column": manifest.target_column,
                "problem_type": manifest.problem_type.value if hasattr(manifest.problem_type, "value") else str(manifest.problem_type),
                "feature_count": len(manifest.feature_names),
                "feature_names": manifest.feature_names,
                "feature_types": manifest.feature_types,
                "class_labels": manifest.class_labels,
            },
            "partitions": {
                "train_rows": train_rows,
                "val_rows": val_rows,
                "test_rows": test_rows,
                "train_sha256": pinfo.get("train_sha256", ""),
                "val_sha256": pinfo.get("val_sha256", ""),
                "test_sha256": pinfo.get("test_sha256", test_evaluation.test_sha256),
                "random_state": pinfo.get("random_state", manifest.random_state),
            },
            "champion_selection": {
                "champion_id": manifest.champion_id,
                "model_name": manifest.model_name,
                "backend_name": manifest.backend_name,
                "primary_metric": manifest.primary_metric,
                "primary_val_score": champion_selection.champion.primary_val_score,
                "selection_rationale": champion_selection.selection_rationale,
                "selection_time_seconds": champion_selection.selection_time_seconds,
                "all_candidates": [
                    {
                        "candidate_id": c.candidate_id,
                        "backend_name": c.backend_name,
                        "model_name": c.model_name,
                        "primary_metric": c.primary_metric,
                        "primary_val_score": c.primary_val_score,
                        "fit_time_seconds": c.fit_time_seconds,
                        "val_metrics": c.val_metrics,
                    }
                    for c in champion_selection.all_candidates
                ],
            },
            "metrics_comparison": {
                "validation_metrics": manifest.val_metrics,
                "final_test_metrics": manifest.test_metrics,
            },
            "explainability": {
                "method": "Permutation Feature Importance",
                "evaluation_data": "EXTERNAL_VALIDATION",
                "primary_metric": importance_result.primary_metric,
                "baseline_score": importance_result.baseline_score,
                "n_repeats": importance_result.n_repeats,
                "sample_size": importance_result.sample_size,
                "computation_time_seconds": importance_result.computation_time_seconds,
                "status": importance_result.status,
                "warnings": importance_result.warnings,
                "feature_importances": [
                    {
                        "rank": e.rank,
                        "feature_name": e.feature_name,
                        "importance_score": e.importance_score,
                        "relative_importance_pct": e.relative_importance_pct,
                        "std_dev": getattr(e, "std_dev", 0.0),
                    }
                    for e in importance_result.importance_entries
                ],
            },
            "visualizations": {
                name: str(Path(path).resolve()) for name, path in plot_paths.items()
            },
            "model_provenance": {
                "artifact_rel_path": manifest.artifact_rel_path,
                "artifact_type": manifest.artifact_type,
                "random_state": manifest.random_state,
                "extra_metadata": extra_metadata or {},
            },
        }

        json_path = out_dir / cls.REPORT_JSON_NAME
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # -------------------------------------------------------------
        # 2. Build Human-Readable Markdown Report
        # -------------------------------------------------------------
        p_type_str = manifest.problem_type.value if hasattr(manifest.problem_type, "value") else str(manifest.problem_type)
        is_classification = p_type_str in ["binary_classification", "multiclass_classification"]

        md_lines = [
            f"# Model Evaluation & Explainability Report",
            f"",
            f"**Dataset:** `{dataset_name}`  ",
            f"**Task Type:** `{p_type_str.upper()}`  ",
            f"**Target Column:** `{manifest.target_column}`  ",
            f"**Report Generated:** `{timestamp}`  ",
            f"",
            f"---",
            f"",
            f"## 1. Dataset & Partition Overview",
            f"",
            f"- **Total Rows:** {calc_total_rows}",
            f"- **Feature Count:** {len(manifest.feature_names)}",
            f"- **Features:** {', '.join([f'`{col}`' for col in manifest.feature_names[:10]])}{' ...' if len(manifest.feature_names) > 10 else ''}",
            f"",
            f"### Three-Way Partitions (Strict Zero Data Leakage)",
            f"",
            f"| Partition | Rows | Ratio | SHA-256 Hash | Role |",
            f"| :--- | :--- | :--- | :--- | :--- |",
            f"| **TRAIN** | {train_rows:,} | {round(train_rows/calc_total_rows*100, 1) if calc_total_rows else 0}% | `{pinfo.get('train_sha256', 'n/a')[:16]}...` | Backend Model Fitting Only |",
            f"| **VALIDATION** | {val_rows:,} | {round(val_rows/calc_total_rows*100, 1) if calc_total_rows else 0}% | `{pinfo.get('val_sha256', 'n/a')[:16]}...` | Champion Selection & Permutation Importance |",
            f"| **TEST** | {test_rows:,} | {round(test_rows/calc_total_rows*100, 1) if calc_total_rows else 0}% | `{pinfo.get('test_sha256', 'n/a')[:16]}...` | Untouched Holdout (Single-Pass Final Evaluation) |",
            f"",
            f"---",
            f"",
            f"## 2. Champion Selection Summary",
            f"",
            f"- **Winning Champion:** `{manifest.model_name}` ({manifest.backend_name})",
            f"- **Selection Primary Metric:** `{manifest.primary_metric}` = **{champion_selection.champion.primary_val_score:.4f}**",
            f"- **Rationale:** {champion_selection.selection_rationale}",
            f"",
            f"### Candidate Comparison on External VALIDATION",
            f"",
            f"| Candidate ID | Backend | Model Name | Primary Metric (`{manifest.primary_metric}`) | Fit Time (s) |",
            f"| :--- | :--- | :--- | :--- | :--- |",
        ]

        for cand in champion_selection.all_candidates:
            md_lines.append(
                f"| `{cand.candidate_id}` | `{cand.backend_name}` | {cand.model_name} | **{cand.primary_val_score:.4f}** | {cand.fit_time_seconds:.2f}s |"
            )

        md_lines.extend([
            f"",
            f"---",
            f"",
            f"## 3. Evaluation Metrics: Validation vs. Final Test",
            f"",
            f"> [!IMPORTANT]",
            f"> **VALIDATION METRICS** were used strictly to compare candidate models and select the champion.",
            f"> **FINAL TEST METRICS** were produced by evaluating the persisted champion on the untouched TEST partition exactly once.",
            f"",
            f"### Metric Side-by-Side Comparison",
            f"",
            f"| Metric | Validation Score (Selection) | Final Test Score (Holdout) | Delta (Test - Val) |",
            f"| :--- | :--- | :--- | :--- |",
        ])

        all_metric_keys = sorted(list(set(manifest.val_metrics.keys()) | set(manifest.test_metrics.keys())))
        for m_name in all_metric_keys:
            val_v = manifest.val_metrics.get(m_name)
            test_v = manifest.test_metrics.get(m_name)
            val_str = f"{val_v:.4f}" if val_v is not None else "N/A"
            test_str = f"{test_v:.4f}" if test_v is not None else "N/A"
            delta_str = f"{test_v - val_v:+.4f}" if (val_v is not None and test_v is not None) else "N/A"
            md_lines.append(f"| **{m_name}** | {val_str} | {test_str} | {delta_str} |")

        md_lines.extend([
            f"",
            f"---",
            f"",
            f"## 4. Explainability: Permutation Feature Importance",
            f"",
            f"*Computed strictly on the external VALIDATION set using {importance_result.n_repeats} shuffle repeats.*",
            f"",
            f"| Rank | Feature Name | Mean Importance Score | Relative Importance (%) | Std Dev |",
            f"| :---: | :--- | :---: | :---: | :---: |",
        ])

        for entry in importance_result.importance_entries[:15]:
            std_val = getattr(entry, "std_dev", 0.0)
            md_lines.append(
                f"| {entry.rank} | `{entry.feature_name}` | {entry.importance_score:+.4f} | {entry.relative_importance_pct:.1f}% | ±{std_val:.4f} |"
            )

        md_lines.extend([
            f"",
            f"---",
            f"",
            f"## 5. Visual Diagnostics & Plots",
            f"",
        ])

        for plot_name, plot_file in plot_paths.items():
            rel_plot_path = Path(plot_file).name
            clean_title = plot_name.replace("_", " ").title()
            md_lines.extend([
                f"### {clean_title}",
                f"![{clean_title}]({rel_plot_path})",
                f"",
            ])

        md_lines.extend([
            f"---",
            f"",
            f"## 6. Model Artifact & Provenance",
            f"- **Champion Manifest:** `{out_dir / 'champion_manifest.json'}`",
            f"- **Artifact Storage:** `{manifest.artifact_rel_path}` ({manifest.artifact_type})",
            f"- **Random State:** `{manifest.random_state}`",
            f"- **Schema Version:** `{manifest.schema_version}`",
        ])

        md_path = out_dir / cls.REPORT_MD_NAME
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        logger.info(f"[M5ReportGenerator] Created reports at: {json_path} and {md_path}")
        return M5ReportArtifacts(
            report_json_path=str(json_path),
            report_md_path=str(md_path),
            plot_paths=plot_paths,
            generated_at=timestamp,
        )
