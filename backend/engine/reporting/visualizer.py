import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Headless matplotlib configuration
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error

from backend.engine.contracts.schemas import FeatureImportanceEntry

logger = logging.getLogger(__name__)


class M5Visualizer:
    """
    Automated visualization generator for classification and regression tasks.
    Produces high-resolution PNG plots using native matplotlib (zero heavy dependencies).
    """

    @classmethod
    def plot_confusion_matrix(
        cls,
        y_true: Union[np.ndarray, List[Any]],
        y_pred: Union[np.ndarray, List[Any]],
        output_path: Union[str, Path],
        labels: Optional[List[Any]] = None,
        title: str = "Confusion Matrix",
    ) -> Path:
        """
        Renders and saves a confusion matrix heatmap.
        """
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        # Harmonize types to avoid string/int mismatch
        if y_true_arr.dtype != y_pred_arr.dtype:
            y_true_arr = y_true_arr.astype(str)
            y_pred_arr = y_pred_arr.astype(str)

        if labels is None:
            resolved_labels = sorted(list(set(y_true_arr) | set(y_pred_arr)))
        else:
            resolved_labels = [str(lbl) for lbl in labels] if y_true_arr.dtype == object or str(y_true_arr.dtype).startswith("<U") else labels

        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=resolved_labels)
        total = np.sum(cm)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        tick_marks = np.arange(len(resolved_labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels([str(lbl) for lbl in resolved_labels], rotation=45, ha="right", fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels([str(lbl) for lbl in resolved_labels], fontsize=9)
        ax.set_ylabel("True Label", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")

        # Annotate cells with counts and percentages
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                pct = (val / total * 100.0) if total > 0 else 0.0
                text = f"{val}\n({pct:.1f}%)"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "black",
                    fontsize=9,
                    fontweight="bold",
                )

        plt.tight_layout()
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"[M5Visualizer] Saved confusion matrix plot to: {out_file}")
        return out_file

    @classmethod
    def plot_feature_importance(
        cls,
        importance_entries: List[FeatureImportanceEntry],
        output_path: Union[str, Path],
        top_n: int = 15,
        title: str = "Permutation Feature Importance (Validation)",
    ) -> Path:
        """
        Renders and saves a horizontal bar chart of ranked permutation feature importances.
        """
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, max(4, min(10, len(importance_entries[:top_n]) * 0.4 + 1.5))), dpi=150)

        if not importance_entries:
            ax.text(0.5, 0.5, "No feature importance data available", ha="center", va="center")
            ax.set_title(title, fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return out_file

        top_entries = importance_entries[:top_n]
        # Invert so rank 1 is at top
        top_entries_rev = list(reversed(top_entries))

        names = [e.feature_name for e in top_entries_rev]
        scores = [e.importance_score for e in top_entries_rev]
        errors = [getattr(e, "std_dev", 0.0) for e in top_entries_rev]
        rel_pcts = [e.relative_importance_pct for e in top_entries_rev]

        y_pos = np.arange(len(names))
        colors = ["#2563eb" if s >= 0 else "#ef4444" for s in scores]

        bars = ax.barh(
            y_pos,
            scores,
            xerr=errors,
            color=colors,
            alpha=0.85,
            edgecolor="#1e3a8a",
            capsize=3,
            height=0.65,
        )

        ax.axvline(0, color="#64748b", linestyle="--", linewidth=1.0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9, fontweight="medium")
        ax.set_xlabel("Mean Score Delta (Score Decrease upon Shuffle)", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.grid(axis="x", linestyle=":", alpha=0.5)

        # Value annotations
        max_abs = max([abs(s) for s in scores] + [0.001])
        offset = max_abs * 0.02
        for idx, (bar, score, pct) in enumerate(zip(bars, scores, rel_pcts)):
            text = f" {score:+.4f} ({pct:.1f}%)" if pct > 0 else f" {score:+.4f}"
            x_pos = score + offset if score >= 0 else offset
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2.0, text, va="center", fontsize=8, color="#1e293b")

        plt.tight_layout()
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"[M5Visualizer] Saved feature importance plot to: {out_file}")
        return out_file

    @classmethod
    def plot_actual_vs_predicted(
        cls,
        y_true: Union[np.ndarray, List[float]],
        y_pred: Union[np.ndarray, List[float]],
        output_path: Union[str, Path],
        title: str = "Actual vs. Predicted Target (Test)",
    ) -> Path:
        """
        Renders and saves an Actual vs. Predicted scatter plot with 45-degree reference line.
        """
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)

        fig, ax = plt.subplots(figsize=(6, 5.5), dpi=150)

        # Scatter
        ax.scatter(y_true_arr, y_pred_arr, color="#2563eb", alpha=0.6, edgecolors="none", s=35, label="Observations")

        # 45-degree line
        min_val = min(float(np.min(y_true_arr)), float(np.min(y_pred_arr)))
        max_val = max(float(np.max(y_true_arr)), float(np.max(y_pred_arr)))
        buffer = (max_val - min_val) * 0.05
        line_min = min_val - buffer
        line_max = max_val + buffer

        ax.plot([line_min, line_max], [line_min, line_max], color="#dc2626", linestyle="--", linewidth=1.5, label="Perfect Fit (y = x)")

        # Summary Metrics Box
        r2 = r2_score(y_true_arr, y_pred_arr)
        rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
        stats_text = f"R² = {r2:.4f}\nRMSE = {rmse:,.2f}"
        ax.text(
            0.05,
            0.92,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9),
        )

        ax.set_xlim(line_min, line_max)
        ax.set_ylim(line_min, line_max)
        ax.set_xlabel("Actual Target Values", fontsize=10, fontweight="bold")
        ax.set_ylabel("Predicted Target Values", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"[M5Visualizer] Saved actual vs predicted plot to: {out_file}")
        return out_file

    @classmethod
    def plot_residuals(
        cls,
        y_true: Union[np.ndarray, List[float]],
        y_pred: Union[np.ndarray, List[float]],
        output_path: Union[str, Path],
        title: str = "Residuals vs. Predicted Values (Test)",
    ) -> Path:
        """
        Renders and saves a Residuals plot with zero-line.
        """
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        residuals = y_true_arr - y_pred_arr

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

        ax.scatter(y_pred_arr, residuals, color="#0284c7", alpha=0.6, edgecolors="none", s=35)
        ax.axhline(0, color="#dc2626", linestyle="--", linewidth=1.5, label="Zero Residual")

        # Mean residual stat
        mean_res = float(np.mean(residuals))
        std_res = float(np.std(residuals))
        stats_text = f"Mean Residual = {mean_res:,.2f}\nStd Dev = {std_res:,.2f}"
        ax.text(
            0.05,
            0.92,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9),
        )

        ax.set_xlabel("Predicted Values", fontsize=10, fontweight="bold")
        ax.set_ylabel("Residual (Actual - Predicted)", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"[M5Visualizer] Saved residuals plot to: {out_file}")
        return out_file

    @classmethod
    def generate_classification_plots(
        cls,
        y_true_test: Union[np.ndarray, List[Any]],
        y_pred_test: Union[np.ndarray, List[Any]],
        importance_entries: List[FeatureImportanceEntry],
        output_dir: Union[str, Path],
        class_labels: Optional[List[Any]] = None,
    ) -> Dict[str, str]:
        """
        Generates standard classification visual suite:
        - Confusion Matrix (Test partition)
        - Permutation Feature Importance (Validation partition)
        """
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        cm_path = out_dir / "confusion_matrix.png"
        fi_path = out_dir / "feature_importance.png"

        cls.plot_confusion_matrix(
            y_true=y_true_test,
            y_pred=y_pred_test,
            output_path=cm_path,
            labels=class_labels,
            title="Confusion Matrix (Untouched Test Partition)",
        )

        cls.plot_feature_importance(
            importance_entries=importance_entries,
            output_path=fi_path,
            title="Permutation Feature Importance (External Validation)",
        )

        return {
            "confusion_matrix": str(cm_path),
            "feature_importance": str(fi_path),
        }

    @classmethod
    def generate_regression_plots(
        cls,
        y_true_test: Union[np.ndarray, List[float]],
        y_pred_test: Union[np.ndarray, List[float]],
        importance_entries: List[FeatureImportanceEntry],
        output_dir: Union[str, Path],
    ) -> Dict[str, str]:
        """
        Generates standard regression visual suite:
        - Actual vs. Predicted (Test partition)
        - Residual Plot (Test partition)
        - Permutation Feature Importance (Validation partition)
        """
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        act_pred_path = out_dir / "actual_vs_predicted.png"
        res_path = out_dir / "residuals.png"
        fi_path = out_dir / "feature_importance.png"

        cls.plot_actual_vs_predicted(
            y_true=y_true_test,
            y_pred=y_pred_test,
            output_path=act_pred_path,
            title="Actual vs. Predicted Target (Untouched Test Partition)",
        )

        cls.plot_residuals(
            y_true=y_true_test,
            y_pred=y_pred_test,
            output_path=res_path,
            title="Residuals vs. Predicted (Untouched Test Partition)",
        )

        cls.plot_feature_importance(
            importance_entries=importance_entries,
            output_path=fi_path,
            title="Permutation Feature Importance (External Validation)",
        )

        return {
            "actual_vs_predicted": str(act_pred_path),
            "residuals": str(res_path),
            "feature_importance": str(fi_path),
        }
