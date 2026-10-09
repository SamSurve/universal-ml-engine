import os
import sys
import uuid
import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.engine.contracts.schemas import ProblemType
from backend.engine.ingestion.service import IngestionService, DataIngestionError
from backend.engine.problem_detection.detector import ProblemDetector
from backend.engine.training.m5_orchestrator import M5PipelineRunner
from backend.engine.inference.unified_predictor import UnifiedPredictor
from backend.engine.artifacts.champion_store import ChampionStore

logger = logging.getLogger("universal_ml_engine.ui")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Universal ML Engine — Judge-Ready Interface",
    version="1.0.0",
    description="Minimal, reliable local presentation layer for Universal ML Engine.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"
UPLOADS_DIR = PROJECT_ROOT / "scratch" / "uploads"
STATIC_DIR = Path(__file__).resolve().parent / "static"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Thread-safe in-memory store for active runs and uploaded datasets
RUNS_LOCK = threading.Lock()
ACTIVE_RUNS: Dict[str, Dict[str, Any]] = {}
UPLOADED_DATASETS: Dict[str, Dict[str, Any]] = {}


# -----------------------------------------------------------------------------
# Pydantic Request Models
# -----------------------------------------------------------------------------
class DetectTargetRequest(BaseModel):
    upload_id: str
    target_column: str


class RunEngineRequest(BaseModel):
    upload_id: str
    target_column: str
    time_limit_seconds: float = 30.0
    presets: str = "medium_quality"


class PredictRequest(BaseModel):
    run_id: str
    features: Dict[str, Any]


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_run_dir(run_id: str) -> Path:
    """Resolves directory for run_id."""
    direct_path = ARTIFACTS_DIR / run_id
    if direct_path.exists():
        return direct_path
    # Check if run_id is an absolute path or scratch path
    p = Path(run_id)
    if p.exists():
        return p
    raise HTTPException(status_code=404, detail=f"Run directory not found for run_id: {run_id}")


def inspect_dataframe(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
    """Generates dataset inspection summary."""
    row_count = len(df)
    col_count = len(df.columns)
    total_cells = row_count * col_count if col_count > 0 else 1
    missing_cells = int(df.isna().sum().sum())
    missing_pct = round((missing_cells / total_cells) * 100.0, 2)

    columns_meta = []
    categorical_options = {}
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        dtype_str = str(s.dtype)
        columns_meta.append({
            "name": str(col),
            "dtype": dtype_str,
            "missing_count": n_missing,
            "missing_pct": round((n_missing / row_count) * 100.0, 2) if row_count > 0 else 0.0,
            "unique_count": n_unique,
        })
        # If categorical or low-cardinality discrete feature, gather sample choices
        if dtype_str == "object" or dtype_str == "category" or (n_unique <= 20 and n_unique > 1):
            unique_vals = [v for v in s.dropna().unique().tolist() if v is not None][:25]
            # Convert non-serializable types to native python
            cleaned_vals = []
            for uv in unique_vals:
                if isinstance(uv, (np.integer, np.floating)):
                    cleaned_vals.append(uv.item())
                else:
                    cleaned_vals.append(str(uv))
            categorical_options[str(col)] = cleaned_vals

    # Priority-based target identification
    candidate_target = None
    lower_filename = filename.lower()
    
    # Priority 1: Exact dataset-to-target known mapping
    if "turnover" in lower_filename and "Employee_Turnover" in df.columns:
        candidate_target = "Employee_Turnover"
    elif ("house" in lower_filename or "price" in lower_filename) and "SalePrice" in df.columns:
        candidate_target = "SalePrice"

    # Priority 2: Exact column name matches (case-insensitive)
    if not candidate_target:
        exact_target_names = [
            "employee_turnover", "saleprice", "target", "label", "class",
            "churn", "turnover", "price", "outcome", "status", "response", "default", "survived"
        ]
        col_lower_map = {str(c).lower(): c for c in df.columns}
        for kw in exact_target_names:
            if kw in col_lower_map:
                candidate_target = col_lower_map[kw]
                break

    # Priority 3: Specific target keywords avoiding feature substring traps like MSSubClass
    if not candidate_target:
        avoid_cols = {"id", "mssubclass", "education_level", "index", "user_id", "row_id", "unnamed: 0"}
        target_keywords = ["saleprice", "employee_turnover", "turnover", "target", "label", "churn", "price", "outcome", "response", "default"]
        for col in df.columns:
            c_low = str(col).lower()
            if c_low in avoid_cols:
                continue
            if any(kw in c_low for kw in target_keywords):
                candidate_target = col
                break

    if not candidate_target and len(df.columns) > 0:
        candidate_target = df.columns[-1]

    # Pre-detect candidate problem if candidate exists
    detected_problem = None
    if candidate_target and candidate_target in df.columns:
        try:
            pd_res = ProblemDetector.detect(df, candidate_target)
            detected_problem = {
                "problem_type": pd_res.problem_type.value,
                "confidence": round(pd_res.confidence, 2),
                "reason": pd_res.reason,
                "unique_target_values": pd_res.unique_target_values,
                "class_distribution": pd_res.class_distribution,
            }
        except Exception as e:
            logger.warning(f"Problem detection error for {candidate_target}: {e}")

    preview_df = df.head(5).copy()
    preview_rows = preview_df.fillna("").to_dict(orient="records")

    return {
        "filename": filename,
        "row_count": row_count,
        "column_count": col_count,
        "missing_cells": missing_cells,
        "missing_percentage": missing_pct,
        "columns": columns_meta,
        "categorical_options": categorical_options,
        "preview_rows": preview_rows,
        "suggested_target": candidate_target,
        "detected_problem": detected_problem,
    }


def load_result_from_disk(run_dir: Path) -> Dict[str, Any]:
    """Loads results from persisted artifacts (champion_manifest.json + m5_report.json)."""
    manifest_file = run_dir / "champion_manifest.json"
    report_json_file = run_dir / "m5_report.json"

    if not manifest_file.exists():
        raise FileNotFoundError(f"Missing champion_manifest.json in {run_dir}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    report_data = {}
    if report_json_file.exists():
        with open(report_json_file, "r", encoding="utf-8") as f:
            report_data = json.load(f)

    # Resolve partition info
    p_info = manifest.get("partition_info") or report_data.get("partitions") or {}
    overview = report_data.get("dataset_overview", {})
    champ_sel = report_data.get("champion_selection", {})
    test_eval = report_data.get("test_evaluation", {})
    metrics_comp = report_data.get("metrics_comparison", {})
    perm_imp = report_data.get("permutation_importance", {})

    # Plots
    plot_urls = {}
    plots_dir = run_dir / "plots"
    if plots_dir.exists():
        for p_file in plots_dir.glob("*.png"):
            plot_name = p_file.stem
            plot_urls[plot_name] = f"/api/artifacts/{run_dir.name}/plots/{p_file.name}"

    # Extract real sample test rows and categorical options from source datasets
    sample_rows = []
    categorical_options = {}
    target_col = manifest.get("target_column", "")
    feature_names = manifest.get("feature_names", [])

    # Locate source dataset if available
    source_df = None
    possible_paths = [
        PROJECT_ROOT / "employee_turnover.csv",
        PROJECT_ROOT / "HousePricePrediction.csv",
    ]
    # Check if dataset_name matches
    ds_name = overview.get("dataset_name", "")
    if ds_name:
        possible_paths.insert(0, PROJECT_ROOT / ds_name)

    # Match dataset by target column or name
    for p in possible_paths:
        if p.exists():
            try:
                candidate_df = pd.read_csv(p)
                if target_col in candidate_df.columns:
                    source_df = candidate_df
                    break
            except Exception:
                pass

    if source_df is not None and not source_df.empty:
        # Extract up to 5 real rows from the dataset
        sample_subset = source_df.head(5).copy()
        for idx, row in sample_subset.iterrows():
            row_dict = {}
            for feat in feature_names:
                if feat in row:
                    val = row[feat]
                    if pd.isna(val):
                        row_dict[feat] = ""
                    elif isinstance(val, (np.integer, np.floating)):
                        row_dict[feat] = val.item()
                    else:
                        row_dict[feat] = str(val)
                else:
                    row_dict[feat] = 0
            if target_col in row:
                gt_val = row[target_col]
                row_dict["_ground_truth"] = gt_val.item() if isinstance(gt_val, (np.integer, np.floating)) else str(gt_val)
            row_dict["_row_label"] = f"Sample Row #{idx + 1}"
            sample_rows.append(row_dict)

        # Gather categorical options for features
        for feat in feature_names:
            if feat in source_df.columns:
                s = source_df[feat]
                if s.dtype == "object" or s.nunique() <= 20:
                    u_vals = [v for v in s.dropna().unique().tolist() if v is not None][:25]
                    cleaned_vals = []
                    for uv in u_vals:
                        if isinstance(uv, (np.integer, np.floating)):
                            cleaned_vals.append(uv.item())
                        else:
                            cleaned_vals.append(str(uv))
                    categorical_options[feat] = cleaned_vals

    # Fallback generic sample row if dataset not found
    if not sample_rows:
        sample_dict = {"_row_label": "Default Benchmark Input"}
        for feat in feature_names:
            feat_type = manifest.get("feature_types", {}).get(feat, "float64")
            if "int" in feat_type:
                sample_dict[feat] = 1
            elif "float" in feat_type:
                sample_dict[feat] = 1.0
            else:
                sample_dict[feat] = "sample"
        sample_rows.append(sample_dict)

    return {
        "run_id": run_dir.name,
        "dataset_name": overview.get("dataset_name", run_dir.name),
        "target_column": manifest.get("target_column", ""),
        "problem_type": manifest.get("problem_type", "unknown"),
        "total_rows": overview.get("total_rows", (p_info.get("train_rows", 0) + p_info.get("val_rows", 0) + p_info.get("test_rows", 0))),
        "partition_info": {
            "train_rows": p_info.get("train_rows", 0),
            "val_rows": p_info.get("val_rows", 0),
            "test_rows": p_info.get("test_rows", 0),
            "train_sha256": p_info.get("train_sha256", ""),
            "val_sha256": p_info.get("val_sha256", ""),
            "test_sha256": p_info.get("test_sha256", ""),
        },
        "champion": {
            "model_name": manifest.get("model_name", "Unknown Champion"),
            "backend_name": manifest.get("backend_name", "unknown"),
            "candidate_id": champ_sel.get("champion_id", manifest.get("champion_id", "")),
            "primary_metric": manifest.get("primary_metric", ""),
            "primary_val_score": champ_sel.get("primary_val_score", 0.0),
            "val_metrics": metrics_comp.get("validation_metrics", manifest.get("val_metrics", {})),
            "selection_rationale": manifest.get("selection_rationale", champ_sel.get("selection_rationale", "")),
            "all_candidates": champ_sel.get("all_candidates", []),
        },
        "final_test": {
            "test_metrics": metrics_comp.get("final_test_metrics", manifest.get("test_metrics", {})),
            "test_rows": p_info.get("test_rows", test_eval.get("test_rows", 0)),
            "is_isolated": True,
            "test_sha256": p_info.get("test_sha256", test_eval.get("test_sha256", "")),
        },
        "importance_entries": perm_imp.get("importance_entries", []),
        "plots": plot_urls,
        "reports": {
            "markdown_url": f"/api/artifacts/{run_dir.name}/report/markdown",
            "json_url": f"/api/artifacts/{run_dir.name}/report/json",
        },
        "features": manifest.get("feature_names", []),
        "feature_types": manifest.get("feature_types", {}),
        "categorical_options": categorical_options,
        "class_labels": manifest.get("class_labels"),
        "sample_test_rows": sample_rows,
    }


def _run_m5_worker(run_id: str, df: pd.DataFrame, target_column: str, out_dir: Path, time_limit: float, presets: str, dataset_name: str):
    """Background worker function executing M5PipelineRunner."""
    with RUNS_LOCK:
        ACTIVE_RUNS[run_id]["status"] = "running"
        ACTIVE_RUNS[run_id]["stage"] = "Executing 3-Way Partition, Candidates & Final Test..."

    try:
        stage_dir = PROJECT_ROOT / "scratch" / f"stage_{run_id}"
        m5_res = M5PipelineRunner.run(
            data_source=df,
            target_column=target_column,
            output_dir=out_dir,
            time_limit_seconds=time_limit,
            presets=presets,
            include_baselines=True,
            stage_dir=stage_dir,
            dataset_name=dataset_name,
            random_state=42,
        )

        # Build sample test rows from df_test
        df_test = m5_res.get("df_test")
        sample_rows = []
        if df_test is not None and not df_test.empty:
            X_test_sample = df_test.drop(columns=[target_column]).head(5)
            sample_rows = X_test_sample.fillna(0).to_dict(orient="records")

        with RUNS_LOCK:
            ACTIVE_RUNS[run_id]["status"] = "completed"
            ACTIVE_RUNS[run_id]["stage"] = "Completed"
            ACTIVE_RUNS[run_id]["m5_result"] = m5_res
            ACTIVE_RUNS[run_id]["sample_test_rows"] = sample_rows
            ACTIVE_RUNS[run_id]["end_time"] = time.time()

    except Exception as e:
        logger.exception(f"M5 execution failed for run {run_id}: {e}")
        with RUNS_LOCK:
            ACTIVE_RUNS[run_id]["status"] = "failed"
            ACTIVE_RUNS[run_id]["stage"] = "Failed"
            ACTIVE_RUNS[run_id]["error_message"] = str(e)
            ACTIVE_RUNS[run_id]["end_time"] = time.time()


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main Judge-Ready Dashboard UI."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h3>Universal ML Engine UI: index.html not yet initialized.</h3>", status_code=200)
    return FileResponse(str(index_path))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Universal ML Engine Judge UI",
        "version": "1.0.0",
        "project_root": str(PROJECT_ROOT),
    }


@app.get("/api/runs")
async def list_runs():
    """Returns all available persisted and active runs."""
    runs = []
    # 1. Check directories in ARTIFACTS_DIR
    if ARTIFACTS_DIR.exists():
        for item in sorted(ARTIFACTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if item.is_dir() and (item / "champion_manifest.json").exists():
                try:
                    with open(item / "champion_manifest.json", "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    runs.append({
                        "run_id": item.name,
                        "model_name": manifest.get("model_name", "Unknown"),
                        "backend_name": manifest.get("backend_name", "unknown"),
                        "problem_type": manifest.get("problem_type", "unknown"),
                        "target_column": manifest.get("target_column", ""),
                        "primary_metric": manifest.get("primary_metric", ""),
                        "created_at": manifest.get("created_at", ""),
                        "is_persisted": True,
                    })
                except Exception:
                    pass

    return {"runs": runs}


class LoadSampleRequest(BaseModel):
    filename: str


@app.post("/api/load-sample")
async def load_sample_dataset(req: LoadSampleRequest):
    """Loads a pre-existing root benchmark dataset (e.g. employee_turnover.csv or HousePricePrediction.csv)."""
    target_path = PROJECT_ROOT / req.filename
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample dataset '{req.filename}' not found at {target_path}")

    try:
        df = IngestionService.load_dataset(target_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to ingest sample dataset: {str(e)}")

    upload_id = f"up_sample_{uuid.uuid4().hex[:8]}"
    inspection = inspect_dataframe(df, req.filename)
    UPLOADED_DATASETS[upload_id] = {
        "upload_id": upload_id,
        "filename": req.filename,
        "path": str(target_path),
        "df": df,
        "inspection": inspection,
    }
    return {"upload_id": upload_id, **inspection}


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Receives tabular CSV/Excel file, saves to scratch, and returns dataset inspection summary.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename specified.")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Only CSV, XLSX, and XLS files are supported.",
        )

    content = await file.read()
    if not content or len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    upload_id = f"up_{uuid.uuid4().hex[:8]}"
    upload_subdir = UPLOADS_DIR / upload_id
    upload_subdir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_subdir / file.filename

    with open(saved_path, "wb") as f:
        f.write(content)

    try:
        df = IngestionService.load_dataset(saved_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to ingest dataset: {str(e)}")

    if df.empty or len(df.columns) == 0:
        raise HTTPException(
            status_code=400,
            detail="Ingested dataset has 0 rows or 0 columns. Valid tabular data required.",
        )

    # Save to memory store
    inspection = inspect_dataframe(df, file.filename)
    UPLOADED_DATASETS[upload_id] = {
        "upload_id": upload_id,
        "filename": file.filename,
        "path": str(saved_path),
        "df": df,
        "inspection": inspection,
    }

    return {"upload_id": upload_id, **inspection}


@app.post("/api/detect-target")
async def detect_target(req: DetectTargetRequest):
    """
    Validates target column and detects problem type (Classification vs Regression).
    """
    upload_info = UPLOADED_DATASETS.get(req.upload_id)
    if not upload_info:
        raise HTTPException(status_code=404, detail=f"Upload session '{req.upload_id}' not found.")

    df: pd.DataFrame = upload_info["df"]
    if req.target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{req.target_column}' does not exist in dataset.",
        )

    target_s = df[req.target_column].dropna()
    if target_s.empty:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{req.target_column}' contains only missing (null) values.",
        )

    if target_s.nunique() <= 1:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{req.target_column}' contains only 1 unique value ({target_s.iloc[0]}). Cannot train model.",
        )

    pd_res = ProblemDetector.detect(df, req.target_column)
    return {
        "target_column": req.target_column,
        "problem_type": pd_res.problem_type.value,
        "confidence": round(pd_res.confidence, 2),
        "reason": pd_res.reason,
        "unique_target_values": pd_res.unique_target_values,
        "class_distribution": pd_res.class_distribution,
    }


@app.post("/api/run")
async def run_engine(req: RunEngineRequest):
    """
    Initiates full end-to-end M5 Engine run on the uploaded dataset.
    """
    upload_info = UPLOADED_DATASETS.get(req.upload_id)
    if not upload_info:
        raise HTTPException(status_code=404, detail=f"Upload session '{req.upload_id}' not found.")

    df: pd.DataFrame = upload_info["df"]
    if req.target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{req.target_column}' does not exist in dataset.",
        )

    target_s = df[req.target_column].dropna()
    if target_s.empty:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{req.target_column}' contains only missing (null) values.",
        )

    if target_s.nunique() <= 1:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{req.target_column}' contains only 1 unique value. Cannot train model.",
        )

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    out_dir = ARTIFACTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with RUNS_LOCK:
        ACTIVE_RUNS[run_id] = {
            "run_id": run_id,
            "upload_id": req.upload_id,
            "dataset_name": upload_info["filename"],
            "target_column": req.target_column,
            "status": "pending",
            "stage": "Initializing...",
            "start_time": time.time(),
            "end_time": None,
            "out_dir": str(out_dir),
            "error_message": None,
        }

    # Launch background thread
    worker_t = threading.Thread(
        target=_run_m5_worker,
        args=(run_id, df, req.target_column, out_dir, req.time_limit_seconds, req.presets, upload_info["filename"]),
        daemon=True,
    )
    worker_t.start()

    return {"run_id": run_id, "status": "running", "output_dir": str(out_dir)}


@app.get("/api/status/{run_id}")
async def get_run_status(run_id: str):
    """
    Returns live execution status and duration of a run.
    """
    with RUNS_LOCK:
        run_info = ACTIVE_RUNS.get(run_id)

    if not run_info:
        # Check if it exists on disk as completed
        direct_path = ARTIFACTS_DIR / run_id
        if direct_path.exists() and (direct_path / "champion_manifest.json").exists():
            return {"run_id": run_id, "status": "completed", "stage": "Persisted on Disk", "duration_seconds": 0.0}
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    start = run_info.get("start_time", time.time())
    end = run_info.get("end_time") or time.time()
    duration = round(end - start, 2)

    return {
        "run_id": run_id,
        "status": run_info["status"],
        "stage": run_info.get("stage", "Running"),
        "duration_seconds": duration,
        "error_message": run_info.get("error_message"),
    }


@app.get("/api/results/{run_id}")
async def get_run_results(run_id: str):
    """
    Returns complete verified results, champion details, validation & final test metrics, plots, and reports.
    """
    # 1. If in-memory completed run with m5_result:
    with RUNS_LOCK:
        active_info = ACTIVE_RUNS.get(run_id)

    if active_info and active_info.get("status") == "completed" and "m5_result" in active_info:
        m5_res = active_info["m5_result"]
        out_path = Path(active_info["out_dir"])
        results = load_result_from_disk(out_path)
        if active_info.get("sample_test_rows"):
            results["sample_test_rows"] = active_info["sample_test_rows"]
        return results

    # 2. Check disk for persisted run
    try:
        run_dir = get_run_dir(run_id)
        results = load_result_from_disk(run_dir)
        return results
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not load results for run '{run_id}': {str(e)}")


@app.get("/api/artifacts/{run_id}/plots/{plot_name}")
async def serve_plot(run_id: str, plot_name: str):
    """Serves high-resolution diagnostic plot PNGs."""
    run_dir = get_run_dir(run_id)
    plot_file = run_dir / "plots" / plot_name
    if not plot_file.exists():
        # Fallback to direct file in run_dir
        plot_file = run_dir / plot_name
        if not plot_file.exists():
            raise HTTPException(status_code=404, detail=f"Plot '{plot_name}' not found.")
    return FileResponse(str(plot_file), media_type="image/png")


@app.get("/api/artifacts/{run_id}/report/markdown")
async def download_markdown_report(run_id: str):
    """Serves human-readable Markdown report."""
    run_dir = get_run_dir(run_id)
    md_file = run_dir / "m5_report.md"
    if not md_file.exists():
        raise HTTPException(status_code=404, detail="Markdown report not found.")
    return FileResponse(str(md_file), media_type="text/markdown", filename=f"{run_id}_report.md")


@app.get("/api/artifacts/{run_id}/report/json")
async def download_json_report(run_id: str):
    """Serves machine-readable JSON report."""
    run_dir = get_run_dir(run_id)
    json_file = run_dir / "m5_report.json"
    if not json_file.exists():
        raise HTTPException(status_code=404, detail="JSON report not found.")
    return FileResponse(str(json_file), media_type="application/json", filename=f"{run_id}_report.json")


@app.post("/api/predict")
async def predict_with_champion(req: PredictRequest):
    """
    Generates real-time inference using the persisted M4 champion reloaded via UnifiedPredictor.
    Strictly reuses UnifiedPredictor.
    """
    try:
        run_dir = get_run_dir(req.run_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        predictor = UnifiedPredictor.load(run_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load champion via UnifiedPredictor: {e}")

    start_infer = time.time()
    try:
        df_input = pd.DataFrame([req.features])

        # Fill any missing feature columns with default 0 if not provided
        for feat in predictor.feature_names:
            if feat not in df_input.columns:
                df_input[feat] = 0

        preds = predictor.predict(df_input)
        raw_pred = preds[0]
        # Convert numpy types to native Python
        if isinstance(raw_pred, (np.integer, np.floating)):
            clean_pred = raw_pred.item()
        elif isinstance(raw_pred, str) and (raw_pred.isdigit() or (raw_pred.startswith("-") and raw_pred[1:].isdigit())):
            clean_pred = int(raw_pred)
        else:
            try:
                if isinstance(raw_pred, (int, float)):
                    clean_pred = raw_pred
                else:
                    clean_pred = int(raw_pred)
            except (ValueError, TypeError):
                clean_pred = str(raw_pred)

        probabilities = None
        if predictor.problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            try:
                probs = predictor.predict_proba(df_input)
                if probs is not None:
                    probabilities = probs[0].tolist()
            except Exception as e:
                logger.warning(f"predict_proba skipped or failed: {e}")

        latency_ms = round((time.time() - start_infer) * 1000.0, 2)

        return {
            "run_id": req.run_id,
            "prediction": clean_pred,
            "probabilities": probabilities,
            "class_labels": predictor.class_labels,
            "problem_type": predictor.problem_type.value,
            "target_column": predictor.target_column,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        logger.exception(f"Inference prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.ui.server:app", host="127.0.0.1", port=8000, reload=False)
