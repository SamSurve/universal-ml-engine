import os
import sys
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ui.server import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_1_app_starts_successfully(client):
    """Test 1: App starts successfully and health endpoint responds."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "Universal ML Engine" in data["service"]

    # Also test index.html root route
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "Universal ML Engine" in root_resp.text
    assert "Thin Judge-Ready Interface" not in root_resp.text
    assert "Verified M4/M5 Architecture" not in root_resp.text


def test_2_csv_uploads_successfully(client):
    """Test 2: CSV uploads successfully and returns schema & missing value summary."""
    turnover_path = PROJECT_ROOT / "employee_turnover.csv"
    assert turnover_path.exists(), "employee_turnover.csv must exist"

    with open(turnover_path, "rb") as f:
        resp = client.post(
            "/api/upload",
            files={"file": ("employee_turnover.csv", f, "text/csv")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "upload_id" in data
    assert data["row_count"] > 0
    assert data["column_count"] > 0
    assert "missing_cells" in data
    assert "columns" in data
    assert len(data["columns"]) == data["column_count"]
    assert "preview_rows" in data
    assert len(data["preview_rows"]) <= 5
    assert data["suggested_target"] == "Employee_Turnover"


def test_3_target_selection_and_problem_detection(client):
    """Test 3: Target selection works and correctly detects ProblemType."""
    # Test on turnover dataset
    turnover_path = PROJECT_ROOT / "employee_turnover.csv"
    with open(turnover_path, "rb") as f:
        upload_resp = client.post(
            "/api/upload",
            files={"file": ("employee_turnover.csv", f, "text/csv")},
        )
    up_id_clf = upload_resp.json()["upload_id"]

    detect_resp_clf = client.post(
        "/api/detect-target",
        json={"upload_id": up_id_clf, "target_column": "Employee_Turnover"},
    )
    assert detect_resp_clf.status_code == 200
    data_clf = detect_resp_clf.json()
    assert data_clf["problem_type"] == "binary_classification"
    assert data_clf["confidence"] >= 0.95
    assert data_clf["unique_target_values"] == 2

    # Test on housing dataset
    house_path = PROJECT_ROOT / "HousePricePrediction.csv"
    assert house_path.exists(), "HousePricePrediction.csv must exist"
    with open(house_path, "rb") as f:
        upload_resp_reg = client.post(
            "/api/upload",
            files={"file": ("HousePricePrediction.csv", f, "text/csv")},
        )
    up_id_reg = upload_resp_reg.json()["upload_id"]

    detect_resp_reg = client.post(
        "/api/detect-target",
        json={"upload_id": up_id_reg, "target_column": "SalePrice"},
    )
    assert detect_resp_reg.status_code == 200
    data_reg = detect_resp_reg.json()
    assert data_reg["problem_type"] == "regression"
    assert data_reg["confidence"] >= 0.90


def test_4_and_6_classification_results_and_metrics(client):
    """Test 4 & 6: Classification results show genuine metrics with strict separation of Val vs Test."""
    # Load verified classification run
    resp = client.get("/api/results/m5_verified_classification")
    assert resp.status_code == 200
    data = resp.json()

    assert data["problem_type"] == "binary_classification"
    assert data["target_column"] == "Employee_Turnover"
    assert data["total_rows"] == 900
    assert data["partition_info"]["train_rows"] == 540
    assert data["partition_info"]["val_rows"] == 180
    assert data["partition_info"]["test_rows"] == 180

    # Champion
    assert "Logistic Regression" in data["champion"]["model_name"]
    assert data["champion"]["primary_metric"] == "f1"
    assert data["champion"]["primary_val_score"] > 0.80

    # Validation Metrics (clearly separated)
    val_metrics = data["champion"]["val_metrics"]
    assert "f1" in val_metrics
    assert "accuracy" in val_metrics
    assert "roc_auc" in val_metrics
    assert val_metrics["f1"] == 0.8831

    # Final Test Metrics (clearly separated)
    test_metrics = data["final_test"]["test_metrics"]
    assert "f1" in test_metrics
    assert "accuracy" in test_metrics
    assert data["final_test"]["is_isolated"] is True
    assert len(data["final_test"]["test_sha256"]) == 64
    assert test_metrics["f1"] == 0.8667

    # Never mix Validation and Test metrics
    assert val_metrics["f1"] != test_metrics["f1"]


def test_5_and_6_regression_results_and_metrics(client):
    """Test 5 & 6: Regression results show genuine metrics with strict separation of Val vs Test."""
    # Load verified regression run
    resp = client.get("/api/results/m5_verified_regression")
    assert resp.status_code == 200
    data = resp.json()

    assert data["problem_type"] == "regression"
    assert data["target_column"] == "SalePrice"
    assert data["total_rows"] == 1460
    assert data["partition_info"]["train_rows"] == 876
    assert data["partition_info"]["val_rows"] == 292
    assert data["partition_info"]["test_rows"] == 292

    # Champion
    assert "WeightedEnsemble_L2" in data["champion"]["model_name"]
    assert data["champion"]["backend_name"] == "autogluon"
    assert data["champion"]["primary_metric"] == "r2"

    # Validation Metrics (clearly separated)
    val_metrics = data["champion"]["val_metrics"]
    assert "r2" in val_metrics
    assert "rmse" in val_metrics
    assert val_metrics["r2"] == 0.7236

    # Final Test Metrics (clearly separated)
    test_metrics = data["final_test"]["test_metrics"]
    assert "r2" in test_metrics
    assert "rmse" in test_metrics
    assert data["final_test"]["is_isolated"] is True
    assert test_metrics["r2"] == 0.8338


def test_7_plots_served_correctly(client):
    """Test 7: Actual diagnostic plots are served as valid image/png."""
    # Classification plots: confusion_matrix.png and feature_importance.png
    cm_resp = client.get("/api/artifacts/m5_verified_classification/plots/confusion_matrix.png")
    assert cm_resp.status_code == 200
    assert cm_resp.headers["content-type"] == "image/png"
    assert len(cm_resp.content) > 1000

    fi_resp = client.get("/api/artifacts/m5_verified_classification/plots/feature_importance.png")
    assert fi_resp.status_code == 200
    assert fi_resp.headers["content-type"] == "image/png"
    assert len(fi_resp.content) > 1000

    # Regression plots: actual_vs_predicted.png, residuals.png, feature_importance.png
    avp_resp = client.get("/api/artifacts/m5_verified_regression/plots/actual_vs_predicted.png")
    assert avp_resp.status_code == 200
    assert avp_resp.headers["content-type"] == "image/png"
    assert len(avp_resp.content) > 1000

    res_resp = client.get("/api/artifacts/m5_verified_regression/plots/residuals.png")
    assert res_resp.status_code == 200
    assert res_resp.headers["content-type"] == "image/png"
    assert len(res_resp.content) > 1000


def test_8_reports_download_work(client):
    """Test 8: Markdown and JSON report links/downloads work and return valid reports."""
    # Markdown report
    md_resp = client.get("/api/artifacts/m5_verified_classification/report/markdown")
    assert md_resp.status_code == 200
    assert "Report" in md_resp.text
    assert "Champion Selection" in md_resp.text

    # JSON report
    json_resp = client.get("/api/artifacts/m5_verified_classification/report/json")
    assert json_resp.status_code == 200
    report_json = json_resp.json()
    assert report_json["schema_version"] == "1.0.0"
    assert "dataset_overview" in report_json
    assert "champion_selection" in report_json


def test_9_persisted_champion_inference(client):
    """Test 9: Persisted champion inference works reusing UnifiedPredictor."""
    # Classification sample
    clf_payload = {
        "run_id": "m5_verified_classification",
        "features": {
            "Job_Satisfaction": 1.0,
            "Performance_Rating": 2.0,
            "Years_At_Company": 2.0,
            "Work_Life_Balance": 1.0,
            "Distance_From_Home": 25.0,
            "Monthly_Income": 3200.0,
            "Education_Level": 2.0,
            "Age": 28.0,
            "Num_Companies_Worked": 3.0,
            "Employee_Role": 1.0,
            "Annual_Bonus": 1000.0,
            "Training_Hours": 10.0,
            "Department": 1.0,
            "Annual_Bonus_Squared": 1000000.0,
            "Annual_Bonus_Training_Hours_Interaction": 10000.0,
        },
    }
    clf_resp = client.post("/api/predict", json=clf_payload)
    assert clf_resp.status_code == 200
    clf_out = clf_resp.json()
    assert "prediction" in clf_out
    assert int(clf_out["prediction"]) in [0, 1]
    assert "probabilities" in clf_out
    assert len(clf_out["probabilities"]) == 2
    assert clf_out["latency_ms"] >= 0

    # Regression sample
    reg_payload = {
        "run_id": "m5_verified_regression",
        "features": {
            "Id": 1,
            "MSSubClass": 60,
            "MSZoning": "RL",
            "LotArea": 8450,
            "LotConfig": "Inside",
            "BldgType": "1Fam",
            "OverallCond": 5,
            "YearBuilt": 2003,
            "YearRemodAdd": 2003,
            "Exterior1st": "VinylSd",
            "BsmtFinSF2": 0.0,
            "TotalBsmtSF": 856.0,
        },
    }
    reg_resp = client.post("/api/predict", json=reg_payload)
    assert reg_resp.status_code == 200
    reg_out = reg_resp.json()
    assert "prediction" in reg_out
    assert float(reg_out["prediction"]) > 0


def test_10_invalid_csv_and_missing_target_clear_errors(client):
    """Test 10: Invalid CSV / missing target gives clear, descriptive 400 errors."""
    # 1. Empty file
    resp_empty = client.post(
        "/api/upload",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert resp_empty.status_code == 400
    assert "empty" in resp_empty.json()["detail"].lower()

    # 2. Corrupted / unsupported file extension
    resp_corrupt = client.post(
        "/api/upload",
        files={"file": ("invalid.txt", b"some random text", "text/plain")},
    )
    assert resp_corrupt.status_code == 400
    assert "unsupported" in resp_corrupt.json()["detail"].lower()

    # 3. Missing target column during detection
    turnover_path = PROJECT_ROOT / "employee_turnover.csv"
    with open(turnover_path, "rb") as f:
        up_resp = client.post(
            "/api/upload",
            files={"file": ("employee_turnover.csv", f, "text/csv")},
        )
    up_id = up_resp.json()["upload_id"]

    resp_missing_target = client.post(
        "/api/detect-target",
        json={"upload_id": up_id, "target_column": "NonExistentColumn"},
    )
    assert resp_missing_target.status_code == 400
    assert "does not exist in dataset" in resp_missing_target.json()["detail"]

    # 4. Missing target during run
    resp_run_missing = client.post(
        "/api/run",
        json={"upload_id": up_id, "target_column": "NonExistentColumn"},
    )
    assert resp_run_missing.status_code == 400
    assert "does not exist in dataset" in resp_run_missing.json()["detail"]
