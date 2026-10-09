"""
Standalone Verification Script for Universal ML Engine Web UI & Inference Pipeline.
Runs directly with standard Python (no pytest required).
"""
import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.ui.server import app

def run_all_checks():
    client = TestClient(app)
    passed = 0
    total = 6

    print("=" * 70)
    print("UNIVERSAL ML ENGINE — VERIFICATION SUITE")
    print("=" * 70)

    # Check 1: Health & Root
    print("[1/6] Checking /api/health and root HTML...")
    h_resp = client.get("/api/health")
    r_resp = client.get("/")
    assert h_resp.status_code == 200, f"Health check failed: {h_resp.status_code}"
    assert "healthy" in h_resp.json().get("status", ""), "Invalid health status"
    assert r_resp.status_code == 200, f"Root route failed: {r_resp.status_code}"
    assert "Universal ML Engine" in r_resp.text, "Title not found in HTML"
    assert "Thin Judge-Ready Interface" not in r_resp.text, "Disallowed subtitle present"
    print("      [OK] PASSED: Server healthy, white theme index.html served without disallowed subtitle.")
    passed += 1

    # Check 2: Sample Dataset Target Identification
    print("[2/6] Checking target detection on benchmark datasets...")
    t_resp = client.post("/api/load-sample", json={"filename": "employee_turnover.csv"})
    assert t_resp.status_code == 200, f"Failed to load turnover: {t_resp.text}"
    t_data = t_resp.json()
    assert t_data["suggested_target"] == "Employee_Turnover", f"Wrong turnover target: {t_data.get('suggested_target')}"
    assert t_data["detected_problem"]["problem_type"] == "binary_classification", "Wrong problem type for turnover"

    h_resp = client.post("/api/load-sample", json={"filename": "HousePricePrediction.csv"})
    assert h_resp.status_code == 200, f"Failed to load housing: {h_resp.text}"
    h_data = h_resp.json()
    assert h_data["suggested_target"] == "SalePrice", f"Wrong housing target: {h_data.get('suggested_target')} (Must NOT be MSSubClass)"
    assert h_data["detected_problem"]["problem_type"] == "regression", "Wrong problem type for housing"
    print("      [OK] PASSED: employee_turnover -> Employee_Turnover (Classification), HousePrice -> SalePrice (Regression).")
    passed += 1

    # Check 3: Verified Classification Run Results
    print("[3/6] Checking classification results (m5_verified_classification)...")
    clf_res = client.get("/api/results/m5_verified_classification")
    assert clf_res.status_code == 200, f"Failed to get clf results: {clf_res.text}"
    clf_d = clf_res.json()
    assert clf_d["problem_type"] == "binary_classification"
    assert clf_d["champion"]["val_metrics"]["f1"] == 0.8831
    assert clf_d["final_test"]["test_metrics"]["f1"] == 0.8667
    assert len(clf_d["sample_test_rows"]) > 0, "Missing sample_test_rows in classification results"
    print(f"      [OK] PASSED: Champion '{clf_d['champion']['model_name']}' loaded. Val F1: {clf_d['champion']['val_metrics']['f1']} | Test F1: {clf_d['final_test']['test_metrics']['f1']}.")
    passed += 1

    # Check 4: Verified Regression Run Results
    print("[4/6] Checking regression results (m5_verified_regression)...")
    reg_res = client.get("/api/results/m5_verified_regression")
    assert reg_res.status_code == 200, f"Failed to get reg results: {reg_res.text}"
    reg_d = reg_res.json()
    assert reg_d["problem_type"] == "regression"
    assert reg_d["champion"]["val_metrics"]["r2"] == 0.7236
    assert reg_d["final_test"]["test_metrics"]["r2"] == 0.8338
    assert len(reg_d["sample_test_rows"]) > 0, "Missing sample_test_rows in regression results"
    assert "categorical_options" in reg_d, "Missing categorical_options in regression results"
    print(f"      [OK] PASSED: Champion '{reg_d['champion']['model_name']}' loaded. Val R2: {reg_d['champion']['val_metrics']['r2']} | Test R2: {reg_d['final_test']['test_metrics']['r2']}.")
    passed += 1

    # Check 5: Prediction Playground Classification Inference
    print("[5/6] Checking live prediction on Classification Champion...")
    clf_sample = clf_d["sample_test_rows"][0]
    clf_feat = {k: v for k, v in clf_sample.items() if not k.startswith("_")}
    clf_pred_resp = client.post("/api/predict", json={"run_id": "m5_verified_classification", "features": clf_feat})
    assert clf_pred_resp.status_code == 200, f"Classification prediction failed: {clf_pred_resp.text}"
    clf_pred_d = clf_pred_resp.json()
    assert "prediction" in clf_pred_d
    assert "probabilities" in clf_pred_d and len(clf_pred_d["probabilities"]) == 2
    print(f"      [OK] PASSED: Predicted Class: {clf_pred_d['prediction']} | Probabilities: {clf_pred_d['probabilities']} | Latency: {clf_pred_d['latency_ms']}ms.")
    passed += 1

    # Check 6: Prediction Playground Regression Inference
    print("[6/6] Checking live prediction on Regression Champion...")
    reg_sample = reg_d["sample_test_rows"][0]
    reg_feat = {k: v for k, v in reg_sample.items() if not k.startswith("_")}
    reg_pred_resp = client.post("/api/predict", json={"run_id": "m5_verified_regression", "features": reg_feat})
    assert reg_pred_resp.status_code == 200, f"Regression prediction failed: {reg_pred_resp.text}"
    reg_pred_d = reg_pred_resp.json()
    assert "prediction" in reg_pred_d
    assert float(reg_pred_d["prediction"]) > 0
    print(f"      [OK] PASSED: Predicted SalePrice: {float(reg_pred_d['prediction']):,.2f} | Latency: {reg_pred_d['latency_ms']}ms.")
    passed += 1

    print("=" * 70)
    print(f"VERIFICATION SUMMARY: {passed}/{total} CHECKS PASSED (100%)")
    print("=" * 70)

if __name__ == "__main__":
    run_all_checks()
