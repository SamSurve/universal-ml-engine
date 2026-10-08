# Universal ML Engine — Python Environment Configuration

> **Milestone:** M2.5 — Isolated Backend Environments Setup  
> **Status:** Configured & Verified  
> **Host Platform:** Windows 11 (AMD64 / x86_64)

---

## 1. Environment Topology & Interpreters

| Role | Environment Path | Python Executable | Python Version | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Main Engine & Development** | System Python 3.12 | `C:\Users\surve\AppData\Local\Programs\Python\Python312\python.exe` | `3.12.7` | Healthy (Active) |
| **AutoGluon Worker** | Isolated venv | `E:\ML MODEL\.venvs\autogluon\Scripts\python.exe` | `3.12.7` | Healthy (Isolated) |
| **MLJAR Worker (Future)** | Isolated venv | `E:\ML MODEL\.venvs\mljar\Scripts\python.exe` | `3.12.7` | Healthy (Isolated) |

---

## 2. Environment Details & Dependency State

### 2.1 Main Engine / Orchestrator Environment
* **Purpose:** Runs the main FastAPI web service, headless AutoML orchestrator, native boosters, FLAML screening, Optuna HPO, SHAP explainability, data preprocessing, and test suite.
* **Interpreter Path:** `C:\Users\surve\AppData\Local\Programs\Python\Python312\python.exe`
* **Python Version:** `3.12.7`
* **Pip Version:** `26.2.1`
* **Core Installed Packages:**
  * `scikit-learn`: `1.9.1`
  * `pandas`: `3.0.6`
  * `numpy`: `2.5.3`
  * `scipy`: `1.18.1`
  * `joblib`: `1.6.0`
  * `xgboost`: `3.4.1`
  * `lightgbm`: `4.7.0`
  * `catboost`: `1.2.10`
  * `flaml`: `2.7.0`
  * `optuna`: `5.0.0`
  * `shap`: `0.52.0`
  * `psutil`: `7.2.2`
  * `pytest`: `9.1.1`
  * `fastapi`: `0.141.1`
  * `uvicorn`: `0.54.0`
  * `openpyxl`: `3.1.5`
  * `xlrd`: `2.0.2`

### 2.2 AutoGluon Backend Worker Environment
* **Purpose:** Dedicated isolated environment for AutoGluon Tabular subprocess worker execution (Milestone M3).
* **Environment Directory:** `E:\ML MODEL\.venvs\autogluon`
* **Interpreter Path:** `E:\ML MODEL\.venvs\autogluon\Scripts\python.exe`
* **Python Version:** `3.12.7`
* **Pip Version:** `24.2`
* **State:** Base environment initialized. Full stack to be installed in M3.

### 2.3 MLJAR Backend Worker Environment
* **Purpose:** Dedicated isolated environment for MLJAR Supervised subprocess worker execution (Milestone M6).
* **Environment Directory:** `E:\ML MODEL\.venvs\mljar`
* **Interpreter Path:** `E:\ML MODEL\.venvs\mljar\Scripts\python.exe`
* **Python Version:** `3.12.7`
* **Pip Version:** `24.2`
* **State:** Base environment initialized. Full stack to be installed in M6.

---

## 3. Verified Verification Commands

### Main Environment Verification
```powershell
& "C:\Users\surve\AppData\Local\Programs\Python\Python312\python.exe" --version
# Output: Python 3.12.7

& "C:\Users\surve\AppData\Local\Programs\Python\Python312\python.exe" -m pip --version
# Output: pip 26.2.1 from C:\Users\surve\AppData\Local\Programs\Python\Python312\Lib\site-packages\pip (python 3.12)

& "C:\Users\surve\AppData\Local\Programs\Python\Python312\python.exe" -m pytest backend/tests/ -q
# Output: 60 passed (subprocess isolation, baselines, data splitting, profiler, orchestrator, shap, cv, hpo)
```

### Isolated Backend Environments Verification
```powershell
& "E:\ML MODEL\.venvs\autogluon\Scripts\python.exe" --version
# Output: Python 3.12.7

& "E:\ML MODEL\.venvs\mljar\Scripts\python.exe" --version
# Output: Python 3.12.7
```

---

## 4. Git Hygiene
* `.venvs/` is explicitly registered in `.gitignore` to prevent virtual environment binaries, caches, and installed packages from being committed.
