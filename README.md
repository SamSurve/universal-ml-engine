# Universal ML Engine

A practical, local machine-learning workbench for turning a tabular dataset into an evaluated model, useful visualizations, and a report.

The project is built with Python and a lightweight web interface. You can upload a dataset, choose what you want to predict, run the ML workflow, compare validation and final-test results, inspect important features, download reports, and make predictions with the saved model.

## What it does

- **Upload and inspect data** — work with CSV and supported spreadsheet files; review columns, data types, missing values, and sample rows.
- **Identify the task** — classification or regression, with a target column you can confirm.
- **Prepare and train** — use the existing preprocessing pipeline, baseline models, and the integrated AutoGluon Tabular backend when available.
- **Choose a model fairly** — compare candidates using validation data rather than selecting the model by its final test score.
- **Evaluate generalization** — keep final-test metrics separate from validation metrics.
- **Explain results** — generate feature-importance analysis and task-appropriate plots.
- **Save and reuse models** — persist the selected champion and make predictions through the shared inference interface.
- **Generate reports** — produce human-readable Markdown and machine-readable JSON outputs.

## How the workflow works

```text
Upload dataset
     ↓
Inspect data and choose target
     ↓
Preprocess and split the data
     ↓
Train candidate models on training data
     ↓
Select the champion using validation results
     ↓
Evaluate the selected champion on the final test set
     ↓
Feature importance, plots, reports and predictions
```

The exact training configuration and metrics depend on the dataset and the run. The reported test score is not a guarantee of performance on other datasets.

## Quick start

### Requirements

- Python 3.11 or 3.12 (64-bit recommended)
- Windows, macOS, or Linux
- Internet access for installing dependencies

### 1. Clone the repository

```bash
git clone https://github.com/SamSurve/universal-ml-engine.git
cd universal-ml-engine
```

### 2. Create an environment and install dependencies

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

If your machine does not have Python 3.12, use an installed supported version and create the environment with that interpreter.

### 3. Start the web dashboard

From the repository root:

```bash
python run_ui.py --port 8000
```

Open **http://127.0.0.1:8000** in your browser. Keep the terminal open while using the dashboard. Press `Ctrl+C` in the terminal to stop the server.

### Optional: Install AutoGluon in its isolated worker environment

The project can use other supported models when AutoGluon is unavailable. To enable the dedicated AutoGluon worker, create the environment from the repository root.

**Windows PowerShell**

```powershell
py -3.12 -m venv .venvs\autogluon
.\.venvs\autogluon\Scripts\python.exe -m pip install --upgrade pip
.\.venvs\autogluon\Scripts\python.exe -m pip install autogluon.tabular
```

**macOS / Linux**

```bash
python3 -m venv .venvs/autogluon
.venvs/autogluon/bin/python -m pip install --upgrade pip
.venvs/autogluon/bin/python -m pip install autogluon.tabular
```

AutoGluon installation support depends on the Python version and operating system. If its separate environment cannot be installed, use the supported fallback models provided by the current setup.

## Using the dashboard

1. Upload your CSV.
2. Review the dataset preview and missing-value summary.
3. Select and confirm the target column.
4. Run the analysis and wait for it to finish.
5. Review the selected model, validation metrics, and final-test metrics separately.
6. Inspect the plots and feature-importance results.
7. Download the generated report or use the Prediction Playground with the saved model.

For a binary star-versus-galaxy task, make sure the target labels and class definitions match the challenge. Do not assume every dataset uses the same target-column name.

## Important evaluation principles

- Fit data-dependent preprocessing on training data only.
- Use validation data for model selection.
- Keep the final test set out of tuning and champion selection.
- Calculate feature importance on the intended development/validation data, not as a way to tune against the final test set.
- Treat feature importance as predictive evidence, not proof of physical causation.
- Report actual metrics from the current run; never reuse example scores as if they were new results.

## Run tests

From the repository root:

```bash
python -m pytest backend/tests/ -q
```

Some tests and model runs can take several minutes, depending on the hardware and installed backends.

## Repository layout

- `backend/engine/` — data ingestion, preprocessing, models, evaluation, persistence, inference, explainability, and reporting.
- `backend/ui/` — FastAPI server and the HTML/CSS/JavaScript dashboard.
- `backend/tests/` — automated tests.
- `run_ui.py` — local web-app entry point.
- `backend/requirements.txt` — main Python dependencies.

Model artifacts, uploaded files, and run reports may be written locally. Keep private or sensitive datasets out of public repositories.

## Related project

The astronomy challenge workspace, selected datasets, and experiment-specific reports live here:

**[Astromanthan — Deep Sky Divide](https://github.com/Shreyas-84524/Astromanthan)**

That workspace contains the astronomy-specific workflow. The general-purpose engine and the astronomy experiment are related, but their model artifacts and reported metrics are not interchangeable.

## Notes

This is an evolving project. Check the current repository files and test outputs when reproducing a run. The local virtual environments are not part of the repository and must be created separately on each machine.
