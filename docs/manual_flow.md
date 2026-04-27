# this is temporary file we will  move this to index.md in the main flow diagram

This is the smartest thing you can do right now. Before writing a single line of Airflow or GitHub Actions code, you should always be able to run the entire Directed Acyclic Graph (DAG) manually in your terminal. This proves that your state management and decoupling actually work.

To simulate the orchestrator, you will run these scripts sequentially. Before you start, open your `config/params.yaml` and set `n_trials_optuna: 2`. This is a classic CI/CD trick: we just want to prove the pipes connect, we don't want to wait 2 hours for a real training run.

Here is the exact step-by-step flow an orchestrator will execute:

### Step 1: The Data Pipeline (The Feeder)
The orchestrator triggers your data pipeline to generate fresh, immutable data artifacts.
```bash
dvc repro
```
* **What happens:** DVC runs `ingest_bronze`, `clean_silver`, `create_gold`, and `chronological_split`.
* **The State Change:** Fresh `train.parquet`, `val.parquet`, and `test.parquet` files now sit on your disk, ready for the ML scripts.

### Step 2: Hyperparameter Tuning (The Explorer)
The orchestrator triggers the optimizer to find the best mathematical configuration for this specific snapshot of data.
```bash
python -m src.training.optimizer
```
* **What happens:** The `HyperparameterOptimizer` class wakes up, reads the YAML config, loads the parquet files, and runs 2 Optuna trials. 
* **The State Change:** MLflow is updated with a new `StoreCast_XGB_Optimization_V2` experiment containing the trial metrics.

### Step 3: Candidate Staging (The Builder)
The orchestrator immediately triggers the staging script to build the final artifact.
```bash
python -m src.deployment.stage_candidate
```
* **What happens:** The `CandidateStager` queries MLflow, finds the winning trial from Step 2, concatenates `train + val` data, trains a final pipeline, evaluates it on the holdout `test` set, and registers it to the DagsHub MLflow Registry.
* **The State Change:** A new model version exists in the cloud registry, explicitly tagged with the alias **`@candidate`**.

explainability.py By the time you wake up and get the notification to run promote_champion.py, you can just open the docs/images/shap/ folder, look at the beautiful global importance charts for the new model, and confidently type "yes".

### Step 4: The Human Gate (The Approver)
The CI/CD pipeline **STOPS** here. It sends an alert (Slack, Email, etc.) saying: *"Hey, a new model candidate is ready for review."*

You log into DagsHub, review the WMAPE and SHAP plots. If it looks good, you (the human) simulate the approval by running:
```bash
python -m src.deployment.promote_champion
```
* **What happens:** You type "yes" in the terminal.
* **The State Change:** The MLflow alias changes. The model is now officially **`@production`**.

### Step 5: Batch Inference (The Consumer)
Later that night (or on a weekly Airflow cron schedule), the inference job runs.
```bash
python -m src.deployment.batch_inference
```
* **What happens:** The `BatchInferencer` wakes up, downloads the shiny new `@production` model, grabs the latest 1,000 rows from the Gold layer, generates predictions, and saves them to a CSV for the dashboard.

---

### The Test
Run those 5 commands in your terminal right now (with `n_trials: 2`). If you can get from Step 1 to Step 5 without editing any Python code in between, **you have successfully built a production-grade ML system.**