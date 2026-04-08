# Enterprise MLOps Tooling & Infrastructure

While Data Engineering (PySpark/DuckDB) prepares the data, the true scalability of StoreCast comes from its strict adherence to modern **MLOps** standards. Building an Machine Learning system is vastly different from building a web app; code, data, and models all iterate independently.

StoreCast utilizes a powerful trio of open-source tooling to manage this complexity reliably: **DVC**, **DagsHub**, and **MLflow**.

## 1. DVC & DagsHub (Data Version Control)
Git is fundamentally designed for source code. Committing a 2GB `.parquet` file or `Delta Lake` directory to a repository instantly breaks it. 
To solve this, StoreCast utilizes **Data Version Control (DVC)**.

- **How it Works:** DVC functions exactly like Git, but engineered for big data. It hashes our massive Bronze, Silver, and Gold datasets, replaces them with tiny `.dvc` pointer files in our IDE, and commits *those pointers* into our normal Git flow. 
- **The Remote Backend:** Instead of purchasing an expensive AWS S3 bucket for the backend storage, we securely push the actual heavily-partitioned data payloads to **DagsHub** via `dvc push`. DagsHub acts as the remote "GitHub for Data," natively hosting our DVC storage for free.
- **The Result:** Anyone pulling the StoreCast repository simply types `dvc pull`, and the entire 2.4B row Medallion Lakehouse automatically materializes on their local machine, matched perfectly to the current branch of source code.

## 2. MLflow (Experiment & Model Tracking)

In standard data science workflows (the "Jupyter Notebook disaster"), a data scientist might run 50 different Random Forest combinations, overwriting variables each time. Two weeks later, no one remembers which hyperparameters produced the 8.49% baseline WMAPE.

StoreCast eliminates this via a local **MLflow Tracking Server**.

![](https://mlflow.org/docs/latest/_images/scenario_1.png)

### The Tracking Strategy
During Phase 4 execution, our ML training scripts (`src/models/train_mlflow.py`) are wrapped in `mlflow.start_run()`. 
Every single time the code runs, it automatically logs the entire lifecycle of the experiment directly into `mlruns/` (or a remote server):
1. **Hyperparameters:** Automatically captures `n_estimators`, `max_depth`, and tree logic.
2. **Custom Metrics:** We explicitly log our 5x-weighted `WMAPE` and `WMAE` metrics, mapping ML performance directly to the financial KPIs the business cares about.
3. **Artifacts:** MLflow implicitly serializes and saves the actual finalized `.pkl` or `.json` model artifact, ensuring that every deployment is perfectly reproducible and rollback-safe.
4. **The UI:** Running `mlflow ui` instantly spins up an interactive web dashboard (typically on `localhost:5000`) for visual comparison of all historic training geometries.
