# Enterprise MLOps Tooling & Infrastructure

While Data Engineering (PySpark/DuckDB) prepares the data, the true scalability of StoreCast comes from its strict adherence to modern **MLOps** standards. Building a Machine Learning system is vastly different from building a web app; code, data, and models all iterate independently.

StoreCast utilizes a powerful trio of open-source tooling to manage this complexity reliably: **DVC**, **DagsHub**, and **MLflow** — all orchestrated through a centralized `ConfigManager` YAML system.

## 1. DVC & DagsHub (Data Version Control)
Git is fundamentally designed for source code. Committing a 2GB `.parquet` file or `Delta Lake` directory to a repository instantly breaks it. 
To solve this, StoreCast utilizes **Data Version Control (DVC)**.

- **How it Works:** DVC functions exactly like Git, but engineered for big data. It hashes our massive Bronze, Silver, and Gold datasets, replaces them with tiny `.dvc` pointer files in our IDE, and commits *those pointers* into our normal Git flow. 
- **The Remote Backend:** Instead of purchasing an expensive AWS S3 bucket for the backend storage, we securely push the actual heavily-partitioned data payloads to **DagsHub** via `dvc push`. DagsHub acts as the remote "GitHub for Data," natively hosting our DVC storage for free.
- **The Result:** Anyone pulling the StoreCast repository simply types `dvc pull`, and the entire 2.4B row Medallion Lakehouse automatically materializes on their local machine, matched perfectly to the current branch of source code.

## 3. Bayesian Optimization (Optuna)

Instead of relying on naive `GridSearch` or random permutations, StoreCast leverages **Optuna** for Bayesian Hyperparameter tracking.
Optuna mathematical convergence is vastly more efficient because it uses prior trial history to infer where the multi-dimensional global minimum exists. In our Phase 4 tuning, the search natively plateaued by Trial 15, drastically saving Cloud compute constraints. 

Furthermore, we explicitly rely on Optuna's **fANOVA (Functional Analysis of Variance)** visualization module to mathematically isolate feature dominance. For instance, fANOVA proved that `max_depth` had a negligible 2% objective impact compared to `learning_rate` (38%), scientifically proving that our models resist overfitting natively.

## 4. MLflow (Experiment & Model Tracking)

In standard data science workflows, a data scientist might run 50 different Random Forest combinations, overwriting variables each time. Two weeks later, no one remembers which hyperparameters produced the baseline WMAPE.

StoreCast eliminates this via a **Remote MLflow Tracking Server**.

![](https://mlflow.org/docs/latest/_images/scenario_1.png)

### The Tracking Strategy
During optimization, our ML training scripts (`src/training/optimizer.py`) are wrapped in `mlflow.start_run()`. 
Every single time the code runs, it automatically logs the entire lifecycle of the experiment directly to a **DagsHub remote MLflow SaaS interceptor**:
1. **Hyperparameters:** Automatically captures `n_estimators`, `max_depth`, and tree logic.
2. **Custom Metrics:** We explicitly log our 5x-weighted `WMAPE` and `WMAE` metrics, mapping ML performance directly to the financial KPIs the business cares about.
3. **Artifacts:** MLflow implicitly serializes and saves the actual finalized `.pkl` or `.json` model artifact using strict `infer_signature` inputs, ensuring that every deployment handles Pandas missing-value arrays predictably.
4. **The UI:** Because we mapped our URI via `dagshub.init` (credentials pulled from `config/params.yaml`), viewing the dashboard doesn't require spinning up an isolated AWS EC2 host. The run histories, parameters, and fANOVA PNG plots are hosted securely in the public cloud.

## 5. The Deployment Orchestrator (Multi-Objective Pareto Tie-Breaking)

A junior architecture blindly extracts the single trace with the lowest WMAPE error. StoreCast utilizes **Multi-Objective Pareto Tie-Breaking** to secure absolute enterprise scale.

In our decoupled `CandidateStager` class (`src/deployment/stage_candidate.py`), the algorithm mathematically rejects single-variable decisions. The MLflow API queries the Top 10 models that successfully passed the strict `WMAPE < 8.5` mathematical boundary. We then dynamically apply a **Production Score** weighting system directly onto the traces:
- **70% Weight:** Absolute Error (`WMAPE_Val`)
- **20% Weight:** Terminal API Speed (`Latency_ms`) 
- **10% Weight:** Kubernetes Ram Scale (`Model_Size_MB`)

This structurally prevents a massive 90-millisecond latency model from winning just because it was 0.01% more accurate mathematically. The true Champion is the trace that minimizes the globally unified financial production score, which is then staged as `@candidate` for Human-in-the-Loop review before being promoted to `@production`.

## 6. Centralized Configuration Management

StoreCast externalizes **all runtime parameters** — data paths, feature lists, split quantiles, hyperparameters, quality gates, experiment names, and observability thresholds — into a single `config/params.yaml` file.

A lightweight `ConfigManager` class (`src/utils/config_manager.py`) provides dot-notation access:
```python
cfg = ConfigManager()
gold_path = cfg.get("data.paths.gold_data")        # → "data/gold/gold_sales.parquet"
features = cfg.get("data.features.numeric")          # → ["store_size", "temperature", ...]
max_wmape = cfg.get("deployment.quality_gates.max_wmape")  # → 8.5
```

This separation of configuration from code is a non-negotiable prerequisite for:
- **CI/CD:** GitHub Actions can override params without touching Python.
- **DVC Reproducibility:** `dvc repro` detects YAML parameter changes and selectively re-runs affected stages.
- **Environment Portability:** Dev, staging, and production environments share the same code but use different YAML files.
