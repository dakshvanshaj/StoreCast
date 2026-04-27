# StoreCast: Enterprise MLOps & Agentic Forecasting System

## Executive Summary
StoreCast is a production-grade retail analytics and forecasting system. This roadmap details the architectural evolution from raw CSVs to an automated, Human-in-the-Loop Machine Learning pipeline, culminating in a live Agentic LLM dashboard powered by modern microservices.

**Architectural Philosophy:** Built for extreme efficiency and low cloud overhead. We explicitly rejected legacy distributed frameworks (like PySpark JVMs) in favor of modern, single-node, Rust-backed execution engines (Polars) and in-process OLAP (DuckDB). Configuration is 100% YAML-driven, ensuring absolute CI/CD reproducibility.

---

## PART 1: THE FOUNDATION (DATA & MLOPS)

### Phase 1: Data Engineering & The Medallion Pipeline
- **Architecture:** Implemented a strict Bronze (Raw) -> Silver (Cleaned) -> Gold (Master) Medallion architecture.
- **Execution:** Used **Polars** for lazy-evaluated, multi-threaded data cleaning and **DuckDB** for blazing-fast SQL aggregations. 
- **Data Contracts:** Enforced schema integrity using Great Expectations (`validate_silver.py`, `validate_gold.py`).
- **Versioning:** Orchestrated the DAG using **DVC**, ensuring data artifacts are immutable before ML training begins.

### Phase 2: Experimentation & Hyperparameter Tuning
- **Tracking:** Integrated **DagsHub** as a remote MLflow tracking server and S3 artifact store.
- **Optimization:** Built an Object-Oriented `HyperparameterOptimizer` using **Optuna** for Bayesian search.
- **Metrics:** Tracked accuracy (WMAPE, R2) alongside strict hardware constraints (`Latency_ms`, `Model_Size_MB`).

### Phase 3: CI/CD & Human-in-the-Loop Deployment
- **Decoupling:** Migrated to a centralized `config/params.yaml` injected via a `ConfigManager` class.
- **Intelligent Staging:** Automated the `CandidateStager` to query MLflow, execute a **Multi-Objective Pareto Sort**, train on full historical data, and register the model as `@candidate`.
- **The Human Gate:** Built `promote_champion.py` to enforce a human-in-the-loop review of the candidate's metrics before flipping the alias to `@production`.

### Phase 4: MLOps Observability & Explainability
- **Explainability:** Built `ModelExplainer` to generate Global and Local **SHAP** plots for the `@candidate` model, exposing the XGBoost decision geometry.
- **Drift Monitoring:** Integrated **Evidently AI** to compare the live inference ledger against baseline training data, algorithmically detecting Covariate Shift.

---

## PART 2: THE SCALE & SERVE SPRINT

### Phase 5: The "100GB Out-of-Core" Drift Simulation
**Goal:** Prove scalability on resource-constrained hardware (16GB RAM) and test the automated retraining loop.
- **Execution:** Write a synthetic data generator to inflate the 2010-2013 retail dataset to 100GB+ spanning up to 2026.
- **Drift Injection:** Mathematically inject macro-economic drift (CPI inflation, Fuel spikes) into the 2025/2026 data.
- **Validation:** Use Polars Out-of-Core Streaming to process the 100GB dataset without OOM errors. Allow the injected drift to trigger the Evidently AI policy engine, forcing an automated model retrain.

### Phase 6: The Feature Store Transition (Feast)
**Goal:** Eradicate static Parquet reads for live online serving.
- Deploy **Feast** as the central feature registry.
- **Offline Store:** Map Feast to the Gold Parquet layer for historical batch training.
- **Online Store:** Sync the latest feature window to a fast SQLite/Redis online store for millisecond latency during API inference.

### Phase 7: Model Containerization & Load Testing
**Goal:** Transition from Batch CSV exports to Live Microservices.
- Wrap the `@production` XGBoost model using **BentoML** to generate a high-performance REST API.
- Build a **FastAPI** gateway to route requests between the UI, the Feast Online Store, and the BentoML inference engine.
- **Stress Testing:** Use **Locust** or **k6** to fire 1,000+ Requests Per Second (RPS) at the FastAPI endpoint to prove production readiness.

---

## PART 3: THE UI & AGENTIC AI SPRINT

### Phase 8: The Agentic RAG Executive (LangChain/LlamaIndex)
**Goal:** Add state-of-the-art GenAI interactivity.
- Build a **ReAct Tool-Calling Agent** served via FastAPI.
- **Equip Tools:**
  1. `DuckDB Text-to-SQL Tool`: Write SQL dynamically to query forecasts, anomalies, and basket rules.
  2. `Vector Search Tool`: Query MkDocs architecture documentation.
  3. `Live Inference Tool`: Hit the BentoML endpoint to run "What-If" pricing scenarios.

### Phase 9: Dual-Frontend Architecture (Streamlit)
**Goal:** Separate business logic from engineering telemetry.
- **App 1 (`app_business.py`):** The Executive Dashboard. Features Demand Forecasting, Market Basket Rules, Anomaly Detection, and the Agentic Chat interface.
- **App 2 (`app_engineering.py`):** The MLOps Dashboard. Visualizes Evidently AI Drift HTML reports, Locust load-test metrics, and MLflow CI/CD pipeline statuses.

### Phase 10: Zero-Dollar Cloud Deployment
**Goal:** Take the project from `localhost` to the internet.
- **CI/CD:** Write GitHub Actions (`.github/workflows/`) to trigger PyTest on pull requests and execute cron-scheduled drift monitoring.
- **Backend Hosting:** Deploy the Dockerized FastAPI/BentoML microservices to **Render** or **Koyeb**.
- **Frontend Hosting:** Deploy both Streamlit applications to **Streamlit Community Cloud**.
