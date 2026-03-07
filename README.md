# StoreCast: Production-Grade Retail Demand Forecasting

StoreCast is an automated machine learning pipeline designed to optimize store performance and inventory levels for a 45-store retail pilot. By replacing manual heuristics with explainable demand forecasting and markdown-sensitivity segmentation, the system targets a **10% improvement in forecast accuracy** and **$1.5M+ in annual margin preservation**.

---

## 1. Project Overview
*   **What it is:** A weekly batch-inference system that predicts department-level demand and segments stores based on promotional elasticity.
*   **Why it matters:** Reactive inventory management and blanket discounts currently lead to significant margin erosion and stock-outs.
*   **Key Target Results:** Reducing MAPE from ~25% to <15% and achieving a clustering Silhouette Score > 0.6.

## 2. Problem Definition
*   **Business Context:** A retail chain managing 4,455 individual time series (45 stores × 99 departments).
*   **The Baseline:** Ordering relies on a manual "last year same week" heuristic, resulting in high error rates during volatile economic periods.
*   **The Proposed Solution:** A modular MLOps pipeline that integrates macroeconomic indicators (CPI, Fuel Price), temporal seasonality, and markdown interactions into a Gradient Boosting framework.
*   **Success Metrics:** Primary business metrics are **Inventory Turnover Ratio** and **Markdown ROI**; secondary ML metrics are **WMAE** (Weighted Mean Absolute Error) and **Silhouette Score**.

## 3. Technical Approach
*   **Data Pipeline:** Uses a "Sacred Raw Data" philosophy. Pipeline stages (Raw -> Validated -> Processed -> Engineered) are orchestrated to handle structural gaps in historical markdown data.
*   **Feature Engineering:** 
    *   **Temporal:** Lagged sales ($t-1, t-52$), holiday proximity, and seasonality.
    *   **Macroeconomic:** Integration of CPI and Unemployment trends.
    *   **Interaction:** Promotional volume weighted by Store Type and Size.
*   **Versioning:** Full data and model lineage tracked via **DVC** and integrated with **DagsHub**.
*   **Infrastructure:** Python 3.13, `uv` for dependency management, and `MkDocs` for stakeholder-facing documentation.

## 4. Project Structure
```text
/
├── data/               # Versioned via DVC (Raw, Processed, Features)
├── docs/               # Technical and Business documentation (MkDocs)
├── notebooks/          # EDA, Data Quality Assessment, Experimentation
├── main.py             # Entry point for the batch-inference pipeline
├── pyproject.toml      # Python dependencies (managed by uv)
├── mkdocs.yaml         # Documentation configuration
└── GEMINI.md           # LLM Instruction Context
```

## 5. How to Run
### Prerequisites
*   [uv](https://github.com/astral-sh/uv) installed.
*   DVC configured with appropriate remote access.

### Setup
1.  **Sync Dependencies:**
    ```bash
    uv sync
    ```
2.  **Pull Data:**
    ```bash
    dvc pull
    ```

### Execution
*   **Run Automated Pipeline:**
    ```bash
    uv run main.py
    ```
*   **View Documentation:**
    ```bash
    mkdocs serve
    ```

