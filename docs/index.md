# StoreCast: Retail Intelligence & Forecasting

Welcome to the internal documentation for **StoreCast**, our production-grade retail data engine and machine learning pipeline. 

This system is designed to optimize inventory and promotional markdown strategies for our 45-store pilot region, targeting a massive **$17.3M reduction in trapped working capital**.

## Project Architecture & Business Deliverables
StoreCast implements a state-of-the-art **Medallion Architecture**, acting as an on-premise Lakehouse:

```mermaid
graph TD
    %% Styling Classes
    classDef bronze fill:#CD7F32,stroke:#8A5A22,stroke-width:2px,color:#FFFFFF;
    classDef silver fill:#E0E0E0,stroke:#808080,stroke-width:2px,color:#000000;
    classDef gold fill:#FFD700,stroke:#B8860B,stroke-width:3px,color:#000000;
    classDef gx fill:#8B0000,stroke:#4A0000,stroke-width:2px,color:#FFFFFF;
    classDef bi fill:#0078D4,stroke:#005A9E,stroke-width:2px,color:#FFFFFF;
    classDef ml fill:#107C10,stroke:#0F540F,stroke-width:2px,color:#FFFFFF;
    classDef roi fill:#D83B01,stroke:#A42600,stroke-width:3px,color:#FFFFFF;

    subgraph Pipeline["1. Enterprise Lakehouse Engine"]
        A[(Raw Data<br>CSVs)] -->|PySpark Ingestion| B(Bronze Layer<br>Raw Delta):::bronze
        B -->|Polars ELT| C(Silver Layer<br>Conformed Delta):::silver
        
        C --> GX1{{Great Expectations<br>Data Contract Gate}}:::gx
        
        GX1 -->|DuckDB Window SQL| D[(Gold Layer<br>Master Parquet)]:::gold
        
        D --> GX2{{Great Expectations<br>ML Feature Gate}}:::gx
    end

    subgraph Deliverables["2. Business Intelligence & Predictive Analytics"]
        GX2 -->|DAX Semantic Modeling| E[PowerBI Dashboards<br>Market Basket Visuals & KPI Reporting]:::bi
        GX2 -->|ML Orchestration| F[Time-Series Pipeline<br>Demand Forecasting]:::ml
        
        E -->|Executive Action| G(Preserve $5.5M in Gross Margin):::roi
        F -->|Reduce WMAE Error| H(Free $17.3M Trapped Inventory):::roi
        
        G -.-> I[[Total Impact: $17.3M Free Capital & $8.9M Profit]]:::gold
        H -.-> I
    end
```

### The Pipeline Stack
1. **Bronze (Raw Ingestion):** Scalable extraction using `PySpark` to partition and store our raw CSVs as **Delta Lake** tables. This guarantees ACID transactions and concurrency.
2. **Silver (Cleaned & Conformed):** Blazing-fast `Polars` pipelines that execute targeted cleaning logic (clipping returns, forward-filling macroeconomics) formally validated by `Great Expectations`.
3. **Gold (Business Aggregation):** `DuckDB` pipelines creating analytic-ready Flat Schemas with highly complex time-series Window logic (lags, rolling averages) ready for reporting and ML modeling.

## Tooling Stack
- **Dependencies:** `uv`
- **Data Versioning:** `dvc`
- **Documentation:** `mkdocs`
- **Data Engineering:** `PySpark`, `Polars`, `DuckDB`, `Delta Lake`
- **Analytics:** `pandas`, `structlog`

Navigate through the menu to explore the Business Metrics, Baseline Models, and EDA findings!
