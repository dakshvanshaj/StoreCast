# Phase 1: Problem Definition & Business Context

**Project Name:** StoreCast  
**Department:** Supply Chain & Advanced Analytics  
**Date:** March 2026  

---

## 1. Executive Summary & Problem Landscape

**The Current Business Environment:**

Our 45-store pilot region generates an exact **$2.45 Billion** in annualized revenue (averaging ~$47.11M weekly across all stores). Despite this high volume, profitability is being constricted by an outdated, reactive approach to inventory management and promotional pricing strategy.

**The Core Problem:**

Currently, our demand planners rely on a rigid "Last-Year-Same-Week" (LYSW) manual heuristic to forecast inventory needs. We have evaluated this heuristic against our historical data (a process fully reproducible via our `baseline.py` script) and found it yields a **Weighted Mean Absolute Percentage Error (WMAPE) of 11.85%** and a **WMAE of $1969.91** per department per week. This forecasting error creates two simultaneous, margin-crushing scenarios:

1. **Safety Stock Bloat:** We must hold excessive safety stock (roughly $216M in physical inventory) to buffer against unpredictable demand, tying up massive amounts of working capital and accumulating high warehousing and spoilage costs.

2. **Untargeted Markdown Bleed:** Promotional markdowns (costing us ~$110M annually) are applied blanketly across all stores without evaluating local price elasticity, market basket associations, or regional economic conditions. We are losing gross margin on predictable items and missing cross-selling opportunities.

**Target Objective & The Tri-Model Strategy:**
StoreCast aims to replace this manual heuristic with an automated machine learning pipeline that attacks margin erosion from three different angles:

1. **Demand Forecasting:** 

Why do we need this? Currently, we over-order inventory (tying up cash) or under-order (missing sales). By using advanced algorithms to learn exactly how Holidays, CPI, and historical weeks interact, we can predict exact inventory thresholds. This directly reduces the $216M in trapped safety stock.

2. **Predictive Market Basket Analysis:**

Why this? We don't have individual customer receipts, but we do have Department-level sales. We will perform *Department-Level Association*. For example, if we heavily markdown Department 90 (Cameras) and lose margin there, does it cause a latent sales spike in Department 87 (Photo printing) later? If no, the markdown was wasted money.

3. **Anomaly Detection:** 

Why this? Retail data is notoriously dirty. If a store is closed for a week, sales drop to 0. If we feed that 0 into our forecasting model, it will skew to predict 0 sales for next year. Anomaly detection isolates these out-of-band events so they don't corrupt the core model and so regional managers are alerted instantly.

By uniting these three models to natively reduce WMAPE to < 8.5% and execute targeted up-selling, we project an **$8.97M annual increase in bottom-line profit** and the recovery of **$17.3M in trapped working capital**.

---

## 2. ROI Targets & Valuation Mechanics

To ensure executive alignment, our targets are bound by strict retail financial mechanics. *(Note: For a detailed step-by-step breakdown of how these simulated metrics were extracted precisely from our raw `sales.csv` data, see `01a_business_metrics_math.md`).*

### A. Working Capital & Holding Cost Savings
- **Current Inventory Value:** $216.16M (Assumes 75% COGS and an 8.5x inventory turnover rate on actual $2.45B annual revenue).
- **Freed Capital Target ($17.3M):** A ~3.5% absolute reduction in WMAPE error allows us to safely reduce our safety stock buffer by 8% without impacting service levels (Fill Rate). Releasing 8% of a $216M inventory frees **$17.29 Million in working capital**.
- **Holding Cost Savings ($3.46M Annual):** Assuming standard warehousing/spoilage costs at 20% of inventory value, removing $17.3M in stock saves **$3.46 Million** in operational overhead.

### B. Promotional Margin Preservation & Market Basket Uplift
- **Current Markdown Spend:** ~$110.24M (4.5% of gross revenue).
- **Target Optimization ($5.51M Annual):** By clustering stores via Silhouette Score segmentation and replacing global discounts with localized, elasticity-driven pricing, we target a 5% improvement in Markdown ROI, preserving **$5.51 Million** in gross margin.
- **Market Basket:** By identifying latent cross-selling opportunities across departments, we will mitigate profit loss on discounted lead items by intelligently suggesting high-margin pairings.

---

## 3. Project Scope, Constraints & Resources

This project operates within a highly constrained, lean environment to maximize immediate ROI.

- **Budget:** **$0 (Zero Dollar Capital Expenditure).** All infrastructure leverages entirely open-source analytics (Polars, XGboost, Pandas) and leverages existing hardware via a lean containerized cluster (Kubernetes/Docker Desktop).
- **Timeline:** **10 Days** from ideation to full operational dashboard deployment.
- **Staffing Requirements:** 1 Senior ML Engineer & Data Analyst serving as a full-stack MLOps lead.
- **Time/Labor Saved:** The pipeline will automate the forecasting of 4,455 individual time series (45 stores × 99 departments). This reallocates **320 manual hours/month** across our 4 Regional Demand Planners, freeing them to handle strategic out-of-stock interventions rather than spreadsheet calculations.

---

## 4. Success Metrics Hierarchy

| Priority Level | Metric Focus | Specific Target |
| :--- | :--- | :--- |
| **Primary** | Enterprise Financials | Achieve $17.3M reduction in inventory working capital. Achieve $8.97M combined net savings in holding costs and margin preservation. |
| **Secondary** | ML Optimization | Reduce Demand Forecasting **WMAPE < 8.5%** (baseline 11.85%). Achieve Store Clustering **Silhouette Score > 0.6**. |
| **Tertiary** | Resiliency & Trust | Maintain Kubernetes API inference latency under load tests (simulating concurrent planner activity). High SHAP-driven explainability to build trust with regional executives. |

---

## 5. Strategic Deliverables

1. **Baseline (`baseline.py`):** An automated python script to instantly validate our baseline manual forecasting error against new historical data.
2. **Targeted EDA & Hypothesis Testing:** Rigorously test the true impact of CPI, Unemployment, and local holidays on specific departments.
3. **Automated Anomaly Detection:** Real-time visibility into out-of-band sales events (e.g., unprecedented weather spikes).
4. **Store Segmentation API:** Grouping 45 stores dynamically based on markdown sensitivity.
5. **Live Stakeholder Dashboard (GenAI Enabled):** A conversational, Kubernetes-hosted dashboard where regional executives can ask an AI questions about the forecasts, view live stock levels, and review market basket strategies.