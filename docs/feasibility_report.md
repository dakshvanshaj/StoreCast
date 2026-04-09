# ML Feasibility & ROI Report

**Project Name:** StoreCast  
**Date:** April 2026  
**Author:** Daksh Vanshaj
**Document Goal:** Having proven that our initial Machine Learning models can outperform our baseline manual heuristics, this report maps the hard financial numbers to answer: *Is it actually worth deploying and maintaining this in a production environment?*

---
### The Verdict: **Highly Feasible**

Deploying the StoreCast ML pipeline is a massive net-positive undertaking. The monthly cost of employing an MLOps Engineer and running the cloud infrastructure ($15,500) represents just **2%** of the monthly savings it generates ($760k+), making the deployment a strategic necessity. Furthermore, the automation frees our regional planners from spreadsheet drudgery, shifting them to high-level strategic problem-solving.

## 1. The ML Performance Baseline

Our initial feasibility test using a naive Random Forest Regressor (`notebooks/04_feasibility_study.ipynb`) yielded the following results against the historical "beat-to-deploy" threshold established in `01_business_context.md`:

- **Manual Baseline Error:** 11.85% WMAPE (Weighted Mean Absolute Percentage Error — meaning our old method of assuming "we will sell whatever we sold this week last year" was historically off by nearly 12% on average).

- **StoreCast ML Baseline Error:** 8.49% WMAPE
- **Absolute Error Reduction:** ~3.36%

This 3.36% improvement isn't just a statistical win; it directly triggers supply chain mechanics that allow us to reduce inventory and preserve margin. All financial baselines below use standard US big-box retail industry ratios, as the pilot region operates within the United States.

---

## 2. Transforming Math into Inventory Reduction

It seems counterintuitive that a ~3.5% improvement in a mathematical metric could lead to an 8% drop in physical inventory without causing empty shelves. However, in supply chain mathematics, **buffer inventory (Safety Stock) isn't driven directly by average sales; it is driven entirely by unpredictability.** 

Here is the breakdown of how that works intuitively and mathematically:

1. **The Uncertainty Penalty:** Safety stock formulas mathematically penalize variance. When a forecast is consistently off by ~12%, the spread of potential demand is wide. The safety buffer needed to protect against unexpected stock-outs scales with the *square* of that uncertainty. When a reliable ML model trims roughly 3.5 percentage points of that absolute blind-spot error, it collapses that wide variance. A tighter variance means an exponentially smaller buffer is needed to maintain the exact same customer service levels.

2. **Eliminating Planner Padding:** Intuitively, when human planners know their manual tool is historically inaccurate, they don't trust it. They manually bloat their orders and pad safety stock "just in case." An accurate ML model establishes a baseline of trust, allowing us to safely eliminate both the mathematical error and the human padding used to cover it up.

3. **The Retail Industry Ratio:** In the US enterprise retail market, it is an established logistical heuristic that for every 1% absolute improvement in forecast accuracy, system-wide safety stock can typically be reduced by 2% to 2.5%.

Thus, reducing our absolute error by ~3.36% comfortably permits a conservative 8% reduction in total buffer inventory.

---

## 3. Business Value Generation (The Upside)

According to the mathematical framework outlined in `01a_business_metrics_math.md`, an 8% reduction in buffer inventory unlocks the following financial gains:


| Value Source | Yearly Impact | Monthly Equivalent |
| :--- | :--- | :--- |
| **Freed Working Capital** <br>*(Cash permanently pulled out of warehouse inventory and put back into the corporate bank account)* | **$17,290,000** | N/A (One-time) |
| **Holding Cost Savings** <br>*(Rent, insurance, and spoilage saved by housing less inventory, assuming a standard US 20% carrying rate)* | **$3,460,000** | $288,333 / month |
| **Markdown Optimization** <br>*(Stopping blanket discounts on items people would buy anyway; targeting a conservative 5% improvement)* | **$5,510,000** | $459,166 / month |
| **Manual Hours Saved** <br>*(Automating the math so our Regional Planners can focus on strategy)* | **$153,600** | $12,800 / month |
| **Total Bottom-Line Profit Increase** | **~$9.12 Million / year** | **~$760,299 / month** |

*Note: The $17.29M is a cash release. The $9.12M is hard operational savings and preserved profit margin.*

---

## 4. Real-World Deployment & Operational Costs (The Downside)

While the initial development footprint for StoreCast utilizes local open-source tools (minimizing cloud overhead), a real-world enterprise cloud rollout will incur costs.

### A. Infrastructure & Cloud Compute
*This is an estimate based on the current architecture and can be optimized further.*

Based on our planned MLOps architecture, an enterprise cloud deployment (AWS/GCP) will require the following components to execute our specific software stack:

- **Core Compute (Kubernetes/EKS): ~$250/month.**

  This cluster will continuously host our **FastAPI** inference server (providing real-time predictions to the stakeholder dashboard), our **Apache Airflow** environment (orchestrating the ETL and ML DAGs), and our **Prometheus / Grafana / Loki** stack for observability and system alerting.

- **ETL & Model Retraining (Spot EC2): ~$100/month.**

  These ephemeral, low-cost instances will spin up natively via Airflow workers to execute our memory-intensive **Polars** data transformations and retrain our core **Scikit-Learn / XGBoost** models. Once the models are updated, the instances terminate.

- **Artifact & Data Storage (S3/Cloud SQL): ~$50/month.**

  S3 bucket storage will act as the remote backend for **DVC (Data Version Control)** to manage our huge Bronze/Silver/Gold datasets without inflating the Git repository. It will also serve as the artifact store for our **MLflow** model registry. A lightweight Cloud SQL database will manage the backend metadata for Airflow and MLflow tracking.
- **Total Infra Cost:** **~$400 / month** ($4,800 / year).

### B. Engineering & Maintenance (People)

The system requires ongoing monitoring to prevent data drift, handle anomalies, and ensure uptime.

- **1x Senior ML Engineer / MLOps Lead:** ~$15,000 / month ($180,000 / year)
- **Monitoring Tools (Dashboards & Logging):** ~$100 / month
- **Total People & Tooling Cost:** **~$15,100 / month** ($181,200 / year).

---

## 5. Final Feasibility Conclusion

| Financial Summary | Amount |
| :--- | :--- |
| **Monthly Operational Savings & Margin Preserved** | $760,299 / month |
| **Monthly Operational Cost (Cloud + People)** | $15,500 / month |
| **Monthly Net Revenue Generated** | **$744,799 / month** |
| **Freed Working Capital (Instantly Available Cash)** | **$17.29 Million** |

### The Verdict: **Highly Feasible**

Deploying the StoreCast ML pipeline is a massive net-positive undertaking. The monthly cost of employing an MLOps Engineer and running the cloud infrastructure ($15,500) represents just **2%** of the monthly savings it generates ($760k+), making the deployment a strategic necessity. Furthermore, the automation frees our regional planners from spreadsheet drudgery, shifting them to high-level strategic problem-solving.
