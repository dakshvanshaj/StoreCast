## PHASE 1: PROBLEM DEFINITION & BUSINESS CONTEXT

**Project Name:** StoreCast

---

### Step 1: Project Brief

**The Business Problem**
The retail chain is experiencing margin erosion and working capital inefficiency due to reactive inventory management and untargeted promotional strategies. The objective is to reduce revenue lost to stock-outs and decrease margin wasted on blanket discounts by accurately predicting weekly store-level demand and segmenting stores based on price sensitivity.

**The Current Baseline**
Inventory forecasting currently relies on a manual heuristic: ordering based on the previous year's sales figures for the same week. Furthermore, promotional markdowns are applied globally across all store types without considering local elasticity. This manual process is time-consuming and results in a high forecast error. *(TODO: update the manual baseline exact numbers after calculating the manual forecasting accuracy using the previous year's trend as a heuristic).*

**Success Definition & Business Impact**
Optimization targets Working Capital, Gross Margin, and Operational Efficiency across a 45-store pilot region. Success is defined by three quantifiable pillars:

* **Inventory Impact (Money Saved):** Reduce Weekly Sales Forecasting error from the manual baseline of ~25% to < 15%. A 10% improvement in forecast accuracy allows Supply Chain to safely reduce "safety stock" buffers by 8%. Across 45 stores carrying an estimated $200M in physical inventory, this frees up $16M in tied-up working capital and saves roughly $3.2M annually in holding costs.
* **Promotional Impact (Margin Saved):** Achieve a clustering Silhouette Score > 0.6 to segment stores based on actual MarkDown sensitivity. By identifying price-inelastic stores and halting unnecessary blanket discounts, a 5% improvement in Markdown ROI is targeted, preserving an estimated $1.5M in gross margin annually.
* **Operational Impact (Time Saved):** Deploy an automated batch-inference pipeline. Automating the current manual spreadsheet tweaks for 4,455 individual time series (45 stores × 99 departments) saves 160 hours per month, allowing 4 full-time planners to pivot from data entry to strategic exception handling.

**Business Assumptions & Value Calculations**
To ensure alignment with enterprise financial metrics, the projected business impacts are derived using standard retail supply chain baselines applied to the pilot region's data:

* **Total Physical Inventory Estimate ($200M):** * **Revenue:** The 45 pilot stores average approximately $1M in weekly sales per store, yielding an estimated $2.34 Billion in annual revenue.
  * **COGS & Turnover:** Assuming a standard big-box retail Cost of Goods Sold (COGS) of 75%, the annual COGS is $1.75 Billion. With an industry-average inventory turnover rate of 8.5 cycles per year, the average holding inventory value (COGS / turnover rate) is calculated at approximately $205 Million (conservatively rounded to $200M).
* **Freed Working Capital ($16M):** * Safety stock buffers are directly proportional to the standard deviation of forecast error. A 10% absolute reduction in error mathematically permits an estimated 8% reduction in safety stock while maintaining identical service levels to prevent stock-outs. 
  * Reducing the $200M inventory by 8% frees up $16 Million in working capital.
* **Holding Cost Savings ($3.2M):** * The universal retail standard for inventory holding costs (warehousing, insurance, shrinkage, spoilage, opportunity cost) is 20% of the inventory value annually.
  * $16M in freed inventory × 20% holding cost rate equates to $3.2 Million in annual operational savings.
* **Markdown Margin Preservation ($1.5M):** * With ~$2.34 Billion in annual revenue, assuming a conservative 1.28% of revenue is lost to promotional markdowns, the annual markdown spend/margin loss is roughly $30 Million.
  * A 5% improvement in Markdown ROI (achieved by replacing global blanket markdowns with targeted, elasticity-based promotions) yields 5% of $30 Million, which equates to $1.5 Million in preserved gross margin.

**Constraints & Requirements**

* **Time:** Predictions are required on a weekly batch schedule, not real-time.
* **Cost:** The system must be built using open-source tools to minimize operational expenditure.
* **Accuracy:** The model must significantly outperform the current baseline heuristic to justify deployment.
* **Explainability:** Supply chain managers require interpretable outputs, particularly regarding how external factors (CPI, Fuel) influence forecasts.
* **Data Constraints & Governance:** 
  * **Privacy Regulations (GDPR/HIPAA):** The dataset consists of aggregated store and department-level sales. It contains no Personally Identifiable Information (PII), negating strict GDPR/HIPAA compliance overhead, though standard corporate data security protocols apply.
  * **Data Availability:** Historical sales data exists and is accessible. However, promotional markdown data is entirely missing prior to November 11, 2011, requiring a specific engineering strategy (e.g., imputation or isolated feature modeling) for historical gaps.
  * **Labeling Resources:** This is a supervised learning task using historical sales as the ground truth, meaning manual data labeling costs are $0.

---

### Step 2: Success Metrics Hierarchy

| Priority Level | Metric Focus | Specific Target |
| :--- | :--- | :--- |
| **Primary** | Business Impact | Inventory Turnover Ratio and Incremental Gross Margin per markdown dollar. |
| **Secondary** | ML Optimization | Weighted Mean Absolute Error (WMAE) for forecasting; Silhouette Score for segmentation. |
| **Guardrails** | System & Trust | Pipeline execution latency (weekend completion); High model explainability for regional managers. |

**Metric Rationale: Why WMAE?**

Weighted Mean Absolute Error (WMAE) is chosen over RMSE, MSE, or MAPE due to the specific business context of retail sales. Retail revenue is disproportionately driven by holiday weeks (e.g., Thanksgiving, Christmas). WMAE allows us to assign a higher mathematical weight to these critical holiday weeks. 

* *Why not RMSE/MSE?* They square the errors, heavily penalizing random, unpredictable noise (outliers) which is common in retail. 
* *Why not MAPE?* MAPE suffers from division-by-zero errors when department sales are zero and inherently biases models toward under-forecasting. WMAE provides a balanced, dollar-relevant optimization metric.

---

### Step 3: Stakeholder Alignment

* **End Users (Store Managers & Planners):** Receive weekly predictions via a dashboard. They require high explainability to trust and act on the system.
* **Decision Makers (VP of Supply Chain & Head of Merchandising):** Care primarily about ROI justification, operational risk assessment, and resource requirements.
* **Maintainers (MLOps Engineering Team):** Require comprehensive documentation, monitoring dashboards, and alert systems to keep the batch pipeline running.
* **Data Providers (IT Operations):** Manage the data warehouse and require predictable access patterns and update frequencies from data extraction jobs.

---

### Step 4: Risk Assessment



| Risk Category | Identified Threat | Likelihood | Impact | Mitigation Strategy | Monitoring Approach |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Data Risks** | **Source/Quality Issues:** APIs deprecate, schemas change, or markdown data is completely missing before Nov 2011. | High | Moderate | Implement robust data validation schemas; use a Champion/Challenger model strategy to separate historical data without markdowns from modern data. | Track missing value rates, out-of-range values, and schema violations in the pipeline. |
| **Model Risks** | **Concept Drift & Edge Cases:** Performance degrades silently as macroeconomic factors (Unemployment, CPI) shift, or edge cases emerge. | Medium | Severe | Develop ensemble models to handle cold-starts and implement automated retraining triggers. | Statistical drift detection (e.g., KS tests) on input distributions; continuous tracking of WMAE. |
| **System Risks** | **Downtime & Scaling:** Batch pipeline failure during the critical weekend processing window, or dependency breakages. | Low | Severe | Implement fallback heuristics (automatically reverting to the historical manual baseline if the ML pipeline fails to execute). | Real-time infrastructure monitoring for latency, error rates, and resource utilization. |
| **Business Risks** | **Bad Decisions & Over-reliance:** Incorrect demand predictions lead to severe stockouts, damaging customer trust and reputation. | Medium | Severe | Establish strict guardrail thresholds preventing the model from reducing safety stock below a critical, mathematically defined minimum bound. | Track ground-truth inventory levels and business outcomes (revenue impact, stockout incidents). |