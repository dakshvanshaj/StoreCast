# Business Metrics Calculation Guide

This document transparently explains the mathematical logic and real-world retail assumptions used to calculate the ROI targets and baseline figures found in `01_business_context.md`. Rather than assuming a flat $1M/store revenue, these figures are modeled precisely off the historical data provided in `data/raw/sales.csv` combined with standard Big Box retail supply chain metrics.

## 1. Top-Line Revenue Estimate ($2.45 Billion)

- **Raw Data Source Extraction:** Across the 143 weeks of historical data in `sales.csv`, total gross sales are exactly $6,737,218,987.11.
- **Average Weekly Sales:** $6.737B / 143 weeks = $47,113,419.49 per week across all 45 stores.
- **Math:** $47.11 Million/week × 52 weeks = **$2,449,897,813.49 ($2.45 Billion)** in annualized top-line revenue.

## 2. Average Inventory Value Target ($216.16 Million)

Working Capital is trapped in physical inventory. To calculate this:

- **COGS (Cost of Goods Sold) Assumption:** Industry standard for this retail tier is around 75% of top-line revenue.
  - $2.45 Billion × 0.75 = **$1.837 Billion** Annual COGS.
- **Inventory Turnover Assumption:** The industry average inventory turnover is roughly 8.5 cycles per year.
- **Math:** $1.837 Billion COGS / 8.5 cycles = **$216,167,454.13 ($216 Million)** average standing inventory.

## 3. Freed Working Capital Target ($17.29 Million)

How does reducing our ML Forecasting Error mathematically reduce inventory without causing stock-outs?

- **Supply Chain Law (Safety Stock):** In enterprise retail logistics, the formula for Safety Stock is:
  `Safety Stock = Z * sqrt( (LT * σ_d^2) + (D_avg^2 * σ_LT^2) )`
- **The Impact of ML:** The standard deviation of demand (`σ_d`) is heavily bloated by forecasting error. When Machine Learning models reduce the absolute forecasting error (WMAPE) by roughly ~3.5%, the standard deviation of that error shrinks substantially. Less unpredictability means supply chain planners no longer need to hoard "just-in-case" palettes of inventory.
- **Target:** Planners can confidently thin out their buffer warehouse stock by approximately 8-10% mathematically without ever risking a drop in service levels (Fill Rate).
- **Math:** An 8% reduction on the $216.16 Million standing inventory = **$17.29 Million in freed Working Capital**.

## 4. Annual Holding Cost Savings ($3.46 Million)

Holding inventory costs money (warehousing space, insurance, depreciation, shrinkage).

- **Assumption:** The universal retail standard for carrying costs is 20% of the total inventory value annually.
- **Math:** $17.29 Million in sustainably removed stock × 20% carrying rate = **$3.46 Million** in raw bottom-line ops savings annually.

## 5. Promotional Markdown Savings ($5.51 Million)

Blanket markdowns destroy margin. By predicting price elasticity, we only offer discounts where volume completely compensates for the margin drop.

- **Current Spend Assumption:** The industry average for promotional markdowns is roughly 4.5% of gross revenue.
  - $2.45 Billion × 0.045 = **$110.24 Million** total markdown "loss" annually.
- **Target:** A 5% optimization in markdown allocation (halting discounts on inelastic items).
- **Math:** 5% of $110.24 Million = **$5.51 Million** in preserved Gross Margin.

## 6. Baseline Forecast Error (11.85% WMAPE)

- **Data Extracted from:** `data/raw/sales.csv` and `data/raw/features.csv`
- **Methodology (Reproducible via `baseline.py`):** We strictly matched every store/department's sales for a given week against its exact sales 52 weeks prior (the Last-Year-Same-Week heuristic). 
- **Weighting:** Weeks flagged as `IsHoliday = True` received a 5x penalty weight due to their disproportionate business importance.
- **Result:** Averaged across the 4,455 timeseries, the error was 11.85% WMAPE and $1969.91 WMAE, establishing our rigid "beat-to-deploy" ML threshold.

## 7. Context-Aware Anomaly Detection Impact

While traditional Isolation Forests often fall into the "Volume Trap" (flagging naturally large supercenters as anomalies), StoreCast converts raw dollars into **Dimensionless Ratios**:

- **YoY Growth Ratio:** `Weekly Sales / Sales Last Year`
- **Trend Deviation Ratio:** `Weekly Sales / 4-Week Rolling Average`

By feeding the algorithm these normalized ratios, it isolates true operational decoupling. 
- **Business Value:** Prevents stock-outs that occur from "silent" logistical failures. Assuming a 1% reduction in stock-out events across a $2.45B pipeline, this directly preserves up to **$24.5 Million** in otherwise lost top-line revenue.

## 8. Store Segmentation (K-Means Clustering)

Instead of using arbitrary geographical regions, StoreCast uses unsupervised learning to group stores by actual behavioral data (Size, Weekly Velocity, Markdown Response).

- **Methodology:** We utilized the Elbow Method on K-Means to identify `K=3` optimal clusters.
- **Business Value:** This organically reveals the retail taxonomy: Supercenters (High Volume, Low Margin), Standard Stores (Average), and Express Stores (Low Volume, High Agility). This allows executives to route tailored inventory assortments rather than blanketing the 45-store network with identical freight, minimizing supply chain waste.
