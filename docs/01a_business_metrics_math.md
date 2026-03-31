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

How does reducing our WMAPE (Forecasting Error) reduce inventory without causing stock-outs?

- **Supply Chain Law:** Safety stock levels are directly proportional to the standard deviation of forecast error. 
- **Target:** Reducing our manual 11.85% WMAPE absolute forecasting error by ~3.5% (down to < 8.5%) allows the supply chain to safely thin their inventory buffers by approximately 8%, as uncertainty decreases.
- **Math:** 8% reduction on the $216.16 Million standing inventory = **$17.29 Million in freed Working Capital**.

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
