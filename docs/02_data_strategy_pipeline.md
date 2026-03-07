## PHASE 2: DATA STRATEGY & PIPELINE DESIGN

**Project Name:** StoreCast

---

### The Data Pipeline Philosophy
*Raw data is sacred. Never modify originals.*
**Pipeline stages:** Raw -> Validated -> Processed -> Engineered Features -> Model-Ready Splits
*Each stage is versioned via DVC, reproducible, and documented.*

---

### Step 1: Data Acquisition & Simulation Strategy

**Source Identification & Production Simulation**
In a live enterprise environment, data is extracted via SQL queries from a central data warehouse. For this pilot, the data is simulated via three static extracts representing the warehouse tables.

*   **Primary Data Sources:**
    *   `sales.csv`: The core transactional fact table. Contains Store, Dept, Date, Weekly_Sales, and IsHoliday. Date range: 2010-02-05 to 2012-10-26.
    *   `features.csv`: Regional macroeconomic and promotional factors. Contains Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment, IsHoliday. Date range: 2010-02-05 to 2013-07-26.
    *   `stores.csv`: Static dimensional data regarding store typography. 45 stores, 99 departments, 3 unique store types (A, B, C).

*   **Pipeline Update Frequency:** Weekly Batch Execution (scheduled for Saturday nights to process the prior week's aggregated sales).
*   **The Drift Simulation Strategy (Out-of-Time Holdout):**
    *   To test for concept drift, the final 16 weeks of available data will be strictly quarantined.
    *   The model will never see this data during training. During the deployment phase, a separate pipeline script will feed this data to the model chronologically (week by week) to evaluate real-world degradation and trigger drift alerts.

**Data Collection Ethics & Governance**
*   **Privacy (GDPR/HIPAA):** The dataset is fully aggregated to the Store and Department level. It contains zero Personally Identifiable Information (PII) or user-level transaction histories, ensuring complete regulatory compliance.

---

### Main Data Insights (from Assessment)

Following the initial assessment in `notebooks/01_data_assessment.ipynb`, the following key patterns were identified:

*   **Granularity:** The data covers 45 stores and 99 departments across 3 unique store types.
*   **Markdown Gap:** There is a total absence of markdown data from **2010-02-05 to 2011-11-04**. This represents a structural change in data recording.
*   **Economic Reporting Lag:** CPI and Unemployment data are missing for the final 3 months (**2013-05-03 to 2013-07-26**). This is Missing At Random (MAR) due to governmental reporting cycles.
*   **Return Volume:** `Weekly_Sales` contains negative values, confirming that returns are aggregated into the weekly figures.
*   **Promotion Anomalies:** Markdowns contain negative values and extreme spikes, suggesting either reversals or aggressive seasonal clearing events.

---

### Step 2: Data Quality Framework (Requirements)

Based on the insights above, the following requirements are established for the "Validated" stage of the pipeline:

*   **Completeness Requirements:**
    *   `Store` and `Dept` IDs must have 0% null values.
    *   `Weekly_Sales` must have 0% null values.
    *   `MarkDown` nulls are acceptable for the historical gap but must be handled before training.
    *   `CPI`/`Unemployment` nulls must trigger a forward-fill or back-fill mechanism using the most recent available regional report.

*   **Validity Requirements:**
    *   `Weekly_Sales`: Use the negative values to create a Return_Heavy_Flag feature, but then clip the actual Weekly_Sales target variable to $0.00 right before training so the model only predicts positive inventory needs.
    *   `MarkDown`: Extreme outliers (> 3 standard deviations) must be capped or transformed to prevent model bias.
    *   `Date`: All records must fall on a Friday. Any non-Friday dates must be rejected.

*   **Consistency Requirements:**
    *   Referential integrity: Every `Store` in `sales.csv` and `features.csv` must exist in `stores.csv`.
    *   Cross-table Date alignment: The date range in `sales.csv` must be a subset of `features.csv`.

*   **Timeliness Requirements:**
    *   Input data must be verified against the expected Saturday execution window.

---

### Step 3: Data Versioning Strategy

**Why Version Data?**
*Reproducibility, Debugging, Auditability, and Experimentation.*

*   **Data Versioning Tool:** DVC (integrated with DagsHub as the remote backend).
*   **What to Version:**
    *   Raw data snapshots (`data/raw/*.csv`)
    *   Processed data outputs (`data/processed/*.csv`)
    *   Feature engineering outputs
    *   Train/validation/test splits
    *   Data processing scripts
*   **Version Metadata Includes:**
    *   Collection timestamp
    *   Source information
    *   Processing pipeline version
    *   Quality metrics
    *   Known issues or anomalies (e.g., tracking the markdown missingness gap)

---

### Step 4: Feature Engineering Strategy

**Feature Design Process:**

1.  **Understand the Problem Domain:** Retail sales are heavily influenced by temporal seasonality (holidays like Thanksgiving and Christmas), pricing promotions (markdowns), and macroeconomic purchasing power (CPI, fuel, unemployment).
2.  **Form Hypotheses:**
    *   *Hypothesis 1:* Holidays significantly alter baseline sales volume, especially in specific departments.
    *   *Hypothesis 2:* Markdowns drive top-line revenue but differ drastically in effectiveness across Store Types (A vs. B vs. C) and Sizes.
    *   *Hypothesis 3:* Markdowns peak heavily on holidays, festive periods and end of year clearance sales.
    *   *Hypothesis 4:* Markdowns are higher for larger stores.
    *   *Hypothesis 5:* High fuel prices and high unemployment negatively impact spending but might increase basic staple purchases.
    *   *Hypothesis 6:* High or Low temperature may be a factor reducing sales, especially for certain product categories.

3.  **Create Features to Test Hypotheses:** *(See Feature Categories below).*
4.  **Validate Feature Importance:** Measure using tree-based feature importances and SHAP values, ensuring alignment with the business explainability constraint.
5.  **Document Reasoning:** Ensure every feature has an explainable reason for existence to satisfy the VP of Supply Chain and Store Managers.

**Feature Categories to Consider:**
*   **Temporal Features:** Week of the year, Month, Year, Days until next major holiday, Lagged sales (e.g., $t-1$ week, $t-52$ weeks).
*   **Aggregations:** Rolling 4-week mean/std of department sales, Store-level total historical sales.
*   **Interactions:** Markdown volume $\times$ Store Type, Store Size $\times$ IsHoliday.
*   **Encodings:** Target encoding for Department IDs, One-hot encoding for Store Types.

*(Note: Every feature should be explainable and monitorable for drift.)*

---

### Step 5: Data Pipeline Architecture

**Design Principles:**

*   **Modularity:** Each step (Preprocessing -> Validation -> Feature Engineering -> Split) is independent and testable.
*   **Reproducibility:** Same input always produces same output (lock seeds, versions).
*   **Observability:** Log what's happening at each step (shapes, errors, null counts).

**Error Handling Strategy:**
*   *Missing Macroeconomic Data:* Alert pipeline and automatically apply forward-fill heuristic.
*   *Validation failure (e.g., new Store ID appears):* Stop pipeline and alert MLOps maintainers.
*   *Processing error:* Log the traceback and trigger the fallback strategy (revert to previous year's manual heuristic).

**Scalability Requirements:**
*   *(TODO: Describe how the pipeline will handle data growth, e.g., batch processing vs streaming)*