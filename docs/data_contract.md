# Data Contract: Silver Layer

**Owner:** Data Engineering Team  
**Status:** Active  
**Enforcement:** Great Expectations (`src/data/validate_silver.py`)

## 1. Overview
The Silver Layer contains cleaned, aggregated, and curated data ready for Gold Layer dimensional modeling. This document defines the exact constraints and business rules programmatically enforced before any data is allowed to pass downstream.

---

## 2. Table: `sales`
**Description:** Weekly transaction history at the Store/Department level.

### Constraints
| Constraint Type | Columns | Rule / Argument |
| :--- | :--- | :--- |
| **Schema Match** | All | `["store", "dept", "date", "weekly_sales", "isholiday"]` (Exact Match) |
| **Primary Keys** | `store, dept, date` | Must be strictly unique (No duplicate combinations) |
| **Non-Nullability**| `date, weekly_sales` | 100% Completeness Required |
| **Value Bounds** | `weekly_sales` | `>= 0.0` (Negative sales stripped during Polars transformation) |
| **Value Bounds** | `store` | Between `1` and `45` |
| **Data Types** | `weekly_sales` | `FLOAT` |
| **Data Types** | `store` | `INT` |

---

## 3. Table: `features`
**Description:** Macro-economic and regional indicators at the Store/Date level.

### Constraints
| Constraint Type | Columns | Rule / Argument |
| :--- | :--- | :--- |
| **Schema Match** | All | `["store", "date", "temperature", "fuel_price", "markdown1", "markdown2", "markdown3", "markdown4", "markdown5", "cpi", "unemployment", "isholiday"]` (Exact Match) |
| **Primary Keys** | `store, date` | Must be strictly unique |
| **Non-Nullability**| `cpi, unemployment` | 100% Completeness Required (Missing values forwarded filled via Polars) |
| **Non-Nullability**| `markdown1 - markdown5`| No NaNs allowed (Filled with `0.0` during Polars ETL) |
| **Value Bounds** | `markdown1 - markdown5`| `>= 0.0` |
| **Data Types** | `cpi, unemployment` | `FLOAT` |
| **Data Types** | `markdown1 - markdown5`| `FLOAT` |

---

## 4. Table: `stores`
**Description:** Static master dimension table for physical retail locations.

### Constraints
| Constraint Type | Columns | Rule / Argument |
| :--- | :--- | :--- |
| **Schema Match** | All | `["store", "type", "size"]` (Exact Match) |
| **Primary Keys** | `store` | Must be strictly unique |
| **Value Bounds** | `store` | Between `1` and `45` |
| **Categorical Set**| `type` | Must be strictly in `["A", "B", "C"]` |
| **Data Types** | `store` | `INT` |

---

## 5. Violations & Alerting
If any of these expectations fail during `validate_silver.py`, the CI/CD pipeline immediately halts, and the `structlog` service alerts the engineering team with a critical error before the corrupted data can poison the Gold Analytical schemas.
