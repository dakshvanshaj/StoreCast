# Documentation: `src/data/ingest_bronze.py`

**Purpose:** 

It takes the raw, messy `.csv` files provided by the business and converts them into **Delta Lake** tables inside our `data/bronze/` folder. Bronze data is completely uncleaned; it's simply stored in a better format.

## What happens if I run it again?

Because the script uses `.mode("overwrite")` on lines 32-35, if you run the script again, it will **completely replace** the existing data in the `data/bronze/` folders. It will not duplicate the data or crash. In an enterprise, an "overwrite" mode is used for complete daily refreshes, whereas "append" mode is used to add only isolated new daily records. 

---

## How to Run It
You can run the script from anywhere using:

```bash
python -m src.data.ingest_bronze
```

---

## Simple Line-by-Line Explanation

### 1. The Setup (Dependencies)

```python
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
import structlog
import config
```

- **`delta` and `pyspark`**: These are the massive big-data engines that will process our files.
- **`structlog`**: This generates our structured JSON logs so our monitoring tools (Grafana) can read them easily.
- **`config`**: This imports the configuration file you created, which securely holds all the file paths so we don't have hardcoded strings floating around.

### 2. The Delta Lake Configuration

```python
builder = SparkSession.builder.appName("StoreCast_Bronze") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    
spark = configure_spark_with_delta_pip(builder).getOrCreate()
```

- A `SparkSession` is basically the "brain" of PySpark. We are turning it on and naming it `"StoreCast_Bronze"`.
- We are injecting specific settings (`.config`) to tell Spark: *"Hey, do not use your default storage logic. Use the Delta Lake open table format instead."*
- **Why it helps:** Delta Lake provides **ACID Transactions**. If the script crashes halfway through writing the sales record, Delta will automatically roll it back so you don't end up with corrupted, half-written files.

### 3. Extracting the Data

```python
sales = spark.read.csv(str(config.RAW_SALES_PATH), header=True, inferSchema=True)
features = spark.read.csv(str(config.RAW_FEATURES_PATH), header=True, inferSchema=True)
stores = spark.read.csv(str(config.RAW_STORES_PATH), header=True, inferSchema=True)
```

- We read the raw CSV files from our `/data/raw/` directory.
- `header=True`: Tells PySpark the first row is column names.
- `inferSchema=True`: Tells PySpark to automatically guess whether a column is a Number, Text, or Date instead of assuming everything is Text.

### 4. Writing the Data (Partitioning)

```python
sales.write.format("delta").partitionBy("Store").mode("overwrite").save(str(config.BRONZE_SALES_PATH))
```

- We are saving the `sales` data into the `/data/bronze/` folder in `delta` format.
- `partitionBy("Store")`: Because the Sales dataset is huge, we divide it into separate physical folders for each Store. This means if a BI Analyst runs a query looking for "Store 5 data", the engine instantly ignores 44 other folders. This makes the query 44x faster.
- `mode("overwrite")`: Replaces any existing data.

### 5. Resilient Error Handling

```python
except Exception as e:
    logger.error("bronze_ingestion_failed", error=str(e))
    raise
finally:
    spark.stop()
```

- **`try/except/finally`**: We wrap the code so that if a CSV file is missing and the script crashes, it triggers the `except` block to log an error.
- **`finally`**: Crucially, whether the script succeeds or completely crashes, it will always drop down and run `spark.stop()`. This cleanly shuts down the PySpark engine, preventing "zombie processes" from permanently draining your laptop's RAM.
