# etl/extract.py

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
import time

from src.config import (
    RAW_FILES,
    ETL_SETTINGS,
    CLEANING_SETTINGS,
    LOG_FILE,
    ETL_LOG_FILE,
    create_directories,
)

# ── Configure Logger ──────────────────────────────────────────────────────
logger.remove()

# Log to console
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | "
           "{level: <8} | "
           "{name} | "
           "{message}",
    level="INFO",
    colorize=False,
)

# Log to file
logger.add(
    ETL_LOG_FILE,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
    level="INFO",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
)


# ════════════════════════════════════════════════════════════
# SCHEMA DEFINITIONS
# Expected columns for each CSV file
# ════════════════════════════════════════════════════════════
EXPECTED_SCHEMAS = {
    "orders": [
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": [
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "shipping_limit_date",
        "price",
        "freight_value",
    ],
    "order_payments": [
        "order_id",
        "payment_sequential",
        "payment_type",
        "payment_installments",
        "payment_value",
    ],
    "order_reviews": [
        "review_id",
        "order_id",
        "review_score",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
        "review_answer_timestamp",
    ],
    "customers": [
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
    ],
    "products": [
        "product_id",
        "product_category_name",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ],
    "sellers": [
        "seller_id",
        "seller_zip_code_prefix",
        "seller_city",
        "seller_state",
    ],
    "geolocation": [
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
        "geolocation_city",
        "geolocation_state",
    ],
    "category_trans": [
        "product_category_name",
        "product_category_name_english",
    ],
}


# ════════════════════════════════════════════════════════════
# DATA TYPE DEFINITIONS
# Expected data types for key columns
# ════════════════════════════════════════════════════════════
EXPECTED_DTYPES = {
    "orders": {
        "order_id"    : "string",
        "customer_id" : "string",
        "order_status": "string",
    },
    "order_items": {
        "order_id"     : "string",
        "product_id"   : "string",
        "seller_id"    : "string",
        "price"        : "float64",
        "freight_value": "float64",
    },
    "order_payments": {
        "order_id"            : "string",
        "payment_value"       : "float64",
        "payment_installments": "int64",
    },
    "customers": {
        "customer_id"       : "string",
        "customer_unique_id": "string",
        "customer_state"    : "string",
    },
    "products": {
        "product_id"           : "string",
        "product_category_name": "string",
    },
    "sellers": {
        "seller_id"   : "string",
        "seller_state": "string",
    },
}


# ════════════════════════════════════════════════════════════
# CORE EXTRACT FUNCTIONS
# ════════════════════════════════════════════════════════════

def load_single_file(name: str, file_path: Path) -> pd.DataFrame:
    """
    Load a single CSV file into a DataFrame.
    Handles encoding issues and logs row/column counts.
    """
    logger.info(f"Loading: {name} -> {file_path.name}")
    start = time.time()

    try:
        # Try UTF-8 first
        df = pd.read_csv(
            file_path,
            encoding="utf-8",
            low_memory=False
        )
    except UnicodeDecodeError:
        
        logger.warning(
            f"UTF-8 failed for {name}, retrying with latin-1..."
        )
        df = pd.read_csv(
            file_path,
            encoding="latin-1",
            low_memory=False
        )

    elapsed = time.time() - start
    logger.info(
        f"[OK] Loaded {name}: "
        f"{len(df):,} rows x {len(df.columns)} cols "
        f"({elapsed:.2f}s)"
    )
    return df


def validate_schema(name: str, df: pd.DataFrame) -> bool:
    """
    Check that all expected columns exist in the DataFrame.
    Logs missing and extra columns.
    """
    if name not in EXPECTED_SCHEMAS:
        logger.warning(
            f"[WARN] No schema defined for {name}, skipping validation"
        )
        return True

    expected = set(EXPECTED_SCHEMAS[name])
    actual   = set(df.columns.tolist())

    missing = expected - actual
    extra   = actual - expected

    if missing:
        logger.error(f"[ERROR] {name} missing columns: {missing}")
        return False

    if extra:
        logger.warning(f"[WARN] {name} has extra columns: {extra}")

    logger.info(f"[OK] Schema validated: {name}")
    return True


def validate_dtypes(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast columns to expected data types.
    Logs any casting failures.
    """
    if name not in EXPECTED_DTYPES:
        return df

    for col, dtype in EXPECTED_DTYPES[name].items():
        if col not in df.columns:
            continue
        try:
            df[col] = df[col].astype(dtype)
        except (ValueError, TypeError) as e:
            logger.warning(
                f"[WARN] Could not cast {name}.{col} to {dtype}: {e}"
            )

    logger.info(f"[OK] Data types validated: {name}")
    return df


def check_null_summary(name: str, df: pd.DataFrame) -> dict:
    """
    Calculate null value counts and percentages per column.
    Returns a summary dictionary.
    """
    null_counts = df.isnull().sum()
    null_pct    = (df.isnull().sum() / len(df) * 100).round(2)

    summary = {}
    for col in df.columns:
        if null_counts[col] > 0:
            summary[col] = {
                "null_count": int(null_counts[col]),
                "null_pct"  : float(null_pct[col]),
            }

    if summary:
        logger.warning(f"[WARN] Null values found in {name}:")
        for col, info in summary.items():
            logger.warning(
                f"  {col}: "
                f"{info['null_count']:,} nulls "
                f"({info['null_pct']}%)"
            )
    else:
        logger.info(f"[OK] No null values found in {name}")

    return summary


def check_duplicates(name: str, df: pd.DataFrame) -> int:
    """
    Check and log duplicate rows in a DataFrame.
    Returns number of duplicates found.
    """
    dup_count = df.duplicated().sum()

    if dup_count > 0:
        logger.warning(
            f"[WARN] {name} has {dup_count:,} duplicate rows "
            f"({dup_count/len(df)*100:.2f}%)"
        )
    else:
        logger.info(f"[OK] No duplicates found in {name}")

    return int(dup_count)


def check_row_count(name: str, df: pd.DataFrame) -> None:
    """
    Log minimum expected row counts per dataset.
    Warns if dataset seems too small.
    """
    min_expected = {
        "orders"        : 90000,
        "order_items"   : 100000,
        "order_payments": 90000,
        "customers"     : 90000,
        "products"      : 30000,
        "sellers"       : 3000,
        "order_reviews" : 90000,
        "geolocation"   : 1000000,
        "category_trans": 70,
    }

    if name in min_expected:
        expected = min_expected[name]
        if len(df) < expected:
            logger.warning(
                f"[WARN] {name} has only {len(df):,} rows, "
                f"expected at least {expected:,}"
            )
        else:
            logger.info(
                f"[OK] Row count OK: {name} -> {len(df):,} rows"
            )


def generate_extract_report(extraction_results: dict) -> dict:
    """
    Generate a summary report of the extraction process.
    Shows rows loaded, nulls, duplicates per file.
    """
    report = {
        "total_files"     : len(extraction_results),
        "successful_loads": 0,
        "failed_loads"    : 0,
        "total_rows"      : 0,
        "files"           : {},
    }

    for name, result in extraction_results.items():
        if result["success"]:
            report["successful_loads"] += 1
            report["total_rows"]       += result["rows"]
        else:
            report["failed_loads"] += 1

        report["files"][name] = {
            "rows"            : result.get("rows", 0),
            "columns"         : result.get("columns", 0),
            "nulls_found"     : result.get("nulls_found", 0),
            "duplicates_found": result.get("duplicates_found", 0),
            "schema_valid"    : result.get("schema_valid", False),
            "success"         : result.get("success", False),
            "load_time_sec"   : result.get("load_time_sec", 0),
        }

    return report


def print_extract_report(report: dict) -> None:
    """
    Print a formatted extraction summary report to console.
    """
    print("\n" + "=" * 65)
    print("   EXTRACTION REPORT")
    print("=" * 65)
    print(f"  Total Files  : {report['total_files']}")
    print(f"  Successful   : {report['successful_loads']}")
    print(f"  Failed       : {report['failed_loads']}")
    print(f"  Total Rows   : {report['total_rows']:,}")
    print("-" * 65)
    print(
        f"  {'File':<20} {'Rows':>10} {'Cols':>5} "
        f"{'Nulls':>8} {'Dups':>6} {'Valid':>6} {'Status':>8}"
    )
    print("-" * 65)

    for name, info in report["files"].items():
        status = "[OK]" if info["success"] else "[FAIL]"
        print(
            f"{name:<20} {info['rows']:>10,} {info['columns']:>5}"
            f"{info['nulls_found']:>8} {info['duplicates_found']:>6} "
            f"{str(info['schema_valid']):>6} {status:>8}"
        )

    print("=" * 65 + "\n")


# ════════════════════════════════════════════════════════════
# MAIN EXTRACT FUNCTION
# ════════════════════════════════════════════════════════════

def extract_all() -> dict:
    """
    Main extraction function.
    Loads all 9 CSV files, validates schemas,
    checks data types, nulls, and duplicates.

    Returns:
        dict: { "orders": DataFrame, "customers": DataFrame, ... }
    """
    logger.info("=" * 55)
    logger.info("EXTRACT PHASE STARTED")
    logger.info("=" * 55)

    # Create directories if not exist
    create_directories()

    dataframes         = {}
    extraction_results = {}
    total_start        = time.time()

    for name, file_path in RAW_FILES.items():

        file_start = time.time()
        result     = {
            "success"         : False,
            "rows"            : 0,
            "columns"         : 0,
            "nulls_found"     : 0,
            "duplicates_found": 0,
            "schema_valid"    : False,
            "load_time_sec"   : 0,
        }

        logger.info(f"-- Processing: {name} --")

        try:
            # Step 1: Load file
            df = load_single_file(name, file_path)

            # Step 2: Validate schema
            schema_valid           = validate_schema(name, df)
            result["schema_valid"] = schema_valid

            # Step 3: Validate and cast data types
            df = validate_dtypes(name, df)

            # Step 4: Check null values
            null_summary          = check_null_summary(name, df)
            result["nulls_found"] = len(null_summary)

            # Step 5: Check duplicates
            dup_count                  = check_duplicates(name, df)
            result["duplicates_found"] = dup_count

            # Step 6: Check row count
            check_row_count(name, df)

            # Step 7: Store results
            result["rows"]          = len(df)
            result["columns"]       = len(df.columns)
            result["success"]       = True
            result["load_time_sec"] = round(time.time() - file_start, 2)

            dataframes[name] = df

        except FileNotFoundError:
            logger.error(f"[ERROR] File not found: {file_path}")
            result["success"] = False

        except Exception as e:
            logger.error(f"[ERROR] Failed to load {name}: {e}")
            result["success"] = False

        extraction_results[name] = result

    # Generate and Print Report
    total_elapsed = time.time() - total_start
    report        = generate_extract_report(extraction_results)
    print_extract_report(report)

    logger.info(
        f"[OK] EXTRACT PHASE COMPLETE -- "
        f"{report['successful_loads']}/{report['total_files']} files loaded "
        f"in {total_elapsed:.2f}s"
    )

    return dataframes


# ── Run directly to test ──────────────────────────────────────────────────
if __name__ == "__main__":
    dataframes = extract_all()

    # Show sample of each loaded dataframe
    for name, df in dataframes.items():
        print(f"\n-- {name} sample --")
        print(df.head(2).to_string())