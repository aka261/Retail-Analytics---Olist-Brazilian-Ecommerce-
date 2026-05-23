# src/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────────────
load_dotenv()

# ── Base Directory ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Data Directories ──────────────────────────────────────────────────────
DATA_DIR          = BASE_DIR / "data"
RAW_DATA_DIR      = DATA_DIR / "raw"
PROCESSED_DATA_DIR= DATA_DIR / "processed"
WAREHOUSE_DIR     = DATA_DIR / "warehouse"

# ── Database ──────────────────────────────────────────────────────────────
DATABASE_DIR      = BASE_DIR / "database"
DATABASE_PATH     = DATABASE_DIR / "retail.db"
DATABASE_URL      = f"sqlite:///{DATABASE_PATH}"

# ── Output Directories ────────────────────────────────────────────────────
OUTPUTS_DIR       = BASE_DIR / "outputs"
PLOTS_DIR         = OUTPUTS_DIR / "plots"
REPORTS_DIR       = OUTPUTS_DIR / "reports"
ETL_LOGS_DIR      = OUTPUTS_DIR / "etl_logs"

# ── Logs ──────────────────────────────────────────────────────────────────
LOGS_DIR          = BASE_DIR / "logs"
LOG_FILE          = LOGS_DIR / "retail_analytics.log"
ETL_LOG_FILE      = LOGS_DIR / "etl_pipeline.log"

# ── Raw Data File Paths ───────────────────────────────────────────────────
RAW_FILES = {
    "orders"        : RAW_DATA_DIR / "olist_orders_dataset.csv",
    "order_items"   : RAW_DATA_DIR / "olist_order_items_dataset.csv",
    "order_payments": RAW_DATA_DIR / "olist_order_payments_dataset.csv",
    "order_reviews" : RAW_DATA_DIR / "olist_order_reviews_dataset.csv",
    "customers"     : RAW_DATA_DIR / "olist_customers_dataset.csv",
    "products"      : RAW_DATA_DIR / "olist_products_dataset.csv",
    "sellers"       : RAW_DATA_DIR / "olist_sellers_dataset.csv",
    "geolocation"   : RAW_DATA_DIR / "olist_geolocation_dataset.csv",
    "category_trans": RAW_DATA_DIR / "product_category_name_translation.csv",
}

# ── Processed Data File Paths ─────────────────────────────────────────────
PROCESSED_FILES = {
    "orders"        : PROCESSED_DATA_DIR / "orders_cleaned.csv",
    "order_items"   : PROCESSED_DATA_DIR / "order_items_cleaned.csv",
    "order_payments": PROCESSED_DATA_DIR / "order_payments_cleaned.csv",
    "order_reviews" : PROCESSED_DATA_DIR / "order_reviews_cleaned.csv",
    "customers"     : PROCESSED_DATA_DIR / "customers_cleaned.csv",
    "products"      : PROCESSED_DATA_DIR / "products_cleaned.csv",
    "sellers"       : PROCESSED_DATA_DIR / "sellers_cleaned.csv",
    "master"        : PROCESSED_DATA_DIR / "master_dataset.csv",
}

# ── Warehouse (Star Schema) File Paths ────────────────────────────────────
WAREHOUSE_FILES = {
    "fact_orders"   : WAREHOUSE_DIR / "fact_orders.csv",
    "dim_customer"  : WAREHOUSE_DIR / "dim_customer.csv",
    "dim_product"   : WAREHOUSE_DIR / "dim_product.csv",
    "dim_seller"    : WAREHOUSE_DIR / "dim_seller.csv",
    "dim_date"      : WAREHOUSE_DIR / "dim_date.csv",
}

# ── Database Table Names ──────────────────────────────────────────────────
DB_TABLES = {
    # Raw tables
    "orders"        : "orders",
    "order_items"   : "order_items",
    "order_payments": "order_payments",
    "order_reviews" : "order_reviews",
    "customers"     : "customers",
    "products"      : "products",
    "sellers"       : "sellers",
    "geolocation"   : "geolocation",
    "category_trans": "category_translation",

    # Star schema tables
    "fact_orders"   : "fact_orders",
    "dim_customer"  : "dim_customer",
    "dim_product"   : "dim_product",
    "dim_seller"    : "dim_seller",
    "dim_date"      : "dim_date",
}

# ── ETL Settings ──────────────────────────────────────────────────────────
ETL_SETTINGS = {
    "chunk_size"        : 10000,      # rows per chunk for large files
    "log_level"         : "INFO",
    "save_processed"    : True,       # save cleaned CSVs to processed
    "save_warehouse"    : True,       # save star schema to warehouse
}

# ── Data Cleaning Settings ────────────────────────────────────────────────
CLEANING_SETTINGS = {
    "drop_duplicate_orders"     : True,
    "fill_missing_reviews"      : True,
    "remove_cancelled_orders"   : False,  # keep cancelled for analysis
    "min_order_value"           : 0.0,
    "date_columns": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
}

# ── RFM Analysis Settings ─────────────────────────────────────────────────
RFM_SETTINGS = {
    "recency_bins"  : 5,
    "frequency_bins": 5,
    "monetary_bins" : 5,
    "segments": {
        "Champions"         : (4, 5),
        "Loyal Customers"   : (3, 4),
        "Potential Loyalists": (3, 3),
        "At Risk"           : (2, 2),
        "Lost"              : (1, 1),
    }
}

# ── Forecasting Settings ──────────────────────────────────────────────────
FORECAST_SETTINGS = {
    "periods"       : 90,       # forecast 90 days ahead
    "frequency"     : "D",      # daily
    "seasonality"   : True,
    "yearly"        : True,
    "weekly"        : True,
}

# ── Visualization Settings ────────────────────────────────────────────────
VIZ_SETTINGS = {
    "theme"         : "plotly_dark",
    "color_palette" : [
        "#3498DB", "#2ECC71", "#E74C3C",
        "#F39C12", "#9B59B6", "#1ABC9C",
        "#E67E22", "#34495E"
    ],
    "figure_size"   : (16, 8),
    "dpi"           : 150,
    "save_format"   : "png",
}

# ── Dashboard Settings ────────────────────────────────────────────────────
DASHBOARD_SETTINGS = {
    "host"          : os.getenv("DASH_HOST", "127.0.0.1"),
    "port"          : int(os.getenv("DASH_PORT", 8050)),
    "debug"         : os.getenv("DASH_DEBUG", "True") == "True",
    "title"         : "Retail Analytics Dashboard",
}

# ── Email Settings (for reports) ──────────────────────────────────────────
EMAIL_SETTINGS = {
    "smtp_server"   : os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    "smtp_port"     : int(os.getenv("SMTP_PORT", 587)),
    "sender_email"  : os.getenv("SENDER_EMAIL", ""),
    "sender_password": os.getenv("SENDER_PASSWORD", ""),
    "receiver_email": os.getenv("RECEIVER_EMAIL", ""),
}

# ── Scheduler Settings ────────────────────────────────────────────────────
SCHEDULER_SETTINGS = {
    "etl_schedule"      : "0 6 * * *",     # run ETL every day at 6AM
    "report_schedule"   : "0 8 * * 1",     # send report every Monday 8AM
    "timezone"          : "Asia/Kolkata",
}

# ── Churn Prediction Settings ─────────────────────────────────────────────
CHURN_SETTINGS = {
    "test_size"         : 0.20,
    "random_state"      : 42,
    "churn_threshold"   : 180,      # days without order = churned
    "model_path"        : OUTPUTS_DIR / "models" / "churn_model.pkl",
}

# ── Create All Directories if Not Exist ───────────────────────────────────
def create_directories():
    dirs = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, WAREHOUSE_DIR,
        DATABASE_DIR, OUTPUTS_DIR, PLOTS_DIR, REPORTS_DIR,
        ETL_LOGS_DIR, LOGS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

# ── Validate Raw Files Exist ──────────────────────────────────────────────
def validate_raw_files():
    missing = []
    for name, path in RAW_FILES.items():
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"\n Missing raw files:\n" + "\n".join(missing)
        )
    print(" All raw files validated successfully!")
    return True

# ── Print Config Summary ──────────────────────────────────────────────────
def print_config():
    print("=" * 55)
    print("   RETAIL ANALYTICS — PROJECT CONFIGURATION")
    print("=" * 55)
    print(f"  Base Directory    : {BASE_DIR}")
    print(f"  Raw Data          : {RAW_DATA_DIR}")
    print(f"  Database          : {DATABASE_PATH}")
    print(f"  Outputs           : {OUTPUTS_DIR}")
    print(f"  Dashboard Port    : {DASHBOARD_SETTINGS['port']}")
    print(f"  Forecast Periods  : {FORECAST_SETTINGS['periods']} days")
    print(f"  Churn Threshold   : {CHURN_SETTINGS['churn_threshold']} days")
    print("=" * 55)


# ── Run directly to test ──────────────────────────────────────────────────
if __name__ == "__main__":
    create_directories()
    print_config()
    validate_raw_files()