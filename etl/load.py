# etl/load.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import sqlite3
import time
from loguru import logger
from sqlalchemy import create_engine, text, inspect

from src.config import (
    DATABASE_URL,
    DATABASE_PATH,
    DATABASE_DIR,
    DB_TABLES,
    ETL_LOG_FILE,
    ETL_SETTINGS,
    create_directories,
)

# ── Configure Logger ──────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
    level="INFO",
    colorize=False,
)
logger.add(
    ETL_LOG_FILE,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
    level="INFO",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
)


# ════════════════════════════════════════════════════════════
# DATABASE CONNECTION
# ════════════════════════════════════════════════════════════

def get_engine():
    """
    Create and return SQLAlchemy engine for SQLite.
    """
    create_directories()
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    logger.info(f"[OK] Database engine created: {DATABASE_PATH}")
    return engine


def get_connection():
    """
    Get a raw SQLite connection.
    Useful for executing raw SQL statements.
    """
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=10000")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a dataframe for loading into SQLite.
    - Convert categorical columns to string
    - Convert timestamps to string
    - Replace inf values with None
    - Convert numpy types to Python native types
    """
    df = df.copy()

    # Convert categorical columns to string
    cat_cols = df.select_dtypes(include="category").columns
    for col in cat_cols:
        df[col] = df[col].astype(str)

    # Convert datetime columns to string
    dt_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
    for col in dt_cols:
        df[col] = df[col].astype(str)

    # Replace inf values with NaN then None
    df = df.replace([np.inf, -np.inf], np.nan)

    # Convert numpy int/float to Python native
    for col in df.select_dtypes(include=[np.integer]).columns:
        df[col] = df[col].astype(int)
    for col in df.select_dtypes(include=[np.floating]).columns:
        df[col] = df[col].astype(float)

    return df


def table_exists(engine, table_name: str) -> bool:
    """
    Check if a table exists in the database.
    """
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def get_table_row_count(engine, table_name: str) -> int:
    """
    Get row count of a table in the database.
    """
    with engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT COUNT(*) FROM {table_name}")
        )
        return result.scalar()


# ════════════════════════════════════════════════════════════
# LOAD INDIVIDUAL TABLES
# ════════════════════════════════════════════════════════════

def load_table(
    engine,
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "replace",
    chunksize: int = None,
) -> dict:
    """
    Load a single dataframe into a SQLite table.

    Args:
        engine     : SQLAlchemy engine
        df         : DataFrame to load
        table_name : Target table name in database
        if_exists  : 'replace' or 'append'
        chunksize  : Rows per batch (None = all at once)

    Returns:
        dict with load results
    """
    logger.info(f"-- Loading table: {table_name} --")
    start = time.time()

    result = {
        "table"     : table_name,
        "rows"      : 0,
        "success"   : False,
        "time_sec"  : 0,
    }

    try:
        # Prepare dataframe
        df_clean = prepare_dataframe(df)

        # Use chunk size from config if not specified
        if chunksize is None:
            chunksize = ETL_SETTINGS.get("chunk_size", 10000)

        # Load into database
        df_clean.to_sql(
        name      = table_name,
        con       = engine,
        if_exists = if_exists,
        index     = False,
        chunksize = chunksize,
)

        # Verify row count
        db_count = get_table_row_count(engine, table_name)
        result["rows"]    = db_count
        result["success"] = True

        elapsed = time.time() - start
        result["time_sec"] = round(elapsed, 2)

        logger.info(
            f"  [OK] Loaded {table_name}: "
            f"{db_count:,} rows in {elapsed:.2f}s"
        )

    except Exception as e:
        logger.error(
            f"  [ERROR] Failed to load {table_name}: {e}"
        )
        result["success"] = False

    return result


# ════════════════════════════════════════════════════════════
# CREATE INDEXES
# ════════════════════════════════════════════════════════════

def create_indexes(conn: sqlite3.Connection) -> None:
    """
    Create indexes on frequently queried columns
    to speed up SQL queries.
    """
    logger.info("-- Creating database indexes --")

    indexes = [
        # Orders table indexes
        "CREATE INDEX IF NOT EXISTS idx_orders_customer_id "
        "ON orders(customer_id)",

        "CREATE INDEX IF NOT EXISTS idx_orders_status "
        "ON orders(order_status)",

        "CREATE INDEX IF NOT EXISTS idx_orders_purchase_date "
        "ON orders(order_purchase_timestamp)",

        # Order items indexes
        "CREATE INDEX IF NOT EXISTS idx_items_order_id "
        "ON order_items(order_id)",

        "CREATE INDEX IF NOT EXISTS idx_items_product_id "
        "ON order_items(product_id)",

        "CREATE INDEX IF NOT EXISTS idx_items_seller_id "
        "ON order_items(seller_id)",

        # Order payments indexes
        "CREATE INDEX IF NOT EXISTS idx_payments_order_id "
        "ON order_payments(order_id)",

        "CREATE INDEX IF NOT EXISTS idx_payments_type "
        "ON order_payments(primary_payment_type)",

        # Order reviews indexes
        "CREATE INDEX IF NOT EXISTS idx_reviews_order_id "
        "ON order_reviews(order_id)",

        "CREATE INDEX IF NOT EXISTS idx_reviews_score "
        "ON order_reviews(review_score)",

        # Customers indexes
        "CREATE INDEX IF NOT EXISTS idx_customers_state "
        "ON customers(customer_state)",

        "CREATE INDEX IF NOT EXISTS idx_customers_city "
        "ON customers(customer_city)",

        # Products indexes
        "CREATE INDEX IF NOT EXISTS idx_products_category "
        "ON products(product_category_name)",

        # Sellers indexes
        "CREATE INDEX IF NOT EXISTS idx_sellers_state "
        "ON sellers(seller_state)",

        # Fact orders indexes
        "CREATE INDEX IF NOT EXISTS idx_fact_order_id "
        "ON fact_orders(order_id)",

        "CREATE INDEX IF NOT EXISTS idx_fact_customer_id "
        "ON fact_orders(customer_id)",

        "CREATE INDEX IF NOT EXISTS idx_fact_product_id "
        "ON fact_orders(product_id)",

        "CREATE INDEX IF NOT EXISTS idx_fact_seller_id "
        "ON fact_orders(seller_id)",

        "CREATE INDEX IF NOT EXISTS idx_fact_date_id "
        "ON fact_orders(date_id)",

        # Dim date indexes
        "CREATE INDEX IF NOT EXISTS idx_dim_date_year "
        "ON dim_date(year)",

        "CREATE INDEX IF NOT EXISTS idx_dim_date_month "
        "ON dim_date(month)",
    ]

    created = 0
    for sql in indexes:
        try:
            conn.execute(sql)
            created += 1
        except Exception as e:
            logger.warning(f"  [WARN] Index creation failed: {e}")

    conn.commit()
    logger.info(f"  [OK] Created {created}/{len(indexes)} indexes")


# ════════════════════════════════════════════════════════════
# CREATE VIEWS
# ════════════════════════════════════════════════════════════

def create_views(conn: sqlite3.Connection) -> None:
    """
    Create useful SQL views for analytics queries.
    """
    logger.info("-- Creating database views --")

    views = {
        # View 1: Complete order summary
        "vw_order_summary": """
            CREATE VIEW IF NOT EXISTS vw_order_summary AS
            SELECT
                o.order_id,
                o.customer_id,
                o.order_status,
                o.order_purchase_timestamp,
                o.delivery_days,
                o.delivery_delay_days,
                o.is_late_delivery,
                o.purchase_year,
                o.purchase_month,
                o.purchase_quarter,
                p.total_payment_value,
                p.primary_payment_type,
                p.max_installments,
                r.review_score,
                r.review_sentiment,
                c.customer_state,
                c.customer_city,
                c.customer_region
            FROM orders o
            LEFT JOIN order_payments p ON o.order_id = p.order_id
            LEFT JOIN order_reviews  r ON o.order_id = r.order_id
            LEFT JOIN customers      c ON o.customer_id = c.customer_id
        """,

        # View 2: Product sales summary
        "vw_product_sales": """
            CREATE VIEW IF NOT EXISTS vw_product_sales AS
            SELECT
                i.product_id,
                pr.product_category_name_english AS category,
                pr.product_size,
                COUNT(DISTINCT i.order_id)        AS total_orders,
                SUM(i.price)                      AS total_revenue,
                AVG(i.price)                      AS avg_price,
                SUM(i.freight_value)              AS total_freight,
                AVG(r.review_score)               AS avg_review_score
            FROM order_items i
            LEFT JOIN products     pr ON i.product_id  = pr.product_id
            LEFT JOIN order_reviews r ON i.order_id    = r.order_id
            GROUP BY i.product_id
        """,

        # View 3: Seller performance summary
        "vw_seller_performance": """
            CREATE VIEW IF NOT EXISTS vw_seller_performance AS
            SELECT
                i.seller_id,
                s.seller_state,
                s.seller_region,
                COUNT(DISTINCT i.order_id)        AS total_orders,
                SUM(i.price)                      AS total_revenue,
                AVG(i.price)                      AS avg_order_value,
                AVG(r.review_score)               AS avg_review_score,
                AVG(o.delivery_days)              AS avg_delivery_days
            FROM order_items i
            LEFT JOIN sellers       s ON i.seller_id  = s.seller_id
            LEFT JOIN order_reviews r ON i.order_id   = r.order_id
            LEFT JOIN orders        o ON i.order_id   = o.order_id
            GROUP BY i.seller_id
        """,

        # View 4: Monthly revenue trend
        "vw_monthly_revenue": """
            CREATE VIEW IF NOT EXISTS vw_monthly_revenue AS
            SELECT
                o.purchase_year,
                o.purchase_month,
                COUNT(DISTINCT o.order_id)        AS total_orders,
                SUM(p.total_payment_value)        AS total_revenue,
                AVG(p.total_payment_value)        AS avg_order_value,
                COUNT(DISTINCT o.customer_id)     AS unique_customers
            FROM orders o
            LEFT JOIN order_payments p ON o.order_id = p.order_id
            WHERE o.order_status = 'delivered'
            GROUP BY o.purchase_year, o.purchase_month
            ORDER BY o.purchase_year, o.purchase_month
        """,

        # View 5: Customer purchase summary
        "vw_customer_summary": """
            CREATE VIEW IF NOT EXISTS vw_customer_summary AS
            SELECT
                c.customer_unique_id,
                c.customer_state,
                c.customer_region,
                COUNT(DISTINCT o.order_id)        AS total_orders,
                SUM(p.total_payment_value)        AS total_spent,
                AVG(p.total_payment_value)        AS avg_order_value,
                MIN(o.order_purchase_timestamp)   AS first_order_date,
                MAX(o.order_purchase_timestamp)   AS last_order_date,
                AVG(r.review_score)               AS avg_review_score
            FROM customers c
            LEFT JOIN orders         o ON c.customer_id   = o.customer_id
            LEFT JOIN order_payments p ON o.order_id      = p.order_id
            LEFT JOIN order_reviews  r ON o.order_id      = r.order_id
            GROUP BY c.customer_unique_id
        """,
    }

    created = 0
    for name, sql in views.items():
        try:
            conn.execute(sql)
            created += 1
            logger.info(f"  [OK] Created view: {name}")
        except Exception as e:
            logger.warning(
                f"  [WARN] View creation failed for {name}: {e}"
            )

    conn.commit()
    logger.info(f"  [OK] Created {created}/{len(views)} views")


# ════════════════════════════════════════════════════════════
# VERIFY LOAD
# ════════════════════════════════════════════════════════════

def verify_load(engine) -> dict:
    """
    Verify all tables were loaded correctly.
    Returns row counts for all tables and views.
    """
    logger.info("-- Verifying database load --")

    verification = {}

    # Check all tables
    tables_to_check = [
        "orders", "order_items", "order_payments",
        "order_reviews", "customers", "products",
        "sellers", "geolocation", "category_translation",
        "fact_orders", "dim_customer", "dim_product",
        "dim_seller", "dim_date",
    ]

    for table in tables_to_check:
        try:
            if table_exists(engine, table):
                count = get_table_row_count(engine, table)
                verification[table] = {
                    "exists"  : True,
                    "rows"    : count,
                    "status"  : "[OK]"
                }
                logger.info(
                    f"  [OK] {table}: {count:,} rows"
                )
            else:
                verification[table] = {
                    "exists": False,
                    "rows"  : 0,
                    "status": "[MISSING]"
                }
                logger.warning(f"  [WARN] {table}: table not found")
        except Exception as e:
            verification[table] = {
                "exists": False,
                "rows"  : 0,
                "status": "[ERROR]"
            }
            logger.error(f"  [ERROR] {table}: {e}")

    return verification


# ════════════════════════════════════════════════════════════
# PRINT LOAD REPORT
# ════════════════════════════════════════════════════════════

def print_load_report(
    load_results: list,
    verification: dict
) -> None:
    """
    Print a formatted load summary report.
    """
    print("\n" + "=" * 65)
    print("   LOAD REPORT")
    print("=" * 65)

    total_rows    = sum(r["rows"] for r in load_results)
    successful    = sum(1 for r in load_results if r["success"])
    total_tables  = len(load_results)

    print(f"  Tables Loaded  : {successful}/{total_tables}")
    print(f"  Total Rows     : {total_rows:,}")
    print("-" * 65)
    print(f"  {'Table':<28} {'Rows':>10} {'Time':>8} {'Status':>8}")
    print("-" * 65)

    for r in load_results:
        print(
            f"{r['table']:<28} "
            f"{r['rows']:>10,} "
            f"{r['time_sec']:>7.2f}s "
            f"{('[OK]' if r['success'] else '[FAIL]'):>8}"
        )

    print("-" * 65)
    print("  Database Verification:")
    print("-" * 65)

    for table, info in verification.items():
        print(
            f"  {table:<28} "
            f"{info['rows']:>10,} "
            f"{info['status']:>8}"
        )

    print("=" * 65 + "\n")


# ════════════════════════════════════════════════════════════
# MAIN LOAD FUNCTION
# ════════════════════════════════════════════════════════════

def load_all(
    cleaned_dataframes: dict,
    master: pd.DataFrame,
    star_schema: dict,
) -> None:
    """
    Main load function.
    Loads all cleaned dataframes and star schema
    into SQLite database.

    Args:
        cleaned_dataframes : dict of cleaned DataFrames
        master             : master merged DataFrame
        star_schema        : dict of star schema DataFrames
    """
    logger.info("=" * 55)
    logger.info("LOAD PHASE STARTED")
    logger.info("=" * 55)

    total_start  = time.time()
    load_results = []

    # Create engine
    engine = get_engine()

    # ── Load Raw/Cleaned Tables ───────────────────────────
    logger.info("-- Loading cleaned tables --")

    raw_tables = {
    DB_TABLES["orders"]        : cleaned_dataframes.get("orders"),
    DB_TABLES["order_items"]   : cleaned_dataframes.get("order_items"),
    DB_TABLES["order_payments"]: cleaned_dataframes.get("order_payments"),
    DB_TABLES["order_reviews"] : cleaned_dataframes.get("order_reviews"),
    DB_TABLES["customers"]     : cleaned_dataframes.get("customers"),
    DB_TABLES["products"]      : cleaned_dataframes.get("products"),
    DB_TABLES["sellers"]       : cleaned_dataframes.get("sellers"),
    DB_TABLES["geolocation"]   : cleaned_dataframes.get("geolocation"),
    "category_translation"     : cleaned_dataframes.get("category_trans"),
    }

    for table_name, df in raw_tables.items():
        if df is not None:
            result = load_table(engine, df, table_name)
            load_results.append(result)

    # Load master dataset
    result = load_table(engine, master, "master_dataset")
    load_results.append(result)

    # ── Load Star Schema Tables ───────────────────────────
    logger.info("-- Loading star schema tables --")

    star_tables = {
        DB_TABLES["fact_orders"]  : star_schema.get("fact_orders"),
        DB_TABLES["dim_customer"] : star_schema.get("dim_customer"),
        DB_TABLES["dim_product"]  : star_schema.get("dim_product"),
        DB_TABLES["dim_seller"]   : star_schema.get("dim_seller"),
        DB_TABLES["dim_date"]     : star_schema.get("dim_date"),
    }

    for table_name, df in star_tables.items():
        if df is not None:
            result = load_table(engine, df, table_name)
            load_results.append(result)

    # ── Create Indexes and Views ──────────────────────────
    conn = get_connection()
    create_indexes(conn)
    create_views(conn)
    conn.close()

    # ── Verify Load ───────────────────────────────────────
    verification = verify_load(engine)

    # ── Print Report ──────────────────────────────────────
    print_load_report(load_results, verification)

    total_elapsed = time.time() - total_start
    logger.info(
        f"[OK] LOAD PHASE COMPLETE in {total_elapsed:.2f}s"
    )
    logger.info(
        f"[OK] Database saved at: {DATABASE_PATH}"
    )


# ── Run directly to test ──────────────────────────────────────────────────
if __name__ == "__main__":
    from etl.extract   import extract_all
    from etl.transform import transform_all

    # Extract
    raw_dataframes = extract_all()

    # Transform
    cleaned, master, star_schema = transform_all(raw_dataframes)

    # Load
    load_all(cleaned, master, star_schema)