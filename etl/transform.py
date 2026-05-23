# etl/transform.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from loguru import logger
import time

from src.config import (
    CLEANING_SETTINGS,
    PROCESSED_FILES,
    ETL_SETTINGS,
    ETL_LOG_FILE,
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
# TRANSFORM ORDERS
# ════════════════════════════════════════════════════════════

def transform_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform the orders dataframe.
    - Parse date columns
    - Add delivery metrics
    - Add order status flags
    """
    logger.info("-- Transforming: orders --")
    start    = time.time()
    original = len(df)

    # Step 1: Parse all date columns
    date_cols = CLEANING_SETTINGS["date_columns"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    logger.info(f"  [OK] Parsed {len(date_cols)} date columns")

    # Step 2: Drop duplicate order_ids
    if CLEANING_SETTINGS["drop_duplicate_orders"]:
        before = len(df)
        df     = df.drop_duplicates(subset=["order_id"])
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(f"  [WARN] Dropped {dropped:,} duplicate orders")
        else:
            logger.info("  [OK] No duplicate orders found")

    # Step 3: Fill missing approved_at with purchase timestamp
    mask = df["order_approved_at"].isnull()
    df.loc[mask, "order_approved_at"] = df.loc[
        mask, "order_purchase_timestamp"
    ]
    logger.info(f"  [OK] Filled {mask.sum():,} missing order_approved_at")

    # Step 4: Add delivery time in days
    df["delivery_days"] = (
        df["order_delivered_customer_date"] -
        df["order_purchase_timestamp"]
    ).dt.days
    logger.info("  [OK] Added delivery_days column")

    # Step 5: Add estimated vs actual delivery difference
    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"] -
        df["order_estimated_delivery_date"]
    ).dt.days
    logger.info("  [OK] Added delivery_delay_days column")

    # Step 6: Add is_late flag
    df["is_late_delivery"] = (
        df["delivery_delay_days"] > 0
    ).astype(int)
    logger.info("  [OK] Added is_late_delivery flag")

    # Step 7: Add purchase date parts
    df["purchase_year"]    = df["order_purchase_timestamp"].dt.year
    df["purchase_month"]   = df["order_purchase_timestamp"].dt.month
    df["purchase_day"]     = df["order_purchase_timestamp"].dt.day
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.day_name()
    df["purchase_hour"]    = df["order_purchase_timestamp"].dt.hour
    df["purchase_quarter"] = df["order_purchase_timestamp"].dt.quarter
    logger.info("  [OK] Added purchase date parts")

    # Step 8: Add order status flags
    df["is_delivered"]  = (df["order_status"] == "delivered").astype(int)
    df["is_cancelled"]  = (df["order_status"] == "canceled").astype(int)
    df["is_processing"] = (
        df["order_status"].isin(["processing", "approved"])
    ).astype(int)
    logger.info("  [OK] Added order status flags")

    elapsed = time.time() - start
    logger.info(
        f"  [OK] Orders transformed: "
        f"{original:,} -> {len(df):,} rows ({elapsed:.2f}s)"
    )
    return df


# ════════════════════════════════════════════════════════════
# TRANSFORM ORDER ITEMS
# ════════════════════════════════════════════════════════════

def transform_order_items(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform order items dataframe.
    - Parse dates
    - Add revenue columns
    - Add price buckets
    """
    logger.info("-- Transforming: order_items --")
    start = time.time()

    # Step 1: Parse shipping date
    df["shipping_limit_date"] = pd.to_datetime(
        df["shipping_limit_date"], errors="coerce"
    )
    logger.info("  [OK] Parsed shipping_limit_date")

    # Step 2: Add total item value
    df["total_item_value"] = (
        df["price"] + df["freight_value"]
    ).round(2)
    logger.info("  [OK] Added total_item_value column")

    # Step 3: Add freight percentage
    df["freight_pct"] = (
        df["freight_value"] / df["total_item_value"] * 100
    ).round(2)
    logger.info("  [OK] Added freight_pct column")

    # Step 4: Add price buckets
    df["price_bucket"] = pd.cut(
        df["price"],
        bins=[0, 50, 100, 250, 500, 1000, float("inf")],
        labels=[
            "Under 50",
            "50-100",
            "100-250",
            "250-500",
            "500-1000",
            "Above 1000"
        ],
    )
    logger.info("  [OK] Added price_bucket column")

    # Step 5: Remove negative prices
    before = len(df)
    df     = df[df["price"] >= 0]
    df     = df[df["freight_value"] >= 0]
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(
            f"  [WARN] Removed {dropped:,} rows with negative prices"
        )
    else:
        logger.info("  [OK] No negative prices found")

    elapsed = time.time() - start
    logger.info(
        f"  [OK] Order items transformed: {len(df):,} rows ({elapsed:.2f}s)"
    )
    return df


# ════════════════════════════════════════════════════════════
# TRANSFORM ORDER PAYMENTS
# ════════════════════════════════════════════════════════════

def transform_order_payments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform order payments dataframe.
    - Aggregate multiple payments per order
    - Add payment method flags
    """
    logger.info("-- Transforming: order_payments --")
    start = time.time()

    # Step 1: Remove zero value payments
    before = len(df)
    df     = df[df["payment_value"] > 0]
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(
            f"  [WARN] Removed {dropped:,} zero-value payments"
        )
    else:
        logger.info("  [OK] No zero-value payments found")

    # Step 2: Add payment type flags
    df["is_credit_card"] = (
        df["payment_type"] == "credit_card"
    ).astype(int)
    df["is_boleto"]      = (
        df["payment_type"] == "boleto"
    ).astype(int)
    df["is_voucher"]     = (
        df["payment_type"] == "voucher"
    ).astype(int)
    df["is_debit_card"]  = (
        df["payment_type"] == "debit_card"
    ).astype(int)
    logger.info("  [OK] Added payment type flags")

    # Step 3: Add installment flag
    df["has_installments"] = (
        df["payment_installments"] > 1
    ).astype(int)
    logger.info("  [OK] Added has_installments flag")

    # Step 4: Aggregate payments per order
    payments_agg = df.groupby("order_id").agg(
        total_payment_value  = ("payment_value",        "sum"),
        payment_count        = ("payment_sequential",   "count"),
        max_installments     = ("payment_installments", "max"),
        primary_payment_type = ("payment_type",         "first"),
        used_credit_card     = ("is_credit_card",       "max"),
        used_boleto          = ("is_boleto",             "max"),
        used_voucher         = ("is_voucher",            "max"),
        used_debit_card      = ("is_debit_card",        "max"),
    ).reset_index()
    logger.info(
        f"  [OK] Aggregated payments: "
        f"{len(df):,} rows -> {len(payments_agg):,} orders"
    )

    elapsed = time.time() - start
    logger.info(
        f"  [OK] Payments transformed: "
        f"{len(payments_agg):,} rows ({elapsed:.2f}s)"
    )
    return payments_agg


# ════════════════════════════════════════════════════════════
# TRANSFORM ORDER REVIEWS
# ════════════════════════════════════════════════════════════

def transform_order_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform order reviews dataframe.
    - Fill missing comments
    - Add sentiment buckets
    - Parse dates
    """
    logger.info("-- Transforming: order_reviews --")
    start = time.time()

    # Step 1: Fill missing comment titles
    df["review_comment_title"] = (
        df["review_comment_title"].fillna("No Title")
    )
    logger.info("  [OK] Filled missing review_comment_title")

    # Step 2: Fill missing comment messages
    df["review_comment_message"] = (
        df["review_comment_message"].fillna("No Comment")
    )
    logger.info("  [OK] Filled missing review_comment_message")

    # Step 3: Parse date columns
    df["review_creation_date"] = pd.to_datetime(
        df["review_creation_date"], errors="coerce"
    )
    df["review_answer_timestamp"] = pd.to_datetime(
        df["review_answer_timestamp"], errors="coerce"
    )
    logger.info("  [OK] Parsed review date columns")

    # Step 4: Add sentiment bucket based on score
    df["review_sentiment"] = pd.cut(
        df["review_score"],
        bins=[0, 2, 3, 5],
        labels=["Negative", "Neutral", "Positive"],
    )
    logger.info("  [OK] Added review_sentiment column")

    # Step 5: Add response time in hours
    df["review_response_hours"] = (
        df["review_answer_timestamp"] -
        df["review_creation_date"]
    ).dt.total_seconds() / 3600
    df["review_response_hours"] = (
        df["review_response_hours"].round(2)
    )
    logger.info("  [OK] Added review_response_hours column")

    # Step 6: Keep only one review per order (latest)
    before = len(df)
    df     = df.sort_values("review_creation_date", ascending=False)
    df     = df.drop_duplicates(subset=["order_id"], keep="first")
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(
            f"  [WARN] Dropped {dropped:,} duplicate order reviews"
        )
    else:
        logger.info("  [OK] No duplicate reviews found")

    elapsed = time.time() - start
    logger.info(
        f"  [OK] Reviews transformed: {len(df):,} rows ({elapsed:.2f}s)"
    )
    return df


# ════════════════════════════════════════════════════════════
# TRANSFORM CUSTOMERS
# ════════════════════════════════════════════════════════════

def transform_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform customers dataframe.
    - Standardize city and state names
    - Add region mapping
    """
    logger.info("-- Transforming: customers --")
    start = time.time()

    # Step 1: Standardize city names to title case
    df["customer_city"] = (
        df["customer_city"].str.strip().str.title()
    )
    logger.info("  [OK] Standardized customer_city to title case")

    # Step 2: Standardize state to uppercase
    df["customer_state"] = (
        df["customer_state"].str.strip().str.upper()
    )
    logger.info("  [OK] Standardized customer_state to uppercase")

    # Step 3: Add Brazil region mapping
    region_map = {
        "SP": "Southeast", "RJ": "Southeast", "MG": "Southeast",
        "ES": "Southeast", "RS": "South",      "SC": "South",
        "PR": "South",      "BA": "Northeast",  "CE": "Northeast",
        "PE": "Northeast",  "MA": "Northeast",  "PB": "Northeast",
        "RN": "Northeast",  "AL": "Northeast",  "SE": "Northeast",
        "PI": "Northeast",  "PA": "North",      "AM": "North",
        "RO": "North",      "AC": "North",      "AP": "North",
        "RR": "North",      "TO": "North",      "MT": "Central-West",
        "MS": "Central-West","GO": "Central-West","DF": "Central-West",
    }
    df["customer_region"] = (
        df["customer_state"].map(region_map).fillna("Unknown")
    )
    logger.info("  [OK] Added customer_region column")

    elapsed = time.time() - start
    logger.info(
        f"  [OK] Customers transformed: {len(df):,} rows ({elapsed:.2f}s)"
    )
    return df


# ════════════════════════════════════════════════════════════
# TRANSFORM PRODUCTS
# ════════════════════════════════════════════════════════════

def transform_products(
    df: pd.DataFrame,
    category_trans: pd.DataFrame
) -> pd.DataFrame:
    """
    Clean and transform products dataframe.
    - Fill missing category names
    - Merge English category names
    - Fill missing dimensions with median
    - Add product size category
    """
    logger.info("-- Transforming: products --")
    start = time.time()

    # Step 1: Fill missing category names
    df["product_category_name"] = (
        df["product_category_name"].fillna("unknown")
    )
    logger.info("  [OK] Filled missing product_category_name")

    # Step 2: Merge English category translation
    df = df.merge(
        category_trans,
        on="product_category_name",
        how="left"
    )
    df["product_category_name_english"] = (
        df["product_category_name_english"].fillna(
            df["product_category_name"]
        )
    )
    logger.info("  [OK] Merged English category names")

    # Step 3: Fill missing dimensions with median
    dim_cols = [
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
    ]
    for col in dim_cols:
        if col in df.columns:
            median_val = df[col].median()
            missing    = df[col].isnull().sum()
            df[col]    = df[col].fillna(median_val)
            if missing > 0:
                logger.info(
                    f"  [OK] Filled {missing:,} nulls in {col} "
                    f"with median ({median_val:.1f})"
                )
    logger.info("  [OK] Filled missing product dimensions")

    # Step 4: Add product volume in cubic cm
    df["product_volume_cm3"] = (
        df["product_length_cm"] *
        df["product_height_cm"] *
        df["product_width_cm"]
    ).round(2)
    logger.info("  [OK] Added product_volume_cm3 column")

    # Step 5: Add product size category
    df["product_size"] = pd.cut(
        df["product_weight_g"],
        bins=[0, 500, 2000, 5000, float("inf")],
        labels=["Small", "Medium", "Large", "Extra Large"],
    )
    logger.info("  [OK] Added product_size column")

    elapsed = time.time() - start
    logger.info(
        f"  [OK] Products transformed: {len(df):,} rows ({elapsed:.2f}s)"
    )
    return df


# ════════════════════════════════════════════════════════════
# TRANSFORM SELLERS
# ════════════════════════════════════════════════════════════

def transform_sellers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform sellers dataframe.
    - Standardize city and state
    - Add region mapping
    """
    logger.info("-- Transforming: sellers --")
    start = time.time()

    # Step 1: Standardize city names
    df["seller_city"] = (
        df["seller_city"].str.strip().str.title()
    )
    logger.info("  [OK] Standardized seller_city")

    # Step 2: Standardize state names
    df["seller_state"] = (
        df["seller_state"].str.strip().str.upper()
    )
    logger.info("  [OK] Standardized seller_state")

    # Step 3: Add seller region
    region_map = {
        "SP": "Southeast", "RJ": "Southeast", "MG": "Southeast",
        "ES": "Southeast", "RS": "South",      "SC": "South",
        "PR": "South",      "BA": "Northeast",  "CE": "Northeast",
        "PE": "Northeast",  "MA": "Northeast",  "PB": "Northeast",
        "RN": "Northeast",  "AL": "Northeast",  "SE": "Northeast",
        "PI": "Northeast",  "PA": "North",      "AM": "North",
        "RO": "North",      "AC": "North",      "AP": "North",
        "RR": "North",      "TO": "North",      "MT": "Central-West",
        "MS": "Central-West","GO": "Central-West","DF": "Central-West",
    }
    df["seller_region"] = (
        df["seller_state"].map(region_map).fillna("Unknown")
    )
    logger.info("  [OK] Added seller_region column")

    elapsed = time.time() - start
    logger.info(
        f"  [OK] Sellers transformed: {len(df):,} rows ({elapsed:.2f}s)"
    )
    return df


# ════════════════════════════════════════════════════════════
# TRANSFORM GEOLOCATION
# ════════════════════════════════════════════════════════════

def transform_geolocation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform geolocation dataframe.
    - Remove duplicates
    - Remove outliers outside Brazil bounds
    - Standardize city and state names
    """
    logger.info("-- Transforming: geolocation --")
    start    = time.time()
    original = len(df)

    # Step 1: Remove duplicate zip codes
    df = df.drop_duplicates(
        subset=["geolocation_zip_code_prefix"]
    )
    logger.info(
        f"  [OK] Removed duplicates: "
        f"{original:,} -> {len(df):,} rows"
    )

    # Step 2: Remove coordinates outside Brazil bounds
    # Brazil lat: -33.75 to 5.27 | lng: -73.99 to -28.85
    before = len(df)
    df     = df[
        (df["geolocation_lat"] >= -33.75) &
        (df["geolocation_lat"] <=   5.27) &
        (df["geolocation_lng"] >= -73.99) &
        (df["geolocation_lng"] <= -28.85)
    ]
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(
            f"  [WARN] Removed {dropped:,} coordinates outside Brazil"
        )
    else:
        logger.info("  [OK] All coordinates within Brazil bounds")

    # Step 3: Standardize city and state
    df["geolocation_city"]  = (
        df["geolocation_city"].str.strip().str.title()
    )
    df["geolocation_state"] = (
        df["geolocation_state"].str.strip().str.upper()
    )
    logger.info("  [OK] Standardized geolocation city and state")

    elapsed = time.time() - start
    logger.info(
        f"  [OK] Geolocation transformed: "
        f"{len(df):,} rows ({elapsed:.2f}s)"
    )
    return df


# ════════════════════════════════════════════════════════════
# BUILD MASTER DATASET
# ════════════════════════════════════════════════════════════

def build_master_dataset(dataframes: dict) -> pd.DataFrame:
    """
    Merge all cleaned dataframes into one master dataset.
    Joins: orders + order_items + payments + reviews
           + customers + products + sellers
    """
    logger.info("-- Building master dataset --")
    start = time.time()

    orders   = dataframes["orders"]
    items    = dataframes["order_items"]
    payments = dataframes["order_payments"]
    reviews  = dataframes["order_reviews"]
    customers= dataframes["customers"]
    products = dataframes["products"]
    sellers  = dataframes["sellers"]

    # Step 1: Orders + Items
    master = orders.merge(items, on="order_id", how="left")
    logger.info(
        f"  [OK] orders + items: {len(master):,} rows"
    )

    # Step 2: + Payments
    master = master.merge(payments, on="order_id", how="left")
    logger.info(
        f"  [OK] + payments: {len(master):,} rows"
    )

    # Step 3: + Reviews
    review_cols = [
        "order_id", "review_score",
        "review_sentiment", "review_response_hours"
    ]
    master = master.merge(
        reviews[review_cols], on="order_id", how="left"
    )
    logger.info(
        f"  [OK] + reviews: {len(master):,} rows"
    )

    # Step 4: + Customers
    master = master.merge(customers, on="customer_id", how="left")
    logger.info(
        f"  [OK] + customers: {len(master):,} rows"
    )

    # Step 5: + Products
    master = master.merge(products, on="product_id", how="left")
    logger.info(
        f"  [OK] + products: {len(master):,} rows"
    )

    # Step 6: + Sellers
    master = master.merge(sellers, on="seller_id", how="left")
    logger.info(
        f"  [OK] + sellers: {len(master):,} rows"
    )

    # Step 7: Fill missing review scores with median
    median_score = master["review_score"].median()
    master["review_score"] = (
        master["review_score"].fillna(median_score)
    )
    logger.info(
        f"  [OK] Filled missing review scores with {median_score}"
    )

    elapsed = time.time() - start
    logger.info(
        f"  [OK] Master dataset built: "
        f"{len(master):,} rows x {len(master.columns)} cols "
        f"({elapsed:.2f}s)"
    )
    return master


# ════════════════════════════════════════════════════════════
# BUILD STAR SCHEMA
# ════════════════════════════════════════════════════════════

def build_dim_date(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Build date dimension table from order purchase timestamps.
    """
    logger.info("-- Building dim_date --")

    dates = orders["order_purchase_timestamp"].dropna().unique()
    dates = pd.to_datetime(dates)

    dim_date = pd.DataFrame({"date": dates})
    dim_date["date_id"]     = dim_date["date"].dt.strftime("%Y%m%d").astype(int)
    dim_date["year"]        = dim_date["date"].dt.year
    dim_date["month"]       = dim_date["date"].dt.month
    dim_date["month_name"]  = dim_date["date"].dt.month_name()
    dim_date["quarter"]     = dim_date["date"].dt.quarter
    dim_date["day"]         = dim_date["date"].dt.day
    dim_date["day_name"]    = dim_date["date"].dt.day_name()
    dim_date["week"]        = dim_date["date"].dt.isocalendar().week.astype(int)
    dim_date["is_weekend"]  = dim_date["date"].dt.dayofweek.isin([5, 6]).astype(int)
    dim_date               = dim_date.drop_duplicates(subset=["date_id"])
    dim_date               = dim_date.sort_values("date_id").reset_index(drop=True)

    logger.info(f"  [OK] dim_date: {len(dim_date):,} rows")
    return dim_date


def build_star_schema(dataframes: dict) -> dict:
    """
    Build star schema dimension and fact tables.

    Returns:
        dict with fact_orders, dim_customer, dim_product,
              dim_seller, dim_date
    """
    logger.info("-- Building star schema --")
    start = time.time()

    orders   = dataframes["orders"]
    items    = dataframes["order_items"]
    payments = dataframes["order_payments"]
    reviews  = dataframes["order_reviews"]
    customers= dataframes["customers"]
    products = dataframes["products"]
    sellers  = dataframes["sellers"]

    # ── Dimension Tables ──────────────────────────────────

    # dim_customer
    dim_customer = customers[[
        "customer_id", "customer_unique_id",
        "customer_city", "customer_state", "customer_region",
        "customer_zip_code_prefix",
    ]].drop_duplicates(subset=["customer_id"])
    logger.info(f"  [OK] dim_customer: {len(dim_customer):,} rows")

    # dim_product
    dim_product = products[[
        "product_id",
        "product_category_name",
        "product_category_name_english",
        "product_weight_g",
        "product_volume_cm3",
        "product_size",
    ]].drop_duplicates(subset=["product_id"])
    logger.info(f"  [OK] dim_product: {len(dim_product):,} rows")

    # dim_seller
    dim_seller = sellers[[
        "seller_id",
        "seller_city",
        "seller_state",
        "seller_region",
        "seller_zip_code_prefix",
    ]].drop_duplicates(subset=["seller_id"])
    logger.info(f"  [OK] dim_seller: {len(dim_seller):,} rows")

    # dim_date
    dim_date = build_dim_date(orders)

    # ── Fact Table ────────────────────────────────────────
    # Start with order items as the grain
    fact = items[[
        "order_id", "product_id", "seller_id",
        "price", "freight_value", "total_item_value",
    ]].copy()

    # Add customer_id and date from orders
    fact = fact.merge(
        orders[[
            "order_id", "customer_id",
            "order_purchase_timestamp",
            "order_status", "delivery_days",
            "delivery_delay_days", "is_late_delivery",
            "is_delivered",
        ]],
        on="order_id", how="left"
    )

    # Add payment info
    fact = fact.merge(
        payments[[
            "order_id", "total_payment_value",
            "primary_payment_type", "max_installments",
        ]],
        on="order_id", how="left"
    )

    # Add review score
    fact = fact.merge(
        reviews[["order_id", "review_score", "review_sentiment"]],
        on="order_id", how="left"
    )

    # Add date_id
    fact["date_id"] = (
        pd.to_datetime(fact["order_purchase_timestamp"])
        .dt.strftime("%Y%m%d")
        .astype(int)
    )

    logger.info(f"  [OK] fact_orders: {len(fact):,} rows")

    elapsed = time.time() - start
    logger.info(
        f"  [OK] Star schema built in {elapsed:.2f}s"
    )

    return {
        "fact_orders" : fact,
        "dim_customer": dim_customer,
        "dim_product" : dim_product,
        "dim_seller"  : dim_seller,
        "dim_date"    : dim_date,
    }


# ════════════════════════════════════════════════════════════
# SAVE PROCESSED FILES
# ════════════════════════════════════════════════════════════

def save_processed_files(
    dataframes: dict,
    master: pd.DataFrame,
    star_schema: dict
) -> None:
    """
    Save all cleaned dataframes and star schema
    to processed and warehouse folders.
    """
    logger.info("-- Saving processed files --")
    create_directories()

    # Save cleaned individual files
    save_map = {
        "orders"        : PROCESSED_FILES["orders"],
        "order_items"   : PROCESSED_FILES["order_items"],
        "order_payments": PROCESSED_FILES["order_payments"],
        "order_reviews" : PROCESSED_FILES["order_reviews"],
        "customers"     : PROCESSED_FILES["customers"],
        "products"      : PROCESSED_FILES["products"],
        "sellers"       : PROCESSED_FILES["sellers"],
    }

    for name, path in save_map.items():
        if name in dataframes:
            dataframes[name].to_csv(path, index=False)
            logger.info(
                f"  [OK] Saved {name} -> {path.name} "
                f"({len(dataframes[name]):,} rows)"
            )

    # Save master dataset
    master.to_csv(PROCESSED_FILES["master"], index=False)
    logger.info(
        f"  [OK] Saved master -> {PROCESSED_FILES['master'].name} "
        f"({len(master):,} rows)"
    )

    # Save star schema tables
    from src.config import WAREHOUSE_FILES
    warehouse_map = {
        "fact_orders" : WAREHOUSE_FILES["fact_orders"],
        "dim_customer": WAREHOUSE_FILES["dim_customer"],
        "dim_product" : WAREHOUSE_FILES["dim_product"],
        "dim_seller"  : WAREHOUSE_FILES["dim_seller"],
        "dim_date"    : WAREHOUSE_FILES["dim_date"],
    }

    for name, path in warehouse_map.items():
        if name in star_schema:
            # Convert categorical columns to string before saving
            df_save = star_schema[name].copy()
            for col in df_save.select_dtypes(include="category").columns:
                df_save[col] = df_save[col].astype(str)
            df_save.to_csv(path, index=False)
            logger.info(
                f"  [OK] Saved {name} -> {path.name} "
                f"({len(star_schema[name]):,} rows)"
            )


# ════════════════════════════════════════════════════════════
# PRINT TRANSFORM REPORT
# ════════════════════════════════════════════════════════════

def print_transform_report(
    dataframes: dict,
    master: pd.DataFrame,
    star_schema: dict
) -> None:
    """
    Print a summary of the transformation results.
    """
    print("\n" + "=" * 65)
    print("   TRANSFORM REPORT")
    print("=" * 65)
    print(f"  {'Table':<25} {'Rows':>10} {'Cols':>6}")
    print("-" * 65)

    for name, df in dataframes.items():
        print(f"  {name:<25} {len(df):>10,} {len(df.columns):>6}")

    print("-" * 65)
    print(f"  {'master_dataset':<25} {len(master):>10,} {len(master.columns):>6}")
    print("-" * 65)
    print("  Star Schema:")
    for name, df in star_schema.items():
        print(f"  {name:<25} {len(df):>10,} {len(df.columns):>6}")
    print("=" * 65 + "\n")


# ════════════════════════════════════════════════════════════
# MAIN TRANSFORM FUNCTION
# ════════════════════════════════════════════════════════════

def transform_all(raw_dataframes: dict) -> tuple:
    """
    Main transformation function.
    Applies all transformations to raw dataframes.

    Args:
        raw_dataframes: dict of raw DataFrames from extract phase

    Returns:
        tuple: (cleaned_dataframes, master_dataset, star_schema)
    """
    logger.info("=" * 55)
    logger.info("TRANSFORM PHASE STARTED")
    logger.info("=" * 55)

    total_start = time.time()

    try:
        # Transform each table
        cleaned = {}
        cleaned["orders"]         = transform_orders(
            raw_dataframes["orders"].copy()
        )
        cleaned["order_items"]    = transform_order_items(
            raw_dataframes["order_items"].copy()
        )
        cleaned["order_payments"] = transform_order_payments(
            raw_dataframes["order_payments"].copy()
        )
        cleaned["order_reviews"]  = transform_order_reviews(
            raw_dataframes["order_reviews"].copy()
        )
        cleaned["customers"]      = transform_customers(
            raw_dataframes["customers"].copy()
        )
        cleaned["products"]       = transform_products(
            raw_dataframes["products"].copy(),
            raw_dataframes["category_trans"].copy()
        )
        cleaned["sellers"]        = transform_sellers(
            raw_dataframes["sellers"].copy()
        )
        cleaned["geolocation"]    = transform_geolocation(
            raw_dataframes["geolocation"].copy()
        )
        cleaned["category_trans"] = raw_dataframes["category_trans"].copy()

        # Build master dataset
        master = build_master_dataset(cleaned)

        # Build star schema
        star_schema = build_star_schema(cleaned)

        # Save all files
        if ETL_SETTINGS["save_processed"]:
            save_processed_files(cleaned, master, star_schema)

        # Print report
        print_transform_report(cleaned, master, star_schema)

        total_elapsed = time.time() - total_start
        logger.info(
            f"[OK] TRANSFORM PHASE COMPLETE in {total_elapsed:.2f}s"
        )

        return cleaned, master, star_schema

    except Exception as e:
        logger.error(f"[ERROR] Transform phase failed: {e}")
        raise


# ── Run directly to test ──────────────────────────────────────────────────
if __name__ == "__main__":
    from etl.extract import extract_all

    # Run extract first
    raw_dataframes = extract_all()

    # Run transform
    cleaned, master, star_schema = transform_all(raw_dataframes)

    # Preview master dataset
    print("\n-- Master Dataset Sample --")
    print(master.head(3).to_string())
    print(f"\nMaster columns: {list(master.columns)}")