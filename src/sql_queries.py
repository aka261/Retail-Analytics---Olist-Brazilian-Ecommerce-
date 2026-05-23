# src/sql_queries.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import sqlite3
import time
from loguru import logger

from src.config import (
    DATABASE_PATH,
    ETL_LOG_FILE,
)

# ── Configure Logger ──────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
    level="INFO",
    colorize=False,
)


# ════════════════════════════════════════════════════════════
# DATABASE CONNECTION
# ════════════════════════════════════════════════════════════

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA cache_size = 10000")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = OFF")
    return conn


def run_query(query: str) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        logger.error(f"[ERROR] Query failed: {e}")
        raise
    finally:
        conn.close()


# ════════════════════════════════════════════════════════════
# SECTION 1: BUSINESS OVERVIEW
# ════════════════════════════════════════════════════════════

def get_business_overview() -> pd.DataFrame:
    query = """
        SELECT
            COUNT(DISTINCT o.order_id)              AS total_orders,
            COUNT(DISTINCT o.customer_id)           AS total_customers,
            COUNT(DISTINCT i.seller_id)             AS total_sellers,
            COUNT(DISTINCT i.product_id)            AS total_products,
            ROUND(SUM(p.total_payment_value), 2)    AS total_revenue,
            ROUND(AVG(p.total_payment_value), 2)    AS avg_order_value,
            ROUND(MIN(p.total_payment_value), 2)    AS min_order_value,
            ROUND(MAX(p.total_payment_value), 2)    AS max_order_value,
            ROUND(SUM(i.freight_value), 2)          AS total_freight,
            ROUND(AVG(i.freight_value), 2)          AS avg_freight,
            ROUND(AVG(o.delivery_days), 1)          AS avg_delivery_days,
            SUM(o.is_late_delivery)                 AS late_deliveries,
            ROUND(AVG(r.review_score), 2)           AS avg_review_score
        FROM orders o
        LEFT JOIN order_items    i ON o.order_id = i.order_id
        LEFT JOIN order_payments p ON o.order_id = p.order_id
        LEFT JOIN order_reviews  r ON o.order_id = r.order_id
        WHERE o.order_status = 'delivered'
    """
    df = run_query(query)
    logger.info("[OK] Q1: Business overview fetched")
    return df


# ════════════════════════════════════════════════════════════
# SECTION 2: REVENUE ANALYSIS
# ════════════════════════════════════════════════════════════

def get_monthly_revenue() -> pd.DataFrame:
    query = """
        WITH monthly AS (
            SELECT
                o.purchase_year,
                o.purchase_month,
                o.purchase_year || '-' ||
                PRINTF('%02d', o.purchase_month)    AS year_month,
                COUNT(DISTINCT o.order_id)          AS total_orders,
                COUNT(DISTINCT o.customer_id)       AS unique_customers,
                ROUND(SUM(p.total_payment_value),2) AS total_revenue,
                ROUND(AVG(p.total_payment_value),2) AS avg_order_value
            FROM orders o
            LEFT JOIN order_payments p
                ON o.order_id = p.order_id
            WHERE o.order_status = 'delivered'
            GROUP BY o.purchase_year, o.purchase_month
        )
        SELECT
            *,
            LAG(total_revenue) OVER (
                ORDER BY purchase_year, purchase_month
            )                                       AS prev_month_revenue,
            ROUND(
                (total_revenue - LAG(total_revenue) OVER (
                    ORDER BY purchase_year, purchase_month
                )) /
                NULLIF(LAG(total_revenue) OVER (
                    ORDER BY purchase_year, purchase_month
                ), 0) * 100, 2
            )                                       AS revenue_growth_pct
        FROM monthly
        ORDER BY purchase_year, purchase_month
    """
    df = run_query(query)
    logger.info("[OK] Q2: Monthly revenue fetched")
    return df


def get_quarterly_revenue() -> pd.DataFrame:
    query = """
        SELECT
            o.purchase_year,
            o.purchase_quarter,
            o.purchase_year || '-Q' ||
            o.purchase_quarter                  AS year_quarter,
            COUNT(DISTINCT o.order_id)          AS total_orders,
            ROUND(SUM(p.total_payment_value),2) AS total_revenue,
            ROUND(AVG(p.total_payment_value),2) AS avg_order_value,
            COUNT(DISTINCT o.customer_id)       AS unique_customers
        FROM orders o
        LEFT JOIN order_payments p
            ON o.order_id = p.order_id
        WHERE o.order_status = 'delivered'
        GROUP BY o.purchase_year, o.purchase_quarter
        ORDER BY o.purchase_year, o.purchase_quarter
    """
    df = run_query(query)
    logger.info("[OK] Q3: Quarterly revenue fetched")
    return df


def get_revenue_by_category() -> pd.DataFrame:
    query = """
        SELECT
            p.product_category_name_english     AS category,
            COUNT(DISTINCT i.order_id)          AS total_orders,
            COUNT(DISTINCT i.product_id)        AS unique_products,
            ROUND(SUM(i.price), 2)              AS total_revenue,
            ROUND(AVG(i.price), 2)              AS avg_price,
            ROUND(SUM(i.freight_value), 2)      AS total_freight,
            ROUND(AVG(r.review_score), 2)       AS avg_review_score,
            ROUND(
                SUM(i.price) * 100.0 /
                SUM(SUM(i.price)) OVER (), 2
            )                                   AS revenue_share_pct
        FROM order_items i
        LEFT JOIN products      p ON i.product_id = p.product_id
        LEFT JOIN order_reviews r ON i.order_id   = r.order_id
        GROUP BY p.product_category_name_english
        ORDER BY total_revenue DESC
        LIMIT 20
    """
    df = run_query(query)
    logger.info("[OK] Q4: Revenue by category fetched")
    return df


def get_revenue_by_state() -> pd.DataFrame:
    query = """
        SELECT
            c.customer_state                    AS state,
            c.customer_region                   AS region,
            COUNT(DISTINCT o.order_id)          AS total_orders,
            COUNT(DISTINCT c.customer_id)       AS total_customers,
            ROUND(SUM(p.total_payment_value),2) AS total_revenue,
            ROUND(AVG(p.total_payment_value),2) AS avg_order_value,
            ROUND(AVG(o.delivery_days), 1)      AS avg_delivery_days
        FROM order_payments p
        JOIN orders   o ON p.order_id    = o.order_id
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_status = 'delivered'
        GROUP BY c.customer_state
        ORDER BY total_revenue DESC
    """
    df = run_query(query)
    logger.info("[OK] Q5: Revenue by state fetched")
    return df

# ════════════════════════════════════════════════════════════
# SECTION 3: CUSTOMER ANALYSIS
# ════════════════════════════════════════════════════════════

def get_customer_segmentation() -> pd.DataFrame:
    query = """
        WITH customer_orders AS (
            SELECT
                c.customer_unique_id,
                c.customer_state,
                c.customer_region,
                COUNT(DISTINCT o.order_id)          AS order_count,
                ROUND(SUM(p.total_payment_value),2) AS total_spent,
                ROUND(AVG(p.total_payment_value),2) AS avg_order_value
            FROM customers c
            LEFT JOIN orders         o ON c.customer_id = o.customer_id
            LEFT JOIN order_payments p ON o.order_id    = p.order_id
            WHERE o.order_status = 'delivered'
            GROUP BY c.customer_unique_id
        )
        SELECT
            CASE
                WHEN order_count = 1
                    THEN '1 - One Time'
                WHEN order_count = 2
                    THEN '2 - Two Times'
                WHEN order_count BETWEEN 3 AND 5
                    THEN '3-5 - Regular'
                ELSE '6+ - Loyal'
            END                                     AS customer_segment,
            COUNT(*)                                AS customer_count,
            ROUND(AVG(total_spent), 2)              AS avg_total_spent,
            ROUND(AVG(avg_order_value), 2)          AS avg_order_value,
            ROUND(
                COUNT(*) * 100.0 /
                SUM(COUNT(*)) OVER (), 2
            )                                       AS customer_pct
        FROM customer_orders
        GROUP BY customer_segment
        ORDER BY customer_segment
    """
    df = run_query(query)
    logger.info("[OK] Q6: Customer segmentation fetched")
    return df


def get_top_customers() -> pd.DataFrame:
    query = """
        SELECT
            c.customer_unique_id,
            c.customer_state,
            c.customer_region,
            COUNT(DISTINCT o.order_id)          AS total_orders,
            ROUND(SUM(p.total_payment_value),2) AS total_spent,
            ROUND(AVG(p.total_payment_value),2) AS avg_order_value,
            ROUND(AVG(r.review_score), 2)       AS avg_review_score,
            MIN(o.order_purchase_timestamp)     AS first_order_date,
            MAX(o.order_purchase_timestamp)     AS last_order_date
        FROM customers c
        LEFT JOIN orders         o ON c.customer_id = o.customer_id
        LEFT JOIN order_payments p ON o.order_id    = p.order_id
        LEFT JOIN order_reviews  r ON o.order_id    = r.order_id
        WHERE o.order_status = 'delivered'
        GROUP BY c.customer_unique_id
        ORDER BY total_spent DESC
        LIMIT 20
    """
    df = run_query(query)
    logger.info("[OK] Q7: Top customers fetched")
    return df


def get_customer_region_analysis() -> pd.DataFrame:
    query = """
        SELECT
            c.customer_region                   AS region,
            COUNT(DISTINCT c.customer_id)       AS total_customers,
            COUNT(DISTINCT o.order_id)          AS total_orders,
            ROUND(SUM(p.total_payment_value),2) AS total_revenue,
            ROUND(AVG(p.total_payment_value),2) AS avg_order_value,
            ROUND(AVG(o.delivery_days), 1)      AS avg_delivery_days,
            ROUND(AVG(r.review_score), 2)       AS avg_review_score,
            ROUND(
                COUNT(DISTINCT o.order_id) * 1.0 /
                COUNT(DISTINCT c.customer_id), 2
            )                                   AS orders_per_customer
        FROM customers c
        LEFT JOIN orders         o ON c.customer_id = o.customer_id
        LEFT JOIN order_payments p ON o.order_id    = p.order_id
        LEFT JOIN order_reviews  r ON o.order_id    = r.order_id
        WHERE o.order_status = 'delivered'
        GROUP BY c.customer_region
        ORDER BY total_revenue DESC
    """
    df = run_query(query)
    logger.info("[OK] Q8: Customer region analysis fetched")
    return df


# ════════════════════════════════════════════════════════════
# SECTION 4: PRODUCT ANALYSIS
# ════════════════════════════════════════════════════════════

def get_top_products() -> pd.DataFrame:
    query = """
        SELECT
            i.product_id,
            p.product_category_name_english     AS category,
            p.product_size,
            COUNT(DISTINCT i.order_id)          AS total_orders,
            SUM(i.order_item_id)                AS total_units_sold,
            ROUND(SUM(i.price), 2)              AS total_revenue,
            ROUND(AVG(i.price), 2)              AS avg_price,
            ROUND(AVG(r.review_score), 2)       AS avg_review_score
        FROM order_items i
        LEFT JOIN products      p ON i.product_id = p.product_id
        LEFT JOIN order_reviews r ON i.order_id   = r.order_id
        GROUP BY i.product_id
        ORDER BY total_revenue DESC
        LIMIT 20
    """
    df = run_query(query)
    logger.info("[OK] Q9: Top products fetched")
    return df


def get_category_performance() -> pd.DataFrame:
    query = """
        SELECT
            p.product_category_name_english     AS category,
            COUNT(DISTINCT i.order_id)          AS total_orders,
            COUNT(DISTINCT i.product_id)        AS unique_products,
            COUNT(DISTINCT i.seller_id)         AS unique_sellers,
            ROUND(SUM(i.price), 2)              AS total_revenue,
            ROUND(AVG(i.price), 2)              AS avg_price,
            ROUND(MIN(i.price), 2)              AS min_price,
            ROUND(MAX(i.price), 2)              AS max_price,
            ROUND(AVG(i.freight_value), 2)      AS avg_freight,
            ROUND(AVG(r.review_score), 2)       AS avg_review_score,
            COUNT(
                CASE WHEN r.review_score >= 4
                THEN 1 END
            )                                   AS positive_reviews,
            COUNT(
                CASE WHEN r.review_score <= 2
                THEN 1 END
            )                                   AS negative_reviews
        FROM order_items i
        LEFT JOIN products      p ON i.product_id = p.product_id
        LEFT JOIN order_reviews r ON i.order_id   = r.order_id
        GROUP BY p.product_category_name_english
        ORDER BY total_revenue DESC
    """
    df = run_query(query)
    logger.info("[OK] Q10: Category performance fetched")
    return df


# ════════════════════════════════════════════════════════════
# SECTION 5: SELLER ANALYSIS
# ════════════════════════════════════════════════════════════

def get_seller_performance() -> pd.DataFrame:
    query = """
        SELECT
            i.seller_id,
            s.seller_state,
            s.seller_region,
            COUNT(DISTINCT i.order_id)          AS total_orders,
            COUNT(DISTINCT i.product_id)        AS unique_products,
            ROUND(SUM(i.price), 2)              AS total_revenue,
            ROUND(AVG(i.price), 2)              AS avg_order_value,
            ROUND(AVG(i.freight_value), 2)      AS avg_freight,
            ROUND(AVG(r.review_score), 2)       AS avg_review_score,
            ROUND(AVG(o.delivery_days), 1)      AS avg_delivery_days,
            SUM(o.is_late_delivery)             AS late_deliveries,
            ROUND(
                SUM(o.is_late_delivery) * 100.0 /
                COUNT(DISTINCT i.order_id), 2
            )                                   AS late_delivery_pct
        FROM order_items i
        LEFT JOIN sellers       s ON i.seller_id = s.seller_id
        LEFT JOIN order_reviews r ON i.order_id  = r.order_id
        LEFT JOIN orders        o ON i.order_id  = o.order_id
        GROUP BY i.seller_id
        ORDER BY total_revenue DESC
    """
    df = run_query(query)
    logger.info("[OK] Q11: Seller performance fetched")
    return df


def get_top_sellers() -> pd.DataFrame:
    query = """
        SELECT
            i.seller_id,
            s.seller_state,
            s.seller_region,
            COUNT(DISTINCT i.order_id)          AS total_orders,
            ROUND(SUM(i.price), 2)              AS total_revenue,
            ROUND(AVG(r.review_score), 2)       AS avg_review_score,
            ROUND(AVG(o.delivery_days), 1)      AS avg_delivery_days
        FROM order_items i
        LEFT JOIN sellers       s ON i.seller_id = s.seller_id
        LEFT JOIN order_reviews r ON i.order_id  = r.order_id
        LEFT JOIN orders        o ON i.order_id  = o.order_id
        GROUP BY i.seller_id
        ORDER BY total_revenue DESC
        LIMIT 20
    """
    df = run_query(query)
    logger.info("[OK] Q12: Top sellers fetched")
    return df


# ════════════════════════════════════════════════════════════
# SECTION 6: ORDER & DELIVERY ANALYSIS
# ════════════════════════════════════════════════════════════

def get_order_status_summary() -> pd.DataFrame:
    query = """
        SELECT
            order_status,
            COUNT(*)                            AS total_orders,
            ROUND(
                COUNT(*) * 100.0 /
                SUM(COUNT(*)) OVER (), 2
            )                                   AS order_pct
        FROM orders
        GROUP BY order_status
        ORDER BY total_orders DESC
    """
    df = run_query(query)
    logger.info("[OK] Q13: Order status fetched")
    return df


def get_delivery_analysis() -> pd.DataFrame:
    query = """
        SELECT
            c.customer_state                    AS state,
            c.customer_region                   AS region,
            COUNT(DISTINCT o.order_id)          AS total_orders,
            ROUND(AVG(o.delivery_days), 1)      AS avg_delivery_days,
            ROUND(MIN(o.delivery_days), 1)      AS min_delivery_days,
            ROUND(MAX(o.delivery_days), 1)      AS max_delivery_days,
            SUM(o.is_late_delivery)             AS late_orders,
            ROUND(
                SUM(o.is_late_delivery) * 100.0 /
                COUNT(DISTINCT o.order_id), 2
            )                                   AS late_pct
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE
            o.order_status = 'delivered'
            AND o.delivery_days IS NOT NULL
            AND o.delivery_days > 0
        GROUP BY c.customer_state
        ORDER BY late_pct DESC
    """
    df = run_query(query)
    logger.info("[OK] Q14: Delivery analysis fetched")
    return df


def get_orders_by_weekday() -> pd.DataFrame:
    query = """
        SELECT
            purchase_weekday                    AS weekday,
            COUNT(DISTINCT order_id)            AS total_orders,
            ROUND(
                COUNT(DISTINCT order_id) * 100.0 /
                SUM(COUNT(DISTINCT order_id)) OVER (), 2
            )                                   AS order_pct
        FROM orders
        WHERE order_status = 'delivered'
        GROUP BY purchase_weekday
        ORDER BY
            CASE purchase_weekday
                WHEN 'Monday'    THEN 1
                WHEN 'Tuesday'   THEN 2
                WHEN 'Wednesday' THEN 3
                WHEN 'Thursday'  THEN 4
                WHEN 'Friday'    THEN 5
                WHEN 'Saturday'  THEN 6
                WHEN 'Sunday'    THEN 7
            END
    """
    df = run_query(query)
    logger.info("[OK] Q15: Orders by weekday fetched")
    return df


def get_orders_by_hour() -> pd.DataFrame:
    query = """
        SELECT
            purchase_hour                       AS hour_of_day,
            COUNT(DISTINCT order_id)            AS total_orders,
            ROUND(
                COUNT(DISTINCT order_id) * 100.0 /
                SUM(COUNT(DISTINCT order_id)) OVER (), 2
            )                                   AS order_pct
        FROM orders
        WHERE order_status = 'delivered'
        GROUP BY purchase_hour
        ORDER BY purchase_hour
    """
    df = run_query(query)
    logger.info("[OK] Q16: Orders by hour fetched")
    return df


# ════════════════════════════════════════════════════════════
# SECTION 7: PAYMENT ANALYSIS
# ════════════════════════════════════════════════════════════

def get_payment_analysis() -> pd.DataFrame:
    query = """
        SELECT
            primary_payment_type                AS payment_type,
            COUNT(DISTINCT order_id)            AS total_orders,
            ROUND(SUM(total_payment_value), 2)  AS total_revenue,
            ROUND(AVG(total_payment_value), 2)  AS avg_order_value,
            ROUND(AVG(max_installments), 1)     AS avg_installments,
            ROUND(
                COUNT(DISTINCT order_id) * 100.0 /
                SUM(COUNT(DISTINCT order_id)) OVER (), 2
            )                                   AS order_pct
        FROM order_payments
        GROUP BY primary_payment_type
        ORDER BY total_revenue DESC
    """
    df = run_query(query)
    logger.info("[OK] Q17: Payment analysis fetched")
    return df


def get_installment_analysis() -> pd.DataFrame:
    query = """
        SELECT
            max_installments                    AS installments,
            COUNT(DISTINCT order_id)            AS total_orders,
            ROUND(AVG(total_payment_value), 2)  AS avg_order_value,
            ROUND(SUM(total_payment_value), 2)  AS total_revenue,
            ROUND(
                COUNT(DISTINCT order_id) * 100.0 /
                SUM(COUNT(DISTINCT order_id)) OVER (), 2
            )                                   AS order_pct
        FROM order_payments
        GROUP BY max_installments
        ORDER BY max_installments
    """
    df = run_query(query)
    logger.info("[OK] Q18: Installment analysis fetched")
    return df


# ════════════════════════════════════════════════════════════
# SECTION 8: REVIEW ANALYSIS
# ════════════════════════════════════════════════════════════

def get_review_analysis() -> pd.DataFrame:
    query = """
        SELECT
            review_score,
            review_sentiment,
            COUNT(*)                            AS total_reviews,
            ROUND(
                COUNT(*) * 100.0 /
                SUM(COUNT(*)) OVER (), 2
            )                                   AS review_pct
        FROM order_reviews
        GROUP BY review_score, review_sentiment
        ORDER BY review_score DESC
    """
    df = run_query(query)
    logger.info("[OK] Q19: Review analysis fetched")
    return df


def get_review_by_category() -> pd.DataFrame:
    query = """
        SELECT
            p.product_category_name_english     AS category,
            COUNT(r.review_id)                  AS total_reviews,
            ROUND(AVG(r.review_score), 2)       AS avg_score,
            COUNT(
                CASE WHEN r.review_score = 5
                THEN 1 END
            )                                   AS five_star,
            COUNT(
                CASE WHEN r.review_score = 1
                THEN 1 END
            )                                   AS one_star,
            ROUND(
                COUNT(
                    CASE WHEN r.review_score >= 4
                    THEN 1 END
                ) * 100.0 /
                COUNT(r.review_score), 2
            )                                   AS positive_pct
        FROM order_reviews r
        LEFT JOIN order_items i ON r.order_id   = i.order_id
        LEFT JOIN products    p ON i.product_id = p.product_id
        GROUP BY p.product_category_name_english
        HAVING total_reviews > 100
        ORDER BY avg_score DESC
        LIMIT 20
    """
    df = run_query(query)
    logger.info("[OK] Q20: Review by category fetched")
    return df


# ════════════════════════════════════════════════════════════
# SECTION 9: ADVANCED SQL
# ════════════════════════════════════════════════════════════

def get_rfm_segment_summary() -> pd.DataFrame:
    query = """
        WITH customer_stats AS (
            SELECT
                c.customer_unique_id,
                COUNT(DISTINCT o.order_id)          AS frequency,
                ROUND(SUM(p.total_payment_value),2) AS monetary,
                JULIANDAY('2018-10-01') -
                JULIANDAY(
                    MAX(o.order_purchase_timestamp)
                )                                   AS recency_days
            FROM customers c
            LEFT JOIN orders         o ON c.customer_id = o.customer_id
            LEFT JOIN order_payments p ON o.order_id    = p.order_id
            WHERE o.order_status = 'delivered'
            GROUP BY c.customer_unique_id
        ),
        rfm_scores AS (
            SELECT
                *,
                NTILE(5) OVER (
                    ORDER BY recency_days ASC
                )                                   AS r_score,
                NTILE(5) OVER (
                    ORDER BY frequency DESC
                )                                   AS f_score,
                NTILE(5) OVER (
                    ORDER BY monetary DESC
                )                                   AS m_score
            FROM customer_stats
        ),
        segments AS (
            SELECT
                *,
                CASE
                    WHEN r_score >= 4
                        AND f_score >= 4             THEN 'Champions'
                    WHEN r_score >= 3
                        AND f_score >= 3             THEN 'Loyal Customers'
                    WHEN r_score >= 3
                        AND f_score <= 2             THEN 'Potential Loyalist'
                    WHEN r_score <= 2
                        AND f_score >= 3             THEN 'At Risk'
                    WHEN r_score <= 2
                        AND f_score <= 2             THEN 'Lost'
                    ELSE                                  'Others'
                END                                 AS rfm_segment
            FROM rfm_scores
        )
        SELECT
            rfm_segment,
            COUNT(*)                                AS customer_count,
            ROUND(AVG(monetary), 2)                 AS avg_monetary,
            ROUND(SUM(monetary), 2)                 AS total_monetary,
            ROUND(AVG(frequency), 2)                AS avg_frequency,
            ROUND(AVG(recency_days), 1)             AS avg_recency_days,
            ROUND(
                COUNT(*) * 100.0 /
                SUM(COUNT(*)) OVER (), 2
            )                                       AS customer_pct
        FROM segments
        GROUP BY rfm_segment
        ORDER BY total_monetary DESC
    """
    df = run_query(query)
    logger.info("[OK] Q21: RFM segment summary fetched")
    return df


def get_running_total_revenue() -> pd.DataFrame:
    query = """
        WITH monthly AS (
            SELECT
                o.purchase_year || '-' ||
                PRINTF('%02d', o.purchase_month)    AS year_month,
                ROUND(SUM(p.total_payment_value),2) AS monthly_revenue
            FROM orders o
            LEFT JOIN order_payments p ON o.order_id = p.order_id
            WHERE o.order_status = 'delivered'
            GROUP BY o.purchase_year, o.purchase_month
        )
        SELECT
            year_month,
            monthly_revenue,
            ROUND(
                SUM(monthly_revenue) OVER (
                    ORDER BY year_month
                    ROWS BETWEEN
                        UNBOUNDED PRECEDING
                        AND CURRENT ROW
                ), 2
            )                                       AS cumulative_revenue,
            ROUND(
                AVG(monthly_revenue) OVER (
                    ORDER BY year_month
                    ROWS BETWEEN 2 PRECEDING
                    AND CURRENT ROW
                ), 2
            )                                       AS rolling_3m_avg
        FROM monthly
        ORDER BY year_month
    """
    df = run_query(query)
    logger.info("[OK] Q22: Running total revenue fetched")
    return df


def get_seller_ranking() -> pd.DataFrame:
    query = """
        WITH seller_stats AS (
            SELECT
                i.seller_id,
                s.seller_state,
                s.seller_region,
                COUNT(DISTINCT i.order_id)          AS total_orders,
                ROUND(SUM(i.price), 2)              AS total_revenue,
                ROUND(AVG(r.review_score), 2)       AS avg_review_score
            FROM order_items i
            LEFT JOIN sellers       s ON i.seller_id = s.seller_id
            LEFT JOIN order_reviews r ON i.order_id  = r.order_id
            GROUP BY i.seller_id
        )
        SELECT
            *,
            RANK() OVER (
                ORDER BY total_revenue DESC
            )                                       AS overall_rank,
            RANK() OVER (
                PARTITION BY seller_state
                ORDER BY total_revenue DESC
            )                                       AS state_rank
        FROM seller_stats
        ORDER BY overall_rank
        LIMIT 30
    """
    df = run_query(query)
    logger.info("[OK] Q23: Seller ranking fetched")
    return df


def get_cohort_data() -> pd.DataFrame:
    query = """
        WITH first_purchase AS (
            SELECT
                customer_id,
                MIN(purchase_year || '-' ||
                PRINTF('%02d', purchase_month)) AS cohort_month
            FROM orders
            WHERE order_status = 'delivered'
            GROUP BY customer_id
        ),
        activity AS (
            SELECT
                customer_id,
                purchase_year || '-' ||
                PRINTF('%02d', purchase_month)  AS activity_month
            FROM orders
            WHERE order_status = 'delivered'
            GROUP BY customer_id, purchase_year, purchase_month
        )
        SELECT
            f.cohort_month,
            a.activity_month,
            COUNT(*)                            AS active_customers
        FROM first_purchase f
        JOIN activity a ON f.customer_id = a.customer_id
        GROUP BY f.cohort_month, a.activity_month
        ORDER BY f.cohort_month, a.activity_month
        LIMIT 500
    """
    df = run_query(query)
    logger.info("[OK] Q24: Cohort data fetched")
    return df


# ════════════════════════════════════════════════════════════
# RUN ALL QUERIES
# ════════════════════════════════════════════════════════════

def run_all_queries() -> dict:
    logger.info("=" * 55)
    logger.info("RUNNING ALL SQL QUERIES")
    logger.info("=" * 55)

    results = {}
    start   = time.time()

    query_functions = {
        "business_overview"    : get_business_overview,
        "monthly_revenue"      : get_monthly_revenue,
        "quarterly_revenue"    : get_quarterly_revenue,
        "revenue_by_category"  : get_revenue_by_category,
        "revenue_by_state"     : get_revenue_by_state,
        "customer_segmentation": get_customer_segmentation,
        "top_customers"        : get_top_customers,
        "customer_region"      : get_customer_region_analysis,
        "top_products"         : get_top_products,
        "category_performance" : get_category_performance,
        "seller_performance"   : get_seller_performance,
        "top_sellers"          : get_top_sellers,
        "order_status"         : get_order_status_summary,
        "delivery_analysis"    : get_delivery_analysis,
        "orders_by_weekday"    : get_orders_by_weekday,
        "orders_by_hour"       : get_orders_by_hour,
        "payment_analysis"     : get_payment_analysis,
        "installment_analysis" : get_installment_analysis,
        "review_analysis"      : get_review_analysis,
        "review_by_category"   : get_review_by_category,
        "rfm_segment_summary"  : get_rfm_segment_summary,
        "running_total_revenue": get_running_total_revenue,
        "seller_ranking"       : get_seller_ranking,
        "cohort_data"          : get_cohort_data,
    }

    passed = 0
    failed = 0

    for name, func in query_functions.items():
        try:
            df            = func()
            results[name] = df
            passed       += 1
        except Exception as e:
            logger.error(f"[ERROR] {name}: {e}")
            failed += 1

    elapsed = time.time() - start

    print("\n" + "=" * 55)
    print("   SQL QUERIES SUMMARY")
    print("=" * 55)
    print(f"  Total   : {len(query_functions)}")
    print(f"  Passed  : {passed}")
    print(f"  Failed  : {failed}")
    print(f"  Time    : {elapsed:.2f}s")
    print("-" * 55)
    for name, df in results.items():
        print(f"  {name:<30} : {len(df):>6,} rows")
    print("=" * 55 + "\n")

    return results


# ── Run directly ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_all_queries()

    print("\n-- Business Overview --")
    print(results["business_overview"].to_string())

    print("\n-- Monthly Revenue --")
    print(results["monthly_revenue"].tail(5).to_string())

    print("\n-- Top 5 Categories --")
    print(results["revenue_by_category"].head(5).to_string())

    print("\n-- RFM Segment Summary --")
    print(results["rfm_segment_summary"].to_string())

    print("\n-- Payment Analysis --")
    print(results["payment_analysis"].to_string())