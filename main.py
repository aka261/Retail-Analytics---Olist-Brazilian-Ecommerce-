# main.py

import sys
import os
import time
import argparse
from datetime import datetime
from loguru import logger
from pathlib import Path

from src.config import (
    create_directories,
    validate_raw_files,
    print_config,
    DATABASE_PATH,
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
    "logs/main.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
    level="INFO",
    rotation="10 MB",
    retention="30 days",
)


# ════════════════════════════════════════════════════════════
# BANNER
# ════════════════════════════════════════════════════════════

def print_banner() -> None:
    print("\n" + "=" * 65)
    print("   RETAIL ANALYTICS - BRAZILIAN E-COMMERCE (OLIST)")
    print("   End-to-End Business Intelligence Project")
    print("=" * 65)
    print(f"   Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65 + "\n")


def print_completion(
    total_elapsed: float,
    steps_run: list
) -> None:
    print("\n" + "=" * 65)
    print("   PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Total Time    : {total_elapsed:.2f}s")
    print(f"  Steps Run     : {len(steps_run)}")
    print("-" * 65)
    for step in steps_run:
        print(f"  [OK] {step}")
    print("=" * 65 + "\n")


# ════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ════════════════════════════════════════════════════════════

def run_etl() -> None:
    """
    Run the full ETL pipeline.
    Extract -> Transform -> Load
    """
    logger.info("=" * 55)
    logger.info("STEP 1: ETL PIPELINE")
    logger.info("=" * 55)

    from etl.pipeline import run_pipeline
    run_pipeline()


def run_sql_analysis() -> dict:
    """
    Run all SQL queries and return results.
    """
    logger.info("=" * 55)
    logger.info("STEP 2: SQL ANALYSIS")
    logger.info("=" * 55)

    from src.sql_queries import run_all_queries
    results = run_all_queries()
    return results


def run_eda() -> None:
    """
    Run EDA and generate all plots.
    """
    logger.info("=" * 55)
    logger.info("STEP 3: EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 55)

    from src.visualization import run_all_visualizations
    run_all_visualizations()


def run_rfm() -> tuple:
    """
    Run RFM customer segmentation analysis.
    """
    logger.info("=" * 55)
    logger.info("STEP 4: RFM ANALYSIS")
    logger.info("=" * 55)

    from analytics.rfm_analysis import run_rfm_analysis
    rfm, segment_summary = run_rfm_analysis()
    return rfm, segment_summary


def run_cohort() -> tuple:
    """
    Run cohort retention analysis.
    """
    logger.info("=" * 55)
    logger.info("STEP 5: COHORT ANALYSIS")
    logger.info("=" * 55)

    from analytics.cohort_analysis import run_cohort_analysis
    cohort_pivot, retention_pivot, metrics = (
        run_cohort_analysis()
    )
    return cohort_pivot, retention_pivot, metrics


def run_forecasting() -> dict:
    """
    Run revenue forecasting.
    """
    logger.info("=" * 55)
    logger.info("STEP 6: FORECASTING")
    logger.info("=" * 55)

    from analytics.forecasting import run_forecasting
    results = run_forecasting()
    return results


def run_seller_scorecard() -> tuple:
    """
    Run seller scorecard analysis.
    """
    logger.info("=" * 55)
    logger.info("STEP 7: SELLER SCORECARD")
    logger.info("=" * 55)

    from analytics.seller_scorecard import run_seller_scorecard
    df, tier_summary = run_seller_scorecard()
    return df, tier_summary


def run_churn() -> tuple:
    """
    Run churn prediction model.
    """
    logger.info("=" * 55)
    logger.info("STEP 8: CHURN PREDICTION")
    logger.info("=" * 55)

    from analytics.churn_prediction import run_churn_prediction
    customer_df, results, importance_df = (
        run_churn_prediction()
    )
    return customer_df, results, importance_df


def run_market_basket() -> tuple:
    """
    Run market basket analysis.
    """
    logger.info("=" * 55)
    logger.info("STEP 9: MARKET BASKET ANALYSIS")
    logger.info("=" * 55)

    from analytics.market_basket import run_market_basket_analysis
    rules, frequent_itemsets, co_occur = (
        run_market_basket_analysis()
    )
    return rules, frequent_itemsets, co_occur


def run_visualization() -> None:
    """
    Run all static visualizations.
    """
    logger.info("=" * 55)
    logger.info("STEP 10: VISUALIZATION")
    logger.info("=" * 55)

    from src.visualization import run_all_visualizations
    run_all_visualizations()


def run_pdf_report() -> Path:
    """
    Generate PDF report.
    """
    logger.info("=" * 55)
    logger.info("STEP 11: PDF REPORT")
    logger.info("=" * 55)

    from reports.pdf_report import generate_pdf_report
    path = generate_pdf_report()
    return path


def run_excel_report() -> Path:
    """
    Generate Excel report.
    """
    logger.info("=" * 55)
    logger.info("STEP 12: EXCEL REPORT")
    logger.info("=" * 55)

    from reports.excel_report import generate_excel_report
    path = generate_excel_report()
    return path


# ════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ════════════════════════════════════════════════════════════

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Retail Analytics Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        choices=[
            "etl", "sql", "eda", "rfm",
            "cohort", "forecast", "seller",
            "churn", "basket", "viz",
            "pdf", "excel", "all"
        ],
        default=["all"],
        help="""Steps to run:
  etl      : Extract Transform Load pipeline
  sql      : SQL analysis queries
  eda      : Exploratory data analysis
  rfm      : RFM customer segmentation
  cohort   : Cohort retention analysis
  forecast : Revenue forecasting
  seller   : Seller scorecard
  churn    : Churn prediction model
  basket   : Market basket analysis
  viz      : Static visualizations
  pdf      : PDF report generation
  excel    : Excel report generation
  all      : Run everything (default)
        """
    )

    parser.add_argument(
        "--skip-etl",
        action="store_true",
        help="Skip ETL if database already exists"
    )

    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="Generate reports only (skip analysis)"
    )

    return parser.parse_args()


# ════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ════════════════════════════════════════════════════════════

def main():
    """
    Main entry point for the retail analytics pipeline.
    Runs all steps in sequence.
    """
    print_banner()
    total_start = time.time()
    steps_run   = []

    # Parse arguments
    args  = parse_args()
    steps = args.steps

    # Run all if "all" specified
    if "all" in steps:
        steps = [
            "etl", "sql", "eda", "rfm",
            "cohort", "forecast", "seller",
            "churn", "basket", "viz",
            "pdf", "excel"
        ]

    # Override if reports only
    if args.reports_only:
        steps = ["pdf", "excel"]

    # Create directories
    create_directories()

    # Print config
    print_config()

    # Check if ETL should be skipped
    if args.skip_etl and DATABASE_PATH.exists():
        logger.info(
            "[OK] Database exists — skipping ETL"
        )
        steps = [s for s in steps if s != "etl"]

    # Run each step
    try:

        if "etl" in steps:
            validate_raw_files()
            run_etl()
            steps_run.append("ETL Pipeline")

        if "sql" in steps:
            run_sql_analysis()
            steps_run.append("SQL Analysis")

        if "eda" in steps:
            run_eda()
            steps_run.append("EDA")

        if "rfm" in steps:
            run_rfm()
            steps_run.append("RFM Analysis")

        if "cohort" in steps:
            run_cohort()
            steps_run.append("Cohort Analysis")

        if "forecast" in steps:
            run_forecasting()
            steps_run.append("Forecasting")

        if "seller" in steps:
            run_seller_scorecard()
            steps_run.append("Seller Scorecard")

        if "churn" in steps:
            run_churn()
            steps_run.append("Churn Prediction")

        if "basket" in steps:
            run_market_basket()
            steps_run.append("Market Basket")

        if "viz" in steps:
            run_visualization()
            steps_run.append("Visualization")

        if "pdf" in steps:
            run_pdf_report()
            steps_run.append("PDF Report")

        if "excel" in steps:
            run_excel_report()
            steps_run.append("Excel Report")

    except Exception as e:
        logger.error(f"[ERROR] Pipeline failed: {e}")
        raise

    total_elapsed = time.time() - total_start
    print_completion(total_elapsed, steps_run)


# ── Run directly ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()