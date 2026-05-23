# etl/pipeline.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from datetime import datetime
from loguru import logger
from pathlib import Path

from src.config import (
    ETL_LOG_FILE,
    ETL_LOGS_DIR,
    LOGS_DIR,
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
# PIPELINE REPORT
# ════════════════════════════════════════════════════════════

def save_pipeline_report(report: dict) -> None:
    """
    Save pipeline run report as JSON file
    with timestamp in etl_logs folder.
    """
    create_directories()

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = ETL_LOGS_DIR / f"pipeline_run_{timestamp}.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4, default=str)

    logger.info(f"[OK] Pipeline report saved: {report_path.name}")


def print_pipeline_banner() -> None:
    """
    Print a banner at the start of the pipeline.
    """
    print("\n" + "=" * 65)
    print("   RETAIL ANALYTICS - ETL PIPELINE")
    print("   Brazilian E-Commerce Dataset (Olist)")
    print("=" * 65)
    print(f"   Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Pipeline : Extract -> Transform -> Load")
    print("=" * 65 + "\n")


def print_pipeline_summary(report: dict) -> None:
    """
    Print a full summary of the pipeline run.
    """
    print("\n" + "=" * 65)
    print("   PIPELINE RUN SUMMARY")
    print("=" * 65)
    print(f"  Status        : {report['status']}")
    print(f"  Started       : {report['started_at']}")
    print(f"  Completed     : {report['completed_at']}")
    print(f"  Total Time    : {report['total_time_sec']:.2f}s")
    print("-" * 65)
    print(f"  {'Phase':<15} {'Status':<10} {'Time':>10}")
    print("-" * 65)

    for phase, info in report["phases"].items():
        status = "[OK]" if info["success"] else "[FAIL]"
        print(
            f"  {phase:<15} "
            f"{status:<10} "
            f"{info['time_sec']:>9.2f}s"
        )

    print("-" * 65)
    print("  Data Summary:")
    print("-" * 65)

    for key, val in report["data_summary"].items():
        print(f"  {key:<30} : {val:,}" if isinstance(val, int)
              else f"  {key:<30} : {val}")

    print("=" * 65 + "\n")


# ════════════════════════════════════════════════════════════
# PIPELINE PHASES
# ════════════════════════════════════════════════════════════

def run_extract() -> tuple:
    """
    Run the Extract phase.
    Loads all 9 raw CSV files into DataFrames.

    Returns:
        tuple: (raw_dataframes, phase_result)
    """
    logger.info("=" * 55)
    logger.info("PHASE 1 of 3 : EXTRACT")
    logger.info("=" * 55)

    phase_result = {
        "success" : False,
        "time_sec": 0,
        "rows"    : 0,
        "files"   : 0,
    }

    start = time.time()

    try:
        from etl.extract import extract_all
        raw_dataframes = extract_all()

        phase_result["success"] = True
        phase_result["files"]   = len(raw_dataframes)
        phase_result["rows"]    = sum(
            len(df) for df in raw_dataframes.values()
        )
        phase_result["time_sec"] = round(time.time() - start, 2)

        logger.info(
            f"[OK] Extract complete: "
            f"{phase_result['files']} files, "
            f"{phase_result['rows']:,} rows in "
            f"{phase_result['time_sec']}s"
        )
        return raw_dataframes, phase_result

    except Exception as e:
        phase_result["time_sec"] = round(time.time() - start, 2)
        phase_result["error"]    = str(e)
        logger.error(f"[ERROR] Extract phase failed: {e}")
        raise


def run_transform(raw_dataframes: dict) -> tuple:
    """
    Run the Transform phase.
    Cleans, enriches and builds master + star schema.

    Returns:
        tuple: (cleaned, master, star_schema, phase_result)
    """
    logger.info("=" * 55)
    logger.info("PHASE 2 of 3 : TRANSFORM")
    logger.info("=" * 55)

    phase_result = {
        "success"      : False,
        "time_sec"     : 0,
        "master_rows"  : 0,
        "master_cols"  : 0,
        "star_tables"  : 0,
    }

    start = time.time()

    try:
        from etl.transform import transform_all
        cleaned, master, star_schema = transform_all(raw_dataframes)

        phase_result["success"]     = True
        phase_result["master_rows"] = len(master)
        phase_result["master_cols"] = len(master.columns)
        phase_result["star_tables"] = len(star_schema)
        phase_result["time_sec"]    = round(time.time() - start, 2)

        logger.info(
            f"[OK] Transform complete: "
            f"master={len(master):,} rows x {len(master.columns)} cols, "
            f"star schema={len(star_schema)} tables in "
            f"{phase_result['time_sec']}s"
        )
        return cleaned, master, star_schema, phase_result

    except Exception as e:
        phase_result["time_sec"] = round(time.time() - start, 2)
        phase_result["error"]    = str(e)
        logger.error(f"[ERROR] Transform phase failed: {e}")
        raise


def run_load(
    cleaned: dict,
    master,
    star_schema: dict
) -> tuple:
    """
    Run the Load phase.
    Loads all data into SQLite database.

    Returns:
        tuple: (phase_result,)
    """
    logger.info("=" * 55)
    logger.info("PHASE 3 of 3 : LOAD")
    logger.info("=" * 55)

    phase_result = {
        "success"     : False,
        "time_sec"    : 0,
        "tables_loaded": 0,
    }

    start = time.time()

    try:
        from etl.load import load_all
        load_all(cleaned, master, star_schema)

        phase_result["success"]      = True
        phase_result["tables_loaded"]= (
            len(cleaned) + len(star_schema) + 1
        )
        phase_result["time_sec"]     = round(time.time() - start, 2)

        logger.info(
            f"[OK] Load complete: "
            f"{phase_result['tables_loaded']} tables loaded in "
            f"{phase_result['time_sec']}s"
        )
        return phase_result,

    except Exception as e:
        phase_result["time_sec"] = round(time.time() - start, 2)
        phase_result["error"]    = str(e)
        logger.error(f"[ERROR] Load phase failed: {e}")
        raise


# ════════════════════════════════════════════════════════════
# MAIN PIPELINE FUNCTION
# ════════════════════════════════════════════════════════════

def run_pipeline() -> dict:
    """
    Main ETL pipeline orchestrator.
    Runs Extract -> Transform -> Load in sequence.
    Saves a JSON report of the pipeline run.

    Returns:
        dict: pipeline run report
    """
    print_pipeline_banner()
    total_start = time.time()
    started_at  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Pipeline report template
    report = {
        "status"       : "RUNNING",
        "started_at"   : started_at,
        "completed_at" : None,
        "total_time_sec": 0,
        "phases": {
            "Extract"  : {"success": False, "time_sec": 0},
            "Transform": {"success": False, "time_sec": 0},
            "Load"     : {"success": False, "time_sec": 0},
        },
        "data_summary" : {},
    }

    try:
        # ── Phase 1: Extract ──────────────────────────────
        raw_dataframes, extract_result = run_extract()
        report["phases"]["Extract"]    = extract_result

        # ── Phase 2: Transform ────────────────────────────
        cleaned, master, star_schema, transform_result = run_transform(
            raw_dataframes
        )
        report["phases"]["Transform"] = transform_result

        # ── Phase 3: Load ─────────────────────────────────
        load_result, = run_load(cleaned, master, star_schema)
        report["phases"]["Load"]      = load_result

        # ── Build Data Summary ────────────────────────────
        report["data_summary"] = {
            "raw_files_loaded"      : extract_result["files"],
            "total_raw_rows"        : extract_result["rows"],
            "master_dataset_rows"   : transform_result["master_rows"],
            "master_dataset_cols"   : transform_result["master_cols"],
            "star_schema_tables"    : transform_result["star_tables"],
            "database_tables_loaded": load_result["tables_loaded"],
            "orders"                : len(cleaned["orders"]),
            "order_items"           : len(cleaned["order_items"]),
            "customers"             : len(cleaned["customers"]),
            "products"              : len(cleaned["products"]),
            "sellers"               : len(cleaned["sellers"]),
        }

        # ── Mark Success ──────────────────────────────────
        total_elapsed            = time.time() - total_start
        report["status"]         = "SUCCESS"
        report["completed_at"]   = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        report["total_time_sec"] = round(total_elapsed, 2)

        # ── Print and Save Summary ────────────────────────
        print_pipeline_summary(report)
        save_pipeline_report(report)

        logger.info(
            f"[OK] PIPELINE COMPLETE in {total_elapsed:.2f}s"
        )
        return report

    except Exception as e:
        total_elapsed            = time.time() - total_start
        report["status"]         = "FAILED"
        report["completed_at"]   = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        report["total_time_sec"] = round(total_elapsed, 2)
        report["error"]          = str(e)

        # Save failed report too
        save_pipeline_report(report)

        logger.error(f"[ERROR] PIPELINE FAILED: {e}")
        logger.error(
            f"[ERROR] Failed after {total_elapsed:.2f}s"
        )
        raise


# ── Run directly ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()