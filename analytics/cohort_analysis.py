# analytics/cohort_analysis.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import time
from loguru import logger

from src.config import (
    PROCESSED_FILES,
    PLOTS_DIR,
    VIZ_SETTINGS,
    create_directories,
)

warnings.filterwarnings("ignore")

# ── Configure Logger ──────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
    level="INFO",
    colorize=False,
)

# ── Plot Settings ─────────────────────────────────────────────────────────
BG      = "#0F1117"
CARD_BG = "#1A1D27"
GRID_C  = "#2C2F3F"
TEXT_C  = "#E8EAF0"
COLORS  = VIZ_SETTINGS["color_palette"]
BLUE    = COLORS[0]
GREEN   = COLORS[1]
RED     = COLORS[2]
ORANGE  = COLORS[3]
PURPLE  = COLORS[4]

plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : CARD_BG,
    "text.color"       : TEXT_C,
    "axes.labelcolor"  : TEXT_C,
    "xtick.color"      : TEXT_C,
    "ytick.color"      : TEXT_C,
    "axes.edgecolor"   : GRID_C,
    "grid.color"       : GRID_C,
    "font.family"      : "DejaVu Sans",
    "axes.titlesize"   : 11,
    "axes.titleweight" : "bold",
})


# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def save_plot(fig, filename: str) -> None:
    create_directories()
    path = PLOTS_DIR / filename
    fig.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=BG, edgecolor="none"
    )
    plt.close(fig)
    logger.info(f"[OK] Saved plot: {filename}")


# ════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """
    Load master dataset for cohort analysis.
    """
    logger.info("-- Loading data for cohort analysis --")

    df = pd.read_csv(PROCESSED_FILES["master"])

    # Convert date columns
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    )

    # Keep only delivered orders
    df = df[df["order_status"] == "delivered"].copy()

    # Add year month column
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M")

    logger.info(f"[OK] Loaded {len(df):,} delivered orders")
    return df


# ════════════════════════════════════════════════════════════
# STEP 2: BUILD COHORT DATA
# ════════════════════════════════════════════════════════════

def build_cohort_data(df: pd.DataFrame) -> tuple:
    """
    Build cohort analysis data.

    Cohort = group of customers who made their
    first purchase in the same month.

    Returns:
        tuple: (cohort_pivot, retention_pivot)
    """
    logger.info("-- Building cohort data --")

    # Get first purchase month per customer
    first_purchase = df.groupby(
        "customer_unique_id"
    )["order_month"].min().reset_index()
    first_purchase.columns = [
        "customer_unique_id", "cohort_month"
    ]

    # Merge back to main dataframe
    df = df.merge(first_purchase, on="customer_unique_id")

    # Calculate cohort index
    # (months since first purchase)
    df["cohort_index"] = (
        df["order_month"] - df["cohort_month"]
    ).apply(lambda x: x.n)

    # Count unique customers per cohort and index
    cohort_data = df.groupby(
        ["cohort_month", "cohort_index"]
    )["customer_unique_id"].nunique().reset_index()

    cohort_data.columns = [
        "cohort_month", "cohort_index", "customers"
    ]

    # Pivot table
    cohort_pivot = cohort_data.pivot_table(
        index="cohort_month",
        columns="cohort_index",
        values="customers"
    )

    # Keep top 12 cohorts and 12 months
    cohort_pivot = cohort_pivot.iloc[:12, :12]
    cohort_pivot = cohort_pivot.fillna(0)

    # Calculate retention rate
    cohort_size     = cohort_pivot.iloc[:, 0]
    retention_pivot = cohort_pivot.divide(
        cohort_size, axis=0
    ) * 100
    retention_pivot = retention_pivot.round(2)

    logger.info(
        f"[OK] Cohort data built: "
        f"{len(cohort_pivot)} cohorts x "
        f"{len(cohort_pivot.columns)} months"
    )

    return cohort_pivot, retention_pivot


# ════════════════════════════════════════════════════════════
# STEP 3: BUILD REVENUE COHORT
# ════════════════════════════════════════════════════════════

def build_revenue_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build revenue cohort analysis.
    Shows avg revenue per customer per cohort month.
    """
    logger.info("-- Building revenue cohort --")

    # Get first purchase month per customer
    first_purchase = df.groupby(
        "customer_unique_id"
    )["order_month"].min().reset_index()
    first_purchase.columns = [
        "customer_unique_id", "cohort_month"
    ]

    df = df.merge(first_purchase, on="customer_unique_id")

    # Calculate cohort index
    df["cohort_index"] = (
        df["order_month"] - df["cohort_month"]
    ).apply(lambda x: x.n)

    # Avg revenue per cohort
    revenue_cohort = df.groupby(
        ["cohort_month", "cohort_index"]
    )["total_payment_value"].mean().reset_index()

    revenue_cohort.columns = [
        "cohort_month", "cohort_index", "avg_revenue"
    ]

    revenue_pivot = revenue_cohort.pivot_table(
        index="cohort_month",
        columns="cohort_index",
        values="avg_revenue"
    ).round(2)

    revenue_pivot = revenue_pivot.iloc[:12, :12]
    revenue_pivot = revenue_pivot.fillna(0)

    logger.info(f"[OK] Revenue cohort built")
    return revenue_pivot


# ════════════════════════════════════════════════════════════
# STEP 4: COHORT METRICS
# ════════════════════════════════════════════════════════════

def calculate_cohort_metrics(
    cohort_pivot: pd.DataFrame,
    retention_pivot: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate key cohort metrics per cohort.
    """
    logger.info("-- Calculating cohort metrics --")

    metrics = pd.DataFrame()
    metrics["cohort_month"]     = cohort_pivot.index
    metrics["cohort_size"]      = cohort_pivot.iloc[:, 0].values
    metrics["month_1_retained"] = cohort_pivot.iloc[:, 1].values \
        if cohort_pivot.shape[1] > 1 else 0
    metrics["month_1_retention_pct"] = retention_pivot.iloc[:, 1].values \
        if retention_pivot.shape[1] > 1 else 0
    metrics["avg_retention_pct"] = retention_pivot.iloc[:, 1:].mean(
        axis=1
    ).values

    metrics = metrics.round(2)
    logger.info(f"[OK] Cohort metrics calculated")
    return metrics


# ════════════════════════════════════════════════════════════
# STEP 5: VISUALIZE COHORTS
# ════════════════════════════════════════════════════════════

def plot_cohort_heatmaps(
    cohort_pivot: pd.DataFrame,
    retention_pivot: pd.DataFrame
) -> None:
    """
    Plot cohort heatmaps for customer count
    and retention rate.
    """
    logger.info("-- Plotting cohort heatmaps --")

    # Convert index to string for display
    cohort_str    = cohort_pivot.copy()
    retention_str = retention_pivot.copy()
    cohort_str.index    = cohort_str.index.astype(str)
    retention_str.index = retention_str.index.astype(str)

    fig, axes = plt.subplots(
        2, 1, figsize=(18, 16), facecolor=BG
    )
    fig.suptitle(
        "COHORT ANALYSIS",
        fontsize=18, fontweight="bold", color=TEXT_C
    )

    # Plot 1: Customer count heatmap
    ax = axes[0]
    sns.heatmap(
        cohort_str,
        ax=ax,
        cmap="Blues",
        annot=True,
        fmt=".0f",
        annot_kws={"size": 8},
        linewidths=0.5,
        linecolor=BG,
        cbar_kws={"shrink": 0.5}
    )
    ax.set_title(
        "Customer Count per Cohort Month",
        color=TEXT_C, pad=10
    )
    ax.set_xlabel("Months Since First Purchase")
    ax.set_ylabel("Cohort Month")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

    # Plot 2: Retention rate heatmap
    ax = axes[1]
    sns.heatmap(
        retention_str,
        ax=ax,
        cmap="RdYlGn",
        annot=True,
        fmt=".1f",
        annot_kws={"size": 8},
        linewidths=0.5,
        linecolor=BG,
        vmin=0,
        vmax=100,
        cbar_kws={
            "shrink": 0.5,
            "label" : "Retention %"
        }
    )
    ax.set_title(
        "Retention Rate (%) per Cohort Month",
        color=TEXT_C, pad=10
    )
    ax.set_xlabel("Months Since First Purchase")
    ax.set_ylabel("Cohort Month")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

    plt.tight_layout()
    save_plot(fig, "cohort_01_heatmaps.png")


def plot_retention_curves(
    retention_pivot: pd.DataFrame
) -> None:
    """
    Plot retention curves for each cohort.
    """
    logger.info("-- Plotting retention curves --")

    fig, ax = plt.subplots(figsize=(18, 8), facecolor=BG)
    fig.suptitle(
        "COHORT RETENTION CURVES",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    cmap   = plt.cm.RdYlGn
    colors = [
        cmap(i / len(retention_pivot))
        for i in range(len(retention_pivot))
    ]

    for i, (idx, row) in enumerate(retention_pivot.iterrows()):
        values = row.values
        months = range(len(values))
        ax.plot(
            months, values,
            color=colors[i], lw=2,
            marker="o", ms=4,
            label=str(idx), alpha=0.8
        )

    ax.set_xlabel("Months Since First Purchase")
    ax.set_ylabel("Retention Rate (%)")
    ax.set_title("Retention Curves by Cohort")
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=7,
        loc="upper right", ncol=2
    )
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    save_plot(fig, "cohort_02_retention_curves.png")


def plot_cohort_size(metrics: pd.DataFrame) -> None:
    """
    Plot cohort size over time.
    """
    logger.info("-- Plotting cohort size --")

    fig, axes = plt.subplots(
        1, 2, figsize=(20, 7), facecolor=BG
    )
    fig.suptitle(
        "COHORT SIZE & RETENTION METRICS",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    # Plot 1: Cohort size
    ax = axes[0]
    bars = ax.bar(
        metrics["cohort_month"].astype(str),
        metrics["cohort_size"],
        color=BLUE, edgecolor="none", alpha=0.85
    )
    ax.set_ylabel("Number of Customers")
    ax.set_title("New Customers per Cohort Month")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, metrics["cohort_size"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 20,
            f"{int(val):,}",
            ha="center", fontsize=7,
            color=TEXT_C
        )

    # Plot 2: Month 1 retention rate
    ax = axes[1]
    colors = [
        GREEN  if v >= 10
        else ORANGE if v >= 5
        else RED
        for v in metrics["month_1_retention_pct"]
    ]
    bars = ax.bar(
        metrics["cohort_month"].astype(str),
        metrics["month_1_retention_pct"],
        color=colors, edgecolor="none", alpha=0.85
    )
    ax.set_ylabel("Month 1 Retention Rate (%)")
    ax.set_title("Month 1 Retention Rate by Cohort")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(
        bars, metrics["month_1_retention_pct"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.1,
            f"{val:.1f}%",
            ha="center", fontsize=7,
            color=TEXT_C
        )

    plt.tight_layout()
    save_plot(fig, "cohort_03_size_retention.png")


def plot_revenue_cohort(
    revenue_pivot: pd.DataFrame
) -> None:
    """
    Plot revenue cohort heatmap.
    """
    logger.info("-- Plotting revenue cohort --")

    revenue_str = revenue_pivot.copy()
    revenue_str.index = revenue_str.index.astype(str)

    fig, ax = plt.subplots(figsize=(18, 8), facecolor=BG)
    fig.suptitle(
        "REVENUE COHORT ANALYSIS - Avg Revenue per Customer",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    sns.heatmap(
        revenue_str,
        ax=ax,
        cmap="RdYlGn",
        annot=True,
        fmt=".0f",
        annot_kws={"size": 8},
        linewidths=0.5,
        linecolor=BG,
        cbar_kws={
            "shrink": 0.5,
            "label" : "Avg Revenue (R$)"
        }
    )
    ax.set_title(
        "Avg Revenue per Customer by Cohort",
        color=TEXT_C, pad=10
    )
    ax.set_xlabel("Months Since First Purchase")
    ax.set_ylabel("Cohort Month")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

    plt.tight_layout()
    save_plot(fig, "cohort_04_revenue.png")


# ════════════════════════════════════════════════════════════
# MAIN COHORT FUNCTION
# ════════════════════════════════════════════════════════════

def run_cohort_analysis() -> tuple:
    """
    Run complete cohort analysis pipeline.

    Returns:
        tuple: (cohort_pivot, retention_pivot, metrics)
    """
    logger.info("=" * 55)
    logger.info("COHORT ANALYSIS STARTED")
    logger.info("=" * 55)

    total_start = time.time()

    # Step 1: Load data
    df = load_data()

    # Step 2: Build cohort data
    cohort_pivot, retention_pivot = build_cohort_data(df)

    # Step 3: Build revenue cohort
    revenue_pivot = build_revenue_cohort(df)

    # Step 4: Calculate metrics
    metrics = calculate_cohort_metrics(
        cohort_pivot, retention_pivot
    )

    # Step 5: Visualize
    plot_cohort_heatmaps(cohort_pivot, retention_pivot)
    plot_retention_curves(retention_pivot)
    plot_cohort_size(metrics)
    plot_revenue_cohort(revenue_pivot)

    # Save results
    reports_dir  = PLOTS_DIR.parent / "reports"
    cohort_path  = reports_dir / "cohort_retention.csv"
    metrics_path = reports_dir / "cohort_metrics.csv"

    retention_pivot.to_csv(cohort_path)
    metrics.to_csv(metrics_path, index=False)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 55)
    print("   COHORT ANALYSIS SUMMARY")
    print("=" * 55)
    print(f"  Total Cohorts    : {len(cohort_pivot)}")
    print(f"  Months Tracked   : {len(cohort_pivot.columns)}")
    print(f"  Plots Generated  : 4")
    print(f"  Time             : {total_elapsed:.2f}s")
    print("-" * 55)
    print(f"  {'Cohort':<12} {'Size':>8} {'M1 Ret%':>10}")
    print("-" * 55)
    for _, row in metrics.iterrows():
        print(
            f"  {str(row['cohort_month']):<12} "
            f"{int(row['cohort_size']):>8,} "
            f"{row['month_1_retention_pct']:>9.1f}%"
        )
    print("=" * 55 + "\n")

    return cohort_pivot, retention_pivot, metrics


# ── Run directly ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    cohort_pivot, retention_pivot, metrics = (
        run_cohort_analysis()
    )