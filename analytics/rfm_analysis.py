# analytics/rfm_analysis.py

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
from datetime import datetime

from src.config import (
    DATABASE_PATH,
    PLOTS_DIR,
    VIZ_SETTINGS,
    PROCESSED_FILES,
    RFM_SETTINGS,
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
    Load master dataset and prepare for RFM analysis.
    """
    logger.info("-- Loading data for RFM analysis --")

    df = pd.read_csv(PROCESSED_FILES["master"])

    # Convert date columns
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    )

    # Keep only delivered orders
    df = df[df["order_status"] == "delivered"].copy()

    logger.info(f"[OK] Loaded {len(df):,} delivered orders")
    return df


# ════════════════════════════════════════════════════════════
# STEP 2: CALCULATE RFM METRICS
# ════════════════════════════════════════════════════════════

def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Recency, Frequency and Monetary
    values for each customer.

    Recency  : Days since last purchase
    Frequency: Number of orders
    Monetary : Total amount spent
    """
    logger.info("-- Calculating RFM metrics --")

    # Reference date — day after last order
    reference_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    logger.info(f"  Reference date: {reference_date.date()}")

    # Calculate RFM per customer
    rfm = df.groupby("customer_unique_id").agg(
        recency   = ("order_purchase_timestamp",
                     lambda x: (reference_date - x.max()).days),
        frequency = ("order_id",   "nunique"),
        monetary  = ("total_payment_value", "sum"),
    ).reset_index()

    rfm["monetary"] = rfm["monetary"].round(2)

    logger.info(f"[OK] RFM calculated for {len(rfm):,} customers")
    logger.info(f"  Avg Recency   : {rfm['recency'].mean():.1f} days")
    logger.info(f"  Avg Frequency : {rfm['frequency'].mean():.2f} orders")
    logger.info(f"  Avg Monetary  : R${rfm['monetary'].mean():.2f}")

    return rfm


# ════════════════════════════════════════════════════════════
# STEP 3: SCORE RFM
# ════════════════════════════════════════════════════════════

def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign scores 1-5 to each RFM metric.

    Recency  : Lower days = higher score (5 = most recent)
    Frequency: Higher count = higher score (5 = most frequent)
    Monetary : Higher amount = higher score (5 = highest spender)
    """
    logger.info("-- Scoring RFM metrics --")

    bins = RFM_SETTINGS["recency_bins"]

    # Recency score (reversed — lower days = better)
    rfm["r_score"] = pd.qcut(
        rfm["recency"],
        q=bins,
        labels=[5, 4, 3, 2, 1],
        duplicates="drop"
    ).astype(int)

    # Frequency score
    rfm["f_score"] = pd.qcut(
        rfm["frequency"].rank(method="first"),
        q=bins,
        labels=[1, 2, 3, 4, 5],
        duplicates="drop"
    ).astype(int)

    # Monetary score
    rfm["m_score"] = pd.qcut(
        rfm["monetary"].rank(method="first"),
        q=bins,
        labels=[1, 2, 3, 4, 5],
        duplicates="drop"
    ).astype(int)

    # Combined RFM score
    rfm["rfm_score"] = (
        rfm["r_score"].astype(str) +
        rfm["f_score"].astype(str) +
        rfm["m_score"].astype(str)
    )

    # Total score
    rfm["rfm_total"] = (
        rfm["r_score"] +
        rfm["f_score"] +
        rfm["m_score"]
    )

    logger.info("[OK] RFM scores assigned")
    return rfm


# ════════════════════════════════════════════════════════════
# STEP 4: SEGMENT CUSTOMERS
# ════════════════════════════════════════════════════════════

def segment_customers(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign customer segments based on RFM scores.
    """
    logger.info("-- Segmenting customers --")

    def assign_segment(row):
        r = row["r_score"]
        f = row["f_score"]
        m = row["m_score"]

        if r >= 4 and f >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 3 and f <= 2:
            return "Potential Loyalist"
        elif r >= 4 and f <= 1:
            return "New Customers"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r <= 2 and f <= 2 and m >= 3:
            return "Cannot Lose Them"
        elif r <= 2 and f <= 2:
            return "Lost"
        else:
            return "Others"

    rfm["segment"] = rfm.apply(assign_segment, axis=1)

    # Segment summary
    segment_summary = rfm.groupby("segment").agg(
        customer_count = ("customer_unique_id", "count"),
        avg_recency    = ("recency",   "mean"),
        avg_frequency  = ("frequency", "mean"),
        avg_monetary   = ("monetary",  "mean"),
        total_monetary = ("monetary",  "sum"),
    ).reset_index()

    segment_summary["avg_recency"]    = segment_summary["avg_recency"].round(1)
    segment_summary["avg_frequency"]  = segment_summary["avg_frequency"].round(2)
    segment_summary["avg_monetary"]   = segment_summary["avg_monetary"].round(2)
    segment_summary["total_monetary"] = segment_summary["total_monetary"].round(2)
    segment_summary["customer_pct"]   = (
        segment_summary["customer_count"] /
        segment_summary["customer_count"].sum() * 100
    ).round(2)

    logger.info("[OK] Customer segments assigned")
    logger.info(f"\n{segment_summary.to_string(index=False)}")

    return rfm, segment_summary


# ════════════════════════════════════════════════════════════
# STEP 5: VISUALIZE RFM
# ════════════════════════════════════════════════════════════

def plot_rfm_distributions(rfm: pd.DataFrame) -> None:
    """
    Plot distributions of R, F, M values.
    """
    logger.info("-- Plotting RFM distributions --")

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor=BG)
    fig.suptitle(
        "RFM METRIC DISTRIBUTIONS",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    # Recency distribution
    ax = axes[0]
    ax.hist(
        rfm["recency"], bins=50,
        color=BLUE, edgecolor="none", alpha=0.85
    )
    ax.axvline(
        rfm["recency"].mean(),
        color=RED, lw=2, ls="--",
        label=f"Mean: {rfm['recency'].mean():.0f} days"
    )
    ax.axvline(
        rfm["recency"].median(),
        color=GREEN, lw=2, ls="--",
        label=f"Median: {rfm['recency'].median():.0f} days"
    )
    ax.set_xlabel("Recency (Days)")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Recency Distribution")
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=8
    )
    ax.grid(axis="y", alpha=0.3)

    # Frequency distribution
    ax = axes[1]
    freq_counts = rfm["frequency"].value_counts().sort_index()
    ax.bar(
        freq_counts.index,
        freq_counts.values,
        color=GREEN, edgecolor="none", alpha=0.85
    )
    ax.set_xlabel("Frequency (Orders)")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Frequency Distribution")
    ax.set_xlim(0, 10)
    ax.grid(axis="y", alpha=0.3)

    # Monetary distribution
    ax = axes[2]
    monetary_clip = rfm["monetary"].clip(0, 2000)
    ax.hist(
        monetary_clip, bins=50,
        color=ORANGE, edgecolor="none", alpha=0.85
    )
    ax.axvline(
        rfm["monetary"].mean(),
        color=RED, lw=2, ls="--",
        label=f"Mean: R${rfm['monetary'].mean():.0f}"
    )
    ax.axvline(
        rfm["monetary"].median(),
        color=GREEN, lw=2, ls="--",
        label=f"Median: R${rfm['monetary'].median():.0f}"
    )
    ax.set_xlabel("Monetary (R$) - clipped at R$2000")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Monetary Distribution")
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=8
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_plot(fig, "rfm_01_distributions.png")


def plot_rfm_scores(rfm: pd.DataFrame) -> None:
    """
    Plot RFM score distributions.
    """
    logger.info("-- Plotting RFM scores --")

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor=BG)
    fig.suptitle(
        "RFM SCORE DISTRIBUTIONS",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    score_cols = ["r_score", "f_score", "m_score"]
    titles     = ["Recency Score", "Frequency Score", "Monetary Score"]
    colors     = [BLUE, GREEN, ORANGE]

    for ax, col, title, color in zip(axes, score_cols, titles, colors):
        score_counts = rfm[col].value_counts().sort_index()
        bars = ax.bar(
            score_counts.index,
            score_counts.values,
            color=color, edgecolor="none", alpha=0.85
        )
        ax.set_xlabel("Score (1-5)")
        ax.set_ylabel("Number of Customers")
        ax.set_title(title)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, score_counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 100,
                f"{val:,}",
                ha="center", fontsize=9,
                fontweight="bold", color=TEXT_C
            )

    plt.tight_layout()
    save_plot(fig, "rfm_02_scores.png")


def plot_segment_analysis(
    rfm: pd.DataFrame,
    segment_summary: pd.DataFrame
) -> None:
    """
    Plot customer segment analysis.
    """
    logger.info("-- Plotting segment analysis --")

    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    fig.suptitle(
        "RFM CUSTOMER SEGMENT ANALYSIS",
        fontsize=18, fontweight="bold", color=TEXT_C
    )

    gs     = gridspec.GridSpec(2, 3, figure=fig,
                               hspace=0.4, wspace=0.35)
    colors = [
        GREEN, BLUE, ORANGE, PURPLE,
        RED, "#1ABC9C", "#E67E22", "#34495E"
    ]

    # Plot 1: Customer count donut
    ax1 = fig.add_subplot(gs[0, 0])
    wedges, texts, autotexts = ax1.pie(
        segment_summary["customer_count"],
        labels=segment_summary["segment"],
        autopct="%1.1f%%",
        colors=colors[:len(segment_summary)],
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(
            width=0.5, edgecolor=BG, linewidth=2
        )
    )
    for at in autotexts:
        at.set(fontsize=8, fontweight="bold", color="white")
    for t in texts:
        t.set(fontsize=7, color=TEXT_C)
    ax1.set_title("Customer Count by Segment")

    # Plot 2: Total revenue by segment
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(
        segment_summary["segment"],
        segment_summary["total_monetary"],
        color=colors[:len(segment_summary)],
        edgecolor="none", alpha=0.85
    )
    ax2.set_ylabel("Total Revenue (R$)")
    ax2.set_title("Total Revenue by Segment")
    ax2.tick_params(axis="x", rotation=30, labelsize=7)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax2.grid(axis="y", alpha=0.3)

    # Plot 3: Avg monetary by segment
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(
        segment_summary["segment"],
        segment_summary["avg_monetary"],
        color=colors[:len(segment_summary)],
        edgecolor="none", alpha=0.85
    )
    ax3.set_ylabel("Avg Monetary (R$)")
    ax3.set_title("Avg Spending by Segment")
    ax3.tick_params(axis="x", rotation=30, labelsize=7)
    ax3.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax3.grid(axis="y", alpha=0.3)

    # Plot 4: Avg recency by segment
    ax4 = fig.add_subplot(gs[1, 0])
    bars = ax4.bar(
        segment_summary["segment"],
        segment_summary["avg_recency"],
        color=colors[:len(segment_summary)],
        edgecolor="none", alpha=0.85
    )
    ax4.set_ylabel("Avg Recency (Days)")
    ax4.set_title("Avg Recency by Segment")
    ax4.tick_params(axis="x", rotation=30, labelsize=7)
    ax4.grid(axis="y", alpha=0.3)

    # Plot 5: Avg frequency by segment
    ax5 = fig.add_subplot(gs[1, 1])
    bars = ax5.bar(
        segment_summary["segment"],
        segment_summary["avg_frequency"],
        color=colors[:len(segment_summary)],
        edgecolor="none", alpha=0.85
    )
    ax5.set_ylabel("Avg Frequency (Orders)")
    ax5.set_title("Avg Frequency by Segment")
    ax5.tick_params(axis="x", rotation=30, labelsize=7)
    ax5.grid(axis="y", alpha=0.3)

    # Plot 6: Segment summary table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    table_data = []
    for _, row in segment_summary.iterrows():
        table_data.append([
            row["segment"],
            f"{row['customer_count']:,}",
            f"{row['customer_pct']}%",
            f"R${row['avg_monetary']:,.0f}",
            f"{row['avg_recency']:.0f}d",
        ])
    t = ax6.table(
        cellText=table_data,
        colLabels=["Segment", "Count", "Pct", "Avg $", "Recency"],
        cellLoc="center",
        loc="center",
        colWidths=[0.30, 0.15, 0.12, 0.20, 0.15]
    )
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    for (row, col), cell in t.get_celld().items():
        cell.set_facecolor(
            CARD_BG if row % 2 == 0 else "#222535"
        )
        cell.set_text_props(color=TEXT_C)
        cell.set_edgecolor(GRID_C)
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(
                fontweight="bold", color="white"
            )
    ax6.set_title(
        "Segment Summary Table",
        color=TEXT_C, fontsize=11,
        fontweight="bold", pad=15
    )

    save_plot(fig, "rfm_03_segment_analysis.png")


def plot_rfm_scatter(rfm: pd.DataFrame) -> None:
    """
    Plot RFM scatter plots.
    """
    logger.info("-- Plotting RFM scatter --")

    segment_colors = {
        "Champions"         : GREEN,
        "Loyal Customers"   : BLUE,
        "Potential Loyalist": ORANGE,
        "New Customers"     : PURPLE,
        "At Risk"           : RED,
        "Cannot Lose Them"  : "#1ABC9C",
        "Lost"              : "#E67E22",
        "Others"            : "#34495E",
    }

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), facecolor=BG)
    fig.suptitle(
        "RFM SCATTER ANALYSIS",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    # Plot 1: Recency vs Frequency
    ax = axes[0]
    for segment, group in rfm.groupby("segment"):
        ax.scatter(
            group["recency"],
            group["frequency"],
            c=segment_colors.get(segment, BLUE),
            label=segment,
            alpha=0.6, s=20,
            edgecolors="none"
        )
    ax.set_xlabel("Recency (Days)")
    ax.set_ylabel("Frequency (Orders)")
    ax.set_title("Recency vs Frequency by Segment")
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=7,
        markerscale=2
    )
    ax.grid(alpha=0.3)

    # Plot 2: Frequency vs Monetary
    ax = axes[1]
    monetary_clip = rfm["monetary"].clip(0, 5000)
    for segment, group in rfm.groupby("segment"):
        ax.scatter(
            group["frequency"],
            group["monetary"].clip(0, 5000),
            c=segment_colors.get(segment, BLUE),
            label=segment,
            alpha=0.6, s=20,
            edgecolors="none"
        )
    ax.set_xlabel("Frequency (Orders)")
    ax.set_ylabel("Monetary (R$) - clipped at R$5000")
    ax.set_title("Frequency vs Monetary by Segment")
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=7,
        markerscale=2
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_plot(fig, "rfm_04_scatter.png")


def plot_rfm_heatmap(rfm: pd.DataFrame) -> None:
    """
    Plot RFM score heatmap.
    R score vs F score with avg monetary as color.
    """
    logger.info("-- Plotting RFM heatmap --")

    pivot = rfm.pivot_table(
        index="r_score",
        columns="f_score",
        values="monetary",
        aggfunc="mean"
    ).round(2)

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
    fig.suptitle(
        "RFM HEATMAP - Avg Monetary by R & F Score",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn",
        annot=True,
        fmt=".0f",
        annot_kws={"size": 10},
        linewidths=0.5,
        linecolor=BG,
        cbar_kws={
            "shrink": 0.7,
            "label" : "Avg Monetary (R$)"
        }
    )
    ax.set_xlabel("Frequency Score")
    ax.set_ylabel("Recency Score")
    ax.set_title(
        "Higher = Better Customer",
        color=TEXT_C, pad=10
    )
    ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    save_plot(fig, "rfm_05_heatmap.png")


# ════════════════════════════════════════════════════════════
# MAIN RFM FUNCTION
# ════════════════════════════════════════════════════════════

def run_rfm_analysis() -> tuple:
    """
    Run complete RFM analysis pipeline.

    Returns:
        tuple: (rfm DataFrame, segment_summary DataFrame)
    """
    logger.info("=" * 55)
    logger.info("RFM ANALYSIS STARTED")
    logger.info("=" * 55)

    total_start = time.time()

    # Step 1: Load data
    df = load_data()

    # Step 2: Calculate RFM
    rfm = calculate_rfm(df)

    # Step 3: Score RFM
    rfm = score_rfm(rfm)

    # Step 4: Segment customers
    rfm, segment_summary = segment_customers(rfm)

    # Step 5: Visualize
    plot_rfm_distributions(rfm)
    plot_rfm_scores(rfm)
    plot_segment_analysis(rfm, segment_summary)
    plot_rfm_scatter(rfm)
    plot_rfm_heatmap(rfm)

    # Save results
    rfm_path = PLOTS_DIR.parent / "reports" / "rfm_results.csv"
    seg_path = PLOTS_DIR.parent / "reports" / "rfm_segments.csv"
    rfm.to_csv(rfm_path, index=False)
    segment_summary.to_csv(seg_path, index=False)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 55)
    print("   RFM ANALYSIS SUMMARY")
    print("=" * 55)
    print(f"  Total Customers  : {len(rfm):,}")
    print(f"  Total Segments   : {rfm['segment'].nunique()}")
    print(f"  Plots Generated  : 5")
    print(f"  Time             : {total_elapsed:.2f}s")
    print("-" * 55)
    print(f"  {'Segment':<22} {'Count':>8} {'Pct':>8} {'Avg R$':>10}")
    print("-" * 55)
    for _, row in segment_summary.sort_values(
        "total_monetary", ascending=False
    ).iterrows():
        print(
            f"  {row['segment']:<22} "
            f"{row['customer_count']:>8,} "
            f"{row['customer_pct']:>7.1f}% "
            f"R${row['avg_monetary']:>8,.0f}"
        )
    print("=" * 55 + "\n")

    return rfm, segment_summary


# ── Run directly ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    rfm, segment_summary = run_rfm_analysis()