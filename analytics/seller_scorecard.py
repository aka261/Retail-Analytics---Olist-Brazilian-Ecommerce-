# analytics/seller_scorecard.py

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
    DATABASE_PATH,
    create_directories,
)
from src.sql_queries import (
    get_seller_performance,
    get_seller_ranking,
    get_top_sellers,
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
# STEP 1: LOAD SELLER DATA
# ════════════════════════════════════════════════════════════

def load_seller_data() -> pd.DataFrame:
    """
    Load seller performance data from SQL queries.
    """
    logger.info("-- Loading seller data --")

    df = get_seller_performance()

    # Fill missing values
    df["avg_review_score"]   = df["avg_review_score"].fillna(0)
    df["avg_delivery_days"]  = df["avg_delivery_days"].fillna(0)
    df["late_delivery_pct"]  = df["late_delivery_pct"].fillna(0)
    df["late_deliveries"]    = df["late_deliveries"].fillna(0)

    logger.info(f"[OK] Loaded {len(df):,} sellers")
    return df


# ════════════════════════════════════════════════════════════
# STEP 2: CALCULATE SELLER SCORES
# ════════════════════════════════════════════════════════════

def calculate_seller_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate composite seller score based on:
    - Revenue Score       (30%)
    - Review Score        (30%)
    - Delivery Score      (20%)
    - Order Volume Score  (20%)
    """
    logger.info("-- Calculating seller scores --")

    # Normalize each metric to 0-100 scale
    def normalize(series, reverse=False):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series([50] * len(series))
        normalized = (series - min_val) / (max_val - min_val) * 100
        if reverse:
            normalized = 100 - normalized
        return normalized.round(2)

    # Revenue score (higher = better)
    df["revenue_score"] = normalize(df["total_revenue"])

    # Review score (higher = better, scale 0-5 to 0-100)
    df["review_score_normalized"] = (
        df["avg_review_score"] / 5 * 100
    ).round(2)

    # Delivery score (lower late % = better)
    df["delivery_score"] = normalize(
        df["late_delivery_pct"], reverse=True
    )

    # Order volume score (higher = better)
    df["order_score"] = normalize(df["total_orders"])

    # Composite score
    df["composite_score"] = (
        df["revenue_score"]           * 0.30 +
        df["review_score_normalized"] * 0.30 +
        df["delivery_score"]          * 0.20 +
        df["order_score"]             * 0.20
    ).round(2)

    # Seller grade
    df["grade"] = pd.cut(
        df["composite_score"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["F", "D", "C", "B", "A"]
    )

    # Seller tier
    df["tier"] = pd.cut(
        df["composite_score"],
        bins=[0, 25, 50, 75, 100],
        labels=["Bronze", "Silver", "Gold", "Platinum"]
    )

    logger.info("[OK] Seller scores calculated")
    logger.info(
        f"  Avg Composite Score : "
        f"{df['composite_score'].mean():.2f}"
    )
    logger.info(
        f"  Top Score           : "
        f"{df['composite_score'].max():.2f}"
    )
    return df


# ════════════════════════════════════════════════════════════
# STEP 3: IDENTIFY TOP & BOTTOM SELLERS
# ════════════════════════════════════════════════════════════

def identify_seller_groups(
    df: pd.DataFrame
) -> tuple:
    """
    Split sellers into top, middle and bottom groups.
    """
    logger.info("-- Identifying seller groups --")

    top_sellers    = df.nlargest(20, "composite_score")
    bottom_sellers = df.nsmallest(20, "composite_score")
    middle_sellers = df[
        ~df["seller_id"].isin(
            top_sellers["seller_id"].tolist() +
            bottom_sellers["seller_id"].tolist()
        )
    ]

    logger.info(
        f"[OK] Top sellers    : {len(top_sellers):,}"
    )
    logger.info(
        f"[OK] Middle sellers : {len(middle_sellers):,}"
    )
    logger.info(
        f"[OK] Bottom sellers : {len(bottom_sellers):,}"
    )

    return top_sellers, middle_sellers, bottom_sellers


# ════════════════════════════════════════════════════════════
# STEP 4: SELLER TIER ANALYSIS
# ════════════════════════════════════════════════════════════

def analyze_seller_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance metrics by seller tier.
    """
    logger.info("-- Analyzing seller tiers --")

    tier_summary = df.groupby("tier").agg(
        seller_count      = ("seller_id",          "count"),
        avg_revenue       = ("total_revenue",       "mean"),
        total_revenue     = ("total_revenue",       "sum"),
        avg_orders        = ("total_orders",        "mean"),
        avg_review        = ("avg_review_score",    "mean"),
        avg_delivery_days = ("avg_delivery_days",   "mean"),
        avg_late_pct      = ("late_delivery_pct",   "mean"),
        avg_score         = ("composite_score",     "mean"),
    ).reset_index()

    tier_summary = tier_summary.round(2)

    logger.info(f"[OK] Tier analysis complete")
    logger.info(f"\n{tier_summary.to_string(index=False)}")

    return tier_summary


# ════════════════════════════════════════════════════════════
# STEP 5: VISUALIZE SELLER SCORECARD
# ════════════════════════════════════════════════════════════

def plot_seller_overview(df: pd.DataFrame) -> None:
    """
    Plot seller performance overview.
    """
    logger.info("-- Plotting seller overview --")

    fig, axes = plt.subplots(
        2, 3, figsize=(22, 14), facecolor=BG
    )
    fig.suptitle(
        "SELLER PERFORMANCE OVERVIEW",
        fontsize=18, fontweight="bold", color=TEXT_C
    )

    # Plot 1: Revenue distribution
    ax = axes[0, 0]
    ax.hist(
        df["total_revenue"].clip(0, 100000),
        bins=50, color=BLUE,
        edgecolor="none", alpha=0.85
    )
    ax.set_xlabel("Total Revenue (R$)")
    ax.set_ylabel("Number of Sellers")
    ax.set_title("Revenue Distribution")
    ax.axvline(
        df["total_revenue"].mean(),
        color=RED, lw=2, ls="--",
        label=f"Mean: R${df['total_revenue'].mean():,.0f}"
    )
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=8
    )
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: Review score distribution
    ax = axes[0, 1]
    review_counts = df["avg_review_score"].round(
        1
    ).value_counts().sort_index()
    ax.bar(
        review_counts.index,
        review_counts.values,
        color=GREEN, edgecolor="none", alpha=0.85,
        width=0.08
    )
    ax.set_xlabel("Avg Review Score")
    ax.set_ylabel("Number of Sellers")
    ax.set_title("Review Score Distribution")
    ax.axvline(
        df["avg_review_score"].mean(),
        color=RED, lw=2, ls="--",
        label=f"Mean: {df['avg_review_score'].mean():.2f}"
    )
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=8
    )
    ax.grid(axis="y", alpha=0.3)

    # Plot 3: Late delivery distribution
    ax = axes[0, 2]
    ax.hist(
        df["late_delivery_pct"],
        bins=30, color=ORANGE,
        edgecolor="none", alpha=0.85
    )
    ax.set_xlabel("Late Delivery %")
    ax.set_ylabel("Number of Sellers")
    ax.set_title("Late Delivery Rate Distribution")
    ax.axvline(
        df["late_delivery_pct"].mean(),
        color=RED, lw=2, ls="--",
        label=f"Mean: {df['late_delivery_pct'].mean():.1f}%"
    )
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=8
    )
    ax.grid(axis="y", alpha=0.3)

    # Plot 4: Composite score distribution
    ax = axes[1, 0]
    ax.hist(
        df["composite_score"],
        bins=30, color=PURPLE,
        edgecolor="none", alpha=0.85
    )
    ax.set_xlabel("Composite Score")
    ax.set_ylabel("Number of Sellers")
    ax.set_title("Composite Score Distribution")
    ax.axvline(
        df["composite_score"].mean(),
        color=RED, lw=2, ls="--",
        label=f"Mean: {df['composite_score'].mean():.1f}"
    )
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=8
    )
    ax.grid(axis="y", alpha=0.3)

    # Plot 5: Seller grade distribution
    ax = axes[1, 1]
    grade_counts = df["grade"].value_counts().sort_index()
    grade_colors = {
        "A": GREEN, "B": BLUE,
        "C": ORANGE, "D": PURPLE, "F": RED
    }
    bars = ax.bar(
        grade_counts.index,
        grade_counts.values,
        color=[
            grade_colors.get(g, BLUE)
            for g in grade_counts.index
        ],
        edgecolor="none", alpha=0.85
    )
    ax.set_xlabel("Grade")
    ax.set_ylabel("Number of Sellers")
    ax.set_title("Seller Grade Distribution")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, grade_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 5,
            f"{val:,}",
            ha="center", fontsize=9,
            fontweight="bold", color=TEXT_C
        )

    # Plot 6: Tier distribution donut
    ax = axes[1, 2]
    tier_counts = df["tier"].value_counts()
    tier_colors = {
        "Platinum": "#E5E4E2",
        "Gold"    : "#FFD700",
        "Silver"  : "#C0C0C0",
        "Bronze"  : "#CD7F32",
    }
    t_colors = [
        tier_colors.get(t, BLUE)
        for t in tier_counts.index
    ]
    wedges, texts, autotexts = ax.pie(
        tier_counts.values,
        labels=tier_counts.index,
        autopct="%1.1f%%",
        colors=t_colors,
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(
            width=0.5,
            edgecolor=BG,
            linewidth=2
        )
    )
    for at in autotexts:
        at.set(fontsize=9, fontweight="bold", color="white")
    for t in texts:
        t.set(fontsize=9, color=TEXT_C)
    ax.set_title("Seller Tier Distribution")

    plt.tight_layout()
    save_plot(fig, "seller_01_overview.png")


def plot_top_sellers(
    top_sellers: pd.DataFrame
) -> None:
    """
    Plot top 20 sellers scorecard.
    """
    logger.info("-- Plotting top sellers --")

    fig, axes = plt.subplots(
        1, 2, figsize=(22, 10), facecolor=BG
    )
    fig.suptitle(
        "TOP 20 SELLERS SCORECARD",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    # Plot 1: Top sellers by composite score
    ax = axes[0]
    grade_colors = {
        "A": GREEN, "B": BLUE,
        "C": ORANGE, "D": PURPLE, "F": RED
    }
    colors = [
        grade_colors.get(str(g), BLUE)
        for g in top_sellers["grade"]
    ]
    bars = ax.barh(
        range(len(top_sellers)),
        top_sellers["composite_score"],
        color=colors, edgecolor="none", alpha=0.85
    )
    ax.set_yticks(range(len(top_sellers)))
    ax.set_yticklabels(
        [f"Seller {i+1}" for i in range(len(top_sellers))],
        fontsize=8
    )
    ax.set_xlabel("Composite Score")
    ax.set_title("Top 20 Sellers by Composite Score")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(
        bars, top_sellers["composite_score"]
    ):
        ax.text(
            val + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            va="center", fontsize=8, color=TEXT_C
        )

    # Grade legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GREEN,  label="Grade A"),
        Patch(facecolor=BLUE,   label="Grade B"),
        Patch(facecolor=ORANGE, label="Grade C"),
        Patch(facecolor=PURPLE, label="Grade D"),
        Patch(facecolor=RED,    label="Grade F"),
    ]
    ax.legend(
        handles=legend_elements,
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=8
    )

    # Plot 2: Revenue vs Review scatter
    ax = axes[1]
    scatter = ax.scatter(
        top_sellers["total_revenue"],
        top_sellers["avg_review_score"],
        c=top_sellers["composite_score"],
        cmap="RdYlGn",
        s=top_sellers["total_orders"] * 3,
        alpha=0.7,
        edgecolors="none",
        vmin=0, vmax=100
    )
    plt.colorbar(
        scatter, ax=ax,
        label="Composite Score"
    )
    ax.set_xlabel("Total Revenue (R$)")
    ax.set_ylabel("Avg Review Score")
    ax.set_title(
        "Revenue vs Review Score\n(bubble = orders)"
    )
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_plot(fig, "seller_02_top_sellers.png")


def plot_tier_analysis(
    tier_summary: pd.DataFrame
) -> None:
    """
    Plot seller tier analysis.
    """
    logger.info("-- Plotting tier analysis --")

    fig, axes = plt.subplots(
        2, 2, figsize=(20, 14), facecolor=BG
    )
    fig.suptitle(
        "SELLER TIER ANALYSIS",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    tier_colors = {
        "Platinum": "#E5E4E2",
        "Gold"    : "#FFD700",
        "Silver"  : "#C0C0C0",
        "Bronze"  : "#CD7F32",
    }
    colors = [
        tier_colors.get(str(t), BLUE)
        for t in tier_summary["tier"]
    ]

    # Plot 1: Seller count per tier
    ax = axes[0, 0]
    bars = ax.bar(
        tier_summary["tier"],
        tier_summary["seller_count"],
        color=colors, edgecolor="none", alpha=0.85
    )
    ax.set_ylabel("Number of Sellers")
    ax.set_title("Sellers per Tier")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, tier_summary["seller_count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1,
            f"{val:,}",
            ha="center", fontsize=10,
            fontweight="bold", color=TEXT_C
        )

    # Plot 2: Avg revenue per tier
    ax = axes[0, 1]
    bars = ax.bar(
        tier_summary["tier"],
        tier_summary["avg_revenue"],
        color=colors, edgecolor="none", alpha=0.85
    )
    ax.set_ylabel("Avg Revenue (R$)")
    ax.set_title("Avg Revenue per Tier")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, tier_summary["avg_revenue"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 100,
            f"R${val:,.0f}",
            ha="center", fontsize=9,
            fontweight="bold", color=TEXT_C
        )

    # Plot 3: Avg review per tier
    ax = axes[1, 0]
    bars = ax.bar(
        tier_summary["tier"],
        tier_summary["avg_review"],
        color=colors, edgecolor="none", alpha=0.85
    )
    ax.set_ylabel("Avg Review Score")
    ax.set_title("Avg Review Score per Tier")
    ax.set_ylim(0, 5)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, tier_summary["avg_review"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.05,
            f"{val:.2f}",
            ha="center", fontsize=10,
            fontweight="bold", color=TEXT_C
        )

    # Plot 4: Avg late delivery per tier
    ax = axes[1, 1]
    late_colors = [
        RED    if v > 15
        else ORANGE if v > 10
        else GREEN
        for v in tier_summary["avg_late_pct"]
    ]
    bars = ax.bar(
        tier_summary["tier"],
        tier_summary["avg_late_pct"],
        color=late_colors, edgecolor="none", alpha=0.85
    )
    ax.set_ylabel("Avg Late Delivery %")
    ax.set_title("Avg Late Delivery Rate per Tier")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(
        bars, tier_summary["avg_late_pct"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.2,
            f"{val:.1f}%",
            ha="center", fontsize=10,
            fontweight="bold", color=TEXT_C
        )

    plt.tight_layout()
    save_plot(fig, "seller_03_tier_analysis.png")


def plot_seller_radar(
    top_sellers: pd.DataFrame
) -> None:
    """
    Plot radar chart for top 5 sellers.
    """
    logger.info("-- Plotting seller radar --")

    # Select top 5 sellers
    top5 = top_sellers.head(5)

    # Metrics for radar
    metrics = [
        "revenue_score",
        "review_score_normalized",
        "delivery_score",
        "order_score",
    ]
    metric_labels = [
        "Revenue",
        "Review",
        "Delivery",
        "Orders",
    ]

    # Number of variables
    N      = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(12, 10),
        subplot_kw=dict(polar=True),
        facecolor=BG
    )
    ax.set_facecolor(CARD_BG)
    fig.suptitle(
        "TOP 5 SELLERS - PERFORMANCE RADAR",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    colors_radar = [GREEN, BLUE, ORANGE, PURPLE, RED]

    for i, (_, row) in enumerate(top5.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]

        ax.plot(
            angles, values,
            color=colors_radar[i], lw=2,
            label=f"Seller {i+1} (Score: {row['composite_score']:.1f})"
        )
        ax.fill(
            angles, values,
            color=colors_radar[i], alpha=0.1
        )

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        metric_labels,
        color=TEXT_C, fontsize=11
    )
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(
        ["20", "40", "60", "80", "100"],
        color=TEXT_C, fontsize=7
    )
    ax.grid(color=GRID_C, alpha=0.5)
    ax.spines["polar"].set_color(GRID_C)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.1),
        facecolor=CARD_BG,
        labelcolor=TEXT_C,
        edgecolor=GRID_C,
        fontsize=9
    )

    plt.tight_layout()
    save_plot(fig, "seller_04_radar.png")


def plot_seller_region(df: pd.DataFrame) -> None:
    """
    Plot seller performance by region.
    """
    logger.info("-- Plotting seller region analysis --")

    region_summary = df.groupby("seller_region").agg(
        seller_count  = ("seller_id",         "count"),
        avg_revenue   = ("total_revenue",      "mean"),
        total_revenue = ("total_revenue",      "sum"),
        avg_score     = ("composite_score",    "mean"),
        avg_review    = ("avg_review_score",   "mean"),
        avg_late_pct  = ("late_delivery_pct",  "mean"),
    ).reset_index().round(2)

    fig, axes = plt.subplots(
        1, 3, figsize=(22, 8), facecolor=BG
    )
    fig.suptitle(
        "SELLER PERFORMANCE BY REGION",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    region_colors = {
        "Southeast"   : BLUE,
        "South"       : GREEN,
        "Northeast"   : ORANGE,
        "North"       : PURPLE,
        "Central-West": RED,
        "Unknown"     : GRID_C,
    }
    colors = [
        region_colors.get(r, BLUE)
        for r in region_summary["seller_region"]
    ]

    # Plot 1: Seller count by region
    ax = axes[0]
    bars = ax.bar(
        region_summary["seller_region"],
        region_summary["seller_count"],
        color=colors, edgecolor="none", alpha=0.85
    )
    ax.set_ylabel("Number of Sellers")
    ax.set_title("Sellers by Region")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(
        bars, region_summary["seller_count"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1,
            f"{val:,}",
            ha="center", fontsize=9,
            fontweight="bold", color=TEXT_C
        )

    # Plot 2: Avg revenue by region
    ax = axes[1]
    bars = ax.bar(
        region_summary["seller_region"],
        region_summary["avg_revenue"],
        color=colors, edgecolor="none", alpha=0.85
    )
    ax.set_ylabel("Avg Revenue (R$)")
    ax.set_title("Avg Revenue by Region")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.grid(axis="y", alpha=0.3)

    # Plot 3: Avg score by region
    ax = axes[2]
    bars = ax.bar(
        region_summary["seller_region"],
        region_summary["avg_score"],
        color=colors, edgecolor="none", alpha=0.85
    )
    ax.set_ylabel("Avg Composite Score")
    ax.set_title("Avg Score by Region")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(
        bars, region_summary["avg_score"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.5,
            f"{val:.1f}",
            ha="center", fontsize=9,
            fontweight="bold", color=TEXT_C
        )

    plt.tight_layout()
    save_plot(fig, "seller_05_region.png")


# ════════════════════════════════════════════════════════════
# MAIN SELLER SCORECARD FUNCTION
# ════════════════════════════════════════════════════════════

def run_seller_scorecard() -> tuple:
    """
    Run complete seller scorecard analysis.

    Returns:
        tuple: (df, tier_summary)
    """
    logger.info("=" * 55)
    logger.info("SELLER SCORECARD STARTED")
    logger.info("=" * 55)

    total_start = time.time()

    # Step 1: Load data
    df = load_seller_data()

    # Step 2: Calculate scores
    df = calculate_seller_scores(df)

    # Step 3: Identify groups
    top_sellers, middle_sellers, bottom_sellers = (
        identify_seller_groups(df)
    )

    # Step 4: Tier analysis
    tier_summary = analyze_seller_tiers(df)

    # Step 5: Visualize
    plot_seller_overview(df)
    plot_top_sellers(top_sellers)
    plot_tier_analysis(tier_summary)
    plot_seller_radar(top_sellers)
    plot_seller_region(df)

    # Save results
    reports_dir = PLOTS_DIR.parent / "reports"
    df.to_csv(
        reports_dir / "seller_scorecard.csv",
        index=False
    )
    tier_summary.to_csv(
        reports_dir / "seller_tier_summary.csv",
        index=False
    )

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 55)
    print("   SELLER SCORECARD SUMMARY")
    print("=" * 55)
    print(f"  Total Sellers     : {len(df):,}")
    print(f"  Plots Generated   : 5")
    print(f"  Time              : {total_elapsed:.2f}s")
    print("-" * 55)
    print(f"  {'Tier':<12} {'Count':>8} {'Avg Score':>12} {'Avg Rev':>12}")
    print("-" * 55)
    for _, row in tier_summary.iterrows():
        print(
            f"  {str(row['tier']):<12} "
            f"{row['seller_count']:>8,} "
            f"{row['avg_score']:>11.1f} "
            f"R${row['avg_revenue']:>9,.0f}"
        )
    print("=" * 55 + "\n")

    return df, tier_summary


# ── Run directly ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, tier_summary = run_seller_scorecard()