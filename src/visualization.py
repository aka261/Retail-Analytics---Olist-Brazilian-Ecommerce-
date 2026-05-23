# src/visualization.py

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
from pathlib import Path

from src.config import (
    PROCESSED_FILES,
    PLOTS_DIR,
    VIZ_SETTINGS,
    DATABASE_PATH,
    create_directories,
)
from src.sql_queries import (
    get_business_overview,
    get_monthly_revenue,
    get_quarterly_revenue,
    get_revenue_by_category,
    get_revenue_by_state,
    get_customer_segmentation,
    get_order_status_summary,
    get_payment_analysis,
    get_review_analysis,
    get_orders_by_weekday,
    get_orders_by_hour,
    get_delivery_analysis,
    get_seller_performance,
    get_rfm_segment_summary,
    get_running_total_revenue,
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
# PLOT 1: EXECUTIVE DASHBOARD
# ════════════════════════════════════════════════════════════

def plot_executive_dashboard() -> None:
    """
    Full executive dashboard with all key KPIs
    and charts in one figure.
    """
    logger.info("-- Plotting executive dashboard --")

    # Load all data
    overview  = get_business_overview()
    monthly   = get_monthly_revenue()
    category  = get_revenue_by_category().head(10)
    state     = get_revenue_by_state().head(10)
    payment   = get_payment_analysis()
    review    = get_review_analysis()

    fig = plt.figure(figsize=(24, 18), facecolor=BG)
    fig.suptitle(
        "RETAIL ANALYTICS - EXECUTIVE DASHBOARD",
        fontsize=20, fontweight="bold",
        color=TEXT_C, y=0.98
    )

    gs = gridspec.GridSpec(
        4, 4, figure=fig,
        hspace=0.5, wspace=0.35
    )

    # ── KPI Cards Row ─────────────────────────────────────
    kpis = [
        ("Total Orders",
         f"{int(overview['total_orders'].iloc[0]):,}",
         BLUE),
        ("Total Revenue",
         f"R${overview['total_revenue'].iloc[0]:,.0f}",
         GREEN),
        ("Avg Order Value",
         f"R${overview['avg_order_value'].iloc[0]:,.2f}",
         ORANGE),
        ("Avg Review Score",
         f"{overview['avg_review_score'].iloc[0]:.2f}/5",
         PURPLE),
    ]

    for i, (title, value, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(color + "22")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(
            0.5, 0.60, value,
            ha="center", va="center",
            fontsize=18, fontweight="bold",
            color=color
        )
        ax.text(
            0.5, 0.25, title,
            ha="center", va="center",
            fontsize=9, color=TEXT_C
        )
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color(color)
            ax.spines[spine].set_linewidth(2)

    # ── Monthly Revenue ───────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    ax.bar(
        monthly["year_month"],
        monthly["total_revenue"],
        color=BLUE, alpha=0.6, edgecolor="none"
    )
    ax.plot(
        monthly["year_month"],
        monthly["total_revenue"],
        color=GREEN, lw=2, marker="o", ms=3
    )
    ax.set_title("Monthly Revenue Trend")
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.grid(axis="y", alpha=0.3)

    # ── Revenue by Category ───────────────────────────────
    ax = fig.add_subplot(gs[1, 2:])
    cat_colors = [
        COLORS[i % len(COLORS)]
        for i in range(len(category))
    ]
    ax.barh(
        category["category"],
        category["total_revenue"],
        color=cat_colors, edgecolor="none", alpha=0.85
    )
    ax.set_title("Top 10 Categories by Revenue")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3)

    # ── Revenue by State ──────────────────────────────────
    ax = fig.add_subplot(gs[2, :2])
    region_colors = {
        "Southeast"   : BLUE,
        "South"       : GREEN,
        "Northeast"   : ORANGE,
        "North"       : PURPLE,
        "Central-West": RED,
    }
    state_colors = [
        region_colors.get(r, BLUE)
        for r in state["region"]
    ]
    ax.barh(
        state["state"],
        state["total_revenue"],
        color=state_colors, edgecolor="none", alpha=0.85
    )
    ax.set_title("Revenue by State")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3)

    # ── Payment Type ──────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    pay_colors = [BLUE, GREEN, ORANGE, PURPLE]
    wedges, texts, autotexts = ax.pie(
        payment["total_orders"],
        labels=payment["payment_type"],
        autopct="%1.1f%%",
        colors=pay_colors[:len(payment)],
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(
            width=0.5,
            edgecolor=BG,
            linewidth=2
        )
    )
    for at in autotexts:
        at.set(fontsize=8, fontweight="bold", color="white")
    for t in texts:
        t.set(fontsize=7, color=TEXT_C)
    ax.set_title("Payment Type Share")

    # ── Review Score ──────────────────────────────────────
    ax = fig.add_subplot(gs[2, 3])
    score_colors = {
        1: RED, 2: ORANGE,
        3: "#F39C12", 4: "#1ABC9C", 5: GREEN
    }
    r_colors = [
        score_colors.get(s, BLUE)
        for s in review["review_score"]
    ]
    ax.bar(
        review["review_score"].astype(str),
        review["total_reviews"],
        color=r_colors, edgecolor="none", alpha=0.85
    )
    ax.set_title("Review Score Distribution")
    ax.set_xlabel("Score")
    ax.grid(axis="y", alpha=0.3)

    # ── Cumulative Revenue ────────────────────────────────
    cum_df = get_running_total_revenue()
    ax = fig.add_subplot(gs[3, :2])
    ax.fill_between(
        cum_df["year_month"],
        cum_df["cumulative_revenue"],
        alpha=0.3, color=BLUE
    )
    ax.plot(
        cum_df["year_month"],
        cum_df["cumulative_revenue"],
        color=BLUE, lw=2
    )
    ax.plot(
        cum_df["year_month"],
        cum_df["rolling_3m_avg"],
        color=GREEN, lw=2, ls="--",
        label="3M Rolling Avg"
    )
    ax.set_title("Cumulative Revenue & Rolling Average")
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=8
    )
    ax.grid(alpha=0.3)

    # ── RFM Segment ───────────────────────────────────────
    rfm = get_rfm_segment_summary()
    ax  = fig.add_subplot(gs[3, 2:])
    rfm_colors = [
        COLORS[i % len(COLORS)]
        for i in range(len(rfm))
    ]
    bars = ax.bar(
        rfm["rfm_segment"],
        rfm["total_monetary"],
        color=rfm_colors, edgecolor="none", alpha=0.85
    )
    ax.set_title("Revenue by RFM Segment")
    ax.tick_params(axis="x", rotation=20, labelsize=7)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.grid(axis="y", alpha=0.3)

    save_plot(fig, "viz_01_executive_dashboard.png")


# ════════════════════════════════════════════════════════════
# PLOT 2: SALES PERFORMANCE DASHBOARD
# ════════════════════════════════════════════════════════════

def plot_sales_dashboard() -> None:
    """
    Sales performance dashboard with
    quarterly, monthly and category views.
    """
    logger.info("-- Plotting sales dashboard --")

    quarterly = get_quarterly_revenue()
    monthly   = get_monthly_revenue()
    category  = get_revenue_by_category().head(15)

    fig = plt.figure(figsize=(22, 16), facecolor=BG)
    fig.suptitle(
        "SALES PERFORMANCE DASHBOARD",
        fontsize=18, fontweight="bold", color=TEXT_C
    )

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.45, wspace=0.35
    )

    # ── Quarterly Revenue ─────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    q_colors = [
        COLORS[i % len(COLORS)]
        for i in range(len(quarterly))
    ]
    bars = ax.bar(
        quarterly["year_quarter"],
        quarterly["total_revenue"],
        color=q_colors, edgecolor="none", alpha=0.85
    )
    ax.set_ylabel("Revenue (R$)")
    ax.set_title("Quarterly Revenue")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, quarterly["total_revenue"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 5000,
            f"R${val:,.0f}",
            ha="center", fontsize=8, color=TEXT_C
        )

    # ── Monthly Orders ────────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    ax.bar(
        monthly["year_month"],
        monthly["total_orders"],
        color=PURPLE, alpha=0.7, edgecolor="none"
    )
    ax2 = ax.twinx()
    ax2.plot(
        monthly["year_month"],
        monthly["avg_order_value"],
        color=ORANGE, lw=2,
        marker="o", ms=4,
        label="Avg Order Value"
    )
    ax.set_ylabel("Total Orders", color=TEXT_C)
    ax2.set_ylabel("Avg Order Value (R$)", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax.set_title("Monthly Orders & Avg Order Value")
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax2.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=8
    )
    ax.grid(axis="y", alpha=0.3)

    # ── Unique Customers Monthly ──────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(
        monthly["year_month"],
        monthly["unique_customers"],
        color=GREEN, alpha=0.7, edgecolor="none"
    )
    ax.set_ylabel("Unique Customers")
    ax.set_title("Monthly Unique Customers")
    ax.tick_params(axis="x", rotation=45, labelsize=5)
    ax.grid(axis="y", alpha=0.3)

    # ── Category Revenue ──────────────────────────────────
    ax = fig.add_subplot(gs[2, :2])
    cat_colors = [
        COLORS[i % len(COLORS)]
        for i in range(len(category))
    ]
    bars = ax.barh(
        category["category"],
        category["total_revenue"],
        color=cat_colors, edgecolor="none", alpha=0.85
    )
    ax.set_xlabel("Revenue (R$)")
    ax.set_title("Top 15 Categories by Revenue")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.grid(axis="x", alpha=0.3)

    # ── Category Avg Price ────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    top5_cat = category.head(5)
    ax.barh(
        top5_cat["category"],
        top5_cat["avg_price"],
        color=cat_colors[:5], edgecolor="none", alpha=0.85
    )
    ax.set_xlabel("Avg Price (R$)")
    ax.set_title("Top 5 Categories Avg Price")
    ax.grid(axis="x", alpha=0.3)

    save_plot(fig, "viz_02_sales_dashboard.png")


# ════════════════════════════════════════════════════════════
# PLOT 3: CUSTOMER ANALYTICS DASHBOARD
# ════════════════════════════════════════════════════════════

def plot_customer_dashboard() -> None:
    """
    Customer analytics dashboard.
    """
    logger.info("-- Plotting customer dashboard --")

    segmentation = get_customer_segmentation()
    region       = get_revenue_by_state()
    rfm          = get_rfm_segment_summary()
    weekday      = get_orders_by_weekday()
    hour         = get_orders_by_hour()

    fig = plt.figure(figsize=(22, 16), facecolor=BG)
    fig.suptitle(
        "CUSTOMER ANALYTICS DASHBOARD",
        fontsize=18, fontweight="bold", color=TEXT_C
    )

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.45, wspace=0.35
    )

    # ── Customer Segments ─────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    seg_colors = [BLUE, GREEN, ORANGE, PURPLE]
    wedges, texts, autotexts = ax.pie(
        segmentation["customer_count"],
        labels=segmentation["customer_segment"],
        autopct="%1.1f%%",
        colors=seg_colors,
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
    ax.set_title("Customer Segments")

    # ── RFM Segments Revenue ──────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    rfm_colors = [
        COLORS[i % len(COLORS)]
        for i in range(len(rfm))
    ]
    ax.bar(
        rfm["rfm_segment"],
        rfm["total_monetary"],
        color=rfm_colors, edgecolor="none", alpha=0.85
    )
    ax.set_title("RFM Segment Revenue")
    ax.tick_params(axis="x", rotation=25, labelsize=7)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.grid(axis="y", alpha=0.3)

    # ── RFM Segment Customers ─────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(
        rfm["rfm_segment"],
        rfm["customer_count"],
        color=rfm_colors, edgecolor="none", alpha=0.85
    )
    ax.set_title("RFM Segment Customer Count")
    ax.tick_params(axis="x", rotation=25, labelsize=7)
    ax.grid(axis="y", alpha=0.3)

    # ── Revenue by State ──────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    top10_state = region.head(10)
    region_colors = {
        "Southeast"   : BLUE,
        "South"       : GREEN,
        "Northeast"   : ORANGE,
        "North"       : PURPLE,
        "Central-West": RED,
    }
    s_colors = [
        region_colors.get(r, BLUE)
        for r in top10_state["region"]
    ]
    ax.barh(
        top10_state["state"],
        top10_state["total_revenue"],
        color=s_colors, edgecolor="none", alpha=0.85
    )
    ax.set_title("Revenue by State")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.grid(axis="x", alpha=0.3)

    # ── Orders per Customer ───────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    if "orders_per_customer" in region.columns:
        ax.barh(
            region.head(10)["state"],
            region.head(10)["orders_per_customer"],
            color=s_colors, edgecolor="none", alpha=0.85
        )
        ax.set_title("Orders per Customer by State")
        ax.grid(axis="x", alpha=0.3)

    # ── Orders by Weekday ─────────────────────────────────
    ax = fig.add_subplot(gs[2, :2])
    w_colors = [
        RED if d in ["Saturday", "Sunday"] else BLUE
        for d in weekday["weekday"]
    ]
    ax.bar(
        weekday["weekday"],
        weekday["total_orders"],
        color=w_colors, edgecolor="none", alpha=0.85
    )
    ax.set_title("Orders by Day of Week")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── Orders by Hour ────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    h_colors = [
        RED    if h < 6
        else ORANGE if h < 9
        else GREEN  if h < 18
        else BLUE
        for h in hour["hour_of_day"]
    ]
    ax.bar(
        hour["hour_of_day"],
        hour["total_orders"],
        color=h_colors, edgecolor="none", alpha=0.85
    )
    ax.set_title("Orders by Hour")
    ax.set_xticks(range(0, 24, 3))
    ax.grid(axis="y", alpha=0.3)

    save_plot(fig, "viz_03_customer_dashboard.png")


# ════════════════════════════════════════════════════════════
# PLOT 4: OPERATIONS DASHBOARD
# ════════════════════════════════════════════════════════════

def plot_operations_dashboard() -> None:
    """
    Operations dashboard — delivery, payments,
    reviews and sellers.
    """
    logger.info("-- Plotting operations dashboard --")

    delivery = get_delivery_analysis().head(15)
    payment  = get_payment_analysis()
    review   = get_review_analysis()
    seller   = get_seller_performance().head(20)
    status   = get_order_status_summary()

    fig = plt.figure(figsize=(22, 16), facecolor=BG)
    fig.suptitle(
        "OPERATIONS ANALYTICS DASHBOARD",
        fontsize=18, fontweight="bold", color=TEXT_C
    )

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.45, wspace=0.35
    )

    # ── Order Status ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    st_colors = [
        GREEN, RED, ORANGE, BLUE, PURPLE,
        "#1ABC9C", "#E67E22"
    ]
    ax.bar(
        status["order_status"],
        status["total_orders"],
        color=st_colors[:len(status)],
        edgecolor="none", alpha=0.85
    )
    ax.set_title("Order Status")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.grid(axis="y", alpha=0.3)

    # ── Delivery Days by State ────────────────────────────
    ax = fig.add_subplot(gs[0, 1:])
    del_colors = [
        RED    if v > 20
        else ORANGE if v > 15
        else GREEN
        for v in delivery["avg_delivery_days"]
    ]
    ax.barh(
        delivery["state"],
        delivery["avg_delivery_days"],
        color=del_colors, edgecolor="none", alpha=0.85
    )
    ax.set_title("Avg Delivery Days by State")
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(
        delivery["avg_delivery_days"].mean(),
        color="white", lw=1.5, ls="--", alpha=0.7
    )

    # ── Payment Revenue ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    pay_colors = [BLUE, GREEN, ORANGE, PURPLE]
    ax.bar(
        payment["payment_type"],
        payment["total_revenue"],
        color=pay_colors[:len(payment)],
        edgecolor="none", alpha=0.85
    )
    ax.set_title("Revenue by Payment Type")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.grid(axis="y", alpha=0.3)

    # ── Review Distribution ───────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    score_colors = {
        1: RED, 2: ORANGE,
        3: "#F39C12", 4: "#1ABC9C", 5: GREEN
    }
    r_colors = [
        score_colors.get(s, BLUE)
        for s in review["review_score"]
    ]
    ax.bar(
        review["review_score"].astype(str),
        review["review_pct"],
        color=r_colors, edgecolor="none", alpha=0.85
    )
    ax.set_title("Review Score %")
    ax.set_xlabel("Score")
    ax.set_ylabel("%")
    ax.grid(axis="y", alpha=0.3)

    # ── Late Delivery by State ────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    late_colors = [
        RED    if v > 15
        else ORANGE if v > 10
        else GREEN
        for v in delivery["late_pct"]
    ]
    ax.barh(
        delivery["state"],
        delivery["late_pct"],
        color=late_colors, edgecolor="none", alpha=0.85
    )
    ax.set_title("Late Delivery % by State")
    ax.grid(axis="x", alpha=0.3)

    # ── Seller Revenue vs Review scatter ──────────────────
    ax = fig.add_subplot(gs[2, :2])
    scatter = ax.scatter(
        seller["total_revenue"],
        seller["avg_review_score"],
        c=seller["late_delivery_pct"].fillna(0),
        cmap="RdYlGn_r",
        s=seller["total_orders"] * 2,
        alpha=0.7, edgecolors="none"
    )
    plt.colorbar(
        scatter, ax=ax,
        label="Late Delivery %"
    )
    ax.set_xlabel("Revenue (R$)")
    ax.set_ylabel("Avg Review Score")
    ax.set_title("Seller Revenue vs Review Score")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.grid(alpha=0.3)

    # ── Avg Installments by Payment ───────────────────────
    ax = fig.add_subplot(gs[2, 2])
    ax.bar(
        payment["payment_type"],
        payment["avg_installments"],
        color=pay_colors[:len(payment)],
        edgecolor="none", alpha=0.85
    )
    ax.set_title("Avg Installments by Payment")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    save_plot(fig, "viz_04_operations_dashboard.png")


# ════════════════════════════════════════════════════════════
# RUN ALL VISUALIZATIONS
# ════════════════════════════════════════════════════════════

def run_all_visualizations() -> None:
    """
    Run all visualization functions and
    save all plots.
    """
    logger.info("=" * 55)
    logger.info("VISUALIZATION STARTED")
    logger.info("=" * 55)

    create_directories()
    total_start = time.time()

    plots = {
        "Executive Dashboard"   : plot_executive_dashboard,
        "Sales Dashboard"       : plot_sales_dashboard,
        "Customer Dashboard"    : plot_customer_dashboard,
        "Operations Dashboard"  : plot_operations_dashboard,
    }

    passed = 0
    failed = 0

    for name, func in plots.items():
        try:
            start = time.time()
            func()
            elapsed = time.time() - start
            logger.info(
                f"[OK] {name} ({elapsed:.2f}s)"
            )
            passed += 1
        except Exception as e:
            logger.error(f"[ERROR] {name}: {e}")
            failed += 1

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 55)
    print("   VISUALIZATION SUMMARY")
    print("=" * 55)
    print(f"  Total Plots : {len(plots)}")
    print(f"  Passed      : {passed}")
    print(f"  Failed      : {failed}")
    print(f"  Time        : {total_elapsed:.2f}s")
    print(f"  Saved to    : {PLOTS_DIR}")
    print("=" * 55 + "\n")


# ── Run directly ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_all_visualizations()