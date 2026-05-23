# analytics/forecasting.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import time
from loguru import logger

from src.config import (
    PROCESSED_FILES,
    PLOTS_DIR,
    VIZ_SETTINGS,
    FORECAST_SETTINGS,
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
# STEP 1: LOAD & PREPARE DATA
# ════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """
    Load and prepare time series data
    for forecasting.
    """
    logger.info("-- Loading data for forecasting --")

    df = pd.read_csv(PROCESSED_FILES["master"])

    # Convert date
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    )

    # Keep delivered orders only
    df = df[df["order_status"] == "delivered"].copy()

    logger.info(f"[OK] Loaded {len(df):,} delivered orders")
    return df


def prepare_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate orders into daily revenue time series.
    """
    logger.info("-- Preparing daily time series --")

    daily = df.groupby(
        df["order_purchase_timestamp"].dt.date
    ).agg(
        revenue = ("total_payment_value", "sum"),
        orders  = ("order_id",            "nunique"),
    ).reset_index()

    daily.columns   = ["date", "revenue", "orders"]
    daily["date"]   = pd.to_datetime(daily["date"])
    daily           = daily.sort_values("date")

    # Fill missing dates with 0
    date_range = pd.date_range(
        daily["date"].min(),
        daily["date"].max(),
        freq="D"
    )
    daily = daily.set_index("date").reindex(
        date_range, fill_value=0
    ).reset_index()
    daily.columns = ["date", "revenue", "orders"]

    logger.info(
        f"[OK] Daily series: {len(daily):,} days "
        f"({daily['date'].min().date()} to "
        f"{daily['date'].max().date()})"
    )
    return daily


def prepare_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate orders into monthly revenue time series.
    """
    logger.info("-- Preparing monthly time series --")

    df["year_month"] = df[
        "order_purchase_timestamp"
    ].dt.to_period("M")

    monthly = df.groupby("year_month").agg(
        revenue  = ("total_payment_value", "sum"),
        orders   = ("order_id",            "nunique"),
        customers= ("customer_unique_id",   "nunique"),
    ).reset_index()

    monthly["year_month"] = monthly[
        "year_month"
    ].dt.to_timestamp()
    monthly = monthly.sort_values("year_month")

    logger.info(
        f"[OK] Monthly series: {len(monthly):,} months"
    )
    return monthly


# ════════════════════════════════════════════════════════════
# STEP 2: MOVING AVERAGE FORECAST
# ════════════════════════════════════════════════════════════

def moving_average_forecast(
    daily: pd.DataFrame,
    periods: int = 30,
    window: int  = 7
) -> pd.DataFrame:
    """
    Simple moving average forecast.
    Uses last N days average to predict next N days.
    """
    logger.info("-- Running moving average forecast --")

    # Calculate rolling averages
    daily["ma_7"]  = daily["revenue"].rolling(7).mean()
    daily["ma_14"] = daily["revenue"].rolling(14).mean()
    daily["ma_30"] = daily["revenue"].rolling(30).mean()

    # Forecast using last window average
    last_avg = daily["revenue"].tail(window).mean()

    # Create forecast dataframe
    last_date     = daily["date"].max()
    forecast_dates= pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=periods,
        freq="D"
    )

    forecast = pd.DataFrame({
        "date"    : forecast_dates,
        "forecast": last_avg,
        "upper"   : last_avg * 1.2,
        "lower"   : last_avg * 0.8,
    })

    logger.info(
        f"[OK] MA forecast: {periods} days, "
        f"avg R${last_avg:,.2f}/day"
    )
    return daily, forecast


# ════════════════════════════════════════════════════════════
# STEP 3: LINEAR TREND FORECAST
# ════════════════════════════════════════════════════════════

def linear_trend_forecast(
    monthly: pd.DataFrame,
    periods: int = 3
) -> pd.DataFrame:
    """
    Simple linear regression forecast on monthly data.
    """
    logger.info("-- Running linear trend forecast --")

    # Create numeric time index
    monthly["time_idx"] = range(len(monthly))

    # Fit linear regression
    x = monthly["time_idx"].values
    y = monthly["revenue"].values

    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs

    # Generate future predictions
    future_idx  = range(
        len(monthly),
        len(monthly) + periods
    )
    last_date   = monthly["year_month"].max()
    future_dates= pd.date_range(
        last_date + pd.DateOffset(months=1),
        periods=periods,
        freq="MS"
    )

    predictions = [
        slope * i + intercept for i in future_idx
    ]
    predictions = [max(0, p) for p in predictions]

    # Calculate residuals for confidence interval
    fitted    = [slope * i + intercept for i in x]
    residuals = y - np.array(fitted)
    std_err   = np.std(residuals)

    forecast = pd.DataFrame({
        "date"    : future_dates,
        "forecast": predictions,
        "upper"   : [p + 1.96 * std_err for p in predictions],
        "lower"   : [max(0, p - 1.96 * std_err) for p in predictions],
    })

    # Add trend line to monthly
    monthly["trend"] = [
        slope * i + intercept for i in x
    ]

    logger.info(
        f"[OK] Linear forecast: {periods} months ahead"
    )
    logger.info(
        f"  Slope     : R${slope:,.2f}/month"
    )
    logger.info(
        f"  Next month: R${predictions[0]:,.2f}"
    )

    return monthly, forecast, slope, intercept


# ════════════════════════════════════════════════════════════
# STEP 4: SEASONAL DECOMPOSITION
# ════════════════════════════════════════════════════════════

def seasonal_analysis(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze seasonal patterns in the data.
    Day of week, month, and holiday effects.
    """
    logger.info("-- Running seasonal analysis --")

    df = daily.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_name"]    = df["date"].dt.day_name()
    df["month"]       = df["date"].dt.month
    df["month_name"]  = df["date"].dt.month_name()
    df["week"]        = df["date"].dt.isocalendar().week
    df["quarter"]     = df["date"].dt.quarter

    # Day of week seasonality
    dow_pattern = df.groupby("day_name").agg(
        avg_revenue = ("revenue", "mean"),
        avg_orders  = ("orders",  "mean"),
    ).reset_index()

    # Month seasonality
    month_pattern = df.groupby("month_name").agg(
        avg_revenue = ("revenue", "mean"),
        avg_orders  = ("orders",  "mean"),
    ).reset_index()

    logger.info("[OK] Seasonal analysis complete")
    return df, dow_pattern, month_pattern


# ════════════════════════════════════════════════════════════
# STEP 5: PROPHET FORECAST (if available)
# ════════════════════════════════════════════════════════════

def prophet_forecast(
    daily: pd.DataFrame,
    periods: int = 90
) -> tuple:
    """
    Facebook Prophet forecast.
    Falls back to moving average if Prophet not installed.
    """
    try:
        from prophet import Prophet

        logger.info("-- Running Prophet forecast --")

        # Prepare data for Prophet
        prophet_df = daily[["date", "revenue"]].copy()
        prophet_df.columns = ["ds", "y"]

        # Remove outliers
        q99 = prophet_df["y"].quantile(0.99)
        prophet_df["y"] = prophet_df["y"].clip(0, q99)

        # Fit model
        model = Prophet(
            yearly_seasonality  = True,
            weekly_seasonality  = True,
            daily_seasonality   = False,
            changepoint_prior_scale = 0.05,
        )
        model.fit(prophet_df)

        # Make forecast
        future   = model.make_future_dataframe(
            periods=periods
        )
        forecast = model.predict(future)

        logger.info(
            f"[OK] Prophet forecast: {periods} days ahead"
        )
        return model, forecast, "prophet"

    except ImportError:
        logger.warning(
            "[WARN] Prophet not installed. "
            "Using moving average instead."
        )
        daily_with_ma, ma_forecast = moving_average_forecast(
            daily, periods=periods
        )
        return None, ma_forecast, "moving_average"


# ════════════════════════════════════════════════════════════
# STEP 6: VISUALIZE FORECASTS
# ════════════════════════════════════════════════════════════

def plot_time_series(daily: pd.DataFrame) -> None:
    """
    Plot the raw time series data.
    """
    logger.info("-- Plotting time series --")

    fig, axes = plt.subplots(
        2, 1, figsize=(20, 12), facecolor=BG
    )
    fig.suptitle(
        "REVENUE TIME SERIES ANALYSIS",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    # Plot 1: Daily revenue
    ax = axes[0]
    ax.plot(
        daily["date"], daily["revenue"],
        color=BLUE, lw=0.8, alpha=0.7,
        label="Daily Revenue"
    )
    if "ma_7" in daily.columns:
        ax.plot(
            daily["date"], daily["ma_7"],
            color=GREEN, lw=2,
            label="7-Day MA"
        )
    if "ma_30" in daily.columns:
        ax.plot(
            daily["date"], daily["ma_30"],
            color=ORANGE, lw=2,
            label="30-Day MA"
        )
    ax.set_ylabel("Revenue (R$)")
    ax.set_title("Daily Revenue with Moving Averages")
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=9
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.grid(alpha=0.3)

    # Plot 2: Daily orders
    ax = axes[1]
    ax.bar(
        daily["date"], daily["orders"],
        color=PURPLE, alpha=0.6,
        edgecolor="none", width=1,
        label="Daily Orders"
    )
    ax.set_ylabel("Number of Orders")
    ax.set_title("Daily Order Volume")
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=9
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_plot(fig, "forecast_01_time_series.png")


def plot_moving_average_forecast(
    daily: pd.DataFrame,
    forecast: pd.DataFrame
) -> None:
    """
    Plot moving average forecast.
    """
    logger.info("-- Plotting MA forecast --")

    fig, ax = plt.subplots(figsize=(20, 8), facecolor=BG)
    fig.suptitle(
        "REVENUE FORECAST - Moving Average",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    # Plot historical (last 90 days)
    recent = daily.tail(90)
    ax.plot(
        recent["date"], recent["revenue"],
        color=BLUE, lw=1, alpha=0.7,
        label="Historical Revenue"
    )
    if "ma_7" in recent.columns:
        ax.plot(
            recent["date"], recent["ma_7"],
            color=GREEN, lw=2,
            label="7-Day MA"
        )

    # Plot forecast
    ax.plot(
        forecast["date"], forecast["forecast"],
        color=ORANGE, lw=2.5,
        ls="--", label="Forecast"
    )
    ax.fill_between(
        forecast["date"],
        forecast["lower"],
        forecast["upper"],
        alpha=0.2, color=ORANGE,
        label="Confidence Interval"
    )

    # Add vertical line at forecast start
    ax.axvline(
        daily["date"].max(),
        color="white", lw=1.5,
        ls="--", alpha=0.7,
        label="Forecast Start"
    )

    ax.set_ylabel("Revenue (R$)")
    ax.set_title("30-Day Revenue Forecast")
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=9
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_plot(fig, "forecast_02_ma_forecast.png")


def plot_linear_forecast(
    monthly: pd.DataFrame,
    forecast: pd.DataFrame,
    slope: float,
    intercept: float
) -> None:
    """
    Plot linear trend forecast on monthly data.
    """
    logger.info("-- Plotting linear forecast --")

    fig, ax = plt.subplots(figsize=(18, 8), facecolor=BG)
    fig.suptitle(
        "MONTHLY REVENUE FORECAST - Linear Trend",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    # Historical bars
    ax.bar(
        monthly["year_month"],
        monthly["revenue"],
        color=BLUE, alpha=0.6,
        edgecolor="none", width=20,
        label="Monthly Revenue"
    )

    # Trend line
    ax.plot(
        monthly["year_month"],
        monthly["trend"],
        color=GREEN, lw=2.5,
        label=f"Trend (R${slope:,.0f}/month)"
    )

    # Forecast
    ax.bar(
        forecast["date"],
        forecast["forecast"],
        color=ORANGE, alpha=0.8,
        edgecolor="none", width=20,
        label="Forecast"
    )
    ax.fill_between(
        forecast["date"],
        forecast["lower"],
        forecast["upper"],
        alpha=0.2, color=ORANGE,
        label="Confidence Interval"
    )

    # Forecast start line
    ax.axvline(
        monthly["year_month"].max(),
        color="white", lw=1.5,
        ls="--", alpha=0.7,
        label="Forecast Start"
    )

    ax.set_ylabel("Revenue (R$)")
    ax.set_title("Monthly Revenue with Linear Trend & Forecast")
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=9
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_plot(fig, "forecast_03_linear_forecast.png")


def plot_seasonal_patterns(
    dow_pattern: pd.DataFrame,
    month_pattern: pd.DataFrame
) -> None:
    """
    Plot seasonal patterns by day of week and month.
    """
    logger.info("-- Plotting seasonal patterns --")

    fig, axes = plt.subplots(
        1, 2, figsize=(20, 8), facecolor=BG
    )
    fig.suptitle(
        "SEASONAL PATTERNS",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    # Day order
    day_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]
    month_order = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    # Filter to available days/months
    dow_pattern = dow_pattern[
        dow_pattern["day_name"].isin(day_order)
    ].copy()
    dow_pattern["day_order"] = dow_pattern[
        "day_name"
    ].map({d: i for i, d in enumerate(day_order)})
    dow_pattern = dow_pattern.sort_values("day_order")

    month_pattern = month_pattern[
        month_pattern["month_name"].isin(month_order)
    ].copy()
    month_pattern["month_order"] = month_pattern[
        "month_name"
    ].map({m: i for i, m in enumerate(month_order)})
    month_pattern = month_pattern.sort_values("month_order")

    # Plot 1: Day of week pattern
    ax = axes[0]
    colors_dow = [
        RED if d in ["Saturday", "Sunday"] else BLUE
        for d in dow_pattern["day_name"]
    ]
    bars = ax.bar(
        dow_pattern["day_name"],
        dow_pattern["avg_revenue"],
        color=colors_dow,
        edgecolor="none", alpha=0.85
    )
    ax.set_ylabel("Avg Daily Revenue (R$)")
    ax.set_title("Avg Revenue by Day of Week")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, dow_pattern["avg_revenue"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 100,
            f"R${val:,.0f}",
            ha="center", fontsize=8, color=TEXT_C
        )

    # Plot 2: Month pattern
    ax = axes[1]
    colors_month = [
        GREEN if v == month_pattern["avg_revenue"].max()
        else ORANGE if v >= month_pattern["avg_revenue"].mean()
        else BLUE
        for v in month_pattern["avg_revenue"]
    ]
    bars = ax.bar(
        month_pattern["month_name"],
        month_pattern["avg_revenue"],
        color=colors_month,
        edgecolor="none", alpha=0.85
    )
    ax.set_ylabel("Avg Daily Revenue (R$)")
    ax.set_title("Avg Revenue by Month")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_plot(fig, "forecast_04_seasonal_patterns.png")


def plot_prophet_forecast(
    daily: pd.DataFrame,
    model,
    forecast: pd.DataFrame,
    method: str
) -> None:
    """
    Plot Prophet or fallback forecast.
    """
    logger.info(f"-- Plotting {method} forecast --")

    fig, axes = plt.subplots(
        2, 1, figsize=(20, 14), facecolor=BG
    )
    fig.suptitle(
        f"REVENUE FORECAST - {method.upper()}",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    if method == "prophet" and model is not None:
        # Plot 1: Full forecast
        ax = axes[0]
        ax.plot(
            daily["date"], daily["revenue"],
            color=BLUE, lw=0.8, alpha=0.6,
            label="Actual Revenue"
        )
        ax.plot(
            pd.to_datetime(forecast["ds"]),
            forecast["yhat"],
            color=GREEN, lw=2,
            label="Forecast"
        )
        ax.fill_between(
            pd.to_datetime(forecast["ds"]),
            forecast["yhat_lower"],
            forecast["yhat_upper"],
            alpha=0.2, color=GREEN,
            label="Confidence Interval"
        )
        ax.axvline(
            daily["date"].max(),
            color="white", lw=1.5,
            ls="--", alpha=0.7,
            label="Forecast Start"
        )
        ax.set_ylabel("Revenue (R$)")
        ax.set_title("Prophet Revenue Forecast")
        ax.legend(
            facecolor=CARD_BG, labelcolor=TEXT_C,
            edgecolor=GRID_C, fontsize=9
        )
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
        )
        ax.grid(alpha=0.3)

        # Plot 2: Components
        ax = axes[1]
        future_forecast = forecast[
            forecast["ds"] > daily["date"].max()
        ]
        ax.plot(
            pd.to_datetime(future_forecast["ds"]),
            future_forecast["yhat"],
            color=ORANGE, lw=2.5,
            marker="o", ms=4
        )
        ax.fill_between(
            pd.to_datetime(future_forecast["ds"]),
            future_forecast["yhat_lower"],
            future_forecast["yhat_upper"],
            alpha=0.3, color=ORANGE
        )
        ax.set_ylabel("Forecasted Revenue (R$)")
        ax.set_title("90-Day Revenue Forecast")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
        )
        ax.grid(alpha=0.3)

    else:
        # Fallback: Moving average forecast
        ax = axes[0]
        ax.plot(
            daily["date"], daily["revenue"],
            color=BLUE, lw=0.8, alpha=0.6,
            label="Actual Revenue"
        )
        ax.plot(
            forecast["date"], forecast["forecast"],
            color=ORANGE, lw=2.5,
            ls="--", label="MA Forecast"
        )
        ax.fill_between(
            forecast["date"],
            forecast["lower"],
            forecast["upper"],
            alpha=0.2, color=ORANGE,
            label="Confidence Interval"
        )
        ax.axvline(
            daily["date"].max(),
            color="white", lw=1.5,
            ls="--", alpha=0.7
        )
        ax.set_ylabel("Revenue (R$)")
        ax.set_title("Moving Average Forecast")
        ax.legend(
            facecolor=CARD_BG, labelcolor=TEXT_C,
            edgecolor=GRID_C, fontsize=9
        )
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
        )
        ax.grid(alpha=0.3)

        axes[1].axis("off")
        axes[1].text(
            0.5, 0.5,
            "Install Prophet for advanced forecasting:\npip install prophet",
            ha="center", va="center",
            fontsize=14, color=TEXT_C,
            transform=axes[1].transAxes
        )

    plt.tight_layout()
    save_plot(fig, "forecast_05_prophet.png")


# ════════════════════════════════════════════════════════════
# MAIN FORECASTING FUNCTION
# ════════════════════════════════════════════════════════════

def run_forecasting() -> dict:
    """
    Run complete forecasting pipeline.

    Returns:
        dict with all forecast results
    """
    logger.info("=" * 55)
    logger.info("FORECASTING STARTED")
    logger.info("=" * 55)

    total_start = time.time()
    results     = {}

    # Step 1: Load data
    df = load_data()

    # Step 2: Prepare series
    daily   = prepare_daily_series(df)
    monthly = prepare_monthly_series(df)

    # Step 3: Moving average forecast
    daily, ma_forecast = moving_average_forecast(
        daily, periods=30
    )
    results["ma_forecast"] = ma_forecast

    # Step 4: Linear trend forecast
    monthly, linear_forecast, slope, intercept = (
        linear_trend_forecast(monthly, periods=3)
    )
    results["linear_forecast"] = linear_forecast

    # Step 5: Seasonal analysis
    daily_seasonal, dow_pattern, month_pattern = (
        seasonal_analysis(daily)
    )
    results["dow_pattern"]   = dow_pattern
    results["month_pattern"] = month_pattern

    # Step 6: Prophet forecast
    model, prophet_forecast_df, method = prophet_forecast(
        daily, periods=FORECAST_SETTINGS["periods"]
    )
    results["prophet_forecast"] = prophet_forecast_df
    results["method"]           = method

    # Step 7: Visualize
    plot_time_series(daily)
    plot_moving_average_forecast(daily, ma_forecast)
    plot_linear_forecast(
        monthly, linear_forecast, slope, intercept
    )
    plot_seasonal_patterns(dow_pattern, month_pattern)
    plot_prophet_forecast(
        daily, model, prophet_forecast_df, method
    )

    # Save forecasts
    reports_dir = PLOTS_DIR.parent / "reports"
    ma_forecast.to_csv(
        reports_dir / "forecast_ma.csv", index=False
    )
    linear_forecast.to_csv(
        reports_dir / "forecast_linear.csv", index=False
    )

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 55)
    print("   FORECASTING SUMMARY")
    print("=" * 55)
    print(f"  Daily Data Points : {len(daily):,}")
    print(f"  Monthly Points    : {len(monthly):,}")
    print(f"  MA Forecast Days  : 30")
    print(f"  Linear Forecast   : 3 months")
    print(f"  Prophet Method    : {method}")
    print(f"  Plots Generated   : 5")
    print(f"  Time              : {total_elapsed:.2f}s")
    print("-" * 55)
    print("  Next 30 Day MA Forecast:")
    print(
        f"  Avg Daily Revenue : "
        f"R${ma_forecast['forecast'].mean():,.2f}"
    )
    print(
        f"  Total 30 Day Rev  : "
        f"R${ma_forecast['forecast'].sum():,.2f}"
    )
    print("-" * 55)
    print("  Next 3 Month Linear Forecast:")
    for _, row in linear_forecast.iterrows():
        print(
            f"  {row['date'].strftime('%Y-%m')} : "
            f"R${row['forecast']:,.2f}"
        )
    print("=" * 55 + "\n")

    return results


# ── Run directly ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_forecasting()