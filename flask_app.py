# flask_app.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from flask import Flask, render_template, jsonify
from pathlib import Path

from src.config import PLOTS_DIR, REPORTS_DIR, DATABASE_PATH
from src.sql_queries import (
    get_business_overview,
    get_monthly_revenue,
    get_revenue_by_category,
    get_revenue_by_state,
    get_customer_segmentation,
    get_payment_analysis,
    get_review_analysis,
    get_rfm_segment_summary,
    get_seller_performance,
    get_delivery_analysis,
    get_top_customers,
    get_top_products,
)

app = Flask(__name__)

BASE_DIR   = Path(__file__).parent
PLOTS_PATH = BASE_DIR / "outputs" / "plots"


# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def get_plot_list() -> list:
    """
    Get all plot filenames from outputs/plots folder.
    """
    plots = []
    if PLOTS_PATH.exists():
        plots = [
            f.name for f in PLOTS_PATH.iterdir()
            if f.suffix == ".png"
        ]
    return sorted(plots)


def df_to_html(df: pd.DataFrame, max_rows: int = 20) -> str:
    """
    Convert DataFrame to HTML table with Bootstrap styling.
    """
    return df.head(max_rows).to_html(
        classes="table table-dark table-striped table-hover table-sm",
        index=False,
        border=0,
        justify="center",
    )


# ════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """
    Home page with KPI cards and overview.
    """
    overview     = get_business_overview().iloc[0]
    monthly      = get_monthly_revenue()
    category     = get_revenue_by_category().head(10)
    plots        = get_plot_list()

    kpis = {
        "total_orders"     : f"{int(overview['total_orders']):,}",
        "total_revenue"    : f"R${overview['total_revenue']:,.0f}",
        "avg_order_value"  : f"R${overview['avg_order_value']:,.2f}",
        "total_customers"  : f"{int(overview['total_customers']):,}",
        "total_sellers"    : f"{int(overview['total_sellers']):,}",
        "avg_review_score" : f"{overview['avg_review_score']:.2f} / 5",
        "avg_delivery_days": f"{overview['avg_delivery_days']:.1f} days",
        "total_products"   : f"{int(overview['total_products']):,}",
    }

    return render_template(
        "index.html",
        kpis         = kpis,
        monthly_table= df_to_html(monthly),
        category_table=df_to_html(category),
        plots        = plots,
    )


@app.route("/sql")
def sql():
    """
    SQL Analysis page with all query results as tables.
    """
    tables = {
        "Business Overview"      : df_to_html(get_business_overview()),
        "Monthly Revenue"        : df_to_html(get_monthly_revenue()),
        "Revenue by Category"    : df_to_html(get_revenue_by_category()),
        "Revenue by State"       : df_to_html(get_revenue_by_state()),
        "Customer Segmentation"  : df_to_html(get_customer_segmentation()),
        "Payment Analysis"       : df_to_html(get_payment_analysis()),
        "Review Analysis"        : df_to_html(get_review_analysis()),
        "RFM Segment Summary"    : df_to_html(get_rfm_segment_summary()),
        "Top Customers"          : df_to_html(get_top_customers()),
        "Top Products"           : df_to_html(get_top_products()),
        "Delivery Analysis"      : df_to_html(get_delivery_analysis()),
        "Seller Performance"     : df_to_html(get_seller_performance().head(20)),
    }
    return render_template("index.html", tables=tables, page="sql")


@app.route("/plots")
def plots():
    """
    EDA plots page showing all generated charts.
    """
    all_plots = get_plot_list()
    return render_template("index.html", plots=all_plots, page="plots")


@app.route("/reports")
def reports():
    """
    Reports download page.
    """
    report_files = []
    if REPORTS_DIR.exists():
        report_files = [
            f.name for f in REPORTS_DIR.iterdir()
            if f.suffix in [".pdf", ".xlsx", ".csv"]
        ]
    return render_template(
        "index.html",
        report_files=sorted(report_files),
        page="reports"
    )


@app.route("/download/<filename>")
def download(filename):
    """
    Download a report file.
    """
    from flask import send_from_directory
    return send_from_directory(
        str(REPORTS_DIR),
        filename,
        as_attachment=True
    )


@app.route("/plot/<filename>")
def serve_plot(filename):
    """
    Serve a plot image.
    """
    from flask import send_from_directory
    return send_from_directory(
        str(PLOTS_PATH),
        filename
    )


# ── Run app ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)