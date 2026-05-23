# reports/pdf_report.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import time
from loguru import logger

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import (
    HexColor, white, black
)
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image, PageBreak,
    HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics import renderPDF

from src.config import (
    PLOTS_DIR,
    REPORTS_DIR,
    create_directories,
)
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

warnings.filterwarnings("ignore")

# ── Configure Logger ──────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
    level="INFO",
    colorize=False,
)

# ── Color Palette ─────────────────────────────────────────────────────────
C_DARK      = HexColor("#0F1117")
C_CARD      = HexColor("#1A1D27")
C_BLUE      = HexColor("#3498DB")
C_GREEN     = HexColor("#2ECC71")
C_RED       = HexColor("#E74C3C")
C_ORANGE    = HexColor("#F39C12")
C_PURPLE    = HexColor("#9B59B6")
C_TEXT      = HexColor("#E8EAF0")
C_GRID      = HexColor("#2C2F3F")
C_HEADER    = HexColor("#2C3E50")
C_WHITE     = white
C_BLACK     = black
C_LIGHT     = HexColor("#F5F6FA")
C_GREY      = HexColor("#BDC3C7")


# ════════════════════════════════════════════════════════════
# STYLES
# ════════════════════════════════════════════════════════════

def get_styles():
    """
    Get all custom paragraph styles for the report.
    """
    styles = getSampleStyleSheet()

    # Title style
    styles.add(ParagraphStyle(
        name      = "ReportTitle",
        fontSize  = 28,
        fontName  = "Helvetica-Bold",
        textColor = C_BLUE,
        alignment = TA_CENTER,
        spaceAfter= 10,
    ))

    # Subtitle style
    styles.add(ParagraphStyle(
        name      = "ReportSubtitle",
        fontSize  = 14,
        fontName  = "Helvetica",
        textColor = C_GREY,
        alignment = TA_CENTER,
        spaceAfter= 5,
    ))

    # Section header style
    styles.add(ParagraphStyle(
        name      = "SectionHeader",
        fontSize  = 16,
        fontName  = "Helvetica-Bold",
        textColor = C_BLUE,
        spaceBefore= 20,
        spaceAfter = 10,
        borderPad  = 5,
    ))

    # Sub section style
    styles.add(ParagraphStyle(
        name      = "SubSection",
        fontSize  = 12,
        fontName  = "Helvetica-Bold",
        textColor = C_HEADER,
        spaceBefore= 10,
        spaceAfter = 5,
    ))

    # Body text style
    styles.add(ParagraphStyle(
        name      = "ReportBody",
        fontSize  = 10,
        fontName  = "Helvetica",
        textColor = C_BLACK,
        spaceAfter= 5,
        leading   = 14,
    ))

    # Insight style
    styles.add(ParagraphStyle(
        name       = "Insight",
        fontSize   = 10,
        fontName   = "Helvetica",
        textColor  = C_HEADER,
        spaceAfter = 4,
        leftIndent = 15,
        leading    = 14,
    ))

    # KPI value style
    styles.add(ParagraphStyle(
        name      = "KPIValue",
        fontSize  = 20,
        fontName  = "Helvetica-Bold",
        textColor = C_BLUE,
        alignment = TA_CENTER,
        spaceAfter= 2,
    ))

    # KPI label style
    styles.add(ParagraphStyle(
        name      = "KPILabel",
        fontSize  = 9,
        fontName  = "Helvetica",
        textColor = C_GREY,
        alignment = TA_CENTER,
    ))

    # Footer style
    styles.add(ParagraphStyle(
        name      = "Footer",
        fontSize  = 8,
        fontName  = "Helvetica",
        textColor = C_GREY,
        alignment = TA_CENTER,
    ))

    return styles


# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def make_table(
    data: list,
    col_widths: list = None,
    header_color: object = None,
    alternate_rows: bool = True,
) -> Table:
    """
    Create a styled table for the report.
    """
    if header_color is None:
        header_color = C_HEADER

    table = Table(data, colWidths=col_widths)

    style = TableStyle([
        # Header
        ("BACKGROUND",  (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR",   (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 9),
        ("ALIGN",       (0, 0), (-1, 0), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",(0, 0), (-1, -1), 8),

        # Body
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), 8),
        ("TEXTCOLOR",   (0, 1), (-1, -1), C_BLACK),

        # Grid
        ("GRID",        (0, 0), (-1, -1), 0.5, C_GREY),
        ("LINEBELOW",   (0, 0), (-1, 0), 1.5, C_BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [C_WHITE, C_LIGHT] if alternate_rows else [C_WHITE]),
    ])

    table.setStyle(style)
    return table


def make_kpi_table(kpis: list) -> Table:
    """
    Create a KPI summary table.
    kpis: list of (label, value, color_hex)
    """
    styles = get_styles()

    # Build table data
    labels = [Paragraph(k[0], styles["KPILabel"]) for k in kpis]
    values = [
        Paragraph(
            k[1],
            ParagraphStyle(
                name=f"KPI_{i}",
                fontSize=16,
                fontName="Helvetica-Bold",
                textColor=HexColor(k[2]),
                alignment=TA_CENTER,
            )
        )
        for i, k in enumerate(kpis)
    ]

    data = [values, labels]

    table = Table(
        data,
        colWidths=[A4[0] / len(kpis) - 20] * len(kpis)
    )
    table.setStyle(TableStyle([
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ("GRID",         (0, 0), (-1, -1), 0.5, C_GREY),
        ("BACKGROUND",   (0, 0), (-1, -1), C_LIGHT),
        ("LINEBELOW",    (0, 0), (-1, 0), 1, C_BLUE),
    ]))
    return table


def add_section_header(
    elements: list,
    title: str,
    styles: dict
) -> None:
    """
    Add a section header with a divider line.
    """
    elements.append(Spacer(1, 10))
    elements.append(
        HRFlowable(
            width="100%",
            thickness=2,
            color=C_BLUE,
            spaceAfter=5
        )
    )
    elements.append(
        Paragraph(title, styles["SectionHeader"])
    )


def add_plot(
    elements: list,
    filename: str,
    width: float = 7 * inch,
    height: float = 4 * inch,
    caption: str = ""
) -> None:
    """
    Add a plot image to the report.
    """
    styles = get_styles()
    plot_path = PLOTS_DIR / filename

    if plot_path.exists():
        img = Image(str(plot_path), width=width, height=height)
        elements.append(img)
        if caption:
            elements.append(
                Paragraph(
                    f"<i>{caption}</i>",
                    styles["Footer"]
                )
            )
        elements.append(Spacer(1, 10))
    else:
        elements.append(
            Paragraph(
                f"[Plot not found: {filename}]",
                styles["ReportBody"]
            )
        )


# ════════════════════════════════════════════════════════════
# REPORT SECTIONS
# ════════════════════════════════════════════════════════════

def build_cover_page(
    elements: list,
    styles: dict
) -> None:
    """
    Build the cover page of the report.
    """
    elements.append(Spacer(1, 2 * inch))

    # Title
    elements.append(Paragraph(
        "RETAIL ANALYTICS REPORT",
        styles["ReportTitle"]
    ))
    elements.append(Spacer(1, 0.2 * inch))

    # Subtitle
    elements.append(Paragraph(
        "Brazilian E-Commerce (Olist) — Business Intelligence Analysis",
        styles["ReportSubtitle"]
    ))
    elements.append(Spacer(1, 0.2 * inch))

    # Date
    elements.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%B %d, %Y')}",
        styles["ReportSubtitle"]
    ))
    elements.append(Spacer(1, 1 * inch))

    # Divider
    elements.append(HRFlowable(
        width="100%", thickness=2,
        color=C_BLUE, spaceAfter=20
    ))

    # Executive Summary
    elements.append(Paragraph(
        "Executive Summary",
        styles["SectionHeader"]
    ))
    elements.append(Paragraph(
        """This report provides a comprehensive analysis of the Brazilian
        E-Commerce platform Olist, covering sales performance, customer
        behavior, product analytics, seller performance, and delivery
        operations. The analysis is based on 99,441 orders from
        September 2016 to August 2018, involving 96,478 unique customers,
        3,095 sellers, and 32,951 products across 73 categories.""",
        styles["ReportBody"]
    ))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(
        "Key Findings:",
        styles["SubSection"]
    ))

    findings = [
        "Total revenue of R$19.7M generated over 2 years",
        "Average order value of R$179.47 with credit card as dominant payment (74%)",
        "97% of customers are one-time buyers — major retention opportunity",
        "Average delivery time of 12 days with 8% late delivery rate",
        "Health & Beauty is the top revenue category",
        "Sao Paulo state contributes 40%+ of total revenue",
        "Average customer review score of 4.08 out of 5",
        "Revenue growing at R$70,439 per month (linear trend)",
    ]

    for finding in findings:
        elements.append(Paragraph(
            f"• {finding}",
            styles["Insight"]
        ))

    elements.append(PageBreak())


def build_kpi_section(
    elements: list,
    styles: dict,
    data: dict
) -> None:
    """
    Build the KPI summary section.
    """
    add_section_header(elements, "1. Business KPIs", styles)

    ov = data["overview"].iloc[0]

    kpis = [
        ("Total Orders",      f"{int(ov['total_orders']):,}",      "#3498DB"),
        ("Total Revenue",     f"R${ov['total_revenue']:,.0f}",      "#2ECC71"),
        ("Avg Order Value",   f"R${ov['avg_order_value']:,.2f}",    "#F39C12"),
        ("Total Customers",   f"{int(ov['total_customers']):,}",    "#9B59B6"),
        ("Avg Review Score",  f"{ov['avg_review_score']:.2f}/5",    "#2ECC71"),
        ("Avg Delivery Days", f"{ov['avg_delivery_days']:.1f}d",    "#E74C3C"),
    ]

    elements.append(make_kpi_table(kpis))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph(
        "Detailed Business Metrics",
        styles["SubSection"]
    ))

    kpi_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Total Orders",      f"{int(ov['total_orders']):,}",
         "Total Revenue",     f"R${ov['total_revenue']:,.0f}"],
        ["Total Customers",   f"{int(ov['total_customers']):,}",
         "Total Sellers",     f"{int(ov['total_sellers']):,}"],
        ["Total Products",    f"{int(ov['total_products']):,}",
         "Avg Order Value",   f"R${ov['avg_order_value']:,.2f}"],
        ["Min Order Value",   f"R${ov['min_order_value']:,.2f}",
         "Max Order Value",   f"R${ov['max_order_value']:,.2f}"],
        ["Total Freight",     f"R${ov['total_freight']:,.0f}",
         "Avg Freight",       f"R${ov['avg_freight']:,.2f}"],
        ["Avg Delivery Days", f"{ov['avg_delivery_days']:.1f} days",
         "Late Deliveries",   f"{int(ov['late_deliveries']):,}"],
        ["Avg Review Score",  f"{ov['avg_review_score']:.2f}/5",
         "Late Delivery Rate",
         f"{ov['late_deliveries']/ov['total_orders']*100:.1f}%"],
    ]

    col_widths = [
        2.2 * inch, 1.8 * inch,
        2.2 * inch, 1.8 * inch
    ]
    elements.append(make_table(kpi_data, col_widths))
    elements.append(Spacer(1, 15))

def build_revenue_section(
    elements: list,
    styles: dict,
    data: dict
) -> None:
    """
    Build revenue analysis section.
    """
    elements.append(Paragraph(
        "2. Revenue Analysis",
        styles["SectionHeader"]
    ))

    # Monthly revenue plot
    add_plot(
        elements,
        "02_monthly_revenue.png",
        width=7 * inch,
        height=3.5 * inch,
        caption="Figure 1: Monthly Revenue Trend and Growth Rate"
    )

    # Monthly revenue table
    elements.append(Paragraph(
        "Monthly Revenue Summary",
        styles["SubSection"]
    ))

    monthly = data["monthly"]
    table_data = [
        ["Month", "Orders", "Customers",
         "Revenue (R$)", "Avg Order (R$)", "Growth %"]
    ]
    for _, row in monthly.iterrows():
        growth = (
            f"{row['revenue_growth_pct']:.1f}%"
            if pd.notna(row.get("revenue_growth_pct"))
            else "N/A"
        )
        table_data.append([
            str(row["year_month"]),
            f"{int(row['total_orders']):,}",
            f"{int(row['unique_customers']):,}",
            f"R${row['total_revenue']:,.0f}",
            f"R${row['avg_order_value']:,.2f}",
            growth,
        ])

    col_widths = [
        1.5*inch, 1.0*inch, 1.0*inch,
        1.5*inch, 1.3*inch, 0.9*inch
    ]
    elements.append(make_table(table_data, col_widths))
    elements.append(Spacer(1, 10))

    # Category revenue plot
    add_plot(
        elements,
        "03_revenue_by_category.png",
        width=7 * inch,
        height=3.5 * inch,
        caption="Figure 2: Revenue by Product Category"
    )

    # Category table
    elements.append(Paragraph(
        "Top 10 Categories by Revenue",
        styles["SubSection"]
    ))

    category = data["category"].head(10)
    cat_data  = [
        ["Category", "Orders", "Revenue (R$)",
         "Avg Price", "Review Score", "Share %"]
    ]
    for _, row in category.iterrows():
        cat_data.append([
            str(row["category"])[:25],
            f"{int(row['total_orders']):,}",
            f"R${row['total_revenue']:,.0f}",
            f"R${row['avg_price']:,.2f}",
            f"{row['avg_review_score']:.2f}",
            f"{row['revenue_share_pct']:.1f}%",
        ])

    col_widths = [
        2.2*inch, 0.9*inch, 1.3*inch,
        1.0*inch, 1.0*inch, 0.8*inch
    ]
    elements.append(make_table(cat_data, col_widths))
    elements.append(PageBreak())


def build_customer_section(
    elements: list,
    styles: dict,
    data: dict
) -> None:
    """
    Build customer analysis section.
    """
    elements.append(Paragraph(
        "3. Customer Analysis",
        styles["SectionHeader"]
    ))

    # Customer segmentation plot
    add_plot(
        elements,
        "05_customer_segmentation.png",
        width=7 * inch,
        height=3 * inch,
        caption="Figure 3: Customer Segmentation Analysis"
    )

    # Segmentation table
    elements.append(Paragraph(
        "Customer Segmentation Summary",
        styles["SubSection"]
    ))

    seg = data["segmentation"]
    seg_data = [
        ["Segment", "Customers", "% of Total",
         "Avg Total Spent", "Avg Order Value"]
    ]
    for _, row in seg.iterrows():
        seg_data.append([
            str(row["customer_segment"]),
            f"{int(row['customer_count']):,}",
            f"{row['customer_pct']:.1f}%",
            f"R${row['avg_total_spent']:,.2f}",
            f"R${row['avg_order_value']:,.2f}",
        ])

    col_widths = [
        2.0*inch, 1.2*inch, 1.0*inch,
        1.5*inch, 1.5*inch
    ]
    elements.append(make_table(seg_data, col_widths))
    elements.append(Spacer(1, 15))

    # RFM segment plot
    add_plot(
        elements,
        "rfm_03_segment_analysis.png",
        width=7 * inch,
        height=3.5 * inch,
        caption="Figure 4: RFM Customer Segment Analysis"
    )

    # RFM table
    elements.append(Paragraph(
        "RFM Segment Summary",
        styles["SubSection"]
    ))

    rfm = data["rfm"]
    rfm_data = [
        ["Segment", "Customers", "%",
         "Avg Monetary", "Total Revenue",
         "Avg Recency"]
    ]
    for _, row in rfm.iterrows():
        rfm_data.append([
            str(row["rfm_segment"]),
            f"{int(row['customer_count']):,}",
            f"{row['customer_pct']:.1f}%",
            f"R${row['avg_monetary']:,.2f}",
            f"R${row['total_monetary']:,.0f}",
            f"{row['avg_recency_days']:.0f}d",
        ])

    col_widths = [
        1.8*inch, 1.0*inch, 0.7*inch,
        1.2*inch, 1.3*inch, 1.0*inch
    ]
    elements.append(make_table(rfm_data, col_widths))
    elements.append(PageBreak())


def build_operations_section(
    elements: list,
    styles: dict,
    data: dict
) -> None:
    """
    Build operations analysis section.
    """
    elements.append(Paragraph(
        "4. Operations Analysis",
        styles["SectionHeader"]
    ))

    # Delivery plot
    add_plot(
        elements,
        "10_delivery_analysis.png",
        width=7 * inch,
        height=3 * inch,
        caption="Figure 5: Delivery Performance Analysis"
    )

    # Delivery table
    elements.append(Paragraph(
        "Delivery Performance by State (Top 15)",
        styles["SubSection"]
    ))

    delivery = data["delivery"].head(15)
    del_data = [
        ["State", "Region", "Orders",
         "Avg Days", "Late Orders", "Late %"]
    ]
    for _, row in delivery.iterrows():
        del_data.append([
            str(row["state"]),
            str(row["region"]),
            f"{int(row['total_orders']):,}",
            f"{row['avg_delivery_days']:.1f}",
            f"{int(row['late_orders']):,}",
            f"{row['late_pct']:.1f}%",
        ])

    col_widths = [
        0.8*inch, 1.2*inch, 1.0*inch,
        1.0*inch, 1.2*inch, 0.8*inch
    ]
    elements.append(make_table(del_data, col_widths))
    elements.append(Spacer(1, 15))

    # Payment plot
    add_plot(
        elements,
        "07_payment_analysis.png",
        width=7 * inch,
        height=3 * inch,
        caption="Figure 6: Payment Analysis"
    )

    # Payment table
    elements.append(Paragraph(
        "Payment Method Summary",
        styles["SubSection"]
    ))

    payment = data["payment"]
    pay_data = [
        ["Payment Type", "Orders", "% of Orders",
         "Revenue (R$)", "Avg Value", "Avg Installments"]
    ]
    for _, row in payment.iterrows():
        pay_data.append([
            str(row["payment_type"]),
            f"{int(row['total_orders']):,}",
            f"{row['order_pct']:.1f}%",
            f"R${row['total_revenue']:,.0f}",
            f"R${row['avg_order_value']:,.2f}",
            f"{row['avg_installments']:.1f}",
        ])

    col_widths = [
        1.5*inch, 1.0*inch, 1.0*inch,
        1.3*inch, 1.0*inch, 1.2*inch
    ]
    elements.append(make_table(pay_data, col_widths))
    elements.append(PageBreak())


def build_seller_section(
    elements: list,
    styles: dict,
    data: dict
) -> None:
    """
    Build seller analysis section.
    """
    elements.append(Paragraph(
        "5. Seller Analysis",
        styles["SectionHeader"]
    ))

    # Seller plot
    add_plot(
        elements,
        "seller_01_overview.png",
        width=7 * inch,
        height=3.5 * inch,
        caption="Figure 7: Seller Performance Overview"
    )

    # Top sellers table
    elements.append(Paragraph(
        "Top 15 Sellers by Revenue",
        styles["SubSection"]
    ))

    seller = data["seller"].head(15)
    sel_data = [
        ["Seller ID", "State", "Orders",
         "Revenue (R$)", "Review Score",
         "Avg Delivery", "Late %"]
    ]
    for _, row in seller.iterrows():
        sel_data.append([
            str(row["seller_id"])[:12] + "...",
            str(row["seller_state"]),
            f"{int(row['total_orders']):,}",
            f"R${row['total_revenue']:,.0f}",
            f"{row['avg_review_score']:.2f}",
            f"{row['avg_delivery_days']:.1f}d",
            f"{row['late_delivery_pct']:.1f}%",
        ])

    col_widths = [
        1.4*inch, 0.6*inch, 0.8*inch,
        1.2*inch, 1.0*inch, 1.0*inch, 0.8*inch
    ]
    elements.append(make_table(sel_data, col_widths))
    elements.append(PageBreak())


def build_insights_section(
    elements: list,
    styles: dict,
) -> None:
    """
    Build key insights and recommendations section.
    """
    elements.append(Paragraph(
        "6. Key Insights & Recommendations",
        styles["SectionHeader"]
    ))

    sections = [
        (
            "Revenue & Sales",
            C_BLUE,
            [
                "Revenue grew consistently from 2016 to 2018 with peak in Nov 2017 (Black Friday)",
                "Health & Beauty, Watches & Gifts, Bed Bath & Table are top 3 categories",
                "Sao Paulo state alone contributes over 40% of total revenue",
                "Credit card is dominant payment method at 74% of transactions",
                "Average customer spends R$179.47 per order",
            ]
        ),
        (
            "Customer Retention",
            C_RED,
            [
                "97%+ of customers are one-time buyers — critical retention issue",
                "Champions segment (high RFM) should receive loyalty rewards",
                "At Risk customers need immediate re-engagement campaigns",
                "Lost customers should receive win-back email campaigns",
                "Recommend implementing a loyalty points program",
            ]
        ),
        (
            "Delivery & Operations",
            C_ORANGE,
            [
                "Average delivery time is 12 days — room for improvement",
                "Northern states (RR, AP, AM) have worst delivery performance (25+ days)",
                "8% late delivery rate — identify and penalize consistently late sellers",
                "Bronze tier sellers have 98% late delivery rate — immediate intervention needed",
                "Consider regional fulfillment centers for northern states",
            ]
        ),
        (
            "Product & Seller Strategy",
            C_GREEN,
            [
                "Only 9 Platinum sellers generating R$167K average revenue",
                "Focus on growing Gold tier sellers (533 sellers, R$8K avg revenue)",
                "Health & Beauty category has highest revenue with good review scores",
                "Encourage top-selling categories with better seller incentives",
                "Implement seller training program for Bronze tier sellers",
            ]
        ),
    ]

    for title, color, insights in sections:
        elements.append(Paragraph(
            title,
            ParagraphStyle(
                name      = f"Insight_{title}",
                fontSize  = 12,
                fontName  = "Helvetica-Bold",
                textColor = color,
                spaceBefore=10,
                spaceAfter =5,
            )
        ))
        for insight in insights:
            elements.append(Paragraph(
                f"• {insight}",
                styles["Insight"]
            ))
        elements.append(Spacer(1, 8))


def build_appendix(
    elements: list,
    styles: dict,
    data: dict
) -> None:
    """
    Build appendix with additional data tables.
    """
    elements.append(PageBreak())
    elements.append(Paragraph(
        "Appendix A: Top 20 Customers",
        styles["SectionHeader"]
    ))

    customers = data["top_customers"]
    cust_data = [
        ["Customer ID", "State", "Region",
         "Orders", "Total Spent", "Avg Order", "Review"]
    ]
    for _, row in customers.iterrows():
        cust_data.append([
            str(row["customer_unique_id"])[:12] + "...",
            str(row["customer_state"]),
            str(row["customer_region"]),
            f"{int(row['total_orders']):,}",
            f"R${row['total_spent']:,.0f}",
            f"R${row['avg_order_value']:,.2f}",
            f"{row['avg_review_score']:.2f}",
        ])

    col_widths = [
        1.5*inch, 0.6*inch, 1.0*inch,
        0.7*inch, 1.1*inch, 1.1*inch, 0.7*inch
    ]
    elements.append(make_table(cust_data, col_widths))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph(
        "Appendix B: Top 20 Products",
        styles["SectionHeader"]
    ))

    products = data["top_products"]
    prod_data = [
        ["Product ID", "Category", "Orders",
         "Revenue (R$)", "Avg Price", "Review"]
    ]
    for _, row in products.iterrows():
        prod_data.append([
            str(row["product_id"])[:12] + "...",
            str(row["category"])[:20],
            f"{int(row['total_orders']):,}",
            f"R${row['total_revenue']:,.0f}",
            f"R${row['avg_price']:,.2f}",
            f"{row['avg_review_score']:.2f}",
        ])

    col_widths = [
        1.5*inch, 1.8*inch, 0.8*inch,
        1.2*inch, 1.0*inch, 0.9*inch
    ]
    elements.append(make_table(prod_data, col_widths))


# ════════════════════════════════════════════════════════════
# MAIN PDF REPORT FUNCTION
# ════════════════════════════════════════════════════════════

def generate_pdf_report(
    filename: str = "retail_analytics_report.pdf"
) -> Path:
    """
    Generate complete PDF report.

    Returns:
        Path to generated PDF file
    """
    logger.info("=" * 55)
    logger.info("PDF REPORT GENERATION STARTED")
    logger.info("=" * 55)

    create_directories()
    total_start = time.time()

    # Load data
    logger.info("-- Loading data --")
    data = {
        "overview"    : get_business_overview(),
        "monthly"     : get_monthly_revenue(),
        "category"    : get_revenue_by_category(),
        "state"       : get_revenue_by_state(),
        "segmentation": get_customer_segmentation(),
        "payment"     : get_payment_analysis(),
        "review"      : get_review_analysis(),
        "rfm"         : get_rfm_segment_summary(),
        "seller"      : get_seller_performance(),
        "delivery"    : get_delivery_analysis(),
        "top_customers": get_top_customers(),
        "top_products" : get_top_products(),
    }
    logger.info("[OK] Data loaded")

    # Output path
    output_path = REPORTS_DIR / filename

    # Create PDF document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Retail Analytics Report",
        author="Retail Analytics System",
    )

    # Get styles
    styles = get_styles()

    # Build content
    elements = []

    # Cover page
    logger.info("-- Building cover page --")
    build_cover_page(elements, styles)

    # KPI section
    logger.info("-- Building KPI section --")
    build_kpi_section(elements, styles, data)

    # Revenue section
    logger.info("-- Building revenue section --")
    build_revenue_section(elements, styles, data)

    # Customer section
    logger.info("-- Building customer section --")
    build_customer_section(elements, styles, data)

    # Operations section
    logger.info("-- Building operations section --")
    build_operations_section(elements, styles, data)

    # Seller section
    logger.info("-- Building seller section --")
    build_seller_section(elements, styles, data)

    # Insights section
    logger.info("-- Building insights section --")
    build_insights_section(elements, styles)

    # Appendix
    logger.info("-- Building appendix --")
    build_appendix(elements, styles, data)

    # Build PDF
    logger.info("-- Generating PDF --")
    doc.build(elements)

    total_elapsed = time.time() - total_start
    file_size     = output_path.stat().st_size / 1024

    print("\n" + "=" * 55)
    print("   PDF REPORT SUMMARY")
    print("=" * 55)
    print(f"  File      : {filename}")
    print(f"  Path      : {output_path}")
    print(f"  Size      : {file_size:.1f} KB")
    print(f"  Sections  : 6 + Appendix")
    print(f"  Time      : {total_elapsed:.2f}s")
    print("=" * 55 + "\n")

    logger.info(
        f"[OK] PDF report generated: {output_path}"
    )

    return output_path


# ── Run directly ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    path = generate_pdf_report()