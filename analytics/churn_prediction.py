# analytics/churn_prediction.py

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

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
import joblib

from src.config import (
    PROCESSED_FILES,
    PLOTS_DIR,
    VIZ_SETTINGS,
    CHURN_SETTINGS,
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
    Load master dataset for churn prediction.
    """
    logger.info("-- Loading data for churn prediction --")

    df = pd.read_csv(PROCESSED_FILES["master"])

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    )

    df = df[df["order_status"] == "delivered"].copy()

    logger.info(f"[OK] Loaded {len(df):,} delivered orders")
    return df


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer level features for churn prediction.
    """
    logger.info("-- Building customer features --")

    reference_date = df["order_purchase_timestamp"].max()
    threshold      = CHURN_SETTINGS["churn_threshold"]

    # Aggregate at customer level
    customer_df = df.groupby("customer_unique_id").agg(
        total_orders      = ("order_id",                "nunique"),
        total_revenue     = ("total_payment_value",     "sum"),
        avg_order_value   = ("total_payment_value",     "mean"),
        total_items       = ("order_item_id",           "sum"),
        avg_review_score  = ("review_score",            "mean"),
        avg_delivery_days = ("delivery_days",           "mean"),
        late_orders       = ("is_late_delivery",        "sum"),
        avg_freight       = ("freight_value",           "mean"),
        avg_installments  = ("max_installments",        "mean"),
        first_order_date  = ("order_purchase_timestamp","min"),
        last_order_date   = ("order_purchase_timestamp","max"),
        unique_categories = ("product_category_name_english","nunique"),
        unique_sellers    = ("seller_id",               "nunique"),
        unique_states     = ("seller_state",            "nunique"),
    ).reset_index()

    # Calculate derived features
    customer_df["recency_days"] = (
        reference_date - customer_df["last_order_date"]
    ).dt.days

    customer_df["customer_age_days"] = (
        customer_df["last_order_date"] -
        customer_df["first_order_date"]
    ).dt.days

    customer_df["avg_days_between_orders"] = (
        customer_df["customer_age_days"] /
        customer_df["total_orders"].clip(1)
    ).round(2)

    customer_df["late_order_pct"] = (
        customer_df["late_orders"] /
        customer_df["total_orders"] * 100
    ).round(2)

    customer_df["revenue_per_item"] = (
        customer_df["total_revenue"] /
        customer_df["total_items"].clip(1)
    ).round(2)

    # Define churn label
    customer_df["is_churned"] = (
        customer_df["recency_days"] > threshold
    ).astype(int)

    churn_rate = customer_df["is_churned"].mean() * 100
    logger.info(
        f"[OK] Customer features built: "
        f"{len(customer_df):,} customers"
    )
    logger.info(
        f"  Churn threshold : {threshold} days"
    )
    logger.info(
        f"  Churn rate      : {churn_rate:.2f}%"
    )

    return customer_df


# ════════════════════════════════════════════════════════════
# STEP 2: PREPARE FEATURES
# ════════════════════════════════════════════════════════════

def prepare_features(
    customer_df: pd.DataFrame
) -> tuple:
    """
    Prepare feature matrix and target variable.
    """
    logger.info("-- Preparing features --")

    feature_cols = [
        "total_orders",
        "total_revenue",
        "avg_order_value",
        "total_items",
        "avg_review_score",
        "avg_delivery_days",
        "late_order_pct",
        "avg_freight",
        "avg_installments",
        "recency_days",
        "customer_age_days",
        "avg_days_between_orders",
        "unique_categories",
        "unique_sellers",
        "revenue_per_item",
    ]

    # Keep only available columns
    feature_cols = [
        c for c in feature_cols
        if c in customer_df.columns
    ]

    X = customer_df[feature_cols].copy()
    y = customer_df["is_churned"].copy()

    # Fill missing values
    X = X.fillna(X.median())

    # Clip outliers
    for col in X.columns:
        q99 = X[col].quantile(0.99)
        q01 = X[col].quantile(0.01)
        X[col] = X[col].clip(q01, q99)

    logger.info(
        f"[OK] Features prepared: "
        f"{X.shape[0]:,} samples x {X.shape[1]} features"
    )
    logger.info(
        f"  Churn: {y.sum():,} ({y.mean()*100:.1f}%)"
    )
    logger.info(
        f"  Active: {(~y.astype(bool)).sum():,} "
        f"({(1-y.mean())*100:.1f}%)"
    )

    return X, y, feature_cols


# ════════════════════════════════════════════════════════════
# STEP 3: TRAIN MODELS
# ════════════════════════════════════════════════════════════

def train_models(
    X: pd.DataFrame,
    y: pd.Series
) -> tuple:
    """
    Train multiple ML models for churn prediction.
    """
    logger.info("-- Training models --")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = CHURN_SETTINGS["test_size"],
        random_state = CHURN_SETTINGS["random_state"],
        stratify     = y
    )

    logger.info(
        f"  Train: {len(X_train):,} | "
        f"Test: {len(X_test):,}"
    )

    # Define models
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(
                max_iter      = 1000,
                class_weight  = "balanced",
                random_state  = CHURN_SETTINGS["random_state"]
            ))
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  DecisionTreeClassifier(
                max_depth    = 8,
                class_weight = "balanced",
                random_state = CHURN_SETTINGS["random_state"]
            ))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestClassifier(
                n_estimators = 100,
                max_depth    = 10,
                class_weight = "balanced",
                random_state = CHURN_SETTINGS["random_state"],
                n_jobs       = -1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingClassifier(
                n_estimators  = 100,
                learning_rate = 0.1,
                max_depth     = 5,
                random_state  = CHURN_SETTINGS["random_state"]
            ))
        ]),
    }

    results = {}

    for name, model in models.items():
        logger.info(f"  Training: {name}")
        start = time.time()

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        auc_roc = roc_auc_score(y_test, y_proba)
        auc_pr  = average_precision_score(y_test, y_proba)
        f1      = f1_score(y_test, y_pred)
        cm      = confusion_matrix(y_test, y_pred)

        elapsed = time.time() - start

        results[name] = {
            "model"   : model,
            "y_pred"  : y_pred,
            "y_proba" : y_proba,
            "auc_roc" : auc_roc,
            "auc_pr"  : auc_pr,
            "f1"      : f1,
            "cm"      : cm,
            "time"    : elapsed,
        }

        logger.info(
            f"  [OK] {name}: "
            f"ROC={auc_roc:.4f} "
            f"PR={auc_pr:.4f} "
            f"F1={f1:.4f} "
            f"({elapsed:.1f}s)"
        )

    return results, X_train, X_test, y_train, y_test


# ════════════════════════════════════════════════════════════
# STEP 4: GET BEST MODEL
# ════════════════════════════════════════════════════════════

def get_best_model(results: dict) -> tuple:
    """
    Get the best model based on AUC-ROC score.
    """
    best_name  = max(
        results, key=lambda k: results[k]["auc_roc"]
    )
    best_model = results[best_name]
    logger.info(
        f"[OK] Best model: {best_name} "
        f"(AUC-ROC={best_model['auc_roc']:.4f})"
    )
    return best_name, best_model


# ════════════════════════════════════════════════════════════
# STEP 5: FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════

def get_feature_importance(
    results: dict,
    feature_cols: list
) -> pd.DataFrame:
    """
    Get feature importance from Random Forest model.
    """
    logger.info("-- Getting feature importance --")

    rf_model = results["Random Forest"]["model"]
    rf       = rf_model.named_steps["model"]

    importance_df = pd.DataFrame({
        "feature"   : feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info("[OK] Feature importance calculated")
    return importance_df


# ════════════════════════════════════════════════════════════
# STEP 6: VISUALIZE RESULTS
# ════════════════════════════════════════════════════════════

def plot_churn_overview(
    customer_df: pd.DataFrame
) -> None:
    """
    Plot churn overview and customer distributions.
    """
    logger.info("-- Plotting churn overview --")

    fig, axes = plt.subplots(
        2, 3, figsize=(22, 14), facecolor=BG
    )
    fig.suptitle(
        "CHURN ANALYSIS OVERVIEW",
        fontsize=18, fontweight="bold", color=TEXT_C
    )

    # Plot 1: Churn distribution donut
    ax = axes[0, 0]
    churn_counts = customer_df["is_churned"].value_counts()
    wedges, texts, autotexts = ax.pie(
        churn_counts.values,
        labels=["Active", "Churned"],
        autopct="%1.1f%%",
        colors=[GREEN, RED],
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(
            width=0.5,
            edgecolor=BG,
            linewidth=2
        )
    )
    for at in autotexts:
        at.set(fontsize=12, fontweight="bold", color="white")
    for t in texts:
        t.set(fontsize=11, color=TEXT_C)
    ax.set_title("Customer Churn Distribution")

    # Plot 2: Recency distribution
    ax = axes[0, 1]
    for label, color, name in [
        (0, GREEN, "Active"),
        (1, RED,   "Churned")
    ]:
        data = customer_df[
            customer_df["is_churned"] == label
        ]["recency_days"]
        ax.hist(
            data.clip(0, 500),
            bins=50, alpha=0.6,
            color=color, label=name,
            edgecolor="none"
        )
    ax.axvline(
        CHURN_SETTINGS["churn_threshold"],
        color="white", lw=2, ls="--",
        label=f"Threshold ({CHURN_SETTINGS['churn_threshold']}d)"
    )
    ax.set_xlabel("Recency Days")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Recency Distribution by Churn Status")
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=8
    )
    ax.grid(axis="y", alpha=0.3)

    # Plot 3: Revenue distribution
    ax = axes[0, 2]
    for label, color, name in [
        (0, GREEN, "Active"),
        (1, RED,   "Churned")
    ]:
        data = customer_df[
            customer_df["is_churned"] == label
        ]["total_revenue"]
        ax.hist(
            data.clip(0, 2000),
            bins=50, alpha=0.6,
            color=color, label=name,
            edgecolor="none"
        )
    ax.set_xlabel("Total Revenue (R$)")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Revenue Distribution by Churn Status")
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=8
    )
    ax.grid(axis="y", alpha=0.3)

    # Plot 4: Orders distribution
    ax = axes[1, 0]
    for label, color, name in [
        (0, GREEN, "Active"),
        (1, RED,   "Churned")
    ]:
        data = customer_df[
            customer_df["is_churned"] == label
        ]["total_orders"]
        ax.hist(
            data.clip(0, 10),
            bins=10, alpha=0.6,
            color=color, label=name,
            edgecolor="none"
        )
    ax.set_xlabel("Total Orders")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Order Count by Churn Status")
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=8
    )
    ax.grid(axis="y", alpha=0.3)

    # Plot 5: Review score distribution
    ax = axes[1, 1]
    for label, color, name in [
        (0, GREEN, "Active"),
        (1, RED,   "Churned")
    ]:
        data = customer_df[
            customer_df["is_churned"] == label
        ]["avg_review_score"].dropna()
        ax.hist(
            data, bins=20, alpha=0.6,
            color=color, label=name,
            edgecolor="none"
        )
    ax.set_xlabel("Avg Review Score")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Review Score by Churn Status")
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=8
    )
    ax.grid(axis="y", alpha=0.3)

    # Plot 6: Churn stats table
    ax = axes[1, 2]
    ax.axis("off")

    churn_stats = customer_df.groupby("is_churned").agg(
        count          = ("customer_unique_id", "count"),
        avg_recency    = ("recency_days",       "mean"),
        avg_orders     = ("total_orders",       "mean"),
        avg_revenue    = ("total_revenue",      "mean"),
        avg_review     = ("avg_review_score",   "mean"),
    ).round(2).reset_index()
    churn_stats["is_churned"] = churn_stats[
        "is_churned"
    ].map({0: "Active", 1: "Churned"})

    table_data = []
    for _, row in churn_stats.iterrows():
        table_data.append([
            row["is_churned"],
            f"{int(row['count']):,}",
            f"{row['avg_recency']:.0f}d",
            f"{row['avg_orders']:.1f}",
            f"R${row['avg_revenue']:,.0f}",
            f"{row['avg_review']:.2f}",
        ])

    t = ax.table(
        cellText=table_data,
        colLabels=[
            "Status", "Count", "Recency",
            "Orders", "Revenue", "Review"
        ],
        cellLoc="center",
        loc="center",
        colWidths=[0.2, 0.15, 0.15, 0.15, 0.20, 0.15]
    )
    t.auto_set_font_size(False)
    t.set_fontsize(9)
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
    ax.set_title(
        "Churn Statistics",
        color=TEXT_C, fontsize=11,
        fontweight="bold", pad=15
    )

    plt.tight_layout()
    save_plot(fig, "churn_01_overview.png")


def plot_model_comparison(
    results: dict,
    y_test: pd.Series
) -> None:
    """
    Plot model comparison — ROC, PR curves
    and confusion matrices.
    """
    logger.info("-- Plotting model comparison --")

    model_colors = {
        "Logistic Regression": PURPLE,
        "Decision Tree"      : ORANGE,
        "Random Forest"      : GREEN,
        "Gradient Boosting"  : BLUE,
    }

    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    fig.suptitle(
        "CHURN MODEL COMPARISON",
        fontsize=18, fontweight="bold", color=TEXT_C
    )
    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        hspace=0.4, wspace=0.35
    )

    # Plot 1: ROC Curves
    ax_roc = fig.add_subplot(gs[0, :2])
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax_roc.plot(
            fpr, tpr,
            color=model_colors[name], lw=2,
            label=f"{name} (AUC={res['auc_roc']:.3f})"
        )
    ax_roc.plot(
        [0, 1], [0, 1],
        "k--", color="gray", alpha=0.5
    )
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves")
    ax_roc.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=8
    )
    ax_roc.grid(alpha=0.3)

    # Plot 2: PR Curves
    ax_pr = fig.add_subplot(gs[0, 2:])
    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(
            y_test, res["y_proba"]
        )
        ax_pr.plot(
            rec, prec,
            color=model_colors[name], lw=2,
            label=f"{name} (AP={res['auc_pr']:.3f})"
        )
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves")
    ax_pr.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=8
    )
    ax_pr.grid(alpha=0.3)

    # Plot 3-6: Confusion matrices
    for i, (name, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1, i])
        cm     = res["cm"]
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(
            cm_pct, ax=ax,
            annot=True, fmt=".1f",
            cmap="RdYlGn",
            linewidths=1, linecolor=BG,
            cbar=False, vmin=0, vmax=100,
            annot_kws={"size": 10, "weight": "bold"},
            xticklabels=["Active", "Churned"],
            yticklabels=["Active", "Churned"]
        )
        ax.set_title(
            f"{name}\nF1={res['f1']:.3f} "
            f"ROC={res['auc_roc']:.3f}",
            fontsize=9
        )
        ax.tick_params(axis="both", labelsize=8)

    save_plot(fig, "churn_02_model_comparison.png")


def plot_feature_importance(
    importance_df: pd.DataFrame
) -> None:
    """
    Plot feature importance from Random Forest.
    """
    logger.info("-- Plotting feature importance --")

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
    fig.suptitle(
        "CHURN PREDICTION - FEATURE IMPORTANCE",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    colors = [
        RED    if v > 0.15
        else ORANGE if v > 0.10
        else BLUE
        for v in importance_df["importance"]
    ]

    bars = ax.barh(
        importance_df["feature"],
        importance_df["importance"],
        color=colors, edgecolor="none", alpha=0.85
    )
    ax.set_xlabel("Feature Importance")
    ax.set_title(
        "Random Forest Feature Importance\n"
        "(Higher = More Important for Churn)"
    )
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, importance_df["importance"]):
        ax.text(
            val + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=9, color=TEXT_C
        )

    plt.tight_layout()
    save_plot(fig, "churn_03_feature_importance.png")


def plot_churn_probability(
    customer_df: pd.DataFrame,
    best_model,
    X: pd.DataFrame,
    feature_cols: list
) -> None:
    """
    Plot churn probability distribution.
    """
    logger.info("-- Plotting churn probability --")

    # Get churn probabilities
    proba = best_model["model"].predict_proba(
        X[feature_cols]
    )[:, 1]

    customer_df = customer_df.copy()
    customer_df["churn_probability"] = proba

    # Risk segments
    customer_df["risk_segment"] = pd.cut(
        customer_df["churn_probability"],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["Low Risk", "Medium Risk",
                "High Risk", "Very High Risk"]
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(22, 8), facecolor=BG
    )
    fig.suptitle(
        "CHURN PROBABILITY ANALYSIS",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    # Plot 1: Probability distribution
    ax = axes[0]
    ax.hist(
        customer_df["churn_probability"],
        bins=50, color=ORANGE,
        edgecolor="none", alpha=0.85
    )
    ax.set_xlabel("Churn Probability")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Churn Probability Distribution")
    ax.axvline(
        0.5, color=RED, lw=2, ls="--",
        label="Decision Threshold (0.5)"
    )
    ax.legend(
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=8
    )
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: Risk segment distribution
    ax = axes[1]
    risk_counts = customer_df[
        "risk_segment"
    ].value_counts().sort_index()
    risk_colors = [GREEN, ORANGE, RED, "#8B0000"]
    bars = ax.bar(
        risk_counts.index,
        risk_counts.values,
        color=risk_colors[:len(risk_counts)],
        edgecolor="none", alpha=0.85
    )
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customers by Risk Segment")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, risk_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 50,
            f"{val:,}",
            ha="center", fontsize=9,
            fontweight="bold", color=TEXT_C
        )

    # Plot 3: Avg revenue by risk segment
    ax = axes[2]
    risk_revenue = customer_df.groupby(
        "risk_segment"
    )["total_revenue"].mean().reset_index()
    bars = ax.bar(
        risk_revenue["risk_segment"],
        risk_revenue["total_revenue"],
        color=risk_colors[:len(risk_revenue)],
        edgecolor="none", alpha=0.85
    )
    ax.set_ylabel("Avg Revenue (R$)")
    ax.set_title("Avg Revenue by Risk Segment")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"R${x:,.0f}")
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_plot(fig, "churn_04_probability.png")

    return customer_df


# ════════════════════════════════════════════════════════════
# MAIN CHURN FUNCTION
# ════════════════════════════════════════════════════════════

def run_churn_prediction() -> tuple:
    """
    Run complete churn prediction pipeline.

    Returns:
        tuple: (customer_df, results, importance_df)
    """
    logger.info("=" * 55)
    logger.info("CHURN PREDICTION STARTED")
    logger.info("=" * 55)

    total_start = time.time()

    # Step 1: Load data
    df = load_data()

    # Step 2: Build features
    customer_df = build_customer_features(df)

    # Step 3: Prepare features
    X, y, feature_cols = prepare_features(customer_df)

    # Step 4: Train models
    results, X_train, X_test, y_train, y_test = (
        train_models(X, y)
    )

    # Step 5: Get best model
    best_name, best_model = get_best_model(results)

    # Step 6: Feature importance
    importance_df = get_feature_importance(
        results, feature_cols
    )

    # Step 7: Visualize
    plot_churn_overview(customer_df)
    plot_model_comparison(results, y_test)
    plot_feature_importance(importance_df)
    customer_df = plot_churn_probability(
        customer_df, best_model, X, feature_cols
    )

    # Save best model
    model_path = CHURN_SETTINGS["model_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model["model"], model_path)
    logger.info(f"[OK] Model saved: {model_path}")

    # Save results
    reports_dir = PLOTS_DIR.parent / "reports"
    customer_df.to_csv(
        reports_dir / "churn_predictions.csv",
        index=False
    )
    importance_df.to_csv(
        reports_dir / "churn_feature_importance.csv",
        index=False
    )

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 55)
    print("   CHURN PREDICTION SUMMARY")
    print("=" * 55)
    print(
        f"  Total Customers  : {len(customer_df):,}"
    )
    print(
        f"  Churned          : "
        f"{customer_df['is_churned'].sum():,} "
        f"({customer_df['is_churned'].mean()*100:.1f}%)"
    )
    print(f"  Best Model       : {best_name}")
    print(
        f"  Best AUC-ROC     : "
        f"{best_model['auc_roc']:.4f}"
    )
    print(f"  Plots Generated  : 4")
    print(f"  Time             : {total_elapsed:.2f}s")
    print("-" * 55)
    print(
        f"  {'Model':<22} {'ROC':>8} "
        f"{'PR':>8} {'F1':>8}"
    )
    print("-" * 55)
    for name, res in sorted(
        results.items(),
        key=lambda x: -x[1]["auc_roc"]
    ):
        print(
            f"  {name:<22} "
            f"{res['auc_roc']:>8.4f} "
            f"{res['auc_pr']:>8.4f} "
            f"{res['f1']:>8.4f}"
        )
    print("-" * 55)
    print("  Top 5 Churn Features:")
    for _, row in importance_df.head(5).iterrows():
        print(
            f"  {row['feature']:<25} : "
            f"{row['importance']:.4f}"
        )
    print("=" * 55 + "\n")

    return customer_df, results, importance_df


# ── Run directly ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    customer_df, results, importance_df = (
        run_churn_prediction()
    )