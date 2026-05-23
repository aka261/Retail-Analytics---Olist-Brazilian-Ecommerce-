# analytics/market_basket.py

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
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from src.config import (
    PROCESSED_FILES,
    PLOTS_DIR,
    VIZ_SETTINGS,
    DATABASE_PATH,
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
    Load master dataset for market basket analysis.
    """
    logger.info("-- Loading data for market basket --")

    df = pd.read_csv(PROCESSED_FILES["master"])

    # Keep delivered orders only
    df = df[df["order_status"] == "delivered"].copy()

    # Fill missing category
    df["product_category_name_english"] = (
        df["product_category_name_english"].fillna("unknown")
    )

    logger.info(f"[OK] Loaded {len(df):,} delivered orders")
    return df


# ════════════════════════════════════════════════════════════
# STEP 2: BUILD TRANSACTION DATA
# ════════════════════════════════════════════════════════════

def build_transaction_data(
    df: pd.DataFrame
) -> tuple:
    """
    Build transaction data for market basket analysis.
    Each transaction = one order
    Each item = product category bought in that order

    Returns:
        tuple: (basket_df, transaction_list)
    """
    logger.info("-- Building transaction data --")

    # Group by order and get categories
    basket = df.groupby(
        ["order_id", "product_category_name_english"]
    )["order_item_id"].count().reset_index()

    basket.columns = ["order_id", "category", "quantity"]

    # Pivot to basket format
    basket_pivot = basket.pivot_table(
        index   = "order_id",
        columns = "category",
        values  = "quantity",
        aggfunc = "sum"
    ).fillna(0)

    # Convert to binary
    basket_binary = (basket_pivot > 0).astype(int)

    # Transaction list for apriori
    transactions = df.groupby("order_id")[
        "product_category_name_english"
    ].apply(list).tolist()

    logger.info(
        f"[OK] Transaction data built: "
        f"{len(basket_binary):,} orders x "
        f"{len(basket_binary.columns)} categories"
    )

    # Filter to orders with multiple items
    multi_item = basket_binary[
        basket_binary.sum(axis=1) > 1
    ]
    logger.info(
        f"[OK] Multi-item orders: {len(multi_item):,}"
    )

    return basket_binary, multi_item, transactions


# ════════════════════════════════════════════════════════════
# STEP 3: CATEGORY CO-OCCURRENCE ANALYSIS
# ════════════════════════════════════════════════════════════

def calculate_co_occurrence(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate category co-occurrence matrix.
    Shows how often categories are bought together.
    """
    logger.info("-- Calculating co-occurrence matrix --")

    # Get top 20 categories
    top_cats = df[
        "product_category_name_english"
    ].value_counts().head(20).index.tolist()

    df_top = df[
        df["product_category_name_english"].isin(top_cats)
    ]

    # Build order-category matrix
    order_cat = df_top.groupby(
        ["order_id", "product_category_name_english"]
    )["order_item_id"].count().unstack(fill_value=0)

    order_cat = (order_cat > 0).astype(int)

    # Calculate co-occurrence
    co_occur = order_cat.T.dot(order_cat)
    np.fill_diagonal(co_occur.values, 0)

    logger.info(
        f"[OK] Co-occurrence matrix: "
        f"{co_occur.shape[0]}x{co_occur.shape[1]}"
    )
    return co_occur


# ════════════════════════════════════════════════════════════
# STEP 4: APRIORI ALGORITHM
# ════════════════════════════════════════════════════════════

def run_apriori(
    basket_binary: pd.DataFrame,
    min_support: float    = 0.01,
    min_confidence: float = 0.1,
    min_lift: float       = 1.0,
) -> tuple:
    """
    Run Apriori algorithm to find association rules.

    Args:
        basket_binary  : Binary basket dataframe
        min_support    : Minimum support threshold
        min_confidence : Minimum confidence threshold
        min_lift       : Minimum lift threshold

    Returns:
        tuple: (frequent_itemsets, rules)
    """
    logger.info("-- Running Apriori algorithm --")
    logger.info(f"  Min Support    : {min_support}")
    logger.info(f"  Min Confidence : {min_confidence}")
    logger.info(f"  Min Lift       : {min_lift}")

    # Use top 30 categories to keep it manageable
    top_cats = basket_binary.sum().sort_values(
        ascending=False
    ).head(30).index
    basket_top = basket_binary[top_cats]

    # Run apriori
    frequent_itemsets = apriori(
        basket_top,
        min_support     = min_support,
        use_colnames    = True,
        max_len         = 3,
        verbose         = 0,
    )

    if len(frequent_itemsets) == 0:
        logger.warning(
            "[WARN] No frequent itemsets found. "
            "Lowering min_support to 0.005"
        )
        frequent_itemsets = apriori(
            basket_top,
            min_support  = 0.005,
            use_colnames = True,
            max_len      = 3,
            verbose      = 0,
        )

    logger.info(
        f"[OK] Frequent itemsets: "
        f"{len(frequent_itemsets):,}"
    )

    if len(frequent_itemsets) == 0:
        logger.warning(
            "[WARN] Still no itemsets. "
            "Returning empty rules."
        )
        return frequent_itemsets, pd.DataFrame()

    # Generate association rules
    rules = association_rules(
        frequent_itemsets,
        metric    = "lift",
        min_threshold = min_lift,
    )

    # Filter by confidence
    rules = rules[
        rules["confidence"] >= min_confidence
    ]

    # Sort by lift
    rules = rules.sort_values(
        "lift", ascending=False
    ).reset_index(drop=True)

    logger.info(f"[OK] Association rules: {len(rules):,}")

    if len(rules) > 0:
        logger.info(
            f"  Top rule: "
            f"{list(rules['antecedents'].iloc[0])} -> "
            f"{list(rules['consequents'].iloc[0])} "
            f"(lift={rules['lift'].iloc[0]:.2f})"
        )

    return frequent_itemsets, rules


# ════════════════════════════════════════════════════════════
# STEP 5: ANALYZE RULES
# ════════════════════════════════════════════════════════════

def analyze_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze and format association rules.
    Add readable columns.
    """
    if len(rules) == 0:
        return rules

    logger.info("-- Analyzing association rules --")

    # Convert frozensets to strings
    rules["antecedents_str"] = rules["antecedents"].apply(
        lambda x: ", ".join(list(x))
    )
    rules["consequents_str"] = rules["consequents"].apply(
        lambda x: ", ".join(list(x))
    )
    rules["rule"] = (
        rules["antecedents_str"] +
        " -> " +
        rules["consequents_str"]
    )

    # Round metrics
    rules["support"]    = rules["support"].round(4)
    rules["confidence"] = rules["confidence"].round(4)
    rules["lift"]       = rules["lift"].round(4)

    # Classify rules by lift
    rules["rule_strength"] = pd.cut(
        rules["lift"],
        bins   = [0, 1.5, 2.5, float("inf")],
        labels = ["Weak", "Moderate", "Strong"]
    )

    logger.info(
        f"[OK] Rules analyzed: {len(rules):,}"
    )
    logger.info(
        f"  Strong rules   : "
        f"{(rules['rule_strength']=='Strong').sum()}"
    )
    logger.info(
        f"  Moderate rules : "
        f"{(rules['rule_strength']=='Moderate').sum()}"
    )
    logger.info(
        f"  Weak rules     : "
        f"{(rules['rule_strength']=='Weak').sum()}"
    )

    return rules


# ════════════════════════════════════════════════════════════
# STEP 6: VISUALIZE
# ════════════════════════════════════════════════════════════

def plot_co_occurrence_heatmap(
    co_occur: pd.DataFrame
) -> None:
    """
    Plot category co-occurrence heatmap.
    """
    logger.info("-- Plotting co-occurrence heatmap --")

    # Normalize co-occurrence
    co_norm = co_occur.copy().astype(float)
    for col in co_norm.columns:
        if co_norm[col].max() > 0:
            co_norm[col] = (
                co_norm[col] / co_norm[col].max()
            )

    # Keep top 15 categories
    top15 = co_occur.sum().sort_values(
        ascending=False
    ).head(15).index
    co_top = co_norm.loc[top15, top15]

    fig, ax = plt.subplots(
        figsize=(14, 10), facecolor=BG
    )
    fig.suptitle(
        "CATEGORY CO-OCCURRENCE HEATMAP",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    sns.heatmap(
        co_top,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.5,
        linecolor=BG,
        cbar_kws={
            "shrink": 0.7,
            "label" : "Normalized Co-occurrence"
        }
    )
    ax.set_title(
        "How Often Categories are Bought Together\n"
        "(darker = more frequent)",
        color=TEXT_C, pad=10
    )
    ax.tick_params(
        axis="x", rotation=45, labelsize=8
    )
    ax.tick_params(
        axis="y", rotation=0, labelsize=8
    )

    plt.tight_layout()
    save_plot(fig, "market_basket_01_cooccurrence.png")


def plot_top_categories(
    basket_binary: pd.DataFrame
) -> None:
    """
    Plot top categories by purchase frequency.
    """
    logger.info("-- Plotting top categories --")

    cat_freq = basket_binary.sum().sort_values(
        ascending=False
    ).head(20)

    fig, ax = plt.subplots(
        figsize=(14, 8), facecolor=BG
    )
    fig.suptitle(
        "TOP 20 CATEGORIES BY PURCHASE FREQUENCY",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    colors = [
        COLORS[i % len(COLORS)]
        for i in range(len(cat_freq))
    ]
    bars = ax.barh(
        cat_freq.index,
        cat_freq.values,
        color=colors, edgecolor="none", alpha=0.85
    )
    ax.set_xlabel("Number of Orders")
    ax.set_title("Category Purchase Frequency")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, cat_freq.values):
        ax.text(
            val + 50,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center", fontsize=8, color=TEXT_C
        )

    plt.tight_layout()
    save_plot(fig, "market_basket_02_top_categories.png")


def plot_association_rules(
    rules: pd.DataFrame
) -> None:
    """
    Plot association rules scatter plot.
    Support vs Confidence colored by Lift.
    """
    if len(rules) == 0:
        logger.warning(
            "[WARN] No rules to plot"
        )
        return

    logger.info("-- Plotting association rules --")

    fig, axes = plt.subplots(
        1, 2, figsize=(20, 8), facecolor=BG
    )
    fig.suptitle(
        "ASSOCIATION RULES ANALYSIS",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    # Plot 1: Support vs Confidence scatter
    ax = axes[0]
    scatter = ax.scatter(
        rules["support"],
        rules["confidence"],
        c=rules["lift"],
        cmap="RdYlGn",
        s=rules["lift"] * 20,
        alpha=0.7,
        edgecolors="none",
    )
    plt.colorbar(scatter, ax=ax, label="Lift")
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title(
        "Support vs Confidence\n(color & size = lift)"
    )
    ax.grid(alpha=0.3)

    # Plot 2: Top 15 rules by lift
    ax = axes[1]
    top_rules = rules.head(15)
    rule_colors = [
        GREEN  if str(s) == "Strong"
        else ORANGE if str(s) == "Moderate"
        else BLUE
        for s in top_rules["rule_strength"]
    ]
    bars = ax.barh(
        range(len(top_rules)),
        top_rules["lift"],
        color=rule_colors, edgecolor="none", alpha=0.85
    )
    ax.set_yticks(range(len(top_rules)))
    ax.set_yticklabels(
        [
            f"{row['antecedents_str'][:20]} ->\n"
            f"{row['consequents_str'][:20]}"
            for _, row in top_rules.iterrows()
        ],
        fontsize=7
    )
    ax.set_xlabel("Lift")
    ax.set_title("Top 15 Rules by Lift")
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GREEN,  label="Strong  (lift > 2.5)"),
        Patch(facecolor=ORANGE, label="Moderate (lift 1.5-2.5)"),
        Patch(facecolor=BLUE,   label="Weak    (lift < 1.5)"),
    ]
    ax.legend(
        handles=legend_elements,
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=8
    )

    plt.tight_layout()
    save_plot(fig, "market_basket_03_rules.png")


def plot_frequent_itemsets(
    frequent_itemsets: pd.DataFrame
) -> None:
    """
    Plot frequent itemsets by support and size.
    """
    if len(frequent_itemsets) == 0:
        logger.warning(
            "[WARN] No itemsets to plot"
        )
        return

    logger.info("-- Plotting frequent itemsets --")

    frequent_itemsets["itemset_size"] = (
        frequent_itemsets["itemsets"].apply(len)
    )
    frequent_itemsets["itemset_str"] = (
        frequent_itemsets["itemsets"].apply(
            lambda x: ", ".join(list(x))
        )
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(20, 8), facecolor=BG
    )
    fig.suptitle(
        "FREQUENT ITEMSETS ANALYSIS",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    # Plot 1: Itemset size distribution
    ax = axes[0]
    size_counts = frequent_itemsets[
        "itemset_size"
    ].value_counts().sort_index()
    bars = ax.bar(
        size_counts.index.astype(str),
        size_counts.values,
        color=[BLUE, GREEN, ORANGE][:len(size_counts)],
        edgecolor="none", alpha=0.85
    )
    ax.set_xlabel("Itemset Size")
    ax.set_ylabel("Number of Itemsets")
    ax.set_title("Frequent Itemsets by Size")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, size_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.5,
            f"{val:,}",
            ha="center", fontsize=10,
            fontweight="bold", color=TEXT_C
        )

    # Plot 2: Top 20 itemsets by support
    ax = axes[1]
    top_items = frequent_itemsets.nlargest(
        20, "support"
    )
    item_colors = [
        BLUE if s == 1 else GREEN if s == 2 else ORANGE
        for s in top_items["itemset_size"]
    ]
    bars = ax.barh(
        range(len(top_items)),
        top_items["support"],
        color=item_colors, edgecolor="none", alpha=0.85
    )
    ax.set_yticks(range(len(top_items)))
    ax.set_yticklabels(
        [
            s[:35] for s in top_items["itemset_str"]
        ],
        fontsize=7
    )
    ax.set_xlabel("Support")
    ax.set_title("Top 20 Frequent Itemsets by Support")
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BLUE,   label="Single Item"),
        Patch(facecolor=GREEN,  label="2-Item Set"),
        Patch(facecolor=ORANGE, label="3-Item Set"),
    ]
    ax.legend(
        handles=legend_elements,
        facecolor=CARD_BG, labelcolor=TEXT_C,
        edgecolor=GRID_C, fontsize=8
    )

    plt.tight_layout()
    save_plot(fig, "market_basket_04_itemsets.png")


def plot_category_network(
    rules: pd.DataFrame
) -> None:
    """
    Plot category association network.
    Shows which categories connect to each other.
    """
    if len(rules) == 0:
        logger.warning(
            "[WARN] No rules for network plot"
        )
        return

    logger.info("-- Plotting category network --")

    fig, ax = plt.subplots(
        figsize=(14, 10), facecolor=BG
    )
    fig.suptitle(
        "CATEGORY ASSOCIATION NETWORK",
        fontsize=16, fontweight="bold", color=TEXT_C
    )

    # Get top 20 rules
    top_rules = rules.head(20)

    # Collect unique categories
    all_cats = set()
    for _, row in top_rules.iterrows():
        for cat in list(row["antecedents"]):
            all_cats.add(cat)
        for cat in list(row["consequents"]):
            all_cats.add(cat)
    all_cats = list(all_cats)

    # Position nodes in circle
    n      = len(all_cats)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos    = {
        cat: (np.cos(a), np.sin(a))
        for cat, a in zip(all_cats, angles)
    }

    # Draw edges
    for _, row in top_rules.iterrows():
        for ant in list(row["antecedents"]):
            for con in list(row["consequents"]):
                if ant in pos and con in pos:
                    x_vals = [pos[ant][0], pos[con][0]]
                    y_vals = [pos[ant][1], pos[con][1]]
                    alpha  = min(0.8, row["lift"] / 5)
                    width  = min(3.0, row["lift"] / 2)
                    ax.plot(
                        x_vals, y_vals,
                        color=BLUE,
                        alpha=alpha,
                        linewidth=width,
                        zorder=1
                    )

    # Draw nodes
    for i, (cat, (x, y)) in enumerate(pos.items()):
        color = COLORS[i % len(COLORS)]
        ax.scatter(
            x, y, s=200,
            color=color, zorder=3,
            edgecolors="white", linewidth=1
        )
        ax.text(
            x * 1.15, y * 1.15,
            cat[:15],
            ha="center", va="center",
            fontsize=7, color=TEXT_C,
            fontweight="bold"
        )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")
    ax.set_title(
        "Category Associations (line thickness = lift strength)",
        color=TEXT_C, pad=10
    )

    plt.tight_layout()
    save_plot(fig, "market_basket_05_network.png")


# ════════════════════════════════════════════════════════════
# MAIN MARKET BASKET FUNCTION
# ════════════════════════════════════════════════════════════

def run_market_basket_analysis() -> tuple:
    """
    Run complete market basket analysis pipeline.

    Returns:
        tuple: (rules, frequent_itemsets, co_occur)
    """
    logger.info("=" * 55)
    logger.info("MARKET BASKET ANALYSIS STARTED")
    logger.info("=" * 55)

    total_start = time.time()

    # Step 1: Load data
    df = load_data()

    # Step 2: Build transaction data
    basket_binary, multi_item, transactions = (
        build_transaction_data(df)
    )

    # Step 3: Co-occurrence analysis
    co_occur = calculate_co_occurrence(df)

    # Step 4: Run Apriori
    frequent_itemsets, rules = run_apriori(
        basket_binary,
        min_support    = 0.01,
        min_confidence = 0.1,
        min_lift       = 1.0,
    )

    # Step 5: Analyze rules
    if len(rules) > 0:
        rules = analyze_rules(rules)

    # Step 6: Visualize
    plot_co_occurrence_heatmap(co_occur)
    plot_top_categories(basket_binary)
    plot_association_rules(rules)
    plot_frequent_itemsets(frequent_itemsets)
    plot_category_network(rules)

    # Save results
    reports_dir = PLOTS_DIR.parent / "reports"
    if len(rules) > 0:
        rules.to_csv(
            reports_dir / "market_basket_rules.csv",
            index=False
        )
    if len(frequent_itemsets) > 0:
        frequent_itemsets["itemsets"] = (
            frequent_itemsets["itemsets"].apply(
                lambda x: ", ".join(list(x))
            )
        )
        frequent_itemsets.to_csv(
            reports_dir / "frequent_itemsets.csv",
            index=False
        )

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 55)
    print("   MARKET BASKET ANALYSIS SUMMARY")
    print("=" * 55)
    print(f"  Total Orders      : {len(basket_binary):,}")
    print(f"  Total Categories  : {len(basket_binary.columns):,}")
    print(f"  Frequent Itemsets : {len(frequent_itemsets):,}")
    print(f"  Association Rules : {len(rules):,}")
    print(f"  Plots Generated   : 5")
    print(f"  Time              : {total_elapsed:.2f}s")

    if len(rules) > 0:
        print("-" * 55)
        print("  Top 5 Association Rules:")
        print("-" * 55)
        for i, (_, row) in enumerate(
            rules.head(5).iterrows()
        ):
            print(
                f"  {i+1}. {row['antecedents_str'][:20]} "
                f"-> {row['consequents_str'][:20]}"
            )
            print(
                f"     Support={row['support']:.4f} "
                f"Confidence={row['confidence']:.4f} "
                f"Lift={row['lift']:.4f}"
            )

    print("=" * 55 + "\n")

    return rules, frequent_itemsets, co_occur


# ── Run directly ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    rules, frequent_itemsets, co_occur = (
        run_market_basket_analysis()
    )