# Retail Analytics — Brazilian E-Commerce (Olist)

End-to-end Business Intelligence and Data Analytics project built on the Olist Brazilian E-Commerce dataset. Covers ETL pipeline, SQL analysis, EDA, advanced analytics, machine learning and automated reporting.

---

## Project Structure

    retail_analytics/
    ├── analytics/
    │   ├── churn_prediction.py
    │   ├── cohort_analysis.py
    │   ├── forecasting.py
    │   ├── market_basket.py
    │   ├── rfm_analysis.py
    │   └── seller_scorecard.py
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── warehouse/
    ├── database/
    │   └── retail.db
    ├── etl/
    │   ├── extract.py
    │   ├── transform.py
    │   ├── load.py
    │   └── pipeline.py
    ├── notebooks/
    │   └── 01_exploration.ipynb
    ├── outputs/
    │   ├── plots/
    │   └── reports/
    ├── reports/
    │   ├── excel_report.py
    │   └── pdf_report.py
    ├── src/
    │   ├── config.py
    │   ├── sql_queries.py
    │   └── visualization.py
    ├── main.py
    └── requirements.txt

---

## Dataset

Brazilian E-Commerce Public Dataset by Olist

| File | Rows | Description |
|---|---|---|
| olist_orders_dataset.csv | 99,441 | Order information |
| olist_order_items_dataset.csv | 112,650 | Items per order |
| olist_order_payments_dataset.csv | 103,886 | Payment details |
| olist_order_reviews_dataset.csv | 99,224 | Customer reviews |
| olist_customers_dataset.csv | 99,441 | Customer info |
| olist_products_dataset.csv | 32,951 | Product details |
| olist_sellers_dataset.csv | 3,095 | Seller info |
| olist_geolocation_dataset.csv | 1,000,163 | Location data |
| product_category_name_translation.csv | 71 | Category names |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Database | SQLite + SQLAlchemy |
| ETL | Custom Python Pipeline |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Forecasting | Prophet, Moving Average |
| Association Rules | MLxtend Apriori |
| Reporting | ReportLab PDF, OpenPyXL Excel |
| Logging | Loguru |
| Environment | VS Code, Jupyter Notebook |

---

## Setup

**1. Clone the repository**

    git clone https://github.com/yourusername/retail_analytics.git
    cd retail_analytics

**2. Create virtual environment**

    py -3.11 -m venv venv
    venv\Scripts\activate

**3. Install dependencies**

    pip install -r requirements.txt

**4. Download dataset**

Download from Kaggle and place all 9 CSV files in data/raw/

    https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

**5. Run ETL pipeline**

    python main.py --steps etl

**6. Run full pipeline**

    python main.py --skip-etl

---

## How to Run

**Run everything**

    python main.py

**Skip ETL if database already exists**

    python main.py --skip-etl

**Run specific steps**

    python main.py --steps rfm cohort forecast

**Generate reports only**

    python main.py --reports-only

**Available steps**

| Step | Description |
|---|---|
| etl | ETL pipeline |
| sql | SQL analysis |
| eda | Exploratory analysis |
| rfm | RFM segmentation |
| cohort | Cohort analysis |
| forecast | Revenue forecasting |
| seller | Seller scorecard |
| churn | Churn prediction |
| basket | Market basket |
| viz | Visualizations |
| pdf | PDF report |
| excel | Excel report |
| all | Everything |

---

## ETL Pipeline

    RAW CSVs -> EXTRACT -> TRANSFORM -> LOAD -> SQLITE DB

**Extract**
- Loads all 9 CSV files
- Validates schema and data types
- Checks null values and duplicates
- Logs extraction report

**Transform**
- Parses date columns
- Fills missing values
- Adds engineered columns
- Builds master dataset
- Creates star schema

**Load**
- Loads to SQLite database
- Creates indexes for performance
- Creates SQL views
- Verifies row counts

**Star Schema**

    fact_orders -> dim_customer
                -> dim_product
                -> dim_seller
                -> dim_date

---

## SQL Analysis

25 SQL queries covering:

| Category | Queries |
|---|---|
| Business Overview | KPIs, totals |
| Revenue Analysis | Monthly, quarterly, by category, by state |
| Customer Analysis | Segmentation, top customers, regions |
| Product Analysis | Top products, category performance |
| Seller Analysis | Performance, ranking, state rank |
| Order Analysis | Status, weekday, hourly patterns |
| Payment Analysis | Payment types, installments |
| Review Analysis | Score distribution, by category |
| Advanced SQL | RFM with CTEs, Cohort, Running totals, Window functions |

---

## Analytics Modules

**RFM Analysis**
- Recency, Frequency, Monetary scoring
- 8 customer segments
- Scatter plots and heatmaps

**Cohort Analysis**
- Customer retention by acquisition month
- Revenue cohort analysis
- Retention curves

**Forecasting**
- Moving average 30 days
- Linear trend 3 months
- Prophet forecast 90 days
- Seasonal pattern analysis

**Seller Scorecard**
- Composite scoring Revenue 30% Review 30% Delivery 20% Orders 20%
- 4 tiers Bronze Silver Gold Platinum
- Radar charts for top sellers

**Churn Prediction**
- 4 ML models Logistic Regression Decision Tree Random Forest Gradient Boosting
- Feature importance analysis
- Risk segmentation

**Market Basket Analysis**
- Apriori algorithm
- Association rules Support Confidence Lift
- Category network visualization

---

## Key Business Insights

**Revenue**
- Total revenue R$19.7M over 2 years
- Peak November 2017 Black Friday
- Top category Health and Beauty
- Top state Sao Paulo 40% of revenue

**Customers**
- 96,478 unique customers
- 97% are one-time buyers
- Average order value R$179.47
- Average review score 4.08 out of 5

**Operations**
- Average delivery 12 days
- Late delivery rate 8%
- Credit card 74% of payments
- Average installments 3.7 months

**Sellers**
- 3,095 total sellers
- 9 Platinum sellers
- Bronze sellers 98% late delivery rate
- Gold sellers best review score 4.87

---

## Outputs

**Plots saved to outputs/plots/**
- Business overview, monthly revenue, category, state
- Customer segmentation, order status, payment, review
- RFM distributions, scores, segments, scatter, heatmap
- Cohort heatmaps, retention curves, revenue cohort
- Forecasting time series, MA forecast, linear, seasonal, Prophet
- Seller overview, top sellers, tier analysis, radar, region
- Churn overview, model comparison, feature importance, probability
- Market basket co-occurrence, categories, rules, itemsets, network
- Executive dashboard, sales dashboard, customer dashboard, operations dashboard

**Reports saved to outputs/reports/**
- retail_analytics_report.pdf
- retail_analytics_report.xlsx with 10 sheets
- rfm_results.csv and rfm_segments.csv
- cohort_retention.csv and cohort_metrics.csv
- forecast_ma.csv and forecast_linear.csv
- seller_scorecard.csv and seller_tier_summary.csv
- churn_predictions.csv and churn_feature_importance.csv
- market_basket_rules.csv and frequent_itemsets.csv

---

## Skills Demonstrated

| Skill | Tools |
|---|---|
| ETL Pipeline | Python SQLite SQLAlchemy |
| SQL | CTEs Window Functions Joins Views Indexes |
| Data Cleaning | Pandas NumPy |
| Data Visualization | Matplotlib Seaborn Plotly |
| Customer Analytics | RFM Cohort Segmentation |
| Machine Learning | Scikit-learn Churn Prediction |
| Time Series | Prophet Moving Average Linear Trend |
| Association Rules | MLxtend Apriori |
| Reporting | ReportLab OpenPyXL |
| Logging | Loguru |
| Project Structure | Modular Python VS Code |

---

## Push to GitHub

    git init
    git add .
    git commit -m "Retail Analytics - Complete BI Project"
    git branch -M main
    git remote add origin https://github.com/aka261/retail_analytics.git
    git push -u origin main

---

## Author

**Akash Tirupapuliyur Srikanth**

- LinkedIn: linkedin.com/in/yourprofile
- GitHub: github.com/aka261
- Email: tsakash999@gmail.com

---

## License

MIT License — free to use for portfolio and learning purposes.