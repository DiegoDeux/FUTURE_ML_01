import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from prophet import Prophet

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import os


# ============================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ============================================================

st.set_page_config(page_title="Enterprise Sales Intelligence", layout="wide",initial_sidebar_state="expanded")

# Dark style
st.markdown("""
<style>
.main { background-color: #0e1117; }
h1, h2, h3 { color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🏢 Enterprise Sales Intelligence Platform")


# ============================================================
# DATA LOADER
# ============================================================

DATA_PATH = "Online Retail.xlsx"


@st.cache_data
def load_data(path):

    df = pd.read_excel(path)

    df = df.dropna()
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Sales"] = df["Quantity"] * df["UnitPrice"]

    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    return df


# ============================================================
# DATA SOURCE
# ============================================================

st.sidebar.header("Data Source")

use_upload = st.sidebar.checkbox("Use custom dataset instead of default",value=False)

if use_upload:
    file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

    if file:
        df = load_data(file)
    else:
        st.stop()

else:
    if os.path.exists(DATA_PATH):
        df = load_data(DATA_PATH)
    else:
        st.error("Online Retail.xlsx not found in project folder")
        st.stop()


# ============================================================
# DAILY AGGREGATION
# ============================================================

daily = df.groupby(df["InvoiceDate"].dt.date)["Sales"].sum().reset_index()
daily.columns = ["Date", "Sales"]
daily["Date"] = pd.to_datetime(daily["Date"])


# ============================================================
# FORECAST (MUST RUN BEFORE EXEC SUMMARY)
# ============================================================

prophet_df = daily.rename(columns={"Date": "ds", "Sales": "y"})

model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

growth = forecast.tail(90)["yhat"].mean() - prophet_df["y"].mean()
growth_pct = (growth / prophet_df["y"].mean()) * 100


# ============================================================
# EXECUTIVE OVERVIEW (FIXED + CLEAN)
# ============================================================

st.markdown("## 📌 Executive Overview")

if growth_pct > 5:
    insight = "Demand is increasing — consider expanding inventory."
elif growth_pct < -5:
    insight = "Demand is declining — reduce stock exposure."
else:
    insight = "Demand is stable — maintain current strategy."

st.info(f"""
This dashboard summarises business performance using historical sales data.

- {insight}
- Forecast Growth: {growth_pct:.2f}%
- Purpose: Support planning for inventory, staffing, and revenue decisions
""")


# ============================================================
# KPI DASHBOARD
# ============================================================

st.subheader("📊 Executive Overview")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Revenue", f"R {df['Sales'].sum():,.0f}")
k2.metric("Avg Daily Sales", f"R {daily['Sales'].mean():,.0f}")
k3.metric("Peak Day", f"R {daily['Sales'].max():,.0f}")
k4.metric("Transactions", f"{len(df):,}")


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trends",
    "🛒 Products",
    "🔮 Forecast",
    "📉 Models",
    "📌 Conclusion"
])

# ============================================================
# TAB 1 - TRENDS
# ============================================================

with tab1:

    c1, c2 = st.columns(2)

    with c1:
        fig = px.line(daily, x="Date", y="Sales", title="Revenue Over Time")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        daily["Day"] = daily["Date"].dt.day_name()

        weekday = daily.groupby("Day")["Sales"].mean().reindex([
            "Monday","Tuesday","Wednesday","Thursday",
            "Friday","Saturday","Sunday"
        ])

        fig = px.bar(x=weekday.index, y=weekday.values, title="Weekly Seasonality")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 2 - PRODUCTS
# ============================================================

with tab2:

    top_products = df.groupby("Description")["Sales"].sum().nlargest(10)

    fig = px.bar(top_products, orientation="h", title="Top Products by Revenue")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(top_products)


# ============================================================
# TAB 3 - FORECAST
# ============================================================

with tab3:

    st.subheader("90-Day Forecast (Prophet)")

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], name="Actual"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))

    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        fill="tonexty",
        name="Confidence Band"
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.success(f"Expected Growth: {growth_pct:.2f}%")

    st.download_button(
        "Download Forecast Data",
        forecast.to_csv(index=False),
        "forecast.csv"
    )

    st.markdown("## 🔍 Key Insights")

    trend = daily["Sales"].diff().mean()

    st.write("Trend:",
             "Sales increasing" if trend > 0 else "Sales stable or declining")

    top_day = daily.loc[daily["Sales"].idxmax(), "Date"]
    st.write(f"Peak activity: **{top_day.date()}**")


# ============================================================
# TAB 4 - MODELS
# ============================================================

with tab4:

    ml = daily.copy()

    ml["DayNum"] = ml["Date"].dt.dayofweek
    ml["MonthNum"] = ml["Date"].dt.month
    ml["Year"] = ml["Date"].dt.year

    X = ml[["DayNum","MonthNum","Year"]]
    y = ml["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=150).fit(X_train, y_train)

    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)

    def eval_model(name, actual, pred):
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        return name, mae, rmse

    results = pd.DataFrame([
        eval_model("Linear Regression", y_test, lr_pred),
        eval_model("Random Forest", y_test, rf_pred)
    ], columns=["Model","MAE","RMSE"])

    st.dataframe(results)

    # ============================================================
    # TAB 5 - CONCLUSION
    # ============================================================

    with tab5:
        st.subheader("📌 Final Business Conclusion")

        st.markdown("""
        ## 🧠 Project Overview

        This platform transforms raw retail transaction data into meaningful
        business insights by analysing historical sales patterns and forecasting
        future demand.

        The system uses machine learning and time-series forecasting techniques
        to help businesses make informed decisions regarding inventory,
        planning, and future growth.
        """)

        st.markdown("---")

        st.markdown("""
        ## 📊 Key Findings

        ### 1. Sales Performance

        Historical sales data was analysed to identify overall revenue patterns
        and changes in customer demand over time.

        The dashboard highlights:
        - Revenue trends
        - Peak sales periods
        - Changes in demand behaviour


        ### 2. Customer Demand Patterns

        The analysis identified repeating patterns in sales behaviour.

        These patterns help businesses understand:
        - When demand increases
        - When demand decreases
        - How seasonal behaviour affects sales


        ### 3. Product Performance

        Product analysis identifies which items contribute the most revenue.

        This allows businesses to:
        - Prioritise popular products
        - Improve inventory decisions
        - Reduce unnecessary stock


        ### 4. Future Forecasting

        The forecasting model predicts expected sales for the next 90 days.

        This supports:
        - Inventory planning
        - Budget forecasting
        - Staffing decisions
        """)

        st.markdown("---")

        st.markdown("""
        ## 💼 Business Recommendations

        Based on the analysis, businesses should:

        ✅ Use forecasts to prepare inventory before periods of high demand

        ✅ Monitor top-performing products to maintain availability

        ✅ Avoid overstocking products with consistently lower demand

        ✅ Continuously update forecasts as new sales data becomes available

        """)

        st.markdown("---")

        st.markdown("""
        ## 🚀 Final Conclusion

        This project demonstrates how data science can transform historical
        transaction data into a practical decision-support system.

        By combining data cleaning, exploratory analysis, machine learning,
        and forecasting, businesses can move from reactive decision-making
        towards proactive planning.

        The result is a platform that helps organisations understand their
        current performance, predict future demand, and make data-driven
        decisions.
        """)

        # Final metrics summary

        st.markdown("## 📌 Model Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Forecast Period",
            "90 Days"
        )

        c2.metric(
            "Forecast Growth",
            f"{growth_pct:.2f}%"
        )

        c3.metric(
            "Models Used",
            "Prophet + ML"
        )


# ============================================================
# FOOTER INSIGHTS
# ============================================================

st.markdown("---")

rolling = daily.copy()
rolling["RollingMean"] = rolling["Sales"].rolling(7).mean()

fig = go.Figure()

fig.add_trace(go.Scatter(x=rolling["Date"], y=rolling["Sales"], name="Actual", opacity=0.4))
fig.add_trace(go.Scatter(x=rolling["Date"], y=rolling["RollingMean"], name="Trend"))

fig.update_layout(title="Market Behaviour Signal Extraction")

st.plotly_chart(fig, use_container_width=True)