
# ENTERPRISE SALES INTELLIGENCE PLATFORM
# Sales Forecasting Dashboard
# Dataset: UCI Online Retail Dataset

import streamlit as st
import pandas as pd
import numpy as np
import os

import plotly.express as px
import plotly.graph_objects as go

from prophet import Prophet

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# PAGE CONFIGURATION


st.set_page_config(
    page_title="Enterprise Sales Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)


# STYLING OF DASHBOARD

st.markdown(
"""
<style>

.main {
    background-color: #0e1117;
}

h1, h2, h3 {
    color: white;
}

</style>
""",
unsafe_allow_html=True
)

# TITLE DESIGNING


st.title("🏢 Enterprise Sales Intelligence Platform")

st.caption(
"""
Transforming historical retail transactions into business insights,
sales forecasts, and planning recommendations.
"""
)

# DATA LOADING + CLEANING

DATA_PATH = "Online Retail.xlsx"

@st.cache_data
def load_data(path):

    # Load dataset
    df = pd.read_excel(path)

    # DATA CLEANING

    # Remove missing values
    df = df.dropna()

    # Remove invalid quantities
    # (negative values represent returns)
    df = df[df["Quantity"] > 0]


    # Remove invalid prices
    df = df[df["UnitPrice"] > 0]


    # Convert date format
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"]
    )


    # Create Sales feature
    df["Sales"] = (
        df["Quantity"] *
        df["UnitPrice"]
    )


    # Remove cancelled transactions
    df = df[
        ~df["InvoiceNo"]
        .astype(str)
        .str.startswith("C")
    ]


    return df



# ============================================================
# DATA SOURCE
# ============================================================

st.sidebar.header("📂 Data Source")


use_upload = st.sidebar.checkbox(
    "Upload custom dataset",
    value=False
)



if use_upload:

    file = st.sidebar.file_uploader(
        "Upload Excel File",
        type=["xlsx"]
    )


    if file:

        df = load_data(file)

    else:

        st.warning(
            "Please upload a dataset"
        )

        st.stop()



else:

    if os.path.exists(DATA_PATH):

        df = load_data(DATA_PATH)

    else:

        st.error(
            "Online Retail.xlsx not found in project folder"
        )

        st.stop()


# DAILY SALES AGGREGATION

daily = (
    df.groupby( df["InvoiceDate"].dt.date )["Sales"].sum().reset_index()
)

daily.columns = [
    "Date",
    "Sales"
]


daily["Date"] = pd.to_datetime(
    daily["Date"]
)


# FORECAST MODEL


prophet_df = daily.rename(
    columns={
        "Date":"ds",
        "Sales":"y"
    }
)

model = Prophet()

model.fit(
    prophet_df
)


future = model.make_future_dataframe(
    periods=90
)


forecast = model.predict(
    future
)


growth = (
    forecast.tail(90)["yhat"].mean()
    -
    prophet_df["y"].mean()
)

growth_pct = (
    growth /
    prophet_df["y"].mean()
) * 100



# EXECUTIVE OVERVIEW

st.markdown("## 📌 Executive Overview")



if growth_pct > 5:

    recommendation = \
    (
        "📈 Demand is increasing. "
        "Consider preparing additional inventory."
    )


elif growth_pct < -5:

    recommendation = (
        "📉 Demand is decreasing. "
        "Review inventory levels."
    )


else:

    recommendation = (
        "📊 Demand is stable. "
        "Maintain current strategy."
    )



st.info(
f"""
This dashboard analyses historical retail sales data and
uses forecasting models to support business planning.

**Current Business Outlook**

- {recommendation}
- Expected Forecast Change: {growth_pct:.2f}%
- Forecast Horizon: Next 90 Days
- Main Purpose: Support inventory, staffing and revenue decisions
"""
)


# KPI DASHBOARD


st.subheader(
"📊 Business Performance Overview"
)

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Total Revenue",
    f"R {df['Sales'].sum():,.0f}"
)


k2.metric(
    "Average Daily Sales",
    f"R {daily['Sales'].mean():,.0f}"
)



k3.metric(
    "Highest Sales Day",
    f"R {daily['Sales'].max():,.0f}"
)



k4.metric(
    "Transactions",
    f"{len(df):,}"
)


# DASHBOARD TABS

tab1, tab2, tab3, tab4, tab5 = st.tabs(
[
    "📈 Trends",
    "🛒 Products",
    "🔮 Forecast",
    "📉 Models",
    "📌 Business Insights"
]
)

# TAB 1 - TRENDS

with tab1:

    st.subheader(
        "Sales Behaviour Over Time"
    )


    c1, c2 = st.columns(2)

    with c1:


        fig = px.line(
            daily,
            x="Date",
            y="Sales",
            title="Revenue Trend"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        daily["Day"] = (
            daily["Date"]
            .dt.day_name()
        )


        weekday = (
            daily.groupby("Day")["Sales"]
            .mean()
            .reindex(
            [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
            ]
            )
        )


        fig = px.bar(
            x=weekday.index,
            y=weekday.values,
            title="Weekly Sales Pattern"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # TAB 2 - PRODUCT ANALYSIS


        with tab2:

            st.subheader(
                "🛒 Product Performance Analysis"
            )

            top_products = (
                df.groupby("Description")["Sales"]
                .sum()
                .nlargest(10)
            )

            fig = px.bar(
                top_products,
                orientation="h",
                title="Top 10 Products by Revenue"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.markdown(
                """
                ### Business Meaning
    
                This analysis identifies which products contribute the most
                revenue. Businesses can use this information to prioritise
                popular products and improve inventory planning.
                """
            )

            st.dataframe(top_products)

        # TAB 3 - FORECAST

        with tab3:

            st.subheader( "🔮 90-Day Sales Forecast" )

            fig = go.Figure()

            # Historical sales

            fig.add_trace(
                go.Scatter(
                    x=prophet_df["ds"],
                    y=prophet_df["y"],
                    name="Historical Sales"
                )
            )

            # Prediction

            fig.add_trace(
                go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat"],
                    name="Forecast"
                )
            )

            # Confidence upper

            fig.add_trace(
                go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat_upper"],
                    line=dict(width=0),
                    showlegend=False
                )
            )

            # Confidence lower

            fig.add_trace(
                go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat_lower"],
                    fill="tonexty",
                    name="Prediction Range"
                )
            )

            fig.update_layout(
                title="Future Demand Prediction",
                height=500
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.success(
                f"Expected Forecast Change: {growth_pct:.2f}%"
            )

            st.download_button(
                label="⬇ Download Forecast Data",
                data=forecast.to_csv(index=False),
                file_name="sales_forecast.csv",
                mime="text/csv"
            )

            st.markdown(
                """
                ### Business Meaning
    
                The forecasting model estimates future sales behaviour
                using historical trends and seasonality.
    
                Businesses can use these predictions to:
                - prepare stock levels
                - plan staffing requirements
                - estimate future revenue
                """
            )

            st.markdown(   "## 🔍 Automated Insights" )

            trend = daily["Sales"].diff().mean()

            if trend > 0:

                trend_message = ("Sales show an overall increasing trend.")

            else:

                trend_message = (
                    "Sales show stable or declining behaviour."
                )

            highest_day = (
                daily.loc[
                    daily["Sales"].idxmax(),
                    "Date"
                ]
            )

            st.write(
                f"📈 {trend_message}"
            )

            st.write(
                f"📅 Highest recorded sales occurred on: {highest_day.date()}"
            )

        # TAB 4 - MODEL COMPARISON

        with tab4:

            st.subheader(  "📉 Machine Learning Model Evaluation" )

            ml = daily.copy()

            # Time features

            ml["DayNum"] = (
                ml["Date"]
                .dt.dayofweek
            )

            ml["MonthNum"] = (
                ml["Date"]
                .dt.month
            )

            ml["Year"] = (
                ml["Date"]
                .dt.year
            )

            X = ml[
                [
                    "DayNum",
                    "MonthNum",
                    "Year"
                ]
            ]

            y = ml["Sales"]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                shuffle=False,
                test_size=0.2
            )

            # Linear Regression

            lr = LinearRegression()

            lr.fit(
                X_train,
                y_train
            )

            lr_predictions = lr.predict(
                X_test
            )

            # Random Forest

            rf = RandomForestRegressor(
                n_estimators=150,
                random_state=42
            )

            rf.fit(
                X_train,
                y_train
            )

            rf_predictions = rf.predict(
                X_test
            )


            def evaluate(name, actual, predicted):
                mae = mean_absolute_error(
                    actual,
                    predicted
                )

                rmse = np.sqrt(
                    mean_squared_error(
                        actual,
                        predicted
                    )
                )

                return [
                    name,
                    mae,
                    rmse
                ]


            results = pd.DataFrame(
                [
                    evaluate(
                        "Linear Regression",
                        y_test,
                        lr_predictions
                    ),

                    evaluate(
                        "Random Forest",
                        y_test,
                        rf_predictions
                    )
                ],
                columns=[
                    "Model",
                    "MAE",
                    "RMSE"
                ]
            )

            st.dataframe(
                results,
                use_container_width=True
            )

            st.markdown(
                """
                ### Business Meaning
    
                Different prediction approaches were tested and compared.
    
                Lower error values indicate a model that predicts sales
                more accurately.
                
                Therefore, we can see that Linear Regression is the better model
                because the Mean Absolute Error is closer to the actual sales values
                by 72 units compared to the Random Forest.
                The Root Mean Squared Error penalizes errors more heavily, so it's lower
                value shows that Linear Regression makes fewer major prediction blunders
                """
            )

        # TAB 5 - BUSINESS INSIGHTS / CONCLUSION

        with tab5:

            st.subheader( "📌 Final Business Insights" )

            st.markdown(
                """
                ## 🧠 Project Summary
    
                This platform converts historical retail transaction data
                into meaningful business intelligence.
    
                The system analyses past sales behaviour, identifies trends,
                predicts future demand, and provides recommendations that
                support business decision-making.
    
                ---
    
                ## 📊 Key Findings
    
    
                **Sales Trends**
    
                Historical sales patterns were analysed to understand how
                revenue changes over time.
    
    
                **Customer Demand Behaviour**
    
                Repeating sales patterns were identified, allowing the
                business to understand periods of higher and lower demand.
    
    
                **Product Performance**
    
                Top-performing products were identified to support better
                inventory decisions.
    
    
                **Future Forecasting**
    
                The forecasting model estimates future demand over the next
                90 days, helping businesses prepare for expected changes.
    
    
                ---
    
                ## 💼 Business Recommendations
    
    
                ✅ Use forecasts to prepare inventory before high-demand periods.
    
    
                ✅ Maintain availability of high-performing products.
    
    
                ✅ Reduce unnecessary stock for low-performing products.
    
    
                ✅ Continuously update the model with new sales data.
    
    
                ---
    
                ## 🚀 Final Conclusion
    
    
                This project demonstrates how data science can transform
                raw business data into a decision-support platform.
    
                By combining data cleaning, exploratory analysis, machine
                learning, and forecasting, businesses can make more informed
                decisions regarding inventory, operations, and future growth.
    
                """
            )

            st.markdown(
                "---"
            )

            st.subheader(
                "📌 System Summary"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Forecast Period",
                "90 Days"
            )

            c2.metric(
                "Expected Change",
                f"{growth_pct:.2f}%"
            )

            c3.metric(
                "Techniques Used",
                "Prophet + ML"
            )

        # FINAL TREND ANALYSIS

        st.markdown("---")

        st.subheader(  "📈 Demand Reality vs Long-Term Trend")

        rolling = daily.copy()

        rolling["RollingMean"] = (
            rolling["Sales"]
            .rolling(7)
            .mean()
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=rolling["Date"],
                y=rolling["Sales"],
                name="Actual Sales",
                opacity=0.4
            )
        )

        fig.add_trace(
            go.Scatter(
                x=rolling["Date"],
                y=rolling["RollingMean"],
                name="7-Day Moving Trend"
            )
        )

        fig.update_layout(
            title="Market Behaviour Signal Extraction",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # FOOTER


        st.markdown("---")

        st.caption(
            """
            📊 Enterprise Sales Intelligence Platform  
            """
        )