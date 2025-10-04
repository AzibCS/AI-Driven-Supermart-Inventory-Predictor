import sqlite3
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
#import pmdarima as pm


# for loading data
def load_data():
    df = pd.read_csv("sales_data.csv")
    con = sqlite3.connect("sales.db")
    df.to_sql("sales", con, if_exists="replace", index=False)
    con.close()
    return df

data = load_data()

# Stramlit frontend 
st.title("🛒 AI-Powered Supermart Inventory Predictor")

items = data["product"].unique()
selectedItem = st.selectbox("Select a Product ", ["-- Choose Here --"] +  list(items))
if selectedItem != "-- Choose Here --":

    col1, col2, col3 = st.columns([2,4,1])
    with col2:
        st.image(f"images/{selectedItem.lower()}.jpg", caption="User Choice", width=250)

    # selected product all records
    selectedItemData = data[data["product"] == selectedItem].reset_index(drop=True)

    st.subheader("📊 Sales History for "+ selectedItem)
    st.line_chart(selectedItemData["sales"])

    # Regression Model x=time steps, y=Target(sales values)
    x = np.arange(len(selectedItemData)).reshape(-1, 1)
    y = selectedItemData["sales"].values

    futureDays = st.number_input("Enter days to predict future forecast: ", min_value=2, max_value=30, step=1)
    modeltype = st.radio("\nChoose Forecast Model: ", ["Linear Regression 📈🔢", "ARIMA ⏳🔮"])
    
    if modeltype.startswith("Linear Regression"):
        model = LinearRegression()
        model.fit(x, y)

        # future forecasting
        future_X = np.arange(len(selectedItemData), len(selectedItemData) + futureDays).reshape(-1,1)
        forecast = model.predict(future_X)

        # --- Plot forecast ---
        fig, ax = plt.subplots()
        ax.plot(range(len(y)), y, label="Past History")
        ax.plot(range(len(y)-1, len(y) + futureDays), np.concatenate(([y[-1]], forecast)), label="Future Forecast", linestyle="--")
        ax.set_title(f"Linear Regression Demand Forecast of {selectedItem}")
        ax.legend()
        st.pyplot(fig)

        st.success(f"📦 Recommended Stock of {selectedItem} for next {futureDays} Days: {int(forecast.sum())} Units.")

    elif modeltype.startswith("ARIMA"):
        try:
            # Fit ARIMA model (p,d,q) = (2,1,2) is use for a common start
            model = sm.tsa.ARIMA(y, order=(2,1,2))
            #model = pm.auto_arima(y, seasonal=False, stepwise=True, suppress_warnings=True)
            model_fit = model.fit()

            # Forecast for next N days
            forecast = model_fit.forecast(steps=futureDays)

            # --- Plot ARIMA Forecast ---
            fig, ax = plt.subplots()
            ax.plot(range(len(y)), y, label="History")
            ax.plot(range(len(y)-1, len(y) + futureDays), np.concatenate(([y[-1]], forecast)), label="ARIMA Forecast", linestyle="--")
            ax.set_title(f"ARIMA Demand Forecast for {selectedItem}")
            ax.legend()
            st.pyplot(fig)

            st.success(f"📦 Recommended stock of {selectedItem} for next {futureDays} days: {int(forecast.sum())} units")
        except Exception as e:
            st.error(f"ARIMA model failed: {e}")
    else:
        st.write("WRong Model Selection")


        