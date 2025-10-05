import sqlite3
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from datetime import date


# for loading data
def load_data():
    if not os.path.exists("sales_data.csv"):
        pd.DataFrame(columns=["date", "product", "sales"]).to_csv("sales_data.csv", index=False)
    df = pd.read_csv("sales_data.csv")
    con = sqlite3.connect("sales.db")
    df.to_sql("sales", con, if_exists="replace", index=False)
    con.close()
    return df

data = load_data()

# Stramlit frontend 
st.title("🛒 AI-Powered Supermart Inventory Predictor")

# Sidebar to add new product sales record 
st.sidebar.header("🆕 Add New/Existing Product Sales Record")

with st.sidebar.form("add_product_form", clear_on_submit=True):
    new_product = st.text_input("Product Name")
    new_date = st.date_input("Sales Date")
    new_sales = st.number_input("Sales Quantity", min_value=1, step=1)
    add_btn = st.form_submit_button("➕ Add Record")

    if add_btn:
        if new_product.strip() == "" or new_sales <= 0:
            st.sidebar.error("❌ Please enter a valid product name or positive sales quantity.")
        else:
            # Append new entry
            new_entry = pd.DataFrame({
                "date": [new_date],
                "product": [new_product.strip()],
                "sales": [new_sales]
            })

            # Append to CSV
            existing_df = pd.read_csv("sales_data.csv")
            updated_df = pd.concat([existing_df, new_entry], ignore_index=True)
            updated_df.to_csv("sales_data.csv", index=False)

            # Append to DB
            with sqlite3.connect("sales.db") as con:
                new_entry.to_sql("sales", con, if_exists="append", index=False)

            st.sidebar.success(f"✅Record for {new_product} with Sales ({new_sales}) added successfully!")

            # Refresh the data
            data = load_data()

items = sorted(data["product"].dropna().astype(str).unique())
selectedItem = st.selectbox("Select a Product ", ["-- Choose Here --"] +  list(items))
if selectedItem != "-- Choose Here --":

    col1, col2, col3 = st.columns([2,4,1])
    with col2:
        imagePath= f"images/{selectedItem.lower()}.jpg" 
        if os.path.exists(imagePath):
            st.image(imagePath, caption="Product Image", width=250)
        else:
             st.warning("⚠️ Sorry, Product Image is not available.")

    # selected product all records
    selectedItemData = data[data["product"] == selectedItem].reset_index(drop=True)

    st.subheader("📊 Sales History for "+ selectedItem)
    st.line_chart(selectedItemData["sales"])
    st.subheader(f"All Sales data for {selectedItem}")
    st.dataframe(selectedItemData)

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
        total = int(forecast.sum())
        if (total <= 0):
            st.success(f"📦 Recommended Stock of {selectedItem} for next {futureDays} Days: No more Product is in Demand. Dont buy it")
        else:
            st.success(f"📦 Recommended Stock of {selectedItem} for next {futureDays} Days: {total} Units.")

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
            total = int(forecast.sum())
            if(total <= 0):
                st.success(f"📦 Recommended Stock of {selectedItem} for next {futureDays} Days: No more Product is in Demand. Dont buy it")
            else:
                st.success(f"📦 Recommended stock of {selectedItem} for next {futureDays} days: {total} units")
        except Exception as e:
            st.error(f"ARIMA model failed: {e}")
    else:
        st.write("WRong Model Selection")


        