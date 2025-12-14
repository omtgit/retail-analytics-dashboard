import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Retail Analytics Dashboard", layout="wide")

st.title("Retail Analytics & AI-Powered Sales Forecasting")

df = pd.read_csv("Retail_Sales_Data_Unlox.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.to_period('M')

st.metric("Total Revenue", f"{df['Revenue'].sum():,.0f}")

st.subheader("Monthly Revenue Trend")
monthly_revenue = df.groupby('Month')['Revenue'].sum().to_timestamp()
st.line_chart(monthly_revenue)

st.subheader("Revenue by Product Category")
st.bar_chart(df.groupby('Product_Category')['Revenue'].sum())

st.subheader("Revenue Forecast (Next 6 Months)")
model = SARIMAX(monthly_revenue, order=(1,1,1), seasonal_order=(1,1,1,12))
results = model.fit(disp=False)
forecast = results.forecast(6)

fig, ax = plt.subplots()
ax.plot(monthly_revenue, label="Actual")
ax.plot(forecast, label="Forecast")
ax.legend()
st.pyplot(fig)
