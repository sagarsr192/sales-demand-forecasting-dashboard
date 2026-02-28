import streamlit as st
import pandas as pd
import joblib
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")

# -----------------------
# Title Section
# -----------------------
st.markdown("""
# 📊 Sales Demand Forecasting Dashboard  
### Machine Learning Based Time-Series Prediction
""")

st.divider()

# -----------------------
# Load Data & Model
# -----------------------
df = pd.read_csv("data/processed/featured_sales_data.csv")
df['Order Date'] = pd.to_datetime(df['Order Date'])

model = joblib.load("models/sales_forecast_model.pkl")

# -----------------------
# Model Performance Section
# -----------------------
features = ['Year', 'Month', 'Day', 'DayOfWeek']
X = df[features]
y = df['Sales']

split_index = int(len(df) * 0.8)
X_test = X[split_index:]
y_test = y[split_index:]

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)
col1.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
col2.metric("RMSE (Root Mean Square Error)", f"{rmse:.2f}")

st.divider()

# -----------------------
# Historical Sales Chart
# -----------------------
st.subheader("📅 Historical Sales Trend")
st.line_chart(df.set_index("Order Date")["Sales"])

st.divider()

# -----------------------
# Forecast Section
# -----------------------
st.subheader("🔮 Future Sales Prediction")

days = st.slider("Select Forecast Days", min_value=7, max_value=90, value=30)

last_date = df['Order Date'].max()
future_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]

future_df = pd.DataFrame({'Order Date': future_dates})
future_df['Year'] = future_df['Order Date'].dt.year
future_df['Month'] = future_df['Order Date'].dt.month
future_df['Day'] = future_df['Order Date'].dt.day
future_df['DayOfWeek'] = future_df['Order Date'].dt.dayofweek

future_df['Predicted Sales'] = model.predict(future_df[features])

st.line_chart(future_df.set_index("Order Date")["Predicted Sales"])

st.dataframe(future_df)

# -----------------------
# Download Button
# -----------------------
st.download_button(
    label="📥 Download Forecast CSV",
    data=future_df.to_csv(index=False),
    file_name="future_sales_forecast.csv",
    mime="text/csv"
)

st.success("Forecast generated successfully!")