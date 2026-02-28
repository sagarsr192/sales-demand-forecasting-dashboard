# src/forecast.py

import pandas as pd
import joblib
from datetime import timedelta

def forecast_future(data_path, model_path, days=30):
    df = pd.read_csv(data_path)
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    model = joblib.load(model_path)

    last_date = df['Order Date'].max()

    future_dates = []
    for i in range(1, days + 1):
        future_dates.append(last_date + timedelta(days=i))

    future_df = pd.DataFrame({'Order Date': future_dates})
    future_df['Year'] = future_df['Order Date'].dt.year
    future_df['Month'] = future_df['Order Date'].dt.month
    future_df['Day'] = future_df['Order Date'].dt.day
    future_df['DayOfWeek'] = future_df['Order Date'].dt.dayofweek

    features = ['Year', 'Month', 'Day', 'DayOfWeek']
    future_df['Predicted Sales'] = model.predict(future_df[features])

    future_df.to_csv("data/processed/future_forecast.csv", index=False)

    print("Next 30 Days Forecast:")
    print(future_df[['Order Date', 'Predicted Sales']])

if __name__ == "__main__":
    forecast_future(
        "data/processed/featured_sales_data.csv",
        "models/sales_forecast_model.pkl"
    )