# src/feature_engineering.py

import pandas as pd

def create_time_features(input_path, output_path):
    df = pd.read_csv(input_path)
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Day'] = df['Order Date'].dt.day
    df['DayOfWeek'] = df['Order Date'].dt.dayofweek

    df.to_csv(output_path, index=False)
    print("Time features created successfully!")

if __name__ == "__main__":
    create_time_features(
        "data/processed/cleaned_sales_data.csv",
        "data/processed/featured_sales_data.csv"
    )