# src/visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import joblib

def plot_historical_sales():
    df = pd.read_csv("data/processed/cleaned_sales_data.csv")
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    plt.figure(figsize=(12,6))
    plt.plot(df['Order Date'], df['Sales'])
    plt.title("Historical Daily Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("visuals/historical_sales.png")
    plt.close()

def plot_actual_vs_predicted():
    df = pd.read_csv("data/processed/featured_sales_data.csv")
    model = joblib.load("models/sales_forecast_model.pkl")

    features = ['Year', 'Month', 'Day', 'DayOfWeek']
    X = df[features]
    y = df['Sales']

    split_index = int(len(df) * 0.8)
    X_test = X[split_index:]
    y_test = y[split_index:]

    predictions = model.predict(X_test)

    plt.figure(figsize=(12,6))
    plt.plot(y_test.values, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted Sales")
    plt.tight_layout()
    plt.savefig("visuals/actual_vs_predicted.png")
    plt.close()

if __name__ == "__main__":
    plot_historical_sales()
    plot_actual_vs_predicted()
    print("Visualizations saved successfully!")