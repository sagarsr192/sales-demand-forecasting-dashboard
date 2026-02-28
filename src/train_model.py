# src/train_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_model(input_path, model_path):
    df = pd.read_csv(input_path)

    features = ['Year', 'Month', 'Day', 'DayOfWeek']
    X = df[features]
    y = df['Sales']

    split_index = int(len(df) * 0.8)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

    print("Model trained & saved successfully!")

if __name__ == "__main__":
    train_model(
        "data/processed/featured_sales_data.csv",
        "models/sales_forecast_model.pkl"
    )