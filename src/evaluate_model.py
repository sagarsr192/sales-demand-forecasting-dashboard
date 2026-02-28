# src/evaluate_model.py

import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model(data_path, model_path):
    df = pd.read_csv(data_path)

    features = ['Year', 'Month', 'Day', 'DayOfWeek']
    X = df[features]
    y = df['Sales']

    split_index = int(len(df) * 0.8)
    X_test = X[split_index:]
    y_test = y[split_index:]

    model = joblib.load(model_path)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("Model Evaluation Results:")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")

if __name__ == "__main__":
    evaluate_model(
        "data/processed/featured_sales_data.csv",
        "models/sales_forecast_model.pkl"
    )