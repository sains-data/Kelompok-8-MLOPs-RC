# src/predict.py
import mlflow.sklearn
import pandas as pd

def load_model(path="models/distraction_model"):
    return mlflow.sklearn.load_model(path)

def predict(model, input_data: dict):
    df = pd.DataFrame([input_data])
    result = model.predict(df)[0]
    return result
