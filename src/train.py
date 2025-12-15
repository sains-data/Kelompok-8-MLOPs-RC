# src/train.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocess import load_and_preprocess
import os

def train_model():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # MLflow tracking
    mlflow.set_experiment("distraction-classification")

    with mlflow.start_run():

        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42
        }

        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)

        # Save model
        os.makedirs("models", exist_ok=True)
        mlflow.sklearn.save_model(model, "models/distraction_model")

        print(f"Model trained. Accuracy = {acc}")

if __name__ == "__main__":
    train_model()
