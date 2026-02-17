import argparse
import json
import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
import mlflow
import mlflow.sklearn

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def prepare_data():
    """
    Placeholder data preparation routine.
    In a real project this would perform cleaning, feature engineering and save processed data.
    """
    raw_path = os.path.join(DATA_DIR, "raw", "loans.csv")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    # For demonstration, if no raw data exists, create synthetic data
    if not os.path.exists(raw_path):
        os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
        df = pd.DataFrame({
            "credit_score": [700, 680, 720, 650],
            "income": [50000, 60000, 55000, 52000],
            "loan_amount": [20000, 15000, 30000, 10000],
            "previous_defaults": [0, 1, 0, 2],
            "defaulted": [0, 1, 0, 1],
        })
        df.to_csv(raw_path, index=False)
    df = pd.read_csv(raw_path)
    # simple processing: no changes
    df.to_csv(os.path.join(PROCESSED_DIR, "loans_processed.csv"), index=False)
    print("Data prepared and saved to processed directory")

def train_model(config_path):
    """
    Train a simple model and log to MLflow.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    processed_file = os.path.join(PROCESSED_DIR, "loans_processed.csv")
    df = pd.read_csv(processed_file)
    X = df[["credit_score", "income", "loan_amount", "previous_defaults"]]
    y = df["defaulted"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.get("random_state", 42)
    )
    model = GradientBoostingClassifier(**config.get("model_params", {}))
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    mlflow.set_experiment("credit_risk")
    with mlflow.start_run():
        mlflow.log_params(config.get("model_params", {}))
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "model")
        # Save metrics to metrics directory for DVC
        metrics_dir = os.path.join(os.path.dirname(__file__), "..", "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
            json.dump({"roc_auc": auc}, f)

    # Save model artifact
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        import pickle
        pickle.dump(model, f)
    print(f"Model trained. AUC: {auc:.4f}. Saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-data", action="store_true", help="Prepare data")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--config", type=str, default="training/config/training_config.yaml", help="Path to config")
    args = parser.parse_args()

    if args.prepare_data:
        prepare_data()
    if args.train:
        train_model(args.config)
