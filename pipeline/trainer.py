# =============================================================
# trainer.py
# Trains a RandomForestClassifier on the sample dataset,
# uses the logic/parameters outlined in the churn_prediction_eda.ipynb
# notebook.
#
# Usage: python -m pipeline.trainer
# Author: Jonathan Cahal
# =============================================================

import numpy as np
import pandas as pd

import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


from pipeline.loader import DataLoader
from pipeline.transformer import DataTransformer

import config

class ModelTrainer:

  def __init__(self):
    pass

  def train(self):
    # Load the sample data
    print(f"[trainer] Loading data from {config.INPUT_FILE}")
    loader = DataLoader(config.INPUT_FILE)
    df_init = loader.load()
    print(f"[trainer] Loaded data shape: {df_init.shape}")

    # Clean the sample data
    print("[trainer] Transforming data)")
    tranformer = DataTransformer(df_init, drop_na=True, normalize=True)
    df_cleaned = tranformer.transform()

    # Drop the customer_id
    df_final = df_cleaned.drop('customer_id', axis=1)
    print(f"[trainer] Transformed data shape: {df_final.shape}")

    # MLFlow
    mlflow.set_experiment("churn_prediction")
    mlflow.sklearn.autolog()

    # Split the dataset
    X = df_final.drop('churned', axis=1)
    y = df_final.churned

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[trainer] Data split - Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train the classifier
    print("[trainer] Fitting Classifier...")

    with mlflow.start_run():
      clf = RandomForestClassifier(random_state=0, criterion='entropy', max_depth=6, n_estimators=25)
      clf = clf.fit(X_train, y_train)
      print(f"[trainer] Training accuracy: {clf.score(X_train, y_train):.4f}, Test accuracy: {clf.score(X_test, y_test):.4f}")

    # Dump/Return the trained classifier
    print("[trainer] Dumping model to models/churn_model_v1.pkl")
    joblib.dump(clf, "models/churn_model_v1.pkl")
    print("[trainer] Done.")
    return clf

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()