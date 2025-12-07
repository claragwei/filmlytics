"""
rebuild_stacking_model.py

This script rebuilds a stacking ensemble model that combines:
    - GNN predictions
    - KGCN predictions
    - XGBoost predictions

The new model is saved as `stacking_meta_model.pkl` in a format that is
fully compatible with Streamlit Cloud (sklearn ≥ 1.3.x).

Run this script locally (NOT in Streamlit Cloud).
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Local path to model_artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "..", "model_artifacts")

# Output paths
STACKING_MODEL_PATH = os.path.join(ARTIFACT_DIR, "stacking_meta_model.pkl")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "stacking_metrics.json")

# OPTIONAL: weight file
ENSEMBLE_WEIGHTS_PATH = os.path.join(ARTIFACT_DIR, "ensemble_weights.json")

# Choose how to load the ground-truth audience scores:
LOAD_FROM_MONGODB = False  # set to True only if you prefer loading directly from DB

# If using MongoDB (your Streamlit secret format)
MONGODB_URI = "mongodb+srv://cinemaniacs:filmlytics@filmlytics.1emhcue.mongodb.net/?retryWrites=true&w=majority"


# -----------------------------------------------------------------------------
# LOAD PREDICTION DATA
# -----------------------------------------------------------------------------

def load_predictions(filename, column_name=None):
    """Generic loader for prediction CSVs."""
    path = os.path.join(ARTIFACT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    df["tmdb_id"] = df["tmdb_id"].astype(int)

    if column_name is None:
        # Auto-detect prediction column
        for col in df.columns:
            if col not in ["tmdb_id"]:
                column_name = col
                break

    df = df[["tmdb_id", column_name]].rename(columns={column_name: filename.replace(".csv", "")})
    return df


def load_ground_truth_from_mongodb():
    """Pull Rotten Tomatoes audience scores from MongoDB."""
    print("Connecting to MongoDB for labels…")
    client = MongoClient(MONGODB_URI, server_api=ServerApi("1"), tlsCAFile=certifi.where())
    db = client["cinemaniacs"]

    cursor = db.movies.find(
        {"rotten_tomatoes.audience_score": {"$ne": None}},
        {"tmdb_id": 1, "rotten_tomatoes.audience_score": 1},
    )

    rows = []
    for doc in cursor:
        rows.append([int(doc["tmdb_id"]), float(doc["rotten_tomatoes"]["audience_score"]) / 100.0])

    df = pd.DataFrame(rows, columns=["tmdb_id", "true_score"])
    print(f"Loaded {len(df):,} ground-truth labels from DB.")
    return df


def load_ground_truth_from_csv():
    """
    Optionally, if you exported a CSV manually:
    This CSV MUST contain:
        tmdb_id, true_score   (true_score in 0–1 scale)
    """
    path = os.path.join(ARTIFACT_DIR, "ground_truth_scores.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Ground truth CSV not found: {path}\n"
            "If you want to use MongoDB labels instead, set LOAD_FROM_MONGODB = True."
        )
    df = pd.read_csv(path)
    df["tmdb_id"] = df["tmdb_id"].astype(int)
    df["true_score"] = df["true_score"].astype(float)
    print(f"Loaded {len(df):,} ground truth labels from CSV.")
    return df


# -----------------------------------------------------------------------------
# BUILD STACKING DATASET
# -----------------------------------------------------------------------------

def build_training_dataframe():
    print("Loading base model predictions…")

    df_gnn = load_predictions("gnn_preds_all_movies.csv")
    df_kgcn = load_predictions("kgcn_preds_all_movies.csv")
    df_xgb = load_predictions("xgb_preds_all_movies.csv")

    print("Merging base predictions…")
    df = df_gnn.merge(df_kgcn, on="tmdb_id", how="inner")
    df = df.merge(df_xgb, on="tmdb_id", how="inner")

    if LOAD_FROM_MONGODB:
        df_truth = load_ground_truth_from_mongodb()
    else:
        df_truth = load_ground_truth_from_csv()

    df = df.merge(df_truth, on="tmdb_id", how="inner")
    print(f"Training samples after full merge: {len(df):,}")

    return df.rename(columns={
        "gnn_preds_all_movies": "gnn_pred",
        "kgcn_preds_all_movies": "kgcn_pred",
        "xgb_preds_all_movies": "xgb_pred"
    })


# -----------------------------------------------------------------------------
# TRAIN STACKING MODEL
# -----------------------------------------------------------------------------

def train_stacking_model(df):
    print("Training stacking ensemble…")

    X = df[["gnn_pred", "kgcn_pred", "xgb_pred"]].values
    y = df["true_score"].values

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )

    model.fit(X, y)
    print("Training complete.")

    # Evaluate on full dataset (train-eval)
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    print("\n=== STACKING MODEL PERFORMANCE ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "meta_model": "GradientBoostingRegressor",
        "train_samples": len(df),
    }

    return model, metrics


# -----------------------------------------------------------------------------
# SAVE OUTPUT FILES
# -----------------------------------------------------------------------------

def save_outputs(model, metrics):
    print("\nSaving new stacking model…")
    with open(STACKING_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved: {STACKING_MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved: {METRICS_PATH}")

    # Also write default ensemble weights (optional)
    weights = {"gnn": 0.33, "kgcn": 0.34, "xg": 0.33}
    with open(ENSEMBLE_WEIGHTS_PATH, "w") as f:
        json.dump(weights, f, indent=4)
    print(f"Saved: {ENSEMBLE_WEIGHTS_PATH}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    print("=== STACKING MODEL REBUILDER ===")

    df = build_training_dataframe()
    model, metrics = train_stacking_model(df)
    save_outputs(model, metrics)

    print("\nAll done! Upload the new `stacking_meta_model.pkl` to Streamlit.")
    print("Your Streamlit app should now load stacking successfully.\n")


if __name__ == "__main__":
    main()
