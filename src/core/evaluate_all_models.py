"""
================================================================================
evaluate_all_models.py
================================================================================
Purpose:
    This script evaluates all federated learning global models (for both IID and 
    non-IID settings) stored in a versioned directory. It performs the following:
        - Loads the centralized test dataset using .npy feature files and label CSV
        - Evaluates each global model across rounds using Accuracy and F1 Score
        - Logs and ranks performance to identify the best model for each tag (IID, non-IID)
        - Saves the best-fit model per category in a dedicated directory
        - Generates a leaderboard CSV and comparative performance plot

Key Features:
    - Modular: Supports multiple rounds and both IID / non-IID experiments
    - Scoring: Uses sklearn metrics (accuracy, F1 score) on binary classification
    - Visualization: Matplotlib-based plot for score trends across rounds
    - Logging: Custom logger integration for traceability and debugging

Input:
    - Centralized test data in .npy format (`test_clients/`)
    - Corresponding ground truth labels in a CSV file
    - Global models stored under `model/global/<version>/<tag>/`

Output:
    - Best model for each tag saved to `model/global/<version>/<tag>/global_best_<tag>_model.h5`
    - Leaderboard CSV: global_model_evaluation_leaderboard_<version>.csv
    - Leaderboard plot: global_model_evaluation_leaderboard_<version>.png

Usage:
    python evaluate_all_models.py

Author:
    Ranjoy Sen

================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score
from typing import Tuple
from logger import get_logger
from config import PLOT_DIR, PLOT_BASE_DIR, CSV_BASE_DIR
from datetime import datetime

# === Constants ===
MODEL_VERSION = "v1.0.0"
LEADERBOARD_CSV_FILE = f"{CSV_BASE_DIR}/global_model_evaluation_leaderboard_{MODEL_VERSION}.csv"
LEADERBOARD_PLOT_FILE = f"{PLOT_BASE_DIR}/global_model_evaluation_leaderboard_{MODEL_VERSION}.png"
BEST_MODEL_DIR = f"model/global"

# === Logger ===
log = get_logger("compare", "compare")

# === Load test data ===
def load_test_data(test_dir: str, label_csv: str) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    df = pd.read_csv(label_csv)

    for _, row in df.iterrows():
        filename = row["filename"]
        label = row["label"]
        npy_filename = filename.replace(".wav", ".npy")
        path = os.path.join(test_dir, npy_filename)

        if not os.path.isfile(path):
            log.warning(f"⚠️ Skipping missing file: {path}")
            continue

        X.append(np.load(path))
        y.append(1 if label == "good" else 0)

    return np.array(X), np.array(y)


# === Evaluate model ===
def evaluate_model(model, X, y) -> Tuple[float, float, np.ndarray]:
    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return acc, f1, y_pred


# === Plot leaderboard ===
def plot_leaderboard(results_df: pd.DataFrame, filename: str):
    iid_df = results_df[results_df["tag"] == "IID"].sort_values("round")
    noniid_df = results_df[results_df["tag"] == "non-IID"].sort_values("round")

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(f"Model Evaluation Leaderboard (Accuracy & F1) - {MODEL_VERSION}", fontsize=16)

    # IID Plot
    axs[0].plot(iid_df["round"], iid_df["accuracy"], marker='o', label="Accuracy")
    axs[0].plot(iid_df["round"], iid_df["f1_score"], marker='s', label="F1 Score")
    axs[0].set_title("IID Models")
    axs[0].set_xlabel("Round")
    axs[0].set_ylabel("Score")
    axs[0].set_ylim(0, 1.05)
    axs[0].grid(True)
    axs[0].legend()

    # non-IID Plot
    axs[1].plot(noniid_df["round"], noniid_df["accuracy"], marker='o', label="Accuracy")
    axs[1].plot(noniid_df["round"], noniid_df["f1_score"], marker='s', label="F1 Score")
    axs[1].set_title("non-IID Models")
    axs[1].set_xlabel("Round")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(filename)
    log.info(f"📈 Saved leaderboard plot to {filename}")
    plt.close()


import re

# === Evaluate all models and pick best ===
def evaluate_all(model_dir: str, tag: str, X_test, y_test):
    results = []

    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith(".h5") and "_round_" in fname:
            match = re.search(r'_round_(\d+)', fname)
            if not match:
                log.warning(f"⚠️ Skipping invalid model filename: {fname}")
                continue

            round_num = int(match.group(1))
            path = os.path.join(model_dir, fname)

            try:
                model = load_model(path)
                acc, f1, y_pred = evaluate_model(model, X_test, y_test)
                log.info(f"🔍 {tag} Round {round_num} — Accuracy: {acc:.4f}, F1: {f1:.4f}")

                results.append({
                    "tag": tag,
                    "round": round_num,
                    "model_path": path,
                    "accuracy": acc,
                    "f1_score": f1
                })
            except Exception as e:
                log.error(f"❌ Failed to evaluate model {fname}: {e}")

    if not results:
        raise ValueError(f"No valid models found in {model_dir} for tag {tag}")

    # Choose best model (highest accuracy)
    best_model = sorted(results, key=lambda x: x["accuracy"], reverse=True)[0]
    log.info(f"🏆 Best model for {tag}: Round {best_model['round']} — Accuracy: {best_model['accuracy']:.4f}")

    # Save a copy of the best model
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    tag_path = os.path.join(BEST_MODEL_DIR, f"{MODEL_VERSION}/{tag}/global_best_{tag}_model.h5")
    os.system(f"cp {best_model['model_path']} {tag_path}")
    log.info(f"✅ Saved best model for {tag} to {tag_path}")

    return results, best_model



# === Entrypoint ===
if __name__ == "__main__":

    log.info(f"=== All model evaluation Logs up and running @ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===")

    test_dir = "resources/material/train-data/federated/IID-npy/test_clients"
    label_csv = "resources/material/train-data/federated/IID/test_clients/labels.csv"

    X_test, y_test = load_test_data(test_dir, label_csv)

    iid_models = f"model/global/{MODEL_VERSION}/IID"
    noniid_models = f"model/global/{MODEL_VERSION}/non-IID"

    iid_results, best_iid = evaluate_all(iid_models, "IID", X_test, y_test)
    noniid_results, best_noniid = evaluate_all(noniid_models, "non-IID", X_test, y_test)

    # Save leaderboard
    results_df = pd.DataFrame(iid_results + noniid_results)
    results_df.to_csv(LEADERBOARD_CSV_FILE, index=False)
    log.info(f"📊 Saved leaderboard CSV to {LEADERBOARD_CSV_FILE}")

    # Plot leaderboard
    plot_leaderboard(results_df, LEADERBOARD_PLOT_FILE)

    log.info(f"=== Logs ended @ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===")
