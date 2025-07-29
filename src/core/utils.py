"""
===============================================================================
utils.py — Shared Utilities for TapNet Federated Learning
===============================================================================

Description:
    This module provides utility functions for:
    - Loading local client `.npy` data
    - Normalizing and flattening input features
    - Logging events with timestamps
    - Evaluating models using classification reports and confusion matrices

Configuration:
    All constants (e.g., label mapping, plot directories) are declared explicitly
    at the top for easy access and reuse.

Usage:
    from utils import load_client_data, evaluate_model, log.info

Author: Ranjoy Sen
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
from config import LABEL_MAP, PLOT_DIR
from logger import get_logger

# Set logger
log = get_logger("client", "client")

# Ensure plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

# === Data Loader ===
def load_client_data(client_id: int, base_path: str):
    """
    Loads and prepares local client data for training.

    Args:
        client_id (int): Unique ID of the federated client.
        base_path (str): Root directory containing client folders.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features (X) and Labels (y)
    """
    X, y = [], []

    for label, val in LABEL_MAP.items():
        dir_path = os.path.join(base_path, f"client_{client_id}", label)
        if not os.path.exists(dir_path):
            log.info(f"⚠️  Directory not found: {dir_path}")
            continue

        for fname in os.listdir(dir_path):
            if fname.endswith(".npy"):
                try:
                    full_path = os.path.join(dir_path, fname)
                    data = np.load(full_path)
                    X.append(data)
                    y.append(val)
                except Exception as e:
                    log.info(f"❌ Error loading file '{fname}' for client {client_id}: {e}")

    X, y = np.array(X), np.array(y)

    # 🔄 Flatten input if 2D or more (for dense models)
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)

    # ⚖️ Normalize input if values exceed [0, 1]
    if np.max(X) > 1.0:
        X = X / np.max(X)

    log.info(f"✅ Loaded {len(X)} samples for client_{client_id}")
    return X, y

# === Evaluation Utility ===

def evaluate_model_with_metrics(model, X, y, label, loss=None, accuracy=None):
    """
    Evaluate model, save confusion matrix and accuracy/loss plot.

    Args:
        model: Trained Keras model.
        X: Feature set (np.ndarray).
        y: True labels (np.ndarray).
        label: String label used for filenames.
        loss: (Optional) Loss value for plotting.
        accuracy: (Optional) Accuracy value for plotting.
    """
    
    y_pred = model.predict(X)
    y_pred_label = (y_pred > 0.5).astype(int).flatten()

    report = classification_report(y, y_pred_label, target_names=LABEL_MAP.keys())
    log.info(f"📋 Classification Report for {label}:\n{report}")

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred_label)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_MAP.keys())
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {label}")
    cm_path = os.path.join(PLOT_DIR, f"confusion_matrix_{label}.png")
    plt.savefig(cm_path)
    plt.close()
    log.info(f"📁 Saved confusion matrix to: {cm_path}")

    # Loss/Accuracy Plot
    if loss is not None and accuracy is not None:
        plt.figure(figsize=(6, 4))
        bars = plt.bar(["Loss", "Accuracy"], [loss, accuracy], color=["red", "green"])
        plt.title(f"Evaluation Metrics - {label}")
        plt.ylim(0, 1.0)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + 0.2, yval + 0.02, f"{yval:.4f}")
        metrics_path = os.path.join(PLOT_DIR, f"metrics_{label}.png")
        plt.savefig(metrics_path)
        plt.close()
        log.info(f"📁 Saved loss/accuracy plot to: {metrics_path}")

def plot_training_history(history, label: str):
    """
    Plot training vs validation accuracy and loss.
    """
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    plt.figure(figsize=(10, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plot_path = os.path.join(PLOT_DIR, f"training_curve_{label}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    log.info(f"📁 Saved training curve to: {plot_path}")