# client.py
"""
===============================================================================
client.py — Federated Learning Client for TapNet Project
===============================================================================

Description:
    This script implements a Flower NumPyClient for federated training of a 
    neural network model using local `.npy` datasets per client.

Execution:
    Run this script from terminal with a specific client ID:
        python client.py --cid <client_id>

    Example:
        python client.py --cid 1

    Multiple clients can be launched in parallel using separate terminals.

Author: Ranjoy Sen
"""

import flwr as fl
import numpy as np
import tensorflow as tf
from typing import Dict
from model import build_model
from utils import load_client_data, plot_training_history
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
from config import VALIDATION_SPLIT, EPOCHS, BATCH_SIZE, EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA, FEDERATED_DATA_DIR
from logger import get_logger

# Set logger
log = get_logger("client", "client")

# ================================
# 🌐 Flower Federated Client
# ================================

class TapNetClient(fl.client.NumPyClient):
    """
    Federated learning client implementation for TapNet.
    """

    def __init__(self, cid: str, base_path: str):
        self.cid = cid
        self.base_path = base_path
        self.model = None
        self.x_train, self.y_train = None, None
        self.x_val, self.y_val = None, None

        self._load_and_prepare_data()
        self.model = build_model(self.x_train.shape[1])
        log.info(f"=== client {cid} Logs up and running @ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===")

        log.info(f"📦 Initialized client {cid} with {len(self.x_train)} samples")

    def _load_and_prepare_data(self):
        """
        Load and split client-specific data into training and validation sets.
        """
        X, y = load_client_data(self.cid, self.base_path)
        from sklearn.model_selection import train_test_split
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            X, y,
            test_size=VALIDATION_SPLIT,
            stratify=y,
            random_state=42
        )

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        """
        Perform local training on client data.
        """
        log.info(f"🧠 Client {self.cid} training started")
        self.model.set_weights(parameters)

        _log_class_distribution(self.cid, self.y_train)

        class_weights = _get_class_weights(self.y_train)
        _log_class_weights(self.cid, class_weights)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            min_delta=EARLY_STOP_MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        )

        history = self.model.fit(
            self.x_train, self.y_train,
            validation_data=(self.x_val, self.y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=1
        )

        val_acc = history.history.get("val_accuracy", [0])[-1]

        log.info(f"📈 Client {self.cid} training finished. Last val_acc: {val_acc:.4f}")

        # 🖼️ Save training curves
        plot_training_history(history, label=f"client_{self.cid}")
        
        log.info(f"📈 Client {self.cid} training finished. Last val_acc: {val_acc:.4f}")

        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """
        Evaluate the model on local validation data.
        """
        self.model.set_weights(parameters)
        y_pred = self.model.predict(self.x_val, verbose=0)
        y_pred_classes = (
            np.argmax(y_pred, axis=1) if y_pred.shape[1] > 1 else (y_pred > 0.5).astype(int).flatten()
        )

        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        cm = confusion_matrix(self.y_val, y_pred_classes)

        print(f"🧪 Eval client {self.cid}: Loss={loss:.4f}, Acc={acc:.4f}")
        print(f"🧾 Confusion Matrix:\n{cm}\n")

        return loss, len(self.x_val), {"accuracy": acc}


# ================================
# 🔁 Client Utilities
# ================================

def start_client(cid: str, base_path: str = FEDERATED_DATA_DIR):
    """
    Start the federated learning client with given ID and dataset path.
    """
    client = TapNetClient(cid, base_path).to_client()
    fl.client.start_client(server_address="localhost:8080", client=client)
    #fl.client.start_client(server_address="localhost:8080", client=TapNetClient().to_client())

def _get_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(cls): float(w) for cls, w in zip(classes, weights)}


def _log_class_distribution(cid: str, y: np.ndarray):
    """
    Print class distribution for the given client data.
    """
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📊 Client {cid} - Class distribution:")
    for cls, count in zip(unique, counts):
        print(f"   Class {cls}: {count} samples")


def _log_class_weights(cid: str, weights: Dict[int, float]):
    """
    Print computed class weights.
    """
    print(f"⚖️  Client {cid} - Computed class weights:")
    for cls, weight in weights.items():
        print(f"   Class {cls}: {weight:.4f}")


# ================================
# 🚀 CLI ENTRY POINT
# ================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start a federated learning client.")
    parser.add_argument("--cid", required=True, help="Client ID (e.g., 1, 2, 3, 4, 5)")
    args = parser.parse_args()

    log.info(f"🔧 Launching client with ID: {args.cid}")
    start_client(args.cid)
