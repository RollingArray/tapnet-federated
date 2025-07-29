# client.py
import flwr as fl
import numpy as np
from typing import Dict
from model import build_model
from utils import load_client_data, log
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import tensorflow as tf

class TapNetClient(fl.client.NumPyClient):
    def __init__(self, cid: str, base_path: str):
        self.cid = cid
        self.base_path = base_path
        self.model = None
        self.x_train, self.y_train = None, None
        self.x_val, self.y_val = None, None
        self.load_data()
        self.model = build_model(self.x_train.shape[1])
        log(f"📦 Initialized client {cid} with {len(self.x_train)} samples")

    def load_data(self):
        X, y = load_client_data(self.cid, self.base_path)
        from sklearn.model_selection import train_test_split
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        log(f"🧠 Client {self.cid} training started")
        self.model.set_weights(parameters)

        # Print class distribution
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"\n📊 Client {self.cid} - Class distribution:")
        for cls, count in zip(unique, counts):
            print(f"   Class {cls}: {count} samples")

        # Compute class weights and print
        class_weights = get_class_weights(self.y_train)
        print(f"⚖️  Client {self.cid} - Computed class weights:")
        for cls, weight in class_weights.items():
            print(f"   Class {cls}: {weight:.4f}")

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )

        # Fit model
        history = self.model.fit(
            self.x_train, self.y_train,
            validation_data=(self.x_val, self.y_val),
            epochs=100, batch_size=32,
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=1
        )

        log(f"📈 Client {self.cid} training finished. Last val_acc: {history.history['val_accuracy'][-1]:.4f}")
        return self.model.get_weights(), len(self.x_train), {}


    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        y_pred = self.model.predict(self.x_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.shape[1] > 1 else (y_pred > 0.5).astype(int).flatten()

        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        cm = confusion_matrix(self.y_val, y_pred_classes)

        print(f"🧪 Eval client {self.cid}: Loss={loss:.4f}, Acc={acc:.4f}")
        print(f"🧾 Confusion Matrix:\n{cm}\n")

        return loss, len(self.x_val), {"accuracy": acc}


def start_client(cid: str, base_path: str):
    client = TapNetClient(cid, base_path)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

def get_class_weights(y):
        """
        Compute class weights for class imbalance.
        Returns a dict like {0: weight_0, 1: weight_1, ...}
        """
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        return {int(cls): float(w) for cls, w in zip(classes, weights)}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", required=True, help="Client ID (e.g., 1, 2, 3, 4)")
    args = parser.parse_args()
    log(f"🔧 Launching client with ID: {args.cid}")
    start_client(args.cid, base_path="resources/material/train-data/federated/IID-npy")
