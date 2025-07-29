# utils.py
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime

LABEL_MAP = {"bad": 0, "good": 1}
PLOT_DIR = "resources/plot"
os.makedirs(PLOT_DIR, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def load_client_data(client_id, base_path):
    X, y = [], []
    for label, val in LABEL_MAP.items():
        dir_path = os.path.join(base_path, f"client_{client_id}", label)
        if not os.path.exists(dir_path): continue
        for fname in os.listdir(dir_path):
            if fname.endswith(".npy"):
                try:
                    X.append(np.load(os.path.join(dir_path, fname)))
                    y.append(val)
                except Exception as e:
                    log(f"❌ Error loading {fname}: {e}")
    X, y = np.array(X), np.array(y)
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    if np.max(X) > 1.0:
        X = X / np.max(X)
    return X, y

def evaluate_model(model, X, y, label):
    y_pred = model.predict(X)
    y_pred_label = (y_pred > 0.5).astype(int).flatten()
    report = classification_report(y, y_pred_label, target_names=LABEL_MAP.keys())
    log(f"=== Eval Report for {label} ===\n{report}")

    cm = confusion_matrix(y, y_pred_label)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_MAP.keys())
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {label}")
    plt.savefig(f"{PLOT_DIR}/confusion_matrix_{label}.png")
    plt.close()
