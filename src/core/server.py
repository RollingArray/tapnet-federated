"""
===============================================================================
server.py — Flower Federated Server for TapNet
===============================================================================

Description:
    Launches a federated learning server using Flower with a custom FedAvg 
    strategy. Evaluates and logs the global model after each round using
    multiple clients' combined data.

Features:
    - Modular logging and evaluation
    - Plotting confusion matrix + loss/accuracy graphs
    - Round-by-round model checkpointing
    - Summary logging at completion

Usage:
    python server.py

Author: Ranjoy Sen
===============================================================================
"""

import os
import numpy as np
import flwr as fl
from datetime import datetime
from collections import Counter

from model import build_model
from utils import load_client_data, evaluate_model_with_metrics
from config import (
    DISTRIBUTION, FEDERATED_DATA_DIR, NUM_CLIENTS,
    ROUNDS, MODEL_BASE_PATH, MODEL_PREFIX
)
from logger import get_logger

# Logger initialization
log = get_logger("server", "server")

# Global dictionary to track round metrics
round_metrics = {}

# ======================== 📊 Evaluation Function =============================
def get_global_eval_fn():
    """
    Returns a server-side evaluation function to be used in FedAvg strategy.
    Evaluates the model on combined data from all clients.
    """

    def evaluate(server_round, parameters, config):
        try:
            log.info(f"\n🎯 [ROUND {server_round}] Evaluation Started")

            # Load model structure and update weights
            input_dim = load_client_data("1", FEDERATED_DATA_DIR)[0].shape[1]
            model = build_model(input_dim=input_dim)
            model.set_weights(parameters)

            # Concatenate data from all clients
            all_x, all_y = [], []
            for cid in range(1, NUM_CLIENTS + 1):
                x, y = load_client_data(str(cid), FEDERATED_DATA_DIR)
                all_x.append(x)
                all_y.append(y)
            X, y = np.concatenate(all_x), np.concatenate(all_y)

            class_counts = Counter(y.tolist())
            log.info(f"📊 [Round {server_round}] Global Class Distribution: {dict(class_counts)}")

            # Evaluate model and save plots
            loss, acc = model.evaluate(X, y, verbose=0)
            evaluate_model_with_metrics(
                model=model,
                X=X,
                y=y,
                label=f"global_{DISTRIBUTION}_round_{server_round}",
                loss=loss,
                accuracy=acc
            )

            # Save model weights
            model_path = os.path.join(MODEL_BASE_PATH, f"{MODEL_PREFIX}_round_{server_round}.h5")
            model.save(model_path)
            log.info(f"💾 Model checkpoint saved to: {model_path}")

            # Track metrics
            round_metrics[server_round] = {"loss": float(loss), "accuracy": float(acc)}

            return loss, {"accuracy": acc}

        except Exception as e:
            log.info(f"❌ Error during evaluation in Round {server_round}: {str(e)}")
            return float("inf"), {"accuracy": 0.0}

    return evaluate

# ======================= 📋 Final Summary Logging ============================
def log_final_summary():
    """
    Logs summary of all training rounds.
    """
    log.info("\n📋 === FINAL FEDERATED TRAINING SUMMARY ===")
    for rnd in sorted(round_metrics):
        metrics = round_metrics[rnd]
        log.info(f"🔁 Round {rnd:02d} — Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

    if round_metrics:
        best_round = max(round_metrics, key=lambda r: round_metrics[r]['accuracy'])
        log.info(f"\n🏆 Best Round: {best_round} | Accuracy: {round_metrics[best_round]['accuracy']:.4f}")
    log.info("✅ Server training complete.\n")


# ============================== 🚀 Main =====================================
def main():
    log.info(f"\n=== 🌐 SERVER STARTED — [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===")
    log.info(f"⚙️  Distribution: {DISTRIBUTION} | Clients: {NUM_CLIENTS} | Rounds: {ROUNDS}\n")

    strategy = fl.server.strategy.FedAvg(
        evaluate_fn=get_global_eval_fn(),
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda rnd: {"rnd": rnd},
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )

    log.info(f"\n🛑 Server stopped — [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    log_final_summary()


if __name__ == "__main__":
    main()
