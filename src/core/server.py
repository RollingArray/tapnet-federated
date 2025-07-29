# server.py
import flwr as fl
import os
import numpy as np
from model import build_model
from utils import load_client_data, evaluate_model, log
from collections import Counter

# Config
DISTRIBUTION = "IID"
BASE_PATH = f"resources/material/train-data/federated/{DISTRIBUTION}"
NUM_CLIENTS = 5
ROUNDS = 5
MODEL_PATH = f"model/global_{DISTRIBUTION}model"

def get_global_eval_fn():
    """Returns a function for evaluating the global model on all client data."""
    def evaluate(server_round, parameters, config):
        try:
            input_dim = load_client_data("1", BASE_PATH)[0].shape[1]
            model = build_model(input_dim=input_dim)
            model.set_weights(parameters)

            # Load all client data
            all_x, all_y = [], []
            for cid in range(1, NUM_CLIENTS + 1):
                x, y = load_client_data(str(cid), BASE_PATH)
                all_x.append(x)
                all_y.append(y)

            X = np.concatenate(all_x)
            y = np.concatenate(all_y)

            # Debug print class distribution
            class_counts = Counter(y.tolist())
            log(f"📊 Global Test Set Class Distribution (Round {server_round}): {class_counts}")

            # Evaluate global model
            loss, acc = model.evaluate(X, y, verbose=0)
            log(f"🌐 [Round {server_round}] Server Eval — Loss: {loss:.4f}, Acc: {acc:.4f}")

            # Save confusion matrix and classification report
            evaluate_model(model, X, y, label=f"global_round_{server_round}")

            # ✅ Save global model
            model_save_path = f"{MODEL_PATH}_{server_round}.h5"
            model.save(model_save_path)
            log(f"💾 Saved global model to: {model_save_path}")

            return loss, {"accuracy": acc}
        
        except Exception as e:
            log(f"❌ Evaluation failed in round {server_round}: {e}")
            return float("inf"), {"accuracy": 0.0}

    return evaluate

def main():
    log("🚀 Starting federated server...")

    strategy = fl.server.strategy.FedAvg(
        evaluate_fn=get_global_eval_fn(),
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda rnd: {"rnd": rnd}
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
