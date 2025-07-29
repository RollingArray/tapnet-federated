# === CONFIGURATION ===

# Data config
DISTRIBUTION = "IID-npy"
BASE_PATH = f"resources/material/train-data/federated/{DISTRIBUTION}"
CLIENT_PREFIX = "client_"
CLIENT_IDS = [1, 2, 3, 4]
LABEL_MAP = {"bad": 0, "good": 1}

# Training config
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Paths
MODEL_DIR = "model"
PLOT_DIR = "resources/plot"
STANDALONE_MODEL_PATH = f"{MODEL_DIR}/global_model_standalone_{DISTRIBUTION}.h5"

# Federated config
SERVER_ADDRESS = "localhost:8080"
EVAL_CLIENT_ID = 1
ROUNDS = 5
MIN_CLIENTS = 2
