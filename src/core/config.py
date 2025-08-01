# config.py
"""
================================================================================
config.py — Central Configuration for Federated TapNet
================================================================================

Purpose:
    Holds all static values such as paths, constants, hyperparameters, labels, etc.

Usage:
    from config import <CONSTANT_NAME>

Author: Ranjoy Sen
================================================================================
"""

import os
from datetime import datetime

# === Label Mapping ===
LABEL_MAP = {
    "bad": 0,
    "good": 1,
}

# === Training Hyperparameters ===
# Data split ratio for validation
VALIDATION_SPLIT = 0.2

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# Early stopping configuration
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 0.001

# === Directories ===
DISTRIBUTION = "IID" 
MODEL_TYPE = "global"
MODEL_VERSION = "v1.0.0"
FEDERATED_DATA_DIR = f"resources/material/train-data/federated/{DISTRIBUTION}-npy"
PLOT_BASE_DIR = f"resources/plot/{MODEL_TYPE}/test-train-split-data/{MODEL_VERSION}"
PLOT_DIR = f"{PLOT_BASE_DIR}/{DISTRIBUTION}"
CSV_BASE_DIR = "resources/csv"

# === Flower Settings ===
SERVER_ADDRESS = "[::]:8080"
NUM_CLIENTS = 5                              # Total number of participating clients
ROUNDS = 5                                   # Total training rounds
MODEL_BASE_PATH = f"model/{MODEL_TYPE}/{MODEL_VERSION}/{DISTRIBUTION}"                     # Folder to save global models
MODEL_PREFIX = f"global_{DISTRIBUTION}_model"  # Prefix for saved models

# === Logging ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_BASE_DIR = f"logs/{MODEL_TYPE}/{MODEL_VERSION}"
os.makedirs(LOG_BASE_DIR, exist_ok=True)

CLIENT_LOG_FILE = f"{LOG_BASE_DIR}/{DISTRIBUTION}/model_train_client_log.log"
SERVER_LOG_FILE = f"{LOG_BASE_DIR}/{DISTRIBUTION}/model_train_server_log.log"
MODEL_COMPARE_LOG_FILE = f"{LOG_BASE_DIR}/all_model_compare_log.log"
