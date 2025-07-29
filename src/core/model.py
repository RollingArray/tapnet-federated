"""
Model Definition for Federated Learning Binary Classifier

This module provides a build_model function that constructs a deep neural network 
using TensorFlow/Keras, suitable for binary classification tasks such as defect detection.

🛠️ Architecture:
    - Input Layer
    - 3 Fully Connected (Dense) Hidden Layers
    - Dropout Regularization
    - Output Layer with Sigmoid Activation

⚙️ Optimizer: Adam with 1e-4 Learning Rate
🎯 Loss Function: Binary Crossentropy
📊 Metric: Accuracy

Author: YourName / Project Team
Version: 1.0
"""

import tensorflow as tf
from config import LEARNING_RATE

from logger import get_logger

# Logger initialization
log = get_logger("client", "client")

def build_model(input_dim):
    
    """
    Builds and compiles a deep neural network model.

    Args:
        input_dim (int): Number of input features for the model.

    Returns:
        tf.keras.Model: A compiled Keras model ready for training.
    """
    
    log.info(f"🔧 Building model with input dimension: {input_dim}")

    model = tf.keras.Sequential([

        # 🔹 Input Layer
        # Accepts feature vectors of shape (num_features,)
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),

        # 🔹 Dense Layer 1
        # 256 neurons with ReLU activation to learn rich, high-level representations
        tf.keras.layers.Dense(256, activation='relu'),
        
        # 🔹 Dropout Layer 1
        # Drop 30% of neurons during training to prevent overfitting
        tf.keras.layers.Dropout(0.3),

        # 🔹 Dense Layer 2
        # Further reduce to 128 neurons to compress learned features
        tf.keras.layers.Dense(128, activation='relu'),
        
        # 🔹 Dropout Layer 2
        # Additional regularization to improve generalization
        tf.keras.layers.Dropout(0.3),

        # 🔹 Dense Layer 3
        # 64 neurons to refine mid-level feature representation
        tf.keras.layers.Dense(64, activation='relu'),

        # 🔹 Dense Layer 4
        # 32 neurons to prepare features for final prediction
        tf.keras.layers.Dense(32, activation='relu'),

        # 🔹 Output Layer
        # Single neuron with sigmoid activation for binary classification output (between 0 and 1)
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


    # ⚙️ Optimizer Configuration
    # Adam is an adaptive learning rate optimizer known for combining the benefits of RMSProp and SGD with momentum.
    # It adjusts the learning rate dynamically based on first- and second-order moments of the gradients.
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # 🔍 Learning rate set to 1e-4 for stable and fine-grained convergence.

    # 🧠 Model Compilation
    # Binary cross-entropy is ideal for binary classification tasks where labels are 0 or 1.
    # Accuracy is used as a basic performance metric.
    model.compile(
        loss='binary_crossentropy',   # 🎯 Objective: Minimize the difference between predicted and true labels
        optimizer=optimizer,          # 🔧 Optimizer: Adam (adaptive and efficient)
        metrics=['accuracy']          # 📈 Metric: Accuracy (fraction of correct predictions)
    )

    log.info("✅ Model compiled successfully with Adam optimizer and binary crossentropy loss.")

    # 🔍 Model summary (optional for dev logs only)
    model.summary(print_fn=lambda x: log.info("📊 " + x))

    return model

