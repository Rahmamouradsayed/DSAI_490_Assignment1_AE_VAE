"""
train.py
--------
Training pipeline for the Autoencoder (AE), Variational Autoencoder (VAE),
and Denoising Autoencoder (DAE).

Usage:
    python src/train.py --data_path /path/to/medical-mnist

Author: Rahma Mourad - 202201407
"""

import argparse
import os
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

from data_processing import (
    CLASS_NAMES,
    load_images,
    split_data,
    make_dataset,
)
from model import Autoencoder, VAE


# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: Dict[str, Any] = {
    "data_path":       "/content/drive/MyDrive/medical-mnist",
    "n_per_class":     200,
    "latent_dim":      16,
    "batch_size":      32,
    "epochs":          20,
    "learning_rate":   1e-3,
    "noise_factor":    0.35,
    "train_ratio":     0.85,
    "patience":        5,
    "seed":            42,
    "models_dir":      "models",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────


def set_seeds(seed: int) -> None:
    """Set global random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_callbacks(patience: int) -> list:
    """Return standard Keras callbacks used for all models.

    Args:
        patience: Number of epochs without improvement before stopping.

    Returns:
        List of Keras callbacks.
    """
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )
    ]


# ─── Training Functions ───────────────────────────────────────────────────────


def train_autoencoder(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    cfg: Dict[str, Any],
    name: str = "ae",
) -> tuple:
    """Instantiate, compile, and train an Autoencoder.

    Args:
        train_ds: Training tf.data.Dataset yielding (input, target) pairs.
        val_ds: Validation tf.data.Dataset.
        cfg: Configuration dictionary.
        name: Short name used for saving the model file.

    Returns:
        Tuple of (trained Autoencoder model, Keras History object).
    """
    model = Autoencoder(latent_dim=cfg["latent_dim"])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss="mse",
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        callbacks=get_callbacks(cfg["patience"]),
        verbose=1,
    )

    # Save model
    os.makedirs(cfg["models_dir"], exist_ok=True)
    save_path = os.path.join(cfg["models_dir"], f"{name}_v1.keras")
    model.save(save_path)
    print(f"[train] Model saved → {save_path}")

    return model, history


def train_vae(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    cfg: Dict[str, Any],
) -> tuple:
    """Instantiate, compile, and train a Variational Autoencoder.

    Args:
        train_ds: Training tf.data.Dataset.
        val_ds: Validation tf.data.Dataset.
        cfg: Configuration dictionary.

    Returns:
        Tuple of (trained VAE model, Keras History object).
    """
    model = VAE(latent_dim=cfg["latent_dim"])
    model.compile(optimizer=optimizers.Adam(learning_rate=cfg["learning_rate"]))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        callbacks=get_callbacks(cfg["patience"]),
        verbose=1,
    )

    os.makedirs(cfg["models_dir"], exist_ok=True)
    save_path = os.path.join(cfg["models_dir"], "vae_v1.keras")
    model.save(save_path)
    print(f"[train] Model saved → {save_path}")

    return model, history


# ─── Main ─────────────────────────────────────────────────────────────────────


def main(cfg: Dict[str, Any]) -> None:
    """Full training pipeline: load data → build pipelines → train all models.

    Args:
        cfg: Configuration dictionary (see DEFAULT_CONFIG).
    """
    set_seeds(cfg["seed"])

    # ── 1. Load & split data ──────────────────────────────────────────────────
    print("[data] Loading images …")
    X, y = load_images(cfg["data_path"], CLASS_NAMES, cfg["n_per_class"])
    X_train, X_val, y_train, y_val = split_data(X, y, cfg["train_ratio"])
    print(f"[data] Train: {X_train.shape}  Val: {X_val.shape}")

    # ── 2. Build tf.data pipelines ────────────────────────────────────────────
    train_ds       = make_dataset(X_train, cfg["batch_size"], shuffle=True)
    val_ds         = make_dataset(X_val,   cfg["batch_size"], shuffle=False)
    train_ds_noisy = make_dataset(
        X_train, cfg["batch_size"], shuffle=True,
        noisy=True, noise_factor=cfg["noise_factor"],
    )
    val_ds_noisy   = make_dataset(
        X_val, cfg["batch_size"], shuffle=False,
        noisy=True, noise_factor=cfg["noise_factor"],
    )

    # ── 3. Train AE ───────────────────────────────────────────────────────────
    print("\n[train] === Autoencoder ===")
    ae, ae_history = train_autoencoder(train_ds, val_ds, cfg, name="ae")

    # ── 4. Train VAE ──────────────────────────────────────────────────────────
    print("\n[train] === Variational Autoencoder ===")
    vae, vae_history = train_vae(train_ds, val_ds, cfg)

    # ── 5. Train Denoising AE ─────────────────────────────────────────────────
    print("\n[train] === Denoising Autoencoder ===")
    dae, dae_history = train_autoencoder(
        train_ds_noisy, val_ds_noisy, cfg, name="dae"
    )

    print("\n[train] All models trained successfully.")
    return ae, vae, dae, ae_history, vae_history, X_train, X_val, y_train, y_val


# ─── CLI Entry Point ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AE / VAE on Medical-MNIST")
    parser.add_argument("--data_path", type=str, default=DEFAULT_CONFIG["data_path"])
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_CONFIG["latent_dim"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG, **vars(args)}
    main(cfg)
