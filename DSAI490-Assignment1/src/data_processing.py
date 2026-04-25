"""
data_processing.py
------------------
Handles data loading, preprocessing, and tf.data pipeline creation
for the Medical-MNIST dataset.

Author: Rahma Mourad - 202201407
"""

import os
from typing import Tuple, List, Optional

import numpy as np
import tensorflow as tf


# ─── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES: List[str] = [
    "AbdomenCT",
    "BreastMRI",
    "ChestCT",
    "CXR",
    "Hand",
    "HeadCT",
]

IMG_SIZE: Tuple[int, int] = (64, 64)
INPUT_SHAPE: Tuple[int, int, int] = (64, 64, 1)


# ─── Data Loading ─────────────────────────────────────────────────────────────


def load_images(
    data_path: str,
    class_names: List[str],
    n_per_class: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess images from the Medical-MNIST dataset.

    Args:
        data_path: Root directory containing one sub-folder per class.
        class_names: List of class folder names to load.
        n_per_class: Maximum number of images to load per class.

    Returns:
        Tuple of (images, labels) as float32 numpy arrays.
        Images are normalized to [0, 1].
    """
    images: List[np.ndarray] = []
    labels: List[int] = []

    for class_idx, cls in enumerate(class_names):
        folder = os.path.join(data_path, cls)
        files = sorted(os.listdir(folder))[:n_per_class]

        for fname in files:
            raw = tf.io.read_file(os.path.join(folder, fname))
            img = tf.image.decode_jpeg(raw, channels=1)
            img = tf.image.resize(img, IMG_SIZE)
            img = tf.cast(img, tf.float32) / 255.0
            images.append(img.numpy())
            labels.append(class_idx)

    images_arr = np.array(images, dtype="float32")
    labels_arr = np.array(labels, dtype="int32")

    # Shuffle
    idx = np.random.permutation(len(images_arr))
    return images_arr[idx], labels_arr[idx]


def split_data(
    images: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into training and validation sets.

    Args:
        images: Array of images, shape (N, H, W, C).
        labels: Array of integer labels, shape (N,).
        train_ratio: Fraction of data to use for training.

    Returns:
        Tuple of (X_train, X_val, y_train, y_val).
    """
    split = int(train_ratio * len(images))
    return images[:split], images[split:], labels[:split], labels[split:]


# ─── tf.data Pipelines ────────────────────────────────────────────────────────


def _add_noise(
    x: tf.Tensor,
    y: tf.Tensor,
    noise_factor: float = 0.35,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Add Gaussian noise to an image (used for denoising AE).

    Args:
        x: Clean input image tensor.
        y: Target image tensor (unchanged).
        noise_factor: Standard deviation of Gaussian noise.

    Returns:
        Tuple of (noisy_image, clean_target).
    """
    noise = tf.random.normal(tf.shape(x), stddev=noise_factor)
    return tf.clip_by_value(x + noise, 0.0, 1.0), y


def make_dataset(
    images: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    noisy: bool = False,
    noise_factor: float = 0.35,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build an efficient tf.data pipeline for autoencoder training.

    The dataset yields (input, target) pairs where target == input
    for standard AE/VAE, or (noisy_input, clean_target) for DAE.

    Args:
        images: Array of images with shape (N, H, W, C).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset each epoch.
        noisy: If True, corrupt inputs with Gaussian noise (for DAE).
        noise_factor: Standard deviation of noise when noisy=True.
        seed: Random seed for reproducibility.

    Returns:
        A batched and prefetched tf.data.Dataset.
    """
    ds = tf.data.Dataset.from_tensor_slices((images, images))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), seed=seed)

    if noisy:
        ds = ds.map(
            lambda x, y: _add_noise(x, y, noise_factor),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
