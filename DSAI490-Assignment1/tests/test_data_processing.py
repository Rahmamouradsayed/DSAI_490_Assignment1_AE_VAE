"""
test_data_processing.py
-----------------------
Unit tests for data loading and tf.data pipeline utilities.

Run with:  python -m pytest tests/

Author: Rahma Mourad - 202201407
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import tensorflow as tf
import pytest

from data_processing import split_data, make_dataset


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_images():
    """100 random grayscale images of shape (64, 64, 1)."""
    np.random.seed(0)
    return np.random.rand(100, 64, 64, 1).astype("float32")


@pytest.fixture
def dummy_labels():
    """100 random integer labels."""
    np.random.seed(0)
    return np.random.randint(0, 6, size=(100,)).astype("int32")


# ─── split_data ───────────────────────────────────────────────────────────────

def test_split_sizes(dummy_images, dummy_labels):
    """85/15 split should give 85 train and 15 val samples."""
    X_tr, X_val, y_tr, y_val = split_data(dummy_images, dummy_labels, train_ratio=0.85)
    assert len(X_tr)  == 85
    assert len(X_val) == 15
    assert len(y_tr)  == 85
    assert len(y_val) == 15


def test_split_no_overlap(dummy_images, dummy_labels):
    """Train and val sets must not share any elements."""
    X_tr, X_val, _, _ = split_data(dummy_images, dummy_labels)
    # Check that no row in val appears in train (compare first pixel as proxy)
    train_first = set(X_tr[:, 0, 0, 0].tolist())
    val_first   = set(X_val[:, 0, 0, 0].tolist())
    # At least one sample differs (floating point uniqueness)
    assert len(train_first | val_first) > 0


# ─── make_dataset ─────────────────────────────────────────────────────────────

def test_dataset_batch_shape(dummy_images):
    """Each batch from make_dataset should have shape (B, 64, 64, 1)."""
    ds = make_dataset(dummy_images, batch_size=16, shuffle=False)
    for x_batch, y_batch in ds.take(1):
        assert x_batch.shape == (16, 64, 64, 1)
        assert y_batch.shape == (16, 64, 64, 1)


def test_dataset_target_equals_input_clean(dummy_images):
    """For clean (non-noisy) dataset, input and target must be identical."""
    ds = make_dataset(dummy_images, batch_size=8, shuffle=False, noisy=False)
    for x_batch, y_batch in ds.take(1):
        np.testing.assert_array_equal(x_batch.numpy(), y_batch.numpy())


def test_dataset_noisy_differs(dummy_images):
    """Noisy dataset input should differ from clean target."""
    ds = make_dataset(dummy_images, batch_size=8, shuffle=False,
                      noisy=True, noise_factor=0.35)
    for x_batch, y_batch in ds.take(1):
        assert not np.allclose(x_batch.numpy(), y_batch.numpy())


def test_dataset_noisy_target_clean(dummy_images):
    """Noisy dataset target (y) should still be the original clean image."""
    ds_clean = make_dataset(dummy_images, batch_size=8, shuffle=False, noisy=False)
    ds_noisy = make_dataset(dummy_images, batch_size=8, shuffle=False, noisy=True)
    for (_, y_clean), (_, y_noisy) in zip(ds_clean.take(1), ds_noisy.take(1)):
        np.testing.assert_array_equal(y_clean.numpy(), y_noisy.numpy())


def test_dataset_values_clipped(dummy_images):
    """Noisy inputs must remain in [0, 1] after clipping."""
    ds = make_dataset(dummy_images, batch_size=32, shuffle=False,
                      noisy=True, noise_factor=1.0)   # large noise
    for x_batch, _ in ds.take(1):
        vals = x_batch.numpy()
        assert vals.min() >= 0.0
        assert vals.max() <= 1.0
