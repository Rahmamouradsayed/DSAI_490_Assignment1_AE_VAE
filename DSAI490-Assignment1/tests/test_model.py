"""
test_model.py
-------------
Unit tests for the AE and VAE model architectures.

Run with:  python -m pytest tests/

Author: Rahma Mourad - 202201407
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import tensorflow as tf
import pytest

from model import Autoencoder, VAE, build_encoder, build_decoder


LATENT_DIM = 16
BATCH = 4
INPUT_SHAPE = (BATCH, 64, 64, 1)


# ─── Encoder / Decoder ────────────────────────────────────────────────────────

def test_encoder_output_shape():
    """Encoder should compress input to (B, latent_dim)."""
    enc = build_encoder(LATENT_DIM)
    x = tf.random.normal(INPUT_SHAPE)
    z = enc(x)
    assert z.shape == (BATCH, LATENT_DIM)


def test_decoder_output_shape():
    """Decoder should reconstruct image with same shape as input."""
    dec = build_decoder(LATENT_DIM)
    z = tf.random.normal((BATCH, LATENT_DIM))
    out = dec(z)
    assert out.shape == INPUT_SHAPE


def test_decoder_output_range():
    """Decoder sigmoid output must be in [0, 1]."""
    dec = build_decoder(LATENT_DIM)
    z = tf.random.normal((BATCH, LATENT_DIM))
    out = dec(z).numpy()
    assert out.min() >= 0.0
    assert out.max() <= 1.0


# ─── Autoencoder ──────────────────────────────────────────────────────────────

def test_ae_forward_shape():
    """AE forward pass should return same shape as input."""
    ae = Autoencoder(latent_dim=LATENT_DIM)
    x = tf.random.normal(INPUT_SHAPE)
    out = ae(x)
    assert out.shape == INPUT_SHAPE


def test_ae_encode_shape():
    """AE encode() should return (B, latent_dim)."""
    ae = Autoencoder(latent_dim=LATENT_DIM)
    x = tf.random.normal(INPUT_SHAPE)
    z = ae.encode(x)
    assert z.shape == (BATCH, LATENT_DIM)


# ─── VAE ──────────────────────────────────────────────────────────────────────

def test_vae_forward_shape():
    """VAE forward pass should return same shape as input."""
    vae = VAE(latent_dim=LATENT_DIM)
    x = tf.random.normal(INPUT_SHAPE)
    out = vae(x)
    assert out.shape == INPUT_SHAPE


def test_vae_encode_returns_mean():
    """VAE encode() returns mean vector of shape (B, latent_dim)."""
    vae = VAE(latent_dim=LATENT_DIM)
    x = tf.random.normal(INPUT_SHAPE)
    mu = vae.encode(x)
    assert mu.shape == (BATCH, LATENT_DIM)


def test_vae_generate_shape():
    """VAE generate() should return correct number of images."""
    vae = VAE(latent_dim=LATENT_DIM)
    n = 8
    imgs = vae.generate(n_samples=n)
    assert imgs.shape == (n, 64, 64, 1)


def test_vae_generate_range():
    """VAE generated images must be in [0, 1] (sigmoid output)."""
    vae = VAE(latent_dim=LATENT_DIM)
    imgs = vae.generate(n_samples=4).numpy()
    assert imgs.min() >= 0.0
    assert imgs.max() <= 1.0
