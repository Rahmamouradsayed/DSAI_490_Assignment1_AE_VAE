"""
model.py
--------
Defines the Autoencoder (AE) and Variational Autoencoder (VAE)
architectures using TensorFlow / Keras.

Author: Rahma Mourad - 202201407
"""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers


# ─── Shared Building Blocks ───────────────────────────────────────────────────


def build_encoder(latent_dim: int, name: str = "encoder") -> Model:
    """Build a convolutional encoder network.

    Compresses an input image into a fixed-size latent vector z.

    Architecture:
        Input (64, 64, 1)
        → Conv2D(32,  3, stride=2) → ReLU
        → Conv2D(64,  3, stride=2) → ReLU
        → Conv2D(128, 3, stride=2) → ReLU
        → Flatten
        → Dense(128) → ReLU
        → Dense(latent_dim)          [z]

    Args:
        latent_dim: Dimensionality of the latent space.
        name: Model name for identification.

    Returns:
        A Keras Model mapping (64, 64, 1) → (latent_dim,).
    """
    inp = layers.Input((64, 64, 1), name="encoder_input")
    x = layers.Conv2D(32,  3, strides=2, padding="same", activation="relu")(inp)
    x = layers.Conv2D(64,  3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    z = layers.Dense(latent_dim, name="z")(x)
    return Model(inp, z, name=name)


def build_decoder(latent_dim: int, name: str = "decoder") -> Model:
    """Build a convolutional decoder network.

    Reconstructs an image from a latent vector.

    Architecture:
        Input (latent_dim,)
        → Dense(8*8*128) → ReLU → Reshape(8, 8, 128)
        → Conv2DTranspose(64, 3, stride=2) → ReLU
        → Conv2DTranspose(32, 3, stride=2) → ReLU
        → Conv2DTranspose(1,  3, stride=2) → Sigmoid   [64×64×1]

    Args:
        latent_dim: Dimensionality of the latent input.
        name: Model name for identification.

    Returns:
        A Keras Model mapping (latent_dim,) → (64, 64, 1).
    """
    inp = layers.Input((latent_dim,), name="decoder_input")
    x = layers.Dense(8 * 8 * 128, activation="relu")(inp)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(1,  3, strides=2, padding="same", activation="sigmoid")(x)
    return Model(inp, x, name=name)


# ─── Autoencoder ──────────────────────────────────────────────────────────────


class Autoencoder(Model):
    """Standard deterministic Autoencoder (AE).

    Learns a compressed latent representation of the input and
    reconstructs it. Trained with MSE reconstruction loss only.

    Args:
        latent_dim: Dimensionality of the latent space.
    """

    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = build_encoder(latent_dim, name="ae_encoder")
        self.decoder = build_decoder(latent_dim, name="ae_decoder")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass: encode then decode.

        Args:
            x: Input image batch, shape (B, 64, 64, 1).
            training: Whether in training mode.

        Returns:
            Reconstructed image batch, shape (B, 64, 64, 1).
        """
        z = self.encoder(x, training=training)
        return self.decoder(z, training=training)

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        """Encode images to latent vectors (inference only).

        Args:
            x: Input image batch.

        Returns:
            Latent vectors, shape (B, latent_dim).
        """
        return self.encoder(x, training=False)


# ─── VAE Components ───────────────────────────────────────────────────────────


class Sampling(layers.Layer):
    """Reparameterization sampling layer for the VAE.

    Samples z = mu + sigma * epsilon, where epsilon ~ N(0, I).
    This allows gradients to flow through the stochastic node.
    """

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Sample a latent vector via the reparameterization trick.

        Args:
            inputs: Tuple of (mu, log_var) tensors, each shape (B, latent_dim).

        Returns:
            Sampled latent vector z, shape (B, latent_dim).
        """
        mu, log_var = inputs
        epsilon = tf.random.normal(tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * epsilon


# ─── Variational Autoencoder ──────────────────────────────────────────────────


class VAE(Model):
    """Variational Autoencoder (VAE).

    Extends the AE with a probabilistic latent space. The encoder
    outputs a distribution (mu, log_var) rather than a fixed point.
    Training uses ELBO: reconstruction loss + KL divergence.

    Args:
        latent_dim: Dimensionality of the latent space.
    """

    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # ── Probabilistic encoder ──────────────────────────────────────────
        inp = layers.Input((64, 64, 1), name="vae_encoder_input")
        x = layers.Conv2D(32,  3, strides=2, padding="same", activation="relu")(inp)
        x = layers.Conv2D(64,  3, strides=2, padding="same", activation="relu")(x)
        x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        mu      = layers.Dense(latent_dim, name="mu")(x)
        log_var = layers.Dense(latent_dim, name="log_var")(x)
        z       = Sampling(name="sampling")([mu, log_var])
        self.encoder = Model(inp, [mu, log_var, z], name="vae_encoder")

        # ── Shared decoder ────────────────────────────────────────────────
        self.decoder = build_decoder(latent_dim, name="vae_decoder")

        # ── Metrics ───────────────────────────────────────────────────────
        self._total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self._recon_loss_tracker = tf.keras.metrics.Mean(name="recon")
        self._kl_loss_tracker    = tf.keras.metrics.Mean(name="kl")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def metrics(self):
        """Return tracked metrics for Keras to reset each epoch."""
        return [
            self._total_loss_tracker,
            self._recon_loss_tracker,
            self._kl_loss_tracker,
        ]

    # ── Forward Pass ──────────────────────────────────────────────────────────

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass using the mean latent vector (no sampling).

        Args:
            x: Input image batch, shape (B, 64, 64, 1).
            training: Whether in training mode.

        Returns:
            Reconstructed images, shape (B, 64, 64, 1).
        """
        mu, _, z = self.encoder(x, training=training)
        return self.decoder(z, training=training)

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        """Return the mean latent vector for visualization (no sampling).

        Args:
            x: Input image batch.

        Returns:
            Mean latent vectors mu, shape (B, latent_dim).
        """
        mu, _, _ = self.encoder(x, training=False)
        return mu

    def generate(self, n_samples: int = 16) -> tf.Tensor:
        """Generate new images by sampling from the prior N(0, I).

        Args:
            n_samples: Number of images to generate.

        Returns:
            Generated images, shape (n_samples, 64, 64, 1).
        """
        z = tf.random.normal((n_samples, self.latent_dim))
        return self.decoder(z, training=False)

    # ── Loss ──────────────────────────────────────────────────────────────────

    def _compute_losses(
        self, x: tf.Tensor, training: bool
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute total ELBO loss = reconstruction + KL divergence.

        Args:
            x: Input image batch.
            training: Whether in training mode.

        Returns:
            Tuple of (total_loss, reconstruction_loss, kl_loss).
        """
        mu, log_var, z = self.encoder(x, training=training)
        x_hat = self.decoder(z, training=training)

        # Pixel-wise MSE summed over spatial dims, averaged over batch
        recon = tf.reduce_mean(
            tf.reduce_sum(tf.square(x - x_hat), axis=[1, 2, 3])
        )
        # Analytical KL: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
        )
        return recon + kl, recon, kl

    # ── Training / Validation Steps ───────────────────────────────────────────

    def train_step(self, data):
        """Custom training step with gradient tape.

        Args:
            data: Tuple (x, _) from the tf.data pipeline.

        Returns:
            Dict of metric names → values.
        """
        x, _ = data
        with tf.GradientTape() as tape:
            total, recon, kl = self._compute_losses(x, training=True)

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self._total_loss_tracker.update_state(total)
        self._recon_loss_tracker.update_state(recon)
        self._kl_loss_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Custom validation step.

        Args:
            data: Tuple (x, _) from the tf.data pipeline.

        Returns:
            Dict of metric names → values.
        """
        x, _ = data
        total, recon, kl = self._compute_losses(x, training=False)

        self._total_loss_tracker.update_state(total)
        self._recon_loss_tracker.update_state(recon)
        self._kl_loss_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}
