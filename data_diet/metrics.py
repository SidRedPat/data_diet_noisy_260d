from flax import linen as nn
from jax import numpy as jnp


def cross_entropy_loss(logits, labels, weights=None):
    """
    Compute weighted cross entropy loss.

    Args:
        logits: Model predictions
        labels: True labels
        weights: Sample weights (optional)
    """
    losses = -jnp.sum(labels * nn.log_softmax(logits), axis=-1)
    if weights is not None:
        losses = losses * weights
    return jnp.mean(losses)


def correct(logits, labels):
    return jnp.argmax(logits, axis=-1) == jnp.argmax(labels, axis=-1)


def accuracy(logits, labels):
    return jnp.mean(correct(logits, labels))
