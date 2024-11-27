from .data import get_class_balanced_random_subset
from .gradients import compute_mean_logit_gradients, flatten_jacobian, get_mean_logit_gradients_fn
from .metrics import cross_entropy_loss
import flax.linen as nn
from jax import jacrev, jit, vmap
import jax.numpy as jnp
import numpy as np


def get_lord_error_fn(fn, params, state, ord):
    @jit
    def lord_error(X, Y):
        errors = nn.softmax(fn(params, state, X)) - Y
        scores = jnp.linalg.norm(errors, ord=ord, axis=-1)
        return scores
    return lambda X, Y: np.array(lord_error(X, Y))


def get_margin_error(fn, params, state, score_type):
    fn_jit = jit(lambda X: fn(params, state, X))

    def margin_error(X, Y):
        batch_sz = X.shape[0]
        P = np.array(nn.softmax(fn_jit(X)))
        correct_logits = Y.astype(bool)
        margins = P[~correct_logits].reshape(batch_sz, -1) - P[correct_logits].reshape(batch_sz, 1)
        if score_type == 'max':
            scores = np.max(margins, -1)
        elif score_type == 'sum':
            scores = np.sum(margins, -1)
        return scores

    return margin_error


def get_grad_norm_fn(fn, params, state):
    @jit
    def score_fn(X, Y):
        per_sample_loss_fn = lambda p, x, y: vmap(cross_entropy_loss)(fn(p, state, x), y)
        loss_grads = flatten_jacobian(jacrev(per_sample_loss_fn)(params, X, Y))
        scores = jnp.linalg.norm(loss_grads, axis=-1)
        return scores

    return lambda X, Y: np.array(score_fn(X, Y))


def get_el2n_score_fn(fn, params, state):
    """
    Creates a function to compute EL2N scores.
    EL2N scores are squared L2 norms of the errors between model predictions
    (softmax probabilities) and one-hot labels.

    Args:
        fn: Forward function for model evaluation.
        params: Model parameters.
        state: Model state.

    Returns:
        A callable function to compute EL2N scores.
    """
    @jit
    def score_fn(X, Y):
        logits = fn(params, state, X)  # Forward pass
        probabilities = nn.softmax(logits)  # Convert logits to probabilities
        errors = probabilities - Y  # Difference from one-hot labels
        scores = jnp.linalg.norm(errors, ord=2, axis=-1) ** 2  # EL2N: Squared L2 norm
        return scores

    return lambda X, Y: np.array(score_fn(X, Y))


def get_score_fn(fn, params, state, score_type):
    if score_type == 'l2_error':
        print(f'compute {score_type}...')
        score_fn = get_lord_error_fn(fn, params, state, 2)
    elif score_type == 'el2n':
        print(f'compute {score_type}...')
        score_fn = get_el2n_score_fn(fn, params, state)
    elif score_type == 'grad_norm':
        print(f'compute {score_type}...')
        score_fn = get_grad_norm_fn(fn, params, state)
    else:
        raise NotImplementedError(f"Score type '{score_type}' not implemented.")
    return score_fn


def compute_scores(fn, params, state, X, Y, batch_sz, score_type):
    """
    Compute scores for a dataset in batches.

    Args:
        fn: Forward function for the model.
        params: Model parameters.
        state: Model state.
        X: Input data.
        Y: Labels (one-hot encoded).
        batch_sz: Batch size for processing.
        score_type: Type of score to compute.

    Returns:
        Scores for all examples in the dataset.
    """
    n_batches = int(np.ceil(X.shape[0] / batch_sz))  # Calculate number of batches
    Xs, Ys = np.array_split(X, n_batches), np.array_split(Y, n_batches)  # Use array_split
    score_fn = get_score_fn(fn, params, state, score_type)
    scores = []
    for i, (X_batch, Y_batch) in enumerate(zip(Xs, Ys)):
        print(f'score batch {i + 1} of {n_batches}')
        scores.append(score_fn(X_batch, Y_batch))
    scores = np.concatenate(scores)
    return scores


def compute_unclog_scores(fn, params, state, X, Y, cls_smpl_sz, seed, batch_sz_mlgs):
    n_batches = X.shape[0]
    Xs = np.split(X, n_batches)
    X_mlgs, _ = get_class_balanced_random_subset(X, Y, cls_smpl_sz, seed)
    mlgs = compute_mean_logit_gradients(fn, params, state, X_mlgs, batch_sz_mlgs)
    logit_grads_fn = get_mean_logit_gradients_fn(fn, params, state)
    score_fn = jit(lambda X: jnp.linalg.norm((logit_grads_fn(X) - mlgs).sum(0)))
    scores = []
    for i, X in enumerate(Xs):
        if i % 500 == 0:
            print(f'images {i} of {n_batches}')
        scores.append(score_fn(X).item())
    scores = np.array(scores)
    return scores
