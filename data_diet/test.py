import numpy as np
from .data import test_batches
from .metrics import cross_entropy_loss, accuracy
from typing import Callable, Any


def get_test_step(f_test: Callable) -> Callable:
    """
    Returns a function to perform a single test step.

    Args:
        f_test: The test function, typically obtained from the model.

    Returns:
        A callable that computes loss and accuracy for a single batch.
    """
    def test_step(params: Any, model_state: Any, x: np.ndarray, y: np.ndarray):
        logits = f_test(params, model_state, x)
        loss = cross_entropy_loss(logits, y)
        acc = accuracy(logits, y)
        return loss, acc
    return test_step


def test(test_step: Callable, state: Any, X: np.ndarray, Y: np.ndarray, batch_size: int):
    """
    Tests the model on the given data.

    Args:
        test_step: The function to perform a single test step.
        state: The current model state, including parameters.
        X: The input test data.
        Y: The corresponding test labels.
        batch_size: The batch size to use for testing.

    Returns:
        Tuple containing the average loss and accuracy across the test set.
    """
    loss, acc, N = 0.0, 0.0, X.shape[0]
    for n, x, y in test_batches(X, Y, batch_size):
        step_loss, step_acc = test_step(state.params, state.model_state, x, y)
        loss += step_loss * n
        acc += step_acc * n
    loss /= N
    acc /= N
    return loss, acc
