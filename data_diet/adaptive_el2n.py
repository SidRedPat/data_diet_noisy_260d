import tensorflow as tf
import numpy as np
from typing import Tuple, Optional

class AdaptiveEL2NPruning:
    def __init__(
        self,
        initial_prune_percent: float = 0.2,
        total_epochs: int = 200,
        device: Optional[str] = None
    ):
        self.initial_prune_percent = initial_prune_percent
        self.total_epochs = total_epochs
        self.device = device or '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

    def compute_el2n_scores(self, state, f_train, x, y) -> Tuple[np.ndarray, np.ndarray]:
        """Compute EL2N scores using JAX/Flax model"""
        with tf.device(self.device):
            # Get model predictions
            logits, _ = f_train(state.params, state.model_state, x)
            
            # Convert targets to one-hot
            targets_one_hot = tf.one_hot(y, depth=logits.shape[-1])
            
            # Compute MSE loss per example
            loss = tf.reduce_mean(tf.square(logits - targets_one_hot), axis=1)
            
            # Convert to numpy for consistent processing
            el2n_scores = loss.numpy()
            indices = np.arange(len(el2n_scores))
            
            return el2n_scores, indices

    def adaptive_pruning_strategy(
        self, el2n_scores: np.ndarray, current_step: int, steps_per_epoch: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        current_epoch = current_step / steps_per_epoch
        epoch_progress = current_epoch / self.total_epochs

        if epoch_progress < 0.5:
            high_noise_threshold = np.percentile(
                el2n_scores, 100 * self.initial_prune_percent
            )
            mask = el2n_scores <= high_noise_threshold
        else:
            prune_percent = min(
                0.5, self.initial_prune_percent * (1 + 2 * (epoch_progress - 0.5))
            )
            low_noise_threshold = np.percentile(el2n_scores, 100 * (1 - prune_percent))
            mask = el2n_scores <= low_noise_threshold

        weights = 1.0 - (el2n_scores / el2n_scores.max())
        weights[~mask] = 0.0
        weights = weights / weights.sum()

        return mask, weights