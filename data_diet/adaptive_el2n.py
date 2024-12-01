import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, List
import logging
import jax
import jax.numpy as jnp

logger = logging.getLogger("PruningManager")
logger.setLevel("INFO")


class AdaptiveEL2NPruning:
    def __init__(
        self,
        initial_prune_percent: float = 0.2,
        total_epochs: int = 200,
        device: Optional[str] = None,
    ):
        self.initial_prune_percent = initial_prune_percent
        self.total_epochs = total_epochs
        self.device = (
            device or "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
        )
        self.pruned_percentages: List[float] = []
        self.pruned_sizes: List[int] = []
        self.total_sizes: List[int] = []
        self.current_step = 0
        self.log_every = 20

    def compute_el2n_scores(self, state, f_train, x, y) -> np.ndarray:
        """Compute EL2N scores using JAX/Flax model"""
        with tf.device(self.device):
            # Get model predictions
            logits, _ = f_train(state.params, state.model_state, x)

            # Convert logits to probabilities using JAX's softmax
            probabilities = jax.nn.softmax(logits)

            # Convert targets to one-hot using JAX
            targets_one_hot = y

            # Ensure targets_one_hot has the same shape as probabilities (128, 10)
            if len(targets_one_hot.shape) > 2:
                targets_one_hot = targets_one_hot.reshape(targets_one_hot.shape[0], -1)

            # Compute EL2N score: squared L2 norm of (probabilities - targets)
            errors = probabilities - targets_one_hot
            el2n_scores = jnp.sum(errors**2, axis=-1)

            # Convert to numpy array for consistent processing
            return np.array(el2n_scores)

    def adaptive_pruning_strategy(
        self, el2n_scores: np.ndarray, current_step: int, steps_per_epoch: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.current_step = current_step
        current_epoch = current_step / steps_per_epoch
        epoch_progress = current_epoch / self.total_epochs
        total_samples = len(el2n_scores)

        if epoch_progress < 0.3:
            high_noise_threshold = np.percentile(
                el2n_scores, 100 * (1 - self.initial_prune_percent)
            )
            mask = el2n_scores <= high_noise_threshold
            pruned_samples = np.sum(~mask)
            prune_percent = self.initial_prune_percent
            if current_step % self.log_every == 0:
                logger.info(
                    f"Early training phase (epoch {current_epoch:.1f}): "
                    f"Pruned {pruned_samples}/{total_samples} samples "
                    f"({100 * pruned_samples/total_samples:.1f}%)"
                )
        else:
            prune_percent = min(
                0.5, self.initial_prune_percent * (1 + 2 * (epoch_progress - 0.5))
            )
            low_noise_threshold = np.percentile(el2n_scores, 100 * prune_percent)
            mask = el2n_scores >= low_noise_threshold
            pruned_samples = np.sum(~mask)
            if current_step % self.log_every == 0:
                logger.info(
                    f"Late training phase (epoch {current_epoch:.1f}): "
                    f"Pruned {pruned_samples}/{total_samples} samples "
                    f"({100 * pruned_samples/total_samples:.1f}%)"
                )

        self.pruned_percentages.append(pruned_samples / total_samples)
        self.pruned_sizes.append(pruned_samples)
        self.total_sizes.append(total_samples)

        # Upweight remaining samples to compensate for pruned ones
        weights = np.ones_like(el2n_scores)
        weights[~mask] = 0.0
        # Scale weights so they sum to total_samples (upweighting remaining samples)
        weights = (
            weights * (total_samples / weights.sum()) if weights.sum() > 0 else weights
        )

        return mask, weights

    def get_prune_stats(self):
        """Return the pruning statistics"""
        return {
            "steps": list(range(self.current_step)),
            "pruned_percentages": self.pruned_percentages,
            "pruned_sizes": self.pruned_sizes,
            "total_sizes": self.total_sizes,
        }
