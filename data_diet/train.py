import logging
from flax.training import checkpoints
from jax import jit, value_and_grad
import optax
from typing import Optional, NamedTuple
from data_diet.plotting import plot_results
from .data import load_data, train_batches
from .metrics import accuracy, cross_entropy_loss
from .models import get_apply_fn_test, get_apply_fn_train, get_model
from .recorder import (
    init_recorder,
    record_prune_stats,
    record_test_stats,
    record_train_stats,
    save_recorder,
)
from .test import get_test_step, test
from .train_state import TrainState, get_train_state
from .utils import make_dir, save_args, set_global_seed
import tensorflow as tf
from .adaptive_el2n import AdaptiveEL2NPruning
from time import time


class EarlyStoppingState(NamedTuple):
    best_loss: float
    patience_counter: int
    best_state: Optional[TrainState]


def get_train_step(loss_and_grad_fn):
    def train_step(state, x, y, lr, weights=None):
        # Forward pass, loss, and gradient computation
        (loss, (acc, logits, new_model_state)), gradients = loss_and_grad_fn(
            state.params, state.model_state, x, y, weights
        )
        return loss, acc, logits, new_model_state, gradients

    return train_step


def get_lr_schedule(args):
    """
    Creates a learning rate schedule based on args.

    Args:
        args: A namespace containing learning rate parameters:
            - lr (float): Initial learning rate.
            - decay_factor (float): Factor to decay learning rate.
            - decay_steps (list[int]): Steps at which to decay learning rate.

    Returns:
        A callable learning rate schedule function.
    """
    if args.lr_vitaly:
        # Custom linear warmup-decay schedule (if specified in args)
        def learning_rate(step):
            base_lr, warmup_steps, total_steps = 0.2, 4680, 31200
            if step <= warmup_steps:
                return base_lr * step / warmup_steps
            else:
                return base_lr * (
                    1 - (step - warmup_steps) / (total_steps - warmup_steps)
                )

        return learning_rate
    elif args.decay_steps:
        # Piecewise constant schedule using optax
        schedule = optax.piecewise_constant_schedule(
            init_value=args.lr,
            boundaries_and_scales={
                step: args.decay_factor for step in args.decay_steps
            },
        )
        return schedule
    else:
        # Constant learning rate
        return lambda step: args.lr


def get_loss_fn(f_train):
    """
    Creates a loss function for the model.

    Args:
        f_train: A callable that computes model outputs during training.

    Returns:
        A loss function that computes cross-entropy loss and auxiliary metrics.
    """

    def loss_fn(params, model_state, x, y, weights=None):
        # Forward pass through the model
        logits, new_model_state = f_train(params, model_state, x)
        loss = cross_entropy_loss(logits, y, weights)  # Calculate cross-entropy loss
        acc = accuracy(logits, y)  # Calculate accuracy for monitoring
        return loss, (acc, logits, new_model_state)

    return loss_fn


########################################################################################################################
#  Train
########################################################################################################################

logger = logging.getLogger("Trainer")
logger.setLevel("INFO")


def train(args):
    print("Available GPUs:", tf.config.list_physical_devices("GPU"))

    # Check for available GPUs
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Set memory growth to avoid TensorFlow using all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Use the first GPU (if multiple GPUs are available)
            tf.config.set_visible_devices(gpus[0], "GPU")
        except RuntimeError as e:
            print(e)

    # setup
    print("Starting training setup...")
    set_global_seed()
    make_dir(args.save_dir)
    make_dir(args.save_dir + "/ckpts")
    if args.track_forgetting:
        make_dir(args.save_dir + "/forget_scores")

    logger.info(
        f"Loading dataset with dataset {args.dataset} and {args.random_label_fraction*100}% noise"
    )
    logger.info(f"{'Running adaptive pruning' if args.adaptive else 'Normal training'}")
    I_train, X_train, Y_train, X_test, Y_test, args = load_data(args)
    logger.info(f"Dataset size: train={len(X_train)}, test={len(X_test)}")
    model = get_model(args)
    logger.info(f"Model {args.model}")
    state, tx, args = get_train_state(args, model)  # Extract tx separately
    f_train, f_test = get_apply_fn_train(model), get_apply_fn_test(model)
    test_step = jit(get_test_step(f_test))
    train_step = jit(get_train_step(value_and_grad(get_loss_fn(f_train), has_aux=True)))

    lr_schedule = get_lr_schedule(args)
    rec = init_recorder()

    # log and save initial state
    save_args(args, args.save_dir)
    print("Initial test: Computing accuracy and loss on the test set...")
    test_loss, test_acc = test(test_step, state, X_test, Y_test, args.test_batch_size)
    rec = record_test_stats(rec, args.ckpt, test_loss, test_acc)
    checkpoints.save_checkpoint(args.save_dir + "/ckpts", state, step=args.ckpt)
    print(f"Initial test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}")

    pruning_manager = AdaptiveEL2NPruning(
        initial_prune_percent=args.initial_prune_percent,
        total_epochs=args.num_steps // args.log_steps,  # Convert steps to epochs
    )
    logger.info(f"Set up pruning manager for {args.num_steps // args.log_steps} epochs")

    # Add early stopping configuration after initial test
    early_stopping = EarlyStoppingState(
        best_loss=float("inf"), patience_counter=0, best_state=None
    )

    # train loop
    logger.info("Starting training loop...")
    start_time = time()
    for t, idxs, x, y in train_batches(I_train, X_train, Y_train, args):
        if args.adaptive:
            # Compute EL2N scores and get pruning mask
            el2n_scores = pruning_manager.compute_el2n_scores(state, f_train, x, y)
            mask, weights = pruning_manager.adaptive_pruning_strategy(
                el2n_scores, t, args.log_steps
            )

            # Apply mask to current batch
            weights = weights[mask]
            x = x[mask]
            y = y[mask]

            # Skip batch if all examples were pruned
            if len(x) == 0:
                continue

            lr = lr_schedule(t)
            loss, acc, logits, new_model_state, gradients = train_step(
                state, x, y, lr, weights
            )
        else:
            lr = lr_schedule(t)
            loss, acc, logits, new_model_state, gradients = train_step(state, x, y, lr)
        # Gradient update using tx (outside JIT)
        updates, new_opt_state = tx.update(gradients, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        # Update TrainState
        state = TrainState(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            model_state=new_model_state,
        )

        rec = record_train_stats(rec, t, loss.item(), acc.item(), lr)

        if t % args.log_steps == 0:
            print(
                f"Step {t}: Training loss: {loss.item():.4f}, accuracy: {acc.item():.4f}, learning rate: {lr:.6f}"
            )
            test_loss, test_acc = test(
                test_step, state, X_test, Y_test, args.test_batch_size
            )
            rec = record_test_stats(rec, t, test_loss, test_acc)
            print(
                f"Step {t}: Test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}"
            )

            # Early stopping logic
            if test_loss < early_stopping.best_loss:
                early_stopping = EarlyStoppingState(
                    best_loss=test_loss, patience_counter=0, best_state=state
                )
                # Save best checkpoint
                checkpoints.save_checkpoint(
                    args.save_dir + "/ckpts/best", state, step=t
                )
                print(f"New best model saved with test loss: {test_loss:.4f}")
            else:
                early_stopping = early_stopping._replace(
                    patience_counter=early_stopping.patience_counter + 1
                )
                if early_stopping.patience_counter >= args.patience:
                    print(f"Early stopping triggered after {t} steps")
                    state = early_stopping.best_state  # Restore best state
                    break

            # save checkpoint
            checkpoints.save_checkpoint(args.save_dir + "/ckpts", state, step=t)
            print(f"Checkpoint saved at step {t}.")

    end_time = time()
    logger.info(f"Train duration {end_time-start_time}s")
    # save prune stats to recorder
    record_prune_stats(rec, pruning_manager.get_prune_stats())

    # wrap up
    save_recorder(args.save_dir, rec)
    print("Training completed. Results saved.")

    plot_results(args, rec)
