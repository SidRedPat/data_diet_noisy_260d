import logging
from flax.training import checkpoints
from jax import jit, value_and_grad
import numpy as np
import os
import time
import optax
from .data import load_data, train_batches
from .forgetting import init_forget_stats, update_forget_stats, save_forget_scores
from .metrics import accuracy, correct, cross_entropy_loss
from .models import get_apply_fn_test, get_apply_fn_train, get_model
from .recorder import init_recorder, record_ckpt, record_test_stats, record_train_stats, save_recorder
from .test import get_test_step, test
from .train_state import TrainState, get_train_state
from .utils import make_dir, save_args, set_global_seed


def get_train_step(loss_and_grad_fn):
    def train_step(state, x, y, lr):
        # Forward pass, loss, and gradient computation
        (loss, (acc, logits, new_model_state)), gradients = loss_and_grad_fn(
            state.params, state.model_state, x, y
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
                return base_lr * (1 - (step - warmup_steps) / (total_steps - warmup_steps))
        return learning_rate
    elif args.decay_steps:
        # Piecewise constant schedule using optax
        schedule = optax.piecewise_constant_schedule(
            init_value=args.lr,
            boundaries_and_scales={step: args.decay_factor for step in args.decay_steps}
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
    def loss_fn(params, model_state, x, y):
        # Forward pass through the model
        logits, new_model_state = f_train(params, model_state, x)
        loss = cross_entropy_loss(logits, y)  # Calculate cross-entropy loss
        acc = accuracy(logits, y)            # Calculate accuracy for monitoring
        return loss, (acc, logits, new_model_state)

    return loss_fn


########################################################################################################################
#  Train
########################################################################################################################

def train(args):
    # setup
    print("Starting training setup...")
    set_global_seed()
    make_dir(args.save_dir)
    make_dir(args.save_dir + '/ckpts')
    if args.track_forgetting:
        make_dir(args.save_dir + '/forget_scores')

    I_train, X_train, Y_train, X_test, Y_test, args = load_data(args)
    model = get_model(args)
    state, tx, args = get_train_state(args, model)  # Extract tx separately
    f_train, f_test = get_apply_fn_train(model), get_apply_fn_test(model)
    test_step = jit(get_test_step(f_test))
    train_step = jit(get_train_step(value_and_grad(get_loss_fn(f_train), has_aux=True)))

    lr_schedule = get_lr_schedule(args)
    rec = init_recorder()
    forget_stats = init_forget_stats(args) if args.track_forgetting else None

    # log and save initial state
    save_args(args, args.save_dir)
    print(f"Initial test: Computing accuracy and loss on the test set...")
    test_loss, test_acc = test(test_step, state, X_test, Y_test, args.test_batch_size)
    rec = record_test_stats(rec, args.ckpt, test_loss, test_acc)
    checkpoints.save_checkpoint(args.save_dir + '/ckpts', state, step=args.ckpt)
    print(f"Initial test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}")

    # train loop
    print("Starting training loop...")
    for t, idxs, x, y in train_batches(I_train, X_train, Y_train, args):
        lr = lr_schedule(t)

        print("Start forward pass and gradient computation")
        # Forward pass and gradient computation
        loss, acc, logits, new_model_state, gradients = train_step(state, x, y, lr)

        print("Start gradient update using tx")
        # Gradient update using tx (outside JIT)
        updates, new_opt_state = tx.update(gradients, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        
        print("Start update to TrainState")
        # Update TrainState
        state = TrainState(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            model_state=new_model_state,
        )

        print("Start recording/logging/checkpointing stats")
        rec = record_train_stats(rec, t, loss.item(), acc.item(), lr)

        if t % args.log_steps == 0:
            print(f"Step {t}: Training loss: {loss.item():.4f}, accuracy: {acc.item():.4f}, learning rate: {lr:.6f}")
            test_loss, test_acc = test(test_step, state, X_test, Y_test, args.test_batch_size)
            rec = record_test_stats(rec, t, test_loss, test_acc)
            print(f"Step {t}: Test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}")

            # save checkpoint
            checkpoints.save_checkpoint(args.save_dir + '/ckpts', state, step=t)
            print(f"Checkpoint saved at step {t}.")

    # wrap up
    save_recorder(args.save_dir, rec)
    print("Training completed. Results saved.")
