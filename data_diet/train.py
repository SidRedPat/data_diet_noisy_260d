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


########################################################################################################################
#  Train
########################################################################################################################

def train(args):
    # setup
    set_global_seed()
    make_dir(args.save_dir)
    make_dir(args.save_dir + '/ckpts')
    if args.track_forgetting:
        make_dir(args.save_dir + '/forget_scores')

    I_train, X_train, Y_train, X_test, Y_test, args = load_data(args)
    model = get_model(args)
    state, args = get_train_state(args, model)
    f_train, f_test = get_apply_fn_train(model), get_apply_fn_test(model)
    test_step = jit(get_test_step(f_test))
    train_step = jit(get_train_step(value_and_grad(get_loss_fn(f_train), has_aux=True), args))
    lr_schedule = get_lr_schedule(args)
    rec = init_recorder()
    forget_stats = init_forget_stats(args) if args.track_forgetting else None

    # log and save initial state
    save_args(args, args.save_dir)
    test_loss, test_acc = test(test_step, state, X_test, Y_test, args.test_batch_size)
    rec = record_test_stats(rec, args.ckpt, test_loss, test_acc)
    checkpoints.save_checkpoint(args.save_dir + '/ckpts', state, step=args.ckpt)

    # train loop
    for t, idxs, x, y in train_batches(I_train, X_train, Y_train, args):
        lr = lr_schedule(t)
        state, logits, loss, acc = train_step(state, x, y, lr)
        rec = record_train_stats(rec, t, loss.item(), acc.item(), lr)

        if args.track_forgetting:
            batch_accs = np.array(correct(logits, y).astype(int))
            forget_stats = update_forget_stats(forget_stats, idxs, batch_accs)

        if t % args.log_steps == 0:
            test_loss, test_acc = test(test_step, state, X_test, Y_test, args.test_batch_size)
            rec = record_test_stats(rec, t, test_loss, test_acc)

            # save checkpoint
            checkpoints.save_checkpoint(args.save_dir + '/ckpts', state, step=t)

    # wrap up
    save_recorder(args.save_dir, rec)
