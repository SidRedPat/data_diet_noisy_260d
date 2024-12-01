import json
import numpy as np
import os
import shutil
from types import SimpleNamespace
from .models import get_apply_fn_test, get_model
from .train_state import get_train_state


def save_args(args, save_dir, verbose=True):
    save_path = os.path.join(save_dir, 'args.json')
    with open(save_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    if verbose:
        print(f"Args saved to {save_path}")


def load_args(load_dir, verbose=True):
    load_path = os.path.join(load_dir, 'args.json')
    with open(load_path, 'r') as f:
        args = json.load(f)
    if verbose:
        print(f"Args loaded from {load_path}")
    return SimpleNamespace(**args)  # Convert to SimpleNamespace



def make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def set_global_seed(seed=0):
    np.random.seed(seed)


def get_fn_params_state(args):
    """
    Prepares the forward pass function, model parameters, and model state.

    Args:
        args: Namespace object containing training arguments.

    Returns:
        fn: Forward pass function for evaluation.
        params: Model parameters.
        state: Model state.
    """
    # Initialize the model
    model = get_model(args)

    # Initialize training state (loads parameters and state from a checkpoint if available)
    state, _, _ = get_train_state(args, model)

    # Extract the forward function for testing
    fn = get_apply_fn_test(model)

    # Extract parameters and state
    params = state.params
    state = state.model_state

    return fn, params, state
