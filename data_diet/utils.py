import json
import numpy as np
import os
import shutil
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
    return args


def make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def set_global_seed(seed=0):
    np.random.seed(seed)
