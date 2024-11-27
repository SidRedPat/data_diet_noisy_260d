# python get_run_score.py <ROOT:str> <EXP:str> <RUN:int> <STEP:int> <BATCH_SZ:int> <TYPE:str>

from data_diet.data import load_data
from data_diet.scores import compute_scores
from data_diet.utils import get_fn_params_state, load_args
import sys
import numpy as np
import os

# Command-line arguments
ROOT = sys.argv[1]
EXP = sys.argv[2]
RUN = int(sys.argv[3])
STEP = int(sys.argv[4])
BATCH_SZ = int(sys.argv[5])
TYPE = sys.argv[6]

# Experiment setup
run_dir = os.path.join(ROOT, f'exps/{EXP}/run_{RUN}')
args = load_args(run_dir)
args.load_dir = run_dir
args.ckpt = STEP

# Load data and model
_, X, Y, _, _, args = load_data(args)
fn, params, state = get_fn_params_state(args)

# Compute scores
scores = compute_scores(fn, params, state, X, Y, BATCH_SZ, TYPE)

# Handle score saving
path_name_map = {
    'l2_error': 'error_l2_norm_scores',
    'grad_norm': 'grad_norm_scores',
    'el2n': 'el2n_scores'
}

path_name = path_name_map.get(TYPE)
if not path_name:
    raise ValueError(f"Unknown score type: {TYPE}")

save_dir = os.path.join(run_dir, path_name)
save_path = os.path.join(save_dir, f'ckpt_{STEP}.npy')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

np.save(save_path, scores)
print(f"Scores saved to {save_path}")