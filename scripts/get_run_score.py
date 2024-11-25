# python get_run_score.py <ROOT:str> <EXP:str> <RUN:int> <STEP:int> <BATCH_SZ:int> <TYPE:str>

from data_diet.data import load_data
from data_diet.scores import compute_scores
from data_diet.utils import get_fn_params_state, load_args
import sys
import os
import numpy as np

# Setup
ROOT = sys.argv[1]
EXP = sys.argv[2]
RUN = int(sys.argv[3])
STEP = int(sys.argv[4])
BATCH_SZ = int(sys.argv[5])
TYPE = sys.argv[6]

# Directories
RUN_DIR = os.path.join(ROOT, 'exps', EXP, f'run_{RUN}')
SAVE_DIR = os.path.join(RUN_DIR, 'error_l2_norm_scores' if TYPE == 'l2_error' else 'grad_norm_scores')
SAVE_PATH = os.path.join(SAVE_DIR, f'ckpt_{STEP}.npy')

# Load arguments
args = load_args(RUN_DIR)
args.load_dir = RUN_DIR
args.ckpt = STEP

# Load data
_, X, Y, _, _, args = load_data(args)

# Load model function, parameters, and state
fn, params, state = get_fn_params_state(args)

# Compute scores
scores = compute_scores(fn, params, state, X, Y, BATCH_SZ, TYPE)

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Save scores
np.save(SAVE_PATH, scores)
