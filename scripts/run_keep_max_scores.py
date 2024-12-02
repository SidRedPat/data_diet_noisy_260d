# python run_keep_max_scores.py <ROOT:str> <EXP:str> <SCORE_PATH:str> <SIZE:int> <RUN:int>

from data_diet.train import train
import os
import sys
from types import SimpleNamespace

# Setup
ROOT = sys.argv[1]
EXP = sys.argv[2]
SCORE_PATH = sys.argv[3]
SIZE = int(sys.argv[4])
RUN = int(sys.argv[5])
RAND_LABEL_FRAC = float(sys.argv[6])
RAND_LABEL_SEED = int(sys.argv[7])
EPOCHS = int(sys.argv[8])
META_MODEL_SEED, META_TRAIN_SEED, SEED_INCR = 42, 4242, 424242
EP_STEPS = 390
DATA_DIR = os.path.join(ROOT, "data")
EXPS_DIR = os.path.join(ROOT, "exps")

# Arguments
args = SimpleNamespace()
# Data
args.data_dir = DATA_DIR
args.dataset = "cifar10"
args.random_label_fraction = RAND_LABEL_FRAC
args.random_label_seed = RAND_LABEL_SEED
args.adaptive = False
# Subset selection
args.subset = "keep_max_scores"  # Specifies subset type for data.py
args.subset_size = SIZE  # Specifies size of the subset to keep
args.scores_path = SCORE_PATH  # Path to the EL2N scores
args.subset_offset = None  # No offset for keep_max_scores
args.random_subset_seed = None  # Not random
# Model
args.model = "resnet18_lowres"
args.model_seed = META_MODEL_SEED + RUN * SEED_INCR
args.load_dir = None
args.ckpt = 0
# Optimizer
args.lr = 0.1
args.beta = 0.9
args.weight_decay = 0.0005
args.nesterov = True
args.lr_vitaly = False
args.decay_factor = 0.2
args.decay_steps = [60 * EP_STEPS, 120 * EP_STEPS, 160 * EP_STEPS]
# Training
args.num_steps = EPOCHS * EP_STEPS
args.train_seed = META_TRAIN_SEED + RUN * SEED_INCR
args.train_batch_size = 1024
args.test_batch_size = 1024
args.augment = True  # Data augmentation enabled
args.track_forgetting = True  # Track forgetting events
# Checkpoints
args.save_dir = os.path.abspath(os.path.join(EXPS_DIR, f"{EXP}/size_{SIZE}/run_{RUN}"))
args.log_steps = EP_STEPS
args.early_step = 0
args.early_save_steps = None
args.save_steps = EP_STEPS
# Image shape (based on CIFAR-10 dataset)
args.image_shape = (32, 32, 3)

# Experiment
print(f"Training with top {SIZE} examples based on max EL2N scores...")
train(args)
