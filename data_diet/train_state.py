from flax.struct import dataclass
from flax.training import checkpoints
import optax
from jax import random
import jax.numpy as jnp
from typing import Any
from flax.core import FrozenDict


@dataclass
class TrainState:
    step: int
    params: FrozenDict
    opt_state: optax.OptState
    model_state: FrozenDict


def create_train_state(args, model):
    key, input_shape = random.PRNGKey(args.model_seed), (1, *args.image_shape)
    init_vars = model.init(key, jnp.ones(input_shape))

    # Extract trainable parameters and model state
    params = init_vars['params']  # Trainable parameters
    model_state = {k: v for k, v in init_vars.items() if k != 'params'}  # Remaining state

    # Create the optimizer
    optimizer = optax.chain(
        optax.add_decayed_weights(args.weight_decay),
        optax.sgd(learning_rate=args.lr, momentum=args.beta, nesterov=args.nesterov)
    )
    opt_state = optimizer.init(params)

    return TrainState(step=0, params=params, opt_state=opt_state, model_state=model_state)


def get_train_state(args, model):
    state = create_train_state(args, model)
    if args.load_dir:
        state = checkpoints.restore_checkpoint(args.load_dir + '/ckpts', state, args.ckpt)
    return state, args
