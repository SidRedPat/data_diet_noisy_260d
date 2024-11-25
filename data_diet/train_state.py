from flax.struct import dataclass
from flax.training import checkpoints
import optax
from jax import random
import jax.numpy as jnp
from typing import Any


@dataclass
class TrainState:
    step: int
    params: Any
    opt_state: Any


def create_train_state(args, model):
    key, input_shape = random.PRNGKey(args.model_seed), (1, *args.image_shape)
    model_state, params = model.init(key, jnp.ones(input_shape)).pop("params")

    optimizer = optax.chain(
        optax.add_decayed_weights(args.weight_decay),
        optax.sgd(learning_rate=args.lr, momentum=args.beta, nesterov=args.nesterov)
    )
    opt_state = optimizer.init(params)
    return TrainState(step=0, params=params, opt_state=opt_state), model_state


def get_train_state(args, model):
    state, model_state = create_train_state(args, model)
    if args.load_dir:
        state = checkpoints.restore_checkpoint(args.load_dir + '/ckpts', state, args.ckpt)
    return state, args
