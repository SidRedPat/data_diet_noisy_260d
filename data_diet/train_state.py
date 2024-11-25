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
    params: Any
    opt_state: optax.OptState
    model_state: Any

    def __post_init__(self):
        # Define the gradient transformation pipeline
        self.tx = optax.chain(
            optax.add_decayed_weights(0.0005),  # Example weight decay
            optax.sgd(learning_rate=0.1, momentum=0.9, nesterov=True)
        )




def create_train_state(args, model):
    """
    Initialize the training state.

    Args:
        args: Training arguments containing hyperparameters.
        model: The model to be trained.

    Returns:
        Initialized TrainState.
    """
    key, input_shape = random.PRNGKey(args.model_seed), (1, *args.image_shape)
    init_vars = model.init(key, jnp.ones(input_shape))

    # Extract trainable parameters and model state
    params = init_vars['params']
    model_state = {k: v for k, v in init_vars.items() if k != 'params'}

    # Create the training state
    train_state = TrainState(
        step=0,
        params=params,
        opt_state=optax.chain(  # Initialize opt_state based on args
            optax.add_decayed_weights(args.weight_decay),
            optax.sgd(learning_rate=args.lr, momentum=args.beta, nesterov=args.nesterov)
        ).init(params),
        model_state=model_state
    )
    return train_state



def get_train_state(args, model):
    state = create_train_state(args, model)
    if args.load_dir:
        state = checkpoints.restore_checkpoint(args.load_dir + '/ckpts', state, args.ckpt)
    return state, args
