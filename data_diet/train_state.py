from flax.training import checkpoints
from flax.struct import dataclass as flax_dataclass
from jax import jit, random
import jax.numpy as jnp
import optax  # Import optax for optimization
import time
from typing import Any
from .models import get_num_params


@flax_dataclass
class TrainState:
    params: Any  # Parameters of the model
    opt_state: Any  # Optimizer state
    model: Any  # Model state (e.g., batch norm stats)


def create_train_state(args, model):
    @jit
    def init(*args):
        return model.init(*args)
    
    key, input = random.PRNGKey(args.model_seed), jnp.ones((1, *args.image_shape), model.dtype)
    output = init(key, input)  # This returns a dictionary with the model state and parameters
    
    # Extract the model parameters and state
    params = output['params']
    model_state = output  # You might want to include more than just 'params' in the model state, adjust as needed
    
    if not hasattr(args, 'nesterov'): 
        args.nesterov = False

    # Create an Optax optimizer
    optimizer = optax.chain(
        optax.add_decayed_weights(args.weight_decay),
        optax.sgd(args.lr, momentum=args.beta, nesterov=args.nesterov)
    )
    opt_state = optimizer.init(params)
    
    # Initialize TrainState
    train_state = TrainState(params=params, opt_state=opt_state, model=model_state)
    return train_state, optimizer


def get_train_state(args, model):
    time_start = time.time()
    print('get train state... ', end='')
    
    train_state, optimizer = create_train_state(args, model)
    
    if args.load_dir:
        print(f'load from {args.load_dir}/ckpts/checkpoint_{args.ckpt}... ', end='')
        train_state = checkpoints.restore_checkpoint(args.load_dir + '/ckpts', train_state, args.ckpt)
    
    args.num_params = get_num_params(train_state.params)
    print(f'{int(time.time() - time_start)}s')
    
    return train_state, optimizer, args