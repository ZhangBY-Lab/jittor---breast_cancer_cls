import jittor as jt
import numpy as np
import os
import errno
import random
import torch


def initialize_random_states(base_value=42):
    """Configure deterministic behavior across all random number generators.
    
    Args:
        base_value (int, optional): Master seed value that propagates to all 
                                   RNG systems. Defaults to 42.
    """

    rng_configurators = {
        'standard_lib': random.seed,
        'numerical': np.random.seed,
        'jittor_engine': jt.set_global_seed
    }
    
    for configure in rng_configurators.values():
        configure(base_value)

    jt.flags.use_cuda = 1  
    
    os.environ['PYTHONHASHSEED'] = str(base_value)


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def device_usr():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device