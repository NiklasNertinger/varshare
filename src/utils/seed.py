"""
Determinism and Seeding Utility

This module centralizes the seeding of all random number generators
(Python random, NumPy, PyTorch, and CUDA) to ensure mathematical
determinism and strict reproducibility across Multi-Task RL runs.
"""

import random
import numpy as np
import torch

def set_global_seeds(seed: int, use_cuda: bool = True) -> None:
    """
    Locks all random number generators to a specific seed.
    
    Args:
        seed (int): The master seed to use.
        use_cuda (bool): Whether to enforce CuDNN determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available() and use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enforce deterministic algorithm selection in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
