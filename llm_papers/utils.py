import random

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_flops(model, inputs, with_backward=False):
    """
    Adapted from https://alessiodevoto.github.io/Compute-Flops-with-Pytorch-built-in-flops-counter/
    """
    istrain = model.training
    model.eval()

    inputs = inputs if isinstance(inputs, torch.Tensor) else torch.randn(inputs)

    flop_counter = FlopCounterMode(display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inputs).sum().backward()
        else:
            model(inputs)
    total_flops = flop_counter.get_total_flops()
    if istrain:
        model.train()
    return total_flops
