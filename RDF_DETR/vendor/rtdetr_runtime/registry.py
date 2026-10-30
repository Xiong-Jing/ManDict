"""Single-process registry shim; no global src namespace or eager imports."""
import torch
def register(value):
    return value
def is_dist_available_and_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()
def get_world_size():
    return torch.distributed.get_world_size() if is_dist_available_and_initialized() else 1
