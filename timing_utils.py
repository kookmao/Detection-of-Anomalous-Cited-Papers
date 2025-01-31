import time
import torch
from functools import wraps

def print_tensor_device(tensor, name):
    """Print device information for a tensor"""
    if isinstance(tensor, torch.Tensor):
        print(f"{name} is on {tensor.device}")
    elif isinstance(tensor, (list, tuple)):
        print(f"{name} is a {type(tensor)} of tensors:")
        for i, t in enumerate(tensor):
            if isinstance(t, torch.Tensor):
                print(f"  {name}[{i}] is on {t.device}")
    elif isinstance(tensor, dict):
        print(f"{name} is a dict of tensors:")
        for k, v in tensor.items():
            if isinstance(v, torch.Tensor):
                print(f"  {name}[{k}] is on {v.device}")



def timing_decorator(func):
    """Decorator to track execution time of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class TimingContext:
    """Context manager for timing code blocks"""
    def __init__(self, name):
        self.name = name
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        self.end_time = time.time()
        print(f"{self.name} took {self.end_time - self.start_time:.4f} seconds")