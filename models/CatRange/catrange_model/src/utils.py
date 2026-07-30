"""
Utility functions for CatRange pipeline

Core functionality:
- Device management (GPU/CPU)
- Random seed setting for reproducibility
- Tensor/numpy conversions
- Data loading (safe PyTorch compatibility)
"""

import torch
import numpy as np
import random
import warnings
from pathlib import Path
from typing import Union, Any

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# ============================================================================
# DEVICE MANAGEMENT
# ============================================================================
def check_device() -> torch.device:
    """Return cuda device if available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info(device: torch.device = None):
    """Print GPU/CPU information."""
    if device is None:
        device = check_device()
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("Using CPU")


# ============================================================================
# RANDOM SEED MANAGEMENT
# ============================================================================
def set_seed(seed: int = 42):
    """Set random seed for reproducibility across PyTorch, NumPy, and random."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# TENSOR/NUMPY CONVERSION
# ============================================================================
def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a NumPy array (handles CUDA tensors)."""
    return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()


# ============================================================================
# FILE I/O
# ============================================================================
def safe_load(path: Union[str, Path], device: torch.device = None) -> Any:
    """Safely load PyTorch files with backward compatibility."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if device is None:
        device = check_device()
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
