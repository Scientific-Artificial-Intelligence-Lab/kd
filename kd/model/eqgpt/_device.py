"""
Auto-detect compute device: CUDA > MPS > CPU.

Usage in other scripts:
    from _device import device, DEVICE_STR
    # device: torch.device object
    # DEVICE_STR: 'cuda' / 'mps' / 'cpu' string (for NN constructor's Device param)
"""

import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

DEVICE_STR = str(device)


def seed_all(seed: int = 525) -> None:
    """Set random seed for reproducibility across all backends."""
    torch.manual_seed(seed)  # covers CPU + CUDA + MPS
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def empty_cache() -> None:
    """Clear device memory cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_checkpoint(path: str):
    """Load a checkpoint with proper device mapping."""
    return torch.load(path, map_location=device)
