import torch

def get_default_dtype():
    """Determine the optimal default data type based on hardware"""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.float16
    else:
        return torch.float32

def get_device():
    """Determine the best available device for PyTorch."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')