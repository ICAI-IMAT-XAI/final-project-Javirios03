import random
import numpy as np
import torch
import os
import sys


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility across various libraries.

    Args:
        seed
        (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def print_system_info():
    """
    Print system and environment information for reproducibility documentation
    """
    import platform
    import torch
    
    print("=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)
    
    # Python version
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    
    # PyTorch
    print(f"\nPyTorch: {torch.__version__}")
    
    # CUDA
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Multi-processors: {props.multi_processor_count}")
    else:
        print(f"CUDA available: No (CPU mode)")