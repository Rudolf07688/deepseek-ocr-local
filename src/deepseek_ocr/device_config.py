"""
Device configuration and detection for DeepSeek-OCR.

Supports:
- CUDA (NVIDIA GPUs on Windows/Linux)
- MPS (Metal Performance Shaders on macOS)
- CPU (fallback)

Usage:
    from src.deepseek_ocr.device_config import DeviceConfig, get_device_config

    # Auto-detect
    config = get_device_config()

    # Explicit selection
    config = get_device_config('mps')
"""

import torch
import logging
from typing import Dict, Literal, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DeviceConfig:
    """Configuration for model inference on a specific device."""

    device: str  # 'cuda', 'mps', 'cpu'
    dtype: torch.dtype  # torch.float32, torch.float16, torch.bfloat16
    use_flash_attention: bool  # Flash Attention 2 support
    device_name: str  # Friendly name (e.g., "NVIDIA RTX 3080 Ti")
    vram_gb: Optional[float]  # Estimated available VRAM in GB
    recommended_mode: str  # Recommended inference mode ('small', 'base', 'large', 'gundam')
    supports_half_precision: bool  # Can use float16/bfloat16

    def __str__(self) -> str:
        """Human-readable device configuration."""
        dtype_name = str(self.dtype).split('.')[-1]
        return (
            f"Device: {self.device_name} ({self.device})\n"
            f"  Data Type: {dtype_name}\n"
            f"  Flash Attention: {self.use_flash_attention}\n"
            f"  Recommended Mode: {self.recommended_mode}\n"
            f"  VRAM: {self.vram_gb}GB" if self.vram_gb else "  VRAM: Unknown"
        )


def detect_cuda() -> Optional[DeviceConfig]:
    """Detect CUDA device (NVIDIA GPU)."""
    if not torch.cuda.is_available():
        return None

    try:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        # Determine recommended mode based on VRAM
        if vram_gb >= 24:  # 4080, A6000, etc.
            recommended_mode = 'gundam'
        elif vram_gb >= 12:  # 3080 Ti, 2080 Ti, etc.
            recommended_mode = 'base'  # Can handle 'large' too
        elif vram_gb >= 8:  # 3070 Ti, etc.
            recommended_mode = 'base'
        else:
            recommended_mode = 'small'

        logger.info(f"CUDA detected: {device_name} ({vram_gb:.1f}GB VRAM)")

        return DeviceConfig(
            device='cuda',
            dtype=torch.bfloat16,  # NVIDIA GPU with bfloat16
            use_flash_attention=True,
            device_name=device_name,
            vram_gb=vram_gb,
            recommended_mode=recommended_mode,
            supports_half_precision=True
        )
    except Exception as e:
        logger.warning(f"CUDA detection failed: {e}")
        return None


def detect_mps() -> Optional[DeviceConfig]:
    """Detect Metal Performance Shaders (macOS GPU)."""
    # MPS is available in PyTorch 1.12+ on M1/M2/M3 Macs
    if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
        return None

    try:
        # Try to allocate a tensor to verify MPS works
        test_tensor = torch.zeros(1, device='mps')
        test_tensor.to('cpu')  # Clean up

        device_name = "Apple Metal (MPS)"

        # macOS Unified Memory (M1/M2/M3) typically: 8-24GB shared
        # Conservative estimate: assume 8GB available for inference
        vram_gb = 8.0

        logger.info(f"MPS (Metal Performance Shaders) detected on macOS")

        return DeviceConfig(
            device='mps',
            dtype=torch.float32,  # MPS doesn't support bfloat16 yet
            use_flash_attention=False,  # MPS doesn't support Flash Attention
            device_name=device_name,
            vram_gb=vram_gb,
            recommended_mode='base',  # Conservative default
            supports_half_precision=False  # MPS has limited precision support
        )
    except Exception as e:
        logger.warning(f"MPS detection failed: {e}")
        return None


def get_device_config(device: Optional[str] = None) -> DeviceConfig:
    """
    Get device configuration with automatic detection.

    Args:
        device: Explicitly specify device ('cuda', 'mps', 'cpu', or None for auto-detect)

    Returns:
        DeviceConfig with optimal settings for the detected device

    Priority (if None):
        1. CUDA (NVIDIA GPU)
        2. MPS (Apple Metal)
        3. CPU (fallback)
    """
    if device is None:
        # Auto-detect: prefer CUDA, then MPS, fall back to CPU
        cuda_config = detect_cuda()
        if cuda_config:
            return cuda_config

        mps_config = detect_mps()
        if mps_config:
            return mps_config

        logger.warning("No GPU detected, falling back to CPU (this will be very slow)")
        return DeviceConfig(
            device='cpu',
            dtype=torch.float32,
            use_flash_attention=False,
            device_name='CPU',
            vram_gb=None,
            recommended_mode='tiny',  # Smallest mode for CPU
            supports_half_precision=False
        )

    # Explicit device selection
    device = device.lower().strip()

    if device == 'cuda':
        cuda_config = detect_cuda()
        if cuda_config:
            return cuda_config
        raise RuntimeError("CUDA requested but not available. Ensure NVIDIA drivers and CUDA toolkit are installed.")

    elif device == 'mps':
        mps_config = detect_mps()
        if mps_config:
            return mps_config
        raise RuntimeError("MPS requested but not available. This requires macOS 12.3+ with M1/M2/M3 chip.")

    elif device == 'cpu':
        return DeviceConfig(
            device='cpu',
            dtype=torch.float32,
            use_flash_attention=False,
            device_name='CPU',
            vram_gb=None,
            recommended_mode='tiny',
            supports_half_precision=False
        )

    else:
        raise ValueError(f"Invalid device: {device}. Choose from 'cuda', 'mps', 'cpu'")


def validate_model_on_device(config: DeviceConfig) -> bool:
    """
    Validate that the model can run on the detected device.

    Performs basic checks without loading the full model.
    """
    try:
        # Test tensor operations
        test_shape = (1, 3, 224, 224)  # Typical image shape
        test_tensor = torch.randn(test_shape, dtype=config.dtype)

        if config.device == 'cuda':
            test_tensor = test_tensor.cuda()
        elif config.device == 'mps':
            test_tensor = test_tensor.to('mps')

        # Test basic operations
        _ = test_tensor.half() if config.supports_half_precision else test_tensor

        logger.info(f"Device validation passed for {config.device}")
        return True

    except Exception as e:
        logger.error(f"Device validation failed: {e}")
        return False


def print_device_info(config: Optional[DeviceConfig] = None) -> None:
    """Pretty-print device information."""
    if config is None:
        config = get_device_config()

    print("\n" + "=" * 60)
    print("Device Configuration")
    print("=" * 60)
    print(config)
    print("=" * 60 + "\n")
