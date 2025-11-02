"""DeepSeek-OCR Package - Multi-device OCR processing with DeepSeek Vision Models."""

try:
    from .device_config import DeviceConfig, get_device_config, print_device_info
    __all__ = ['DeviceConfig', 'get_device_config', 'print_device_info']
except ImportError:
    # Fallback if device_config is not available
    __all__ = []


def main() -> None:
    """Print package information."""
    print("DeepSeek-OCR - Multi-device Vision-Language OCR Pipeline")
    print("Supports: NVIDIA CUDA, Apple Metal (MPS), CPU")
    try:
        config = get_device_config()
        print(f"Auto-detected device: {config.device_name}")
    except Exception:
        print("Could not auto-detect device configuration")
