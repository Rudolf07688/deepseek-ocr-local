# Device Support for DeepSeek-OCR

DeepSeek-OCR now supports multiple devices for inference:
- **NVIDIA CUDA** (Windows/Linux with NVIDIA GPUs)
- **Apple Metal (MPS)** (macOS with Apple Silicon: M1/M2/M3/M4)
- **CPU** (fallback, not recommended for production)

---

## Quick Start

### Windows (NVIDIA 2080 Ti / RTX 3080 Ti)

```bash
# Docker (recommended)
docker-compose run --rm deepseek-ocr uv run python3 quick_test.py /app/input/sample.pdf

# Native (with Python installed)
python quick_test.py ./input/sample.pdf
```

### macOS (Apple Silicon M1/M2/M3/M4)

```bash
# Using --mac flag (recommended)
python quick_test.py --mac ./input/sample.pdf

# Or explicit device selection
python quick_test.py --device mps ./input/sample.pdf
```

---

## Detailed Setup by Platform

### Windows (NVIDIA GPU)

#### Requirements
- NVIDIA GPU (2080 Ti, RTX 3080 Ti, 3090, etc.)
- CUDA 12.1+ and cuDNN 8
- NVIDIA Docker runtime (for Docker setup)
- Python 3.11+ (for native setup)

#### Docker Setup (Recommended)
No changes needed! The existing `docker-compose.yml` and `Dockerfile` are already configured for Windows + NVIDIA.

```bash
# Build image
docker-compose build

# Run quick test
docker-compose run --rm deepseek-ocr uv run python3 quick_test.py /app/input/sample.pdf

# Run with specific mode
docker-compose run --rm deepseek-ocr uv run python3 quick_test.py --mode large /app/input/sample.pdf

# Interactive shell
docker-compose run --rm deepseek-ocr /bin/bash
```

#### Native Setup (No Docker)

1. **Install Python 3.11**:
   ```bash
   # Windows: Use python.org or Windows Store
   # Verify: python --version
   ```

2. **Install CUDA 12.1 and cuDNN**:
   - https://developer.nvidia.com/cuda-12-1-0-download-archive
   - https://developer.nvidia.com/cudnn (requires free registration)

3. **Install dependencies**:
   ```bash
   # Create virtual environment
   python -m venv venv
   venv\Scripts\activate

   # Install PyTorch with CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # Install other dependencies
   pip install -r requirements.txt
   ```

4. **Verify GPU access**:
   ```bash
   python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
   ```

5. **Run OCR**:
   ```bash
   python quick_test.py ./input/sample.pdf
   ```

---

### macOS (Apple Silicon M1/M2/M3/M4)

#### Requirements
- macOS 12.3 or later
- Apple Silicon chip (M1, M2, M3, M4)
- Python 3.11+ (native installation)
- Xcode Command Line Tools

#### Native Setup (Recommended for macOS)

1. **Install Python 3.11**:
   ```bash
   # Using Homebrew
   brew install python@3.11

   # Verify
   python3 --version
   ```

2. **Install Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

3. **Install system dependencies**:
   ```bash
   # poppler-utils for PDF processing
   brew install poppler
   ```

4. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

5. **Install PyTorch with MPS support**:
   ```bash
   # PyTorch 2.0+ includes native MPS support
   pip install torch torchvision torchaudio
   ```

   > **Note**: PyTorch will automatically detect and use MPS on Apple Silicon. No special installation needed!

6. **Install other dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

7. **Verify MPS access**:
   ```bash
   python3 -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"
   ```

8. **Run OCR with MPS**:
   ```bash
   # Using --mac flag (easiest)
   python quick_test.py --mac ./input/sample.pdf

   # Or explicit device selection
   python quick_test.py --device mps ./input/sample.pdf
   ```

---

## CLI Usage

### quick_test.py (Single PDF Processing)

```bash
# Basic usage (auto-detect device)
python quick_test.py ./input/sample.pdf

# macOS with MPS
python quick_test.py --mac ./input/sample.pdf

# Explicit device selection
python quick_test.py --device cuda ./input/sample.pdf
python quick_test.py --device mps ./input/sample.pdf
python quick_test.py --device cpu ./input/sample.pdf

# Specify OCR mode (default: base)
python quick_test.py --mode large ./input/sample.pdf
python quick_test.py --mode gundam ./input/sample.pdf

# Custom DPI for PDF conversion
python quick_test.py --dpi 300 ./input/sample.pdf

# Enable Flash Attention 2 (NVIDIA only)
python quick_test.py --flash-attention ./input/sample.pdf

# Combination
python quick_test.py --mac --mode base --dpi 200 ./input/sample.pdf
```

**Help**:
```bash
python quick_test.py --help
```

### advanced_examples.py (Batch Processing & Examples)

```bash
# Interactive menu
python advanced_examples.py

# Run specific example with auto-detect
python advanced_examples.py 1

# Run specific example with explicit device
python advanced_examples.py --device cuda 2
python advanced_examples.py --mac 3

# Examples:
# 1. Batch Processing
# 2. Structured Extraction
# 3. Progressive Quality
# 4. Memory Efficient (Large PDF)
# 5. LangChain Integration
```

**Help**:
```bash
python advanced_examples.py --help
```

---

## Mode Selection by Device & Hardware

### NVIDIA GPUs

| GPU Model | VRAM | Recommended Mode | Alternative Modes |
|-----------|------|------------------|--------------------|
| RTX 2080 Ti | 11GB | `base` | `small`, `large` |
| RTX 3080 Ti | 12GB | `base` | `small`, `large`, `gundam` |
| RTX 4090 | 24GB | `gundam` | `large` |
| A6000 | 48GB | `gundam` | All modes |

### Apple Metal (MPS)

| Chip | Unified Memory | Recommended Mode | Notes |
|------|---|------------------|----|
| M1 (8GB base) | 8GB | `base` | May need `small` for complex docs |
| M1 Pro/Max | 16-32GB | `large` | Can handle most modes |
| M2/M3/M4 | 8-24GB | `base` to `large` | Performance scales with memory |

**MPS-Specific Notes**:
- MPS uses unified memory (GPU shares system RAM)
- No bfloat16 support (uses float32)
- No Flash Attention 2 support
- Generally 30-40% slower than comparable NVIDIA GPU

---

## Performance Benchmarks

### Model Loading Time
- **NVIDIA CUDA**: 5-10s (first run), 2-3s (cached)
- **Apple MPS**: 8-15s (first run), 3-5s (cached)
- **CPU**: 15-30s (not recommended)

### Per-Page Processing (Base Mode)
- **NVIDIA RTX 3080 Ti**: ~2-3s per page
- **Apple M1/M2**: ~4-6s per page
- **Apple M1 Pro**: ~3-4s per page
- **CPU (Intel i7)**: ~30-60s per page

### Memory Usage
- **Model**: ~6.6GB (constant)
- **Inference overhead**: ~1-2GB
- **Total**: ~8-9GB (NVIDIA), ~7-8GB (MPS)

---

## Device Configuration Module

The `src/deepseek_ocr/device_config.py` module provides:

```python
from src.deepseek_ocr.device_config import get_device_config, print_device_info

# Auto-detect device
config = get_device_config()
print_device_info(config)

# Explicit selection
config = get_device_config('mps')
config = get_device_config('cuda')
config = get_device_config('cpu')
```

**Device Configuration includes**:
- Optimal dtype (bfloat16 for CUDA, float32 for MPS/CPU)
- Flash Attention 2 availability
- Recommended inference mode
- Available VRAM estimate

---

## Troubleshooting

### Windows (NVIDIA)

| Issue | Solution |
|-------|----------|
| CUDA not available | Install NVIDIA driver + CUDA 12.1 + cuDNN |
| Out of memory | Use `--mode small` instead of `--mode large` |
| docker-compose command not found | Install Docker Desktop for Windows |
| GPU memory not released | Close Docker and restart, or `torch.cuda.empty_cache()` |

### macOS (Apple Metal)

| Issue | Solution |
|-------|----------|
| MPS not available | Update macOS to 12.3+ and PyTorch to 2.0+ |
| `RuntimeError: MPS requested but not available` | Run `python3 -c "import torch; print(torch.backends.mps.is_available())"` to verify |
| Slow performance | MPS is inherently slower than NVIDIA. Use `--mode small` for speed. |
| `torch.mps.is_available() returns False` | Ensure PyTorch 2.0+ is installed: `pip install --upgrade torch` |

### Both Platforms

| Issue | Solution |
|-------|----------|
| Model download fails | Check internet + HuggingFace accessibility. Model is ~6.6GB. |
| `ImportError: pdf2image not found` | Install: `pip install pdf2image` (Windows needs Poppler, macOS needs `brew install poppler`) |
| Out of disk space | Model cache is ~6.6GB. Ensure 15GB free disk space. |

---

## Programmatic Usage

### Using the Device Config

```python
from src.deepseek_ocr.device_config import get_device_config

# Auto-detect
config = get_device_config()
print(f"Device: {config.device}")
print(f"Recommended mode: {config.recommended_mode}")

# Explicit
config = get_device_config('mps')
if config.use_flash_attention:
    print("Flash Attention available!")
```

### Using the Processor

```python
from deepseek_ocr_pdf import DeepSeekOCRProcessor

# Auto-detect device
processor = DeepSeekOCRProcessor()

# Explicit device
processor_cuda = DeepSeekOCRProcessor(device='cuda')
processor_mps = DeepSeekOCRProcessor(device='mps')

# Process PDF
results = processor.process_pdf(
    pdf_path='./input/sample.pdf',
    output_dir='./output',
    mode='base',
    dpi=200
)
```

---

## Docker on macOS

If you want to use Docker on macOS (not recommended, as native is faster):

```bash
# Build for macOS (CPU fallback, no GPU acceleration)
docker build -f Dockerfile.macos -t deepseek-ocr:macos .

# Run
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output deepseek-ocr:macos \
  python quick_test.py /app/input/sample.pdf
```

> **Note**: Docker on macOS cannot directly access GPU. Use native installation for best performance.

---

## Environment Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `HF_HOME` | `.cache/huggingface` | HuggingFace model cache location |
| `TRANSFORMERS_CACHE` | `HF_HOME` | Model cache for transformers |
| `CUDA_VISIBLE_DEVICES` | `0` | Which GPU to use (NVIDIA only) |

**Example**:
```bash
export HF_HOME="/custom/path/to/models"
python quick_test.py ./input/sample.pdf
```

---

## Contributing

When adding features, please ensure:
1. Device-agnostic operations (no CUDA-specific code)
2. Test on both NVIDIA (if available) and CPU
3. Handle device-specific optimizations gracefully
4. Document any device limitations

---

## References

- PyTorch Device API: https://pytorch.org/docs/stable/torch.html#device
- PyTorch MPS Backends: https://pytorch.org/docs/stable/backends.html#torch.backends.mps
- NVIDIA CUDA Setup: https://docs.nvidia.com/cuda/cuda-installation-guide-windows/
- macOS Metal Performance Shaders: https://developer.apple.com/metal/
