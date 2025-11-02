# Quick Start Guide - Multi-Device DeepSeek-OCR

This guide gets you up and running with DeepSeek-OCR on Windows (NVIDIA GPU) or macOS (Apple Silicon).

---

## 30-Second Setup

### Windows (NVIDIA GPU)

```bash
# Using Docker (recommended)
docker-compose run --rm deepseek-ocr uv run python3 quick_test.py /app/input/sample.pdf

# Or native Python
python quick_test.py ./input/sample.pdf
```

### macOS (Apple Silicon M1/M2/M3)

```bash
python quick_test.py --mac ./input/sample.pdf
```

---

## Installation

### Windows (Native - No Docker)

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install CUDA (if not already installed)
# Download from: https://developer.nvidia.com/cuda-12-1-0-download-archive

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 4. Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 5. Run
python quick_test.py ./input/sample.pdf
```

### macOS (Native - Recommended)

```bash
# 1. Install Python 3.11 (if needed)
brew install python@3.11

# 2. Install system dependencies
brew install poppler

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install PyTorch (MPS support is automatic)
pip install torch torchvision torchaudio

# 5. Install other dependencies
pip install -r requirements.txt

# 6. Verify MPS
python3 -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# 7. Run with MPS
python quick_test.py --mac ./input/sample.pdf
```

---

## Common Commands

### Processing a Single PDF

```bash
# Auto-detect device (best)
python quick_test.py ./input/sample.pdf

# Explicit device selection
python quick_test.py --device cuda ./input/sample.pdf
python quick_test.py --device mps ./input/sample.pdf
python quick_test.py --device cpu ./input/sample.pdf

# macOS shorthand
python quick_test.py --mac ./input/sample.pdf

# Different quality modes
python quick_test.py --mode small ./input/sample.pdf   # Fast, lower quality
python quick_test.py --mode base ./input/sample.pdf    # Balanced (default)
python quick_test.py --mode large ./input/sample.pdf   # Slower, better quality
python quick_test.py --mode gundam ./input/sample.pdf  # Slowest, best quality

# Higher DPI for better scanned document quality
python quick_test.py --dpi 300 ./input/sample.pdf

# Combine options
python quick_test.py --mac --mode large --dpi 300 ./input/sample.pdf
```

### Batch Processing

```bash
# Interactive menu
python advanced_examples.py

# Run example 1 (Batch Processing) with auto-detect
python advanced_examples.py 1

# Run example 1 with MPS
python advanced_examples.py --mac 1

# Run example 1 with explicit CUDA
python advanced_examples.py --device cuda 1
```

---

## Mode Selection Guide

Choose based on your needs:

| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| `small` | ⚡⚡⚡ Fast | Low | Quick text extraction |
| `base` | ⚡⚡ Moderate | Medium | **Default, most use cases** |
| `large` | ⚡ Slower | High | Complex documents, tables |
| `gundam` | Very slow | Excellent | Publication-quality OCR |

---

## Device Selection Guide

### Auto-Detect (Recommended)

```bash
python quick_test.py ./input/sample.pdf
```

DeepSeek-OCR automatically selects:
1. **NVIDIA CUDA** if available (fastest on Windows/Linux)
2. **Apple Metal (MPS)** if available (good on macOS)
3. **CPU** as fallback (slow, not recommended)

### Explicit Selection

```bash
# Windows/Linux
python quick_test.py --device cuda ./input/sample.pdf

# macOS (Apple Metal)
python quick_test.py --device mps ./input/sample.pdf
# or shorthand
python quick_test.py --mac ./input/sample.pdf

# CPU (fallback)
python quick_test.py --device cpu ./input/sample.pdf
```

---

## Troubleshooting

### "CUDA not available"

**Problem**: You're on Windows but CUDA isn't detected.

**Solution**:
```bash
# Check if NVIDIA driver is installed
nvidia-smi

# If not found, install from:
# https://www.nvidia.com/Download/driverDetails.aspx

# Then install CUDA 12.1
# https://developer.nvidia.com/cuda-12-1-0-download-archive

# Reinstall PyTorch
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### "MPS not available" (macOS)

**Problem**: PyTorch can't find Metal Performance Shaders.

**Solution**:
```bash
# Update PyTorch to latest (2.0+)
pip install --upgrade torch torchvision torchaudio

# Verify
python3 -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"

# If still not available, update macOS to 12.3+
```

### "Out of Memory"

**Problem**: GPU runs out of memory.

**Solution**:
```bash
# Use a smaller mode
python quick_test.py --mode small ./input/sample.pdf

# Or process fewer pages at once
# (advanced_examples.py handles batching automatically)
```

### "poppler-utils not found" (macOS)

**Problem**: PDF to image conversion fails.

**Solution**:
```bash
# Install poppler
brew install poppler

# Verify
pdfinfo --version
```

---

## Output Structure

```
output/
└── sample/                    # PDF name
    ├── pages/
    │   ├── page_001.png
    │   ├── page_002.png
    │   └── ...
    ├── page_001/
    │   ├── result.mmd        # Single page OCR result
    │   └── ...
    ├── page_002/
    │   ├── result.mmd
    │   └── ...
    └── sample_combined.md    # All pages combined
```

---

## Performance Expectations

### Model Loading
- **First run**: 8-15 seconds (downloads ~6.6GB model)
- **Subsequent runs**: 2-5 seconds (cached)

### Per-Page Processing (Base Mode)
- **NVIDIA RTX 3080 Ti**: 2-3 seconds
- **NVIDIA RTX 2080 Ti**: 3-4 seconds
- **Apple M1/M2**: 4-6 seconds
- **Apple M1 Pro**: 3-4 seconds
- **CPU**: 30-60 seconds (not recommended)

---

## Getting Help

### View Command Help

```bash
python quick_test.py --help
python advanced_examples.py --help
```

### Detailed Documentation

See `DEVICE_SUPPORT.md` for:
- Complete device setup instructions
- Platform-specific troubleshooting
- Advanced configuration options
- Performance benchmarks

---

## Next Steps

1. **Test with your PDFs**: Place them in `./input/` and run
2. **Try different modes**: Compare speed vs quality
3. **Batch process**: Use `advanced_examples.py` for multiple PDFs
4. **Integrate**: Use `DeepSeekOCRProcessor` in your code

---

## File Structure

```
deepseek-ocr/
├── quick_test.py              # Single PDF processing (START HERE)
├── advanced_examples.py        # Batch & advanced usage
├── deepseek_ocr_pdf.py        # Core processor class
├── src/
│   └── deepseek_ocr/
│       ├── __init__.py        # Package exports
│       └── device_config.py   # Device detection & config
├── Dockerfile                  # NVIDIA CUDA setup (Windows/Linux)
├── Dockerfile.macos           # macOS Docker setup (optional)
├── docker-compose.yml         # Docker orchestration
├── requirements.txt           # Python dependencies
├── DEVICE_SUPPORT.md          # Complete device guide
└── QUICK_START.md             # This file
```

---

## Tips & Tricks

### Use in Python Scripts

```python
from deepseek_ocr_pdf import DeepSeekOCRProcessor

# Auto-detect device
processor = DeepSeekOCRProcessor()

# Or specify device
processor = DeepSeekOCRProcessor(device='mps')

# Process PDF
results = processor.process_pdf(
    pdf_path='./input/sample.pdf',
    output_dir='./output',
    mode='base',
    dpi=200
)
```

### Check Device Configuration

```python
from src.deepseek_ocr.device_config import get_device_config, print_device_info

config = get_device_config()
print_device_info(config)
```

### Process Large PDFs Efficiently

Use `advanced_examples.py` example 4 (Memory Efficient Large PDF) for 100+ page PDFs.

---

## Known Limitations

- **MPS (macOS)**: ~30-40% slower than comparable NVIDIA GPU
- **CPU**: Very slow, ~10-20x slower than GPU (not recommended for production)
- **Model**: Requires 6.6GB disk space for model cache
- **VRAM**: Minimum 8GB for `base` mode; 12GB+ recommended

---

## Questions?

Check `DEVICE_SUPPORT.md` for comprehensive documentation, or see `CLAUDE.md` for architecture details.
