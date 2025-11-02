# Testing DeepSeek-OCR on Windows with NVIDIA 2080 Ti

This guide helps you test the refactored multi-device DeepSeek-OCR on your Windows machine with an NVIDIA RTX 2080 Ti.

## Prerequisites

You'll need to have installed on your Windows machine:
- ✅ **Python 3.11** (verified)
- ✅ **NVIDIA CUDA 12.1+** (verified in Dockerfile)
- ✅ **cuDNN 8** (for CUDA)
- ✅ **NVIDIA drivers** (up-to-date)

---

## Quick Setup (Windows Native - Recommended for Testing)

### Step 1: Create Virtual Environment

Open PowerShell in your project directory:

```powershell
# Navigate to project
cd C:\Users\rudol\Documents\dev\devcontainers\deepseek-ocr

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\Activate.ps1

# If activation fails, try:
venv\Scripts\activate.bat
```

### Step 2: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support (this is THE critical step)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch installed correctly with CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA Available: True
CUDA Device: NVIDIA GeForce RTX 2080 Ti
```

### Step 3: Install DeepSeek-OCR Dependencies

```powershell
# From requirements.txt
pip install -r requirements.txt

# Verify critical packages
python -c "from transformers import AutoModel; from pdf2image import convert_from_path; print('✓ All dependencies installed')"
```

### Step 4: Verify Device Configuration Works

```powershell
# Test device auto-detection
python -c "from src.deepseek_ocr.device_config import get_device_config, print_device_info; config = get_device_config(); print_device_info(config)"
```

Expected output:
```
============================================================
Device Configuration
============================================================
Device: NVIDIA GeForce RTX 2080 Ti (cuda)
  Data Type: bfloat16
  Flash Attention: True
  Recommended Mode: base
  VRAM: 11.0GB
============================================================
```

---

## Run the Test

Now test with the DeepSeek paper PDF:

```powershell
# Test with auto-detection (should pick CUDA)
python quick_test.py ./inputs/deepseek_paper.pdf

# Or explicitly specify CUDA
python quick_test.py --device cuda ./inputs/deepseek_paper.pdf

# With different quality modes
python quick_test.py --mode small ./inputs/deepseek_paper.pdf      # Fast
python quick_test.py --mode base ./inputs/deepseek_paper.pdf       # Balanced (default)
python quick_test.py --mode large ./inputs/deepseek_paper.pdf      # Better quality
```

---

## Expected Output

When running successfully, you should see:

```
============================================================
DeepSeek-OCR Quick Test
============================================================
Input: ./inputs/deepseek_paper.pdf
Device: cuda (auto-detected)
Mode: base
DPI: 200
============================================================

[1/3] Loading DeepSeek-OCR model...
(First run will download ~6.6GB from HuggingFace)
2025-11-01 10:30:45,123 - INFO - Initializing DeepSeek-OCR model: deepseek-ai/DeepSeek-OCR
2025-11-01 10:30:45,124 - INFO - Device: NVIDIA GeForce RTX 2080 Ti
2025-11-01 10:30:45,124 - INFO - Data Type: bfloat16
2025-11-01 10:30:45,124 - INFO - Recommended Mode: base
✓ Model loaded in 45.2s

[2/3] Processing PDF...
2025-11-01 10:31:30,456 - INFO - Converting PDF to images: ./inputs/deepseek_paper.pdf (DPI: 200)
2025-11-01 10:31:30,789 - INFO - PDF has 42 pages
2025-11-01 10:31:30,790 - INFO - Processing page 1/42
2025-11-01 10:31:32,456 - INFO - Processing complete. Results saved to: ./output/deepseek_paper/page_001
... (process each page) ...
✓ Processed 42 pages in 105.3s
  Average: 2.5s per page

[3/3] Results saved:
  Combined output: ./output/deepseek_paper/deepseek_paper_combined.md
  Individual pages: ./output/deepseek_paper/page_XXX/

--- Preview (first 500 chars) ---
# Page 1

# DeepSeek-OCR: Efficient PDF Document OCR with Multimodal Large Language Model

## Abstract

This paper presents DeepSeek-OCR, a...

============================================================
✓ Test completed successfully!
============================================================
```

---

## Troubleshooting

### ❌ "CUDA not available"

**Problem**: CUDA is not being detected even with drivers installed.

**Solutions**:
```powershell
# 1. Check NVIDIA driver
nvidia-smi

# 2. If not found, install from nvidia.com
# https://www.nvidia.com/Download/driverDetails.aspx

# 3. Check CUDA Toolkit installed
# https://developer.nvidia.com/cuda-12-1-0-download-archive

# 4. Force reinstall PyTorch
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

# 5. Verify again
python -c "import torch; print(torch.cuda.is_available())"
```

### ❌ "Out of Memory"

**Problem**: GPU runs out of memory (common with complex PDFs).

**Solutions**:
```powershell
# Use smaller mode
python quick_test.py --mode small ./inputs/deepseek_paper.pdf

# Process individual pages instead of all at once
# (advanced_examples.py example 4 handles this)
```

### ❌ "poppler not found" (PDF processing fails)

**Problem**: pdf2image can't convert PDFs to images.

**Solution for Windows**:
```powershell
# Option 1: Download poppler manually from:
# https://github.com/oschwartz10612/poppler-windows/releases/

# Option 2: Use conda (if you have it)
conda install poppler

# Option 3: Install via scoop
scoop install poppler

# Then set environment variable:
$env:PATH += ";C:\path\to\poppler\bin"
```

### ❌ "Device config module not found"

**Problem**: Import error for device_config module.

**Solution**:
```powershell
# Ensure you're in project root directory
cd C:\Users\rudol\Documents\dev\devcontainers\deepseek-ocr

# Check PYTHONPATH includes src/
python -c "import sys; print(sys.path)"

# Add src to path if needed (in PowerShell)
$env:PYTHONPATH = "$pwd\src;$env:PYTHONPATH"
python quick_test.py --device cuda ./inputs/deepseek_paper.pdf
```

---

## Docker Alternative (If Native Setup Fails)

If you have Docker Desktop installed with NVIDIA container toolkit:

```powershell
# Build image (might take 10-15 minutes on first build)
docker-compose build

# Run test
docker-compose run --rm deepseek-ocr uv run python3 quick_test.py /app/input/deepseek_paper.pdf

# Or run with mode selection
docker-compose run --rm deepseek-ocr uv run python3 quick_test.py --mode base /app/input/deepseek_paper.pdf
```

---

## Performance Expectations (RTX 2080 Ti, 11GB VRAM)

| Metric | Value | Notes |
|--------|-------|-------|
| **Model Loading** | 8-15s | First run (6.6GB download), 2-5s cached |
| **Per-Page (base mode)** | 3-4s | RTX 2080 Ti slower than RTX 3080 Ti |
| **Recommended Mode** | `base` | Can use `small` for speed, `large` for quality |
| **Max Mode** | `large` | `gundam` will likely OOM on 2080 Ti |
| **Memory Usage** | ~9GB | Model (6.6GB) + inference (~2-3GB) |

---

## Verify the Refactoring Works

Test all device selection modes:

```powershell
# 1. Test auto-detection (should pick CUDA)
python quick_test.py ./inputs/deepseek_paper.pdf
echo "✓ Auto-detection works"

# 2. Test explicit CUDA
python quick_test.py --device cuda ./inputs/deepseek_paper.pdf
echo "✓ Explicit CUDA works"

# 3. Test CPU fallback
python quick_test.py --device cpu ./inputs/deepseek_paper.pdf
echo "✓ CPU fallback works (will be slow)"

# 4. Test with different modes
python quick_test.py --mode small ./inputs/deepseek_paper.pdf
python quick_test.py --mode base ./inputs/deepseek_paper.pdf
python quick_test.py --mode large ./inputs/deepseek_paper.pdf
echo "✓ All modes work"

# 5. Test batch processing
python advanced_examples.py 1  # Interactive menu
python advanced_examples.py --device cuda 1  # Batch processing with explicit CUDA
echo "✓ Batch processing works"
```

---

## What the Refactoring Enables

After successful testing, you now have:

✅ **Auto-detection** - Automatically uses best available device (CUDA > MPS > CPU)
✅ **Explicit control** - `--device cuda` for explicit selection
✅ **macOS support** - Can use `--mac` flag on macOS (for your future M1/M2 Mac if you get one)
✅ **Device-specific optimization** - Auto-selects dtype and features per device
✅ **Backward compatibility** - All old code still works

---

## Next Steps

1. ✅ Run the test above
2. 📊 Check output in `./output/deepseek_paper/`
3. 📖 Review `DEVICE_SUPPORT.md` for advanced usage
4. 🚀 Integrate into your projects using:
   ```python
   from deepseek_ocr_pdf import DeepSeekOCRProcessor

   # Auto-detect device
   processor = DeepSeekOCRProcessor()

   # Or explicit
   processor = DeepSeekOCRProcessor(device='cuda')

   # Process PDF
   results = processor.process_pdf('./input.pdf', './output')
   ```

---

## Support

If something doesn't work:
1. Check **DEVICE_SUPPORT.md** - Comprehensive troubleshooting
2. Check **QUICK_START.md** - Quick reference
3. Check **CLAUDE.md** - Architecture details
4. Verify all prerequisites are installed correctly

Good luck! 🚀
