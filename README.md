# DeepSeek-OCR PDF Processing

High-performance OCR pipeline for extracting structured text from PDFs using DeepSeek-OCR vision-language model.

## Quick Start

### Prerequisites
- Docker with NVIDIA GPU support
- 12GB+ VRAM (tested on RTX 3080 Ti)
- ~7GB disk space for model weights

### Build & Run

```bash
# Build Docker image
docker compose build

# Process a PDF
docker compose run --rm deepseek-ocr python3 quick_test.py /app/input/your_document.pdf
```

## Usage

### Quick Test Script
```bash
docker compose run --rm deepseek-ocr python3 quick_test.py /app/input/sample.pdf
```

**What it does:**
1. Loads the DeepSeek-OCR model (~200s first run, ~3s cached)
2. Converts PDF pages to images (DPI: 200)
3. Extracts text with document structure
4. Saves results as markdown

**Output:**
- Individual page results: `./output/<pdf_name>/page_XXXX/result.mmd`
- Combined document: `./output/<pdf_name>/<pdf_name>_combined.md`

### Custom Usage

```python
from deepseek_ocr_pdf import DeepSeekOCRProcessor

# Initialize
processor = DeepSeekOCRProcessor(
    device='cuda',              # or 'cpu'
    use_flash_attention=False   # requires compilation
)

# Process PDF
results = processor.process_pdf(
    pdf_path='/path/to/document.pdf',
    output_dir='./output',
    mode='base',     # tiny, small, base, large, gundam
    dpi=200          # higher = better quality, slower
)

# Process single image
result = processor.process_image(
    image_path='/path/to/image.jpg',
    output_dir='./output',
    mode='base',
    prompt="<image>\n<|grounding|>Convert to markdown."  # custom prompt
)
```

## Quality Modes

| Mode | Resolution | Speed | Quality | Use Case |
|------|-----------|-------|---------|----------|
| tiny | 512×512 | Fast | Low | Quick previews |
| small | 640×640 | Fast | Medium | Simple documents |
| **base** | 1024×1024 | Moderate | Good | Default/recommended |
| large | 1280×1280 | Slow | High | Complex layouts |
| gundam | Dynamic | Very slow | Best | High-precision work |

## Expected Output

Extracted markdown includes:
- **Titles & headings** as markdown headers
- **Body text** with structure preserved
- **Tables** in markdown format
- **Mathematical expressions** (LaTeX compatible)
- **Image references** with captions
- **Bounding boxes** for structural elements (in tags)

Example:
```markdown
# DeepSeek-OCR: Context Optical Compression

Haoran Wei, Yaofeng Sun, Yukun Li

## Abstract

We present DeepSeek-OCR as an initial investigation...
```

## Performance

On RTX 3080 Ti:
- Model load: ~200s (first run), ~3s (cached)
- Per-page processing: ~30-40s (base mode)
- Memory: ~6GB model + ~2GB inference = 8GB typical

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA not available | Ensure NVIDIA Docker runtime installed (`nvidia-smi` on host) |
| Out of memory | Use `tiny` or `small` mode, reduce PDF pages |
| Model download fails | Check internet connectivity, HuggingFace API access |
| Slow processing | GPU may be busy; reduce DPI or page count |

## Docker Volume Mounts

- **Input**: `./input/` ’ `/app/input/` (read-only)
- **Output**: `./output/` ’ `/app/output/` (results saved here)
- **Model cache**: `huggingface-cache` volume (persistent across runs)

Place your PDFs in `./input/` and results will appear in `./output/`.

## Notes

- First run downloads ~6.6GB model weights (HuggingFace)
- Model weights cached in Docker volume for fast subsequent runs
- Flash Attention 2 support available (requires `flash-attn` compilation)
- Uses `trust_remote_code=True` for custom model implementations
