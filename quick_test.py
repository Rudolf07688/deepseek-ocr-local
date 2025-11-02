# Quick Test Script for DeepSeek-OCR
# Supports NVIDIA CUDA (Windows/Linux) and Apple MPS (macOS)

from deepseek_ocr_pdf import DeepSeekOCRProcessor
from pathlib import Path
import sys
import time
import argparse

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='DeepSeek-OCR Quick Test - Process a single PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Windows/Linux with NVIDIA GPU (auto-detect)
  python quick_test.py ./input/sample.pdf

  # macOS with Apple Silicon (MPS)
  python quick_test.py --mac ./input/sample.pdf

  # Explicit device selection
  python quick_test.py --device mps ./input/sample.pdf
  python quick_test.py --device cuda ./input/sample.pdf
  python quick_test.py --device cpu ./input/sample.pdf

  # Specify OCR mode
  python quick_test.py --mode large ./input/sample.pdf
        """
    )

    parser.add_argument(
        'pdf_path',
        help='Path to the PDF file to process'
    )

    parser.add_argument(
        '--mac',
        action='store_true',
        help='Use Apple Metal (MPS) on macOS (shorthand for --device mps)'
    )

    parser.add_argument(
        '--device',
        choices=['cuda', 'mps', 'cpu', None],
        default=None,
        help='Device to use: cuda (NVIDIA), mps (Apple Metal), cpu, or auto-detect (default)'
    )

    parser.add_argument(
        '--mode',
        choices=['tiny', 'small', 'base', 'large', 'gundam'],
        default='base',
        help='OCR resolution mode (default: base)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
        help='DPI for PDF to image conversion (default: 200)'
    )

    parser.add_argument(
        '--flash-attention',
        action='store_true',
        help='Enable Flash Attention 2 (NVIDIA GPUs only)'
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Handle --mac flag
    device = args.device
    if args.mac:
        if args.device:
            print("Warning: --mac flag overrides --device argument")
        device = 'mps'

    pdf_path = args.pdf_path

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print("="*60)
    print("DeepSeek-OCR Quick Test")
    print("="*60)
    print(f"Input: {pdf_path}")
    print(f"Device: {device if device else 'auto-detect'}")
    print(f"Mode: {args.mode}")
    print(f"DPI: {args.dpi}")
    print("="*60)

    # Initialize processor
    print("\n[1/3] Loading DeepSeek-OCR model...")
    print("(First run will download ~6.6GB from HuggingFace)")

    start_time = time.time()

    try:
        processor = DeepSeekOCRProcessor(
            device=device,
            use_flash_attention=args.flash_attention if device == 'cuda' else None
        )
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.1f}s")
        
        # Process PDF
        print(f"\n[2/3] Processing PDF...")
        process_start = time.time()

        results = processor.process_pdf(
            pdf_path=pdf_path,
            output_dir='./output',
            mode=args.mode,
            dpi=args.dpi
        )
        
        process_time = time.time() - process_start
        pages = len(results)
        
        print(f"✓ Processed {pages} pages in {process_time:.1f}s")
        print(f"  Average: {process_time/pages:.1f}s per page")
        
        # Output location
        pdf_name = Path(pdf_path).stem
        output_file = f"./output/{pdf_name}/{pdf_name}_combined.md"
        
        print(f"\n[3/3] Results saved:")
        print(f"  Combined output: {output_file}")
        print(f"  Individual pages: ./output/{pdf_name}/page_XXX/")
        
        # Display first few lines of output
        if Path(output_file).exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                preview = f.read(500)
            print(f"\n--- Preview (first 500 chars) ---")
            print(preview)
            print("...")
        
        print("\n" + "="*60)
        print("✓ Test completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
