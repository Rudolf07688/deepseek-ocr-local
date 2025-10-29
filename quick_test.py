# Quick Test Script for RTX 3080 Ti
# Processes a single PDF with optimal settings for 12GB VRAM

from deepseek_ocr_pdf import DeepSeekOCRProcessor
from pathlib import Path
import sys
import time

def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_test.py <path_to_pdf>")
        print("Example: python quick_test.py ./input/sample.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    print("="*60)
    print("DeepSeek-OCR Quick Test")
    print("="*60)
    print(f"GPU: NVIDIA GeForce RTX 3080 Ti (12GB VRAM)")
    print(f"Input: {pdf_path}")
    print(f"Mode: base (1024x1024, 256 tokens)")
    print("="*60)
    
    # Initialize processor
    print("\n[1/3] Loading DeepSeek-OCR model...")
    print("(First run will download ~6.6GB from HuggingFace)")
    
    start_time = time.time()
    
    try:
        processor = DeepSeekOCRProcessor(
            device='cuda',
            use_flash_attention=False  # Set True if you compiled flash-attn
        )
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.1f}s")
        
        # Process PDF
        print(f"\n[2/3] Processing PDF...")
        process_start = time.time()
        
        results = processor.process_pdf(
            pdf_path=pdf_path,
            output_dir='./output',
            mode='base',  # Use 'large' or 'gundam' for better quality
            dpi=200
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
