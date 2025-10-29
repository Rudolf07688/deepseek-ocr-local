"""
Advanced DeepSeek-OCR Usage Examples
Demonstrates batch processing, error handling, and integration patterns
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from deepseek_ocr_pdf import DeepSeekOCRProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchOCRProcessor:
    """
    Batch processor for handling multiple PDFs with error recovery and progress tracking.
    """
    
    def __init__(self, processor: DeepSeekOCRProcessor):
        self.processor = processor
        self.results_cache = {}
        
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        mode: str = 'base',
        file_pattern: str = '*.pdf',
        skip_existing: bool = True
    ) -> Dict[str, any]:
        """
        Process all PDFs in a directory.
        
        Args:
            input_dir: Directory containing PDFs
            output_dir: Base output directory
            mode: OCR resolution mode
            file_pattern: Glob pattern for file selection
            skip_existing: Skip files that have already been processed
            
        Returns:
            Dictionary with processing statistics and results
        """
        input_path = Path(input_dir)
        pdf_files = list(input_path.glob(file_pattern))
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = {
            'total': len(pdf_files),
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'files': {}
        }
        
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {i}/{len(pdf_files)}: {pdf_path.name}")
            logger.info('='*60)
            
            try:
                # Check if already processed
                pdf_output_dir = Path(output_dir) / pdf_path.stem
                combined_file = pdf_output_dir / f"{pdf_path.stem}_combined.md"
                
                if skip_existing and combined_file.exists():
                    logger.info(f"Skipping {pdf_path.name} (already processed)")
                    results['skipped'] += 1
                    continue
                
                # Process PDF
                page_results = self.processor.process_pdf(
                    pdf_path=str(pdf_path),
                    output_dir=output_dir,
                    mode=mode
                )
                
                results['successful'] += 1
                results['files'][str(pdf_path)] = {
                    'status': 'success',
                    'pages': len(page_results),
                    'output': str(combined_file)
                }
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {str(e)}")
                results['failed'] += 1
                results['files'][str(pdf_path)] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        logger.info(f"\n{'='*60}")
        logger.info("Batch Processing Summary")
        logger.info('='*60)
        logger.info(f"Total files: {results['total']}")
        logger.info(f"Successful: {results['successful']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Skipped: {results['skipped']}")
        
        return results


def example_batch_processing():
    """Example: Batch process multiple PDFs in a directory."""
    
    # Initialize processor
    processor = DeepSeekOCRProcessor(
        device='cuda',
        use_flash_attention=True
    )
    
    # Create batch processor
    batch_processor = BatchOCRProcessor(processor)
    
    # Process entire directory
    results = batch_processor.process_directory(
        input_dir='./input_pdfs',
        output_dir='./output',
        mode='base',
        file_pattern='*.pdf',
        skip_existing=True
    )
    
    # Save processing report
    import json
    with open('./output/processing_report.json', 'w') as f:
        json.dump(results, f, indent=2)


def example_adaptive_mode_selection(image_path: str) -> str:
    """
    Example: Automatically select the best mode based on image characteristics.
    """
    from PIL import Image
    
    # Load image to check dimensions
    img = Image.open(image_path)
    width, height = img.size
    total_pixels = width * height
    
    # Heuristic mode selection
    if total_pixels < 500_000:  # Small image
        mode = 'small'
    elif total_pixels < 1_500_000:  # Medium image
        mode = 'base'
    elif total_pixels < 3_000_000:  # Large image
        mode = 'large'
    else:  # Very large or high-resolution
        mode = 'gundam'
    
    logger.info(f"Image: {width}x{height} ({total_pixels:,} pixels) → Selected mode: {mode}")
    return mode


def example_structured_extraction():
    """
    Example: Extract specific structured information from documents.
    """
    processor = DeepSeekOCRProcessor()
    
    # Example: Extract tables only
    tables_result = processor.process_image(
        image_path='financial_report.jpg',
        output_dir='./output/tables',
        prompt="<image>\n<|grounding|>Extract all tables in markdown format. Focus only on tabular data.",
        mode='large'
    )
    
    # Example: Extract headers and structure
    structure_result = processor.process_image(
        image_path='document.jpg',
        output_dir='./output/structure',
        prompt="<image>\n<|grounding|>Extract document structure: titles, headings, and section organization.",
        mode='base'
    )
    
    # Example: Multilingual document
    multilingual_result = processor.process_image(
        image_path='chinese_english_doc.jpg',
        output_dir='./output/multilingual',
        prompt="<image>\n<|grounding|>Convert to markdown preserving both Chinese and English text.",
        mode='base'
    )


def example_quality_validation(result_path: str, expected_keywords: List[str]) -> bool:
    """
    Example: Validate OCR output quality by checking for expected content.
    """
    with open(result_path, 'r', encoding='utf-8') as f:
        content = f.read().lower()
    
    found_keywords = [kw for kw in expected_keywords if kw.lower() in content]
    quality_score = len(found_keywords) / len(expected_keywords)
    
    logger.info(f"Quality validation: {quality_score:.1%} ({len(found_keywords)}/{len(expected_keywords)} keywords found)")
    
    return quality_score >= 0.7  # 70% threshold


def example_progressive_quality():
    """
    Example: Start with fast mode, upgrade to higher quality if needed.
    """
    processor = DeepSeekOCRProcessor()
    image_path = 'complex_document.jpg'
    
    # First pass: Quick extraction with 'small' mode
    logger.info("Pass 1: Quick extraction (small mode)")
    result_small = processor.process_image(
        image_path=image_path,
        output_dir='./output/pass1_small',
        mode='small',
        save_results=True
    )
    
    # Validate quality
    result_file = './output/pass1_small/result.mmd'
    expected_keywords = ['revenue', 'expenses', 'profit', 'quarter']
    
    if not example_quality_validation(result_file, expected_keywords):
        # Second pass: Higher quality if first pass failed validation
        logger.info("Pass 2: High-quality extraction (gundam mode)")
        result_gundam = processor.process_image(
            image_path=image_path,
            output_dir='./output/pass2_gundam',
            mode='gundam',
            save_results=True
        )


def example_memory_efficient_large_pdf():
    """
    Example: Process very large PDFs with memory management.
    """
    processor = DeepSeekOCRProcessor()
    
    # For large PDFs (100+ pages), process in chunks
    from pdf2image import convert_from_path
    
    pdf_path = 'large_document.pdf'
    output_dir = './output/large_pdf'
    chunk_size = 10  # Process 10 pages at a time
    
    # Get total page count
    import fitz  # PyMuPDF for counting pages
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()
    
    logger.info(f"Processing {total_pages} pages in chunks of {chunk_size}")
    
    all_results = []
    
    for start_page in range(1, total_pages + 1, chunk_size):
        end_page = min(start_page + chunk_size - 1, total_pages)
        logger.info(f"Processing pages {start_page}-{end_page}")
        
        # Convert chunk to images
        pages = convert_from_path(
            pdf_path, 
            first_page=start_page,
            last_page=end_page,
            dpi=200
        )
        
        # Process each page
        for i, page_img in enumerate(pages, start=start_page):
            page_path = f'./temp/page_{i:04d}.png'
            page_img.save(page_path)
            
            result = processor.process_image(
                image_path=page_path,
                output_dir=f'{output_dir}/page_{i:04d}',
                mode='base'
            )
            
            all_results.append(result)
            
            # Clean up temporary file
            os.remove(page_path)
        
        # Optional: Clear CUDA cache between chunks
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def example_integration_with_langchain():
    """
    Example: Integration pattern for LangChain document processing.
    """
    from typing import Iterator
    
    class DeepSeekDocumentLoader:
        """Custom LangChain-compatible document loader."""
        
        def __init__(self, file_path: str, mode: str = 'base'):
            self.file_path = file_path
            self.processor = DeepSeekOCRProcessor()
            self.mode = mode
        
        def lazy_load(self) -> Iterator[Dict]:
            """Lazy load documents page by page."""
            results = self.processor.process_pdf(
                pdf_path=self.file_path,
                output_dir='./temp_langchain',
                mode=self.mode
            )
            
            for page_result in results:
                page_num = page_result['page']
                output_dir = page_result['output_dir']
                
                # Read result file
                result_file = Path(output_dir) / 'result.mmd'
                if result_file.exists():
                    content = result_file.read_text(encoding='utf-8')
                    
                    yield {
                        'page_content': content,
                        'metadata': {
                            'source': self.file_path,
                            'page': page_num,
                            'mode': self.mode
                        }
                    }
    
    # Usage
    loader = DeepSeekDocumentLoader('document.pdf', mode='base')
    for doc in loader.lazy_load():
        print(f"Page {doc['metadata']['page']}: {len(doc['page_content'])} chars")


def main():
    """Run example based on command-line argument or menu."""
    import sys
    
    examples = {
        '1': ('Batch Processing', example_batch_processing),
        '2': ('Structured Extraction', example_structured_extraction),
        '3': ('Progressive Quality', example_progressive_quality),
        '4': ('Memory Efficient (Large PDF)', example_memory_efficient_large_pdf),
        '5': ('LangChain Integration', example_integration_with_langchain)
    }
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("\nDeepSeek-OCR Advanced Examples:")
        for key, (name, _) in examples.items():
            print(f"{key}. {name}")
        choice = input("\nSelect example (1-5): ")
    
    if choice in examples:
        name, func = examples[choice]
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print('='*60)
        func()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
