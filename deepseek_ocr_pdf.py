"""
DeepSeek-OCR PDF Processing Script
Based on official DeepSeek-OCR repository and documentation
Sources:
- https://github.com/deepseek-ai/DeepSeek-OCR/tree/main
- https://arxiv.org/html/2510.18234v1
- https://huggingface.co/deepseek-ai/DeepSeek-OCR
"""

import os
import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from typing import Optional, List, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepSeekOCRProcessor:
    """
    DeepSeek-OCR Processor for PDF and Image OCR tasks.
    
    Supports multiple resolution modes:
    - Tiny: 512x512 (64 vision tokens) - fastest, lowest quality
    - Small: 640x640 (100 vision tokens) - good balance for simple docs
    - Base: 1024x1024 (256 vision tokens) - recommended for most docs
    - Large: 1280x1280 (400 vision tokens) - high quality
    - Gundam: n×640 + 1024 (dynamic) - best quality for complex docs
    """
    
    # Resolution modes configuration from official paper (Table 1)
    MODES = {
        'tiny': {'base_size': 512, 'image_size': 512, 'crop_mode': False, 'tokens': 64},
        'small': {'base_size': 640, 'image_size': 640, 'crop_mode': False, 'tokens': 100},
        'base': {'base_size': 1024, 'image_size': 1024, 'crop_mode': False, 'tokens': 256},
        'large': {'base_size': 1280, 'image_size': 1280, 'crop_mode': False, 'tokens': 400},
        'gundam': {'base_size': 1024, 'image_size': 640, 'crop_mode': True, 'tokens': 'dynamic'}
    }
    
    def __init__(
        self, 
        model_name: str = 'deepseek-ai/DeepSeek-OCR',
        device: str = 'cuda',
        use_flash_attention: bool = True
    ):
        """
        Initialize the DeepSeek-OCR model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on ('cuda' or 'cpu')
            use_flash_attention: Whether to use Flash Attention 2 (requires NVIDIA GPU)
        """
        logger.info(f"Initializing DeepSeek-OCR model: {model_name}")
        
        # Set CUDA device
        if device == 'cuda' and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available, falling back to CPU (this will be slow)")
            device = 'cpu'
        
        self.device = device
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # Load model
        logger.info("Loading model (this may take a few minutes)...")
        model_kwargs = {
            'trust_remote_code': True,
            'use_safetensors': True
        }
        
        if use_flash_attention and device == 'cuda':
            model_kwargs['_attn_implementation'] = 'flash_attention_2'
            logger.info("Using Flash Attention 2 for improved performance")
        
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model = self.model.eval()
        
        if device == 'cuda':
            self.model = self.model.cuda().to(torch.bfloat16)
        
        logger.info("Model loaded successfully!")
    
    def process_image(
        self,
        image_path: str,
        output_dir: str,
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown. ",
        mode: str = 'base',
        save_results: bool = True,
        test_compress: bool = False
    ) -> Dict:
        """
        Process a single image file with DeepSeek-OCR.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            prompt: OCR prompt (use <|grounding|> tag for structured output)
            mode: Resolution mode ('tiny', 'small', 'base', 'large', 'gundam')
            save_results: Whether to save results to disk
            test_compress: Whether to test compression ratios
            
        Returns:
            Dictionary containing OCR results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if mode not in self.MODES:
            raise ValueError(f"Invalid mode: {mode}. Choose from {list(self.MODES.keys())}")
        
        mode_config = self.MODES[mode]
        logger.info(f"Processing image: {image_path} with mode '{mode}' ({mode_config['tokens']} tokens)")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run inference using the official model.infer() method
        result = self.model.infer(
            self.tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=output_dir,
            base_size=mode_config['base_size'],
            image_size=mode_config['image_size'],
            crop_mode=mode_config['crop_mode'],
            save_results=save_results,
            test_compress=test_compress
        )
        
        logger.info(f"Processing complete. Results saved to: {output_dir}")
        return result
    
    def process_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown. ",
        mode: str = 'base',
        dpi: int = 200
    ) -> List[Dict]:
        """
        Process a PDF file by converting each page to an image and running OCR.
        
        Note: This method requires pdf2image and poppler-utils to be installed.
        Install with: pip install pdf2image
        System requirement: poppler (brew install poppler on macOS, apt-get install poppler-utils on Ubuntu)
        
        Args:
            pdf_path: Path to input PDF
            output_dir: Directory to save results
            prompt: OCR prompt
            mode: Resolution mode
            dpi: DPI for PDF to image conversion (higher = better quality, slower)
            
        Returns:
            List of dictionaries containing results for each page
        """
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError(
                "pdf2image is required for PDF processing. "
                "Install with: pip install pdf2image\n"
                "Also ensure poppler is installed on your system."
            )
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Converting PDF to images: {pdf_path} (DPI: {dpi})")
        
        # Convert PDF pages to images
        pages = convert_from_path(pdf_path, dpi=dpi)
        logger.info(f"PDF has {len(pages)} pages")
        
        # Create output directory structure
        pdf_name = Path(pdf_path).stem
        pdf_output_dir = os.path.join(output_dir, pdf_name)
        pages_dir = os.path.join(pdf_output_dir, 'pages')
        os.makedirs(pages_dir, exist_ok=True)
        
        results = []
        
        # Process each page
        for i, page_image in enumerate(pages, start=1):
            logger.info(f"Processing page {i}/{len(pages)}")
            
            # Save page as temporary image
            page_image_path = os.path.join(pages_dir, f'page_{i:03d}.png')
            page_image.save(page_image_path, 'PNG')
            
            # Run OCR on page
            page_output_dir = os.path.join(pdf_output_dir, f'page_{i:03d}')
            result = self.process_image(
                image_path=page_image_path,
                output_dir=page_output_dir,
                prompt=prompt,
                mode=mode,
                save_results=True,
                test_compress=False
            )
            
            results.append({
                'page': i,
                'result': result,
                'output_dir': page_output_dir
            })
        
        # Combine all page results into a single markdown file
        combined_path = os.path.join(pdf_output_dir, f'{pdf_name}_combined.md')
        self._combine_results(results, combined_path, pdf_output_dir)
        
        logger.info(f"PDF processing complete. Combined output: {combined_path}")
        return results
    
    def _combine_results(self, results: List[Dict], output_path: str, pdf_output_dir: str):
        """Combine individual page results into a single document."""
        combined_text = []
        
        for page_result in results:
            page_num = page_result['page']
            page_output_dir = page_result['output_dir']
            
            # Look for result.mmd file in the page output directory
            result_file = os.path.join(page_output_dir, 'result.mmd')
            
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    page_text = f.read()
                    combined_text.append(f"# Page {page_num}\n\n{page_text}\n\n")
            else:
                logger.warning(f"Result file not found for page {page_num}")
        
        # Write combined output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(combined_text))
        
        logger.info(f"Combined {len(results)} pages into: {output_path}")


def main():
    """Example usage of DeepSeek-OCR for PDF processing."""
    
    # Initialize processor
    processor = DeepSeekOCRProcessor(
        model_name='deepseek-ai/DeepSeek-OCR',
        device='cuda',  # Use 'cpu' if no GPU available
        use_flash_attention=True
    )
    
    # Example 1: Process a single image
    logger.info("\n" + "="*50)
    logger.info("Example 1: Single Image Processing")
    logger.info("="*50)
    
    # image_result = processor.process_image(
    #     image_path='path/to/your/image.jpg',
    #     output_dir='./output/single_image',
    #     prompt="<image>\n<|grounding|>Convert the document to markdown. ",
    #     mode='base',  # Use 'gundam' for best quality on complex documents
    #     save_results=True,
    #     test_compress=True
    # )
    
    # Example 2: Process a PDF
    logger.info("\n" + "="*50)
    logger.info("Example 2: PDF Processing")
    logger.info("="*50)
    
    # pdf_results = processor.process_pdf(
    #     pdf_path='path/to/your/document.pdf',
    #     output_dir='./output/pdf',
    #     prompt="<image>\n<|grounding|>Convert the document to markdown. ",
    #     mode='base',  # Adjust based on document complexity
    #     dpi=200  # Increase for better quality on scanned documents
    # )
    
    # Example 3: Free OCR (no structure preservation)
    logger.info("\n" + "="*50)
    logger.info("Example 3: Free OCR (Plain Text)")
    logger.info("="*50)
    
    # text_result = processor.process_image(
    #     image_path='path/to/your/image.jpg',
    #     output_dir='./output/free_ocr',
    #     prompt="<image>\nFree OCR. ",  # No <|grounding|> tag for plain text
    #     mode='small',  # Faster mode for simple text extraction
    #     save_results=True
    # )
    
    logger.info("\n" + "="*50)
    logger.info("Examples completed! Uncomment the code above to run.")
    logger.info("="*50)


if __name__ == "__main__":
    main()
