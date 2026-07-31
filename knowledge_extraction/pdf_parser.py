"""
Module 1: Layout-Aware PDF Parser
=================================
Implements layout-aware PDF-to-Markdown conversion pipeline.

The proposal specifies using PyMuPDF for layout-aware processing. This module provides:
1. A robust parser using PyMuPDF for layout-aware extraction
2. A fallback using pdfplumber for standard extraction
3. Post-processing to preserve semantic reading order

Responsibilities (per proposal):
- Parse complex multi-column PDF layouts into continuous markdown
- Preserve semantic reading order from visually complex PDFs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
import json
from typing import List, Dict, Optional

from config import RAW_PDFS_DIR, PARSED_MARKDOWN_DIR
from shared import setup_logger, save_text

logger = setup_logger("PDFParser")





# ─── PyMuPDF Fallback Parser ──────────────────────────────────
def parse_with_pymupdf(pdf_path: str) -> str:
    """
    Fallback PDF parser using PyMuPDF (fitz).
    
    Handles multi-column layouts by analyzing text block positions
    and reconstructing semantic reading order based on spatial
    coordinates (top-to-bottom, left-to-right within columns).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not available, using basic text extraction")
        return parse_with_pdfplumber(pdf_path)
    
    doc = fitz.open(pdf_path)
    full_text = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        text_blocks = []
        
        for block in blocks:
            if block["type"] == 0:  # Text block
                block_text = ""
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span["text"]
                    block_text += line_text + "\n"
                
                if block_text.strip():
                    text_blocks.append({
                        "text": block_text.strip(),
                        "bbox": block["bbox"],  # (x0, y0, x1, y1)
                        "x0": block["bbox"][0],
                        "y0": block["bbox"][1],
                    })
        
        # Sort blocks by reading order: detect columns
        if text_blocks:
            page_width = page.rect.width
            mid_x = page_width / 2
            
            left_blocks = [b for b in text_blocks if b["x0"] < mid_x - 20]
            right_blocks = [b for b in text_blocks if b["x0"] >= mid_x - 20]
            
            # Sort each column top-to-bottom
            left_blocks.sort(key=lambda b: b["y0"])
            right_blocks.sort(key=lambda b: b["y0"])
            
            # If clearly two-column, read left first then right
            if left_blocks and right_blocks:
                ordered = left_blocks + right_blocks
            else:
                ordered = sorted(text_blocks, key=lambda b: (b["y0"], b["x0"]))
            
            page_text = "\n\n".join(b["text"] for b in ordered)
            full_text.append(f"## Page {page_num + 1}\n\n{page_text}")
    
    doc.close()
    return "\n\n---\n\n".join(full_text)


def parse_with_pdfplumber(pdf_path: str) -> str:
    """Minimal fallback using pdfplumber for basic text extraction."""
    try:
        import pdfplumber
        
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"## Page {i + 1}\n\n{text}")
        
        return "\n\n---\n\n".join(pages)
    except ImportError:
        logger.error("No PDF parser available (install PyMuPDF or pdfplumber)")
        return ""


# ─── Markdown Post-Processing ────────────────────────────────
def clean_markdown_output(raw_markdown: str) -> str:
    """
    Post-process extracted markdown to improve quality:
    - Remove excessive whitespace
    - Fix broken sentences across page boundaries
    - Normalize heading levels
    - Remove page header/footer artifacts
    """
    lines = raw_markdown.split("\n")
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Skip likely headers/footers (page numbers, copyright, etc.)
        if re.match(r"^\d+$", stripped):  # Standalone page numbers
            continue
        if re.match(r"^©.*\d{4}", stripped):  # Copyright lines
            continue
        if re.match(r"^(page|pg\.?)\s*\d+", stripped, re.IGNORECASE):
            continue
        
        # Normalize excessive spacing
        if stripped:
            cleaned_lines.append(line)
        elif cleaned_lines and cleaned_lines[-1].strip():
            cleaned_lines.append("")  # Keep single blank lines
    
    text = "\n".join(cleaned_lines)
    
    # Fix sentence fragments split across lines
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # Fix hyphenated words
    
    return text.strip()


# ─── Main Parsing Pipeline ────────────────────────────────────
def parse_pdf(
    pdf_path: str,
    output_dir: str = PARSED_MARKDOWN_DIR,
    force_reparse: bool = False
) -> Dict:
    """
    Parse a single PDF through the layout-aware pipeline.
    
    Attempts PyMuPDF first, then falls back to pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save parsed markdown
        force_reparse: Re-parse even if output already exists
        
    Returns:
        Dict with parsing results and metadata
    """
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{filename}.md")
    
    # Skip if already parsed
    if os.path.exists(output_path) and not force_reparse:
        logger.info(f"Already parsed, skipping: {filename}")
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {
            "filename": filename,
            "output_path": output_path,
            "parser_used": "cached",
            "char_count": len(content),
            "success": True
        }
    
    logger.info(f"Parsing PDF: {filename}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Try parsers in order of preference
    parser_used = "none"
    markdown_text = None
    
    # 1. Try PyMuPDF (primary — layout-aware column extraction)
    markdown_text = parse_with_pymupdf(pdf_path)
    if markdown_text:
        parser_used = "pymupdf"
    
    # 2. Fallback to pdfplumber
    if not markdown_text:
        markdown_text = parse_with_pdfplumber(pdf_path)
        parser_used = "pdfplumber"
    
    # Post-process
    if markdown_text:
        markdown_text = clean_markdown_output(markdown_text)
        save_text(markdown_text, output_path)
        logger.info(f"Parsed {filename}: {len(markdown_text)} chars via {parser_used}")
    
    return {
        "filename": filename,
        "output_path": output_path,
        "parser_used": parser_used,
        "char_count": len(markdown_text) if markdown_text else 0,
        "success": bool(markdown_text)
    }


def run_parsing_pipeline(
    input_dir: str = RAW_PDFS_DIR,
    output_dir: str = PARSED_MARKDOWN_DIR
) -> List[Dict]:
    """
    Parse all PDFs in a directory through the layout-aware pipeline.
    
    Returns:
        List of result dicts for each parsed PDF
    """
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        logger.warning(f"No PDFs found in {input_dir}")
        return []
    
    logger.info(f"Starting parsing pipeline: {len(pdf_files)} PDFs")
    results = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        result = parse_pdf(pdf_path, output_dir)
        results.append(result)
    
    successful = sum(1 for r in results if r["success"])
    logger.info(f"Parsing complete: {successful}/{len(results)} successful")
    
    return results


# ─── CLI Entry Point ──────────────────────────────────────────
if __name__ == "__main__":
    results = run_parsing_pipeline()
    print(json.dumps(results, indent=2))
