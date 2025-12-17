#!/usr/bin/env python3
"""
Phase 3: Document Ingestion Pipeline
- Parse all PDFs in data/raw_pdfs/
- Extract clean text using unstructured + pypdf fallback
- Save as .jsonl in data/processed/
- Validate no empty outputs
"""
import os
import json
import logging
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("parse_pdfs.log")
    ]
)
logger = logging.getLogger(__name__)

# Paths
RAW_PDF_DIR = Path("data/raw_pdfs")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(exist_ok=True)

def parse_pdf_unstructured(pdf_path: Path) -> str:
    """Primary parser using unstructured."""
    try:
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="fast",  # CPU-optimized
            include_page_breaks=False
        )
        return "\n\n".join([str(el) for el in elements if str(el).strip()])
    except Exception as e:
        logger.warning(f"unstructured failed on {pdf_path.name}: {e}")
        return ""

def parse_pdf_pypdf(pdf_path: Path) -> str:
    """Fallback parser using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        return text
    except Exception as e:
        logger.error(f"pypdf failed on {pdf_path.name}: {e}")
        return ""

def process_pdfs() -> List[Path]:
    """Process all PDFs and return list of successfully processed files."""
    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDFs found in data/raw_pdfs/")
        return []

    processed_files = []
    for pdf_path in sorted(pdf_files):
        logger.info(f"Processing: {pdf_path.name}")

        # Try primary parser
        text = parse_pdf_unstructured(pdf_path)
        if not text.strip():
            # Fallback
            logger.info(f"Falling back to pypdf for {pdf_path.name}")
            text = parse_pdf_pypdf(pdf_path)

        if not text.strip():
            logger.error(f"Failed to extract text from {pdf_path.name}")
            continue

        # Save as JSONL (one doc = one line)
        output_path = PROCESSED_DIR / f"{pdf_path.stem}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            doc = {
                "file_name": pdf_path.name,
                "source_path": str(pdf_path),
                "content": text.strip(),
                "char_count": len(text.strip()),
                "word_count": len(text.strip().split())
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        processed_files.append(output_path)
        logger.info(f"Saved: {output_path} ({doc['char_count']} chars)")

    return processed_files

def validate_outputs(processed_files: List[Path]):
    """Ensure no empty or corrupted outputs."""
    if not processed_files:
        raise RuntimeError("No documents were successfully processed")
    
    empty_files = []
    for f in processed_files:
        if f.stat().st_size == 0:
            empty_files.append(f)
    
    if empty_files:
        raise RuntimeError(f"Empty output files detected: {[str(f) for f in empty_files]}")
    
    logger.info(f"✅ Validation passed: {len(processed_files)} documents processed")

if __name__ == "__main__":
    processed = process_pdfs()
    validate_outputs(processed)
    logger.info("Phase 3 complete: All PDFs parsed and validated")
