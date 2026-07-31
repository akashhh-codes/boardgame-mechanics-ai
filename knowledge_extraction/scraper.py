"""
Module 1: Web Scraper for Board Game Rulebooks
===============================================
Automated, fault-tolerant data pipeline for scraping and retrieving
board game rulebook PDFs from public repositories and community archives.

Responsibilities (per proposal):
- Architect scraping bots to navigate game repositories
- Handle rate limiting, retries, and fault tolerance
- Maintain a download manifest to avoid re-downloads
"""

import os
import json
import time
import hashlib
import requests
from typing import List, Dict, Optional
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RAW_PDFS_DIR, DATA_DIR
from shared import setup_logger

logger = setup_logger("Scraper")


# ─── Download Manifest ─────────────────────────────────────────
MANIFEST_PATH = os.path.join(DATA_DIR, "download_manifest.json")


def load_manifest() -> Dict:
    """Load the download tracking manifest."""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {"downloads": [], "failed": [], "last_run": None}


def save_manifest(manifest: Dict):
    """Persist the download manifest."""
    manifest["last_run"] = datetime.now().isoformat()
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


# ─── Scraping Targets ──────────────────────────────────────────
# Public domain / CC-licensed rulebook sources for demonstration
SAMPLE_RULEBOOK_SOURCES = [
    {
        "name": "Catan_Rules",
        "url": "https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf",
        "game": "Catan"
    },
    {
        "name": "Pandemic_Rules",
        "url": "https://images.zmangames.com/filer_public/25/12/251252a1-8b2c-4e50-a983-4c9b3f2e8e9a/zm7101_pandemic_rules.pdf",
        "game": "Pandemic"
    },
    {
        "name": "Ticket_to_Ride_Rules",
        "url": "https://ncdn0.daysofwonder.com/tickettoride/en/img/tt_rules_2015_en.pdf",
        "game": "Ticket to Ride"
    },
]


# ─── Core Scraping Functions ──────────────────────────────────
def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file for deduplication."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_pdf(
    url: str,
    filename: str,
    output_dir: str = RAW_PDFS_DIR,
    max_retries: int = 3,
    timeout: int = 30,
    rate_limit_delay: float = 1.0
) -> Optional[str]:
    """
    Download a PDF with fault-tolerant retry logic.
    
    Args:
        url: URL of the PDF to download
        filename: Name to save the file as
        output_dir: Directory to save PDFs into
        max_retries: Maximum retry attempts on failure
        timeout: Request timeout in seconds
        rate_limit_delay: Delay between requests to respect rate limits
        
    Returns:
        Filepath of the downloaded PDF, or None if failed
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename}.pdf")
    
    # Skip if already downloaded
    if os.path.exists(filepath):
        logger.info(f"Already exists, skipping: {filename}")
        return filepath
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Academic Research Bot - Board Game Ontology Project)"
    }
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading [{attempt}/{max_retries}]: {filename}")
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Verify it's actually a PDF
            content_type = response.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower() and not url.endswith(".pdf"):
                logger.warning(f"Content may not be PDF: {content_type}")
            
            # Write to file
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(filepath)
            logger.info(f"Downloaded: {filename} ({file_size / 1024:.1f} KB)")
            
            # Rate limiting
            time.sleep(rate_limit_delay)
            return filepath
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt} failed for {filename}: {e}")
            if attempt < max_retries:
                wait_time = rate_limit_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
    
    logger.error(f"Failed to download after {max_retries} attempts: {filename}")
    return None


# ─── Batch Scraping Pipeline ──────────────────────────────────
def run_scraping_pipeline(
    sources: List[Dict] = None,
    output_dir: str = RAW_PDFS_DIR
) -> Dict:
    """
    Execute the full scraping pipeline with manifest tracking.
    
    Args:
        sources: List of dicts with 'name', 'url', 'game' keys
        output_dir: Directory to save downloaded PDFs
        
    Returns:
        Summary dict with counts of successful/failed downloads
    """
    if sources is None:
        sources = SAMPLE_RULEBOOK_SOURCES
    
    manifest = load_manifest()
    already_downloaded = {d["name"] for d in manifest["downloads"]}
    
    results = {"successful": 0, "failed": 0, "skipped": 0, "files": []}
    
    logger.info(f"Starting scraping pipeline: {len(sources)} sources")
    
    for source in sources:
        name = source["name"]
        
        if name in already_downloaded:
            results["skipped"] += 1
            continue
        
        filepath = download_pdf(
            url=source["url"],
            filename=name,
            output_dir=output_dir
        )
        
        if filepath:
            record = {
                "name": name,
                "game": source.get("game", "Unknown"),
                "url": source["url"],
                "filepath": filepath,
                "hash": compute_file_hash(filepath),
                "downloaded_at": datetime.now().isoformat(),
                "size_bytes": os.path.getsize(filepath)
            }
            manifest["downloads"].append(record)
            results["successful"] += 1
            results["files"].append(filepath)
        else:
            manifest["failed"].append({
                "name": name,
                "url": source["url"],
                "failed_at": datetime.now().isoformat()
            })
            results["failed"] += 1
    
    save_manifest(manifest)
    
    logger.info(
        f"Scraping complete: {results['successful']} downloaded, "
        f"{results['failed']} failed, {results['skipped']} skipped"
    )
    
    return results


# ─── CLI Entry Point ──────────────────────────────────────────
if __name__ == "__main__":
    results = run_scraping_pipeline()
    print(json.dumps(results, indent=2))
