import os
import sys
import time
import csv
import requests
import xml.etree.ElementTree as ET
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR
from shared import setup_logger

logger = setup_logger("BGG_Scraper")

BGG_API_URL = "https://boardgamegeek.com/xmlapi2/thing"
OUTPUT_CSV = os.path.join(DATA_DIR, "bgg_massive_dataset.csv")
BATCH_SIZE = 400  # Number of IDs to request at once
DELAY_BETWEEN_BATCHES = 2.0  # Seconds to wait to avoid rate limiting
MAX_RETRIES = 5

def fetch_batch(start_id: int, end_id: int):
    """Fetch a batch of board games from BGG XML API."""
    ids = ",".join(str(i) for i in range(start_id, end_id + 1))
    params = {
        "id": ids,
        "type": "boardgame"
    }
    headers = {
        "User-Agent": "AIBoardGameOntologyBot/1.0 (contact: your-email@example.com) - Educational AI Project"
    }
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(BGG_API_URL, params=params, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 429:
                logger.warning(f"Rate limited (429) on batch {start_id}-{end_id}. Sleeping for 10s...")
                time.sleep(10)
            else:
                logger.warning(f"Failed with status {response.status_code} on batch {start_id}-{end_id}.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception on batch {start_id}-{end_id}: {e}")
            
        wait_time = DELAY_BETWEEN_BATCHES * (2 ** attempt)
        logger.info(f"Retrying in {wait_time}s...")
        time.sleep(wait_time)
        
    return None

def parse_xml_and_save(xml_text: str, csv_writer):
    """Parse the XML and append rows to the CSV."""
    if not xml_text:
        return 0
        
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML: {e}")
        return 0
        
    count = 0
    for item in root.findall("item"):
        item_type = item.get("type")
        if item_type != "boardgame":
            continue
            
        game_id = item.get("id")
        
        # Get Primary Name
        name_tag = item.find("name[@type='primary']")
        if name_tag is None:
            continue
        title = name_tag.get("value", "Unknown")
        
        # Get Description
        desc_tag = item.find("description")
        description = desc_tag.text if desc_tag is not None and desc_tag.text else ""
        
        # Get Mechanics
        mechanics = []
        for link in item.findall("link[@type='boardgamemechanic']"):
            mechanics.append(link.get("value"))
            
        # Get Categories
        categories = []
        for link in item.findall("link[@type='boardgamecategory']"):
            categories.append(link.get("value"))
            
        # Only save games that actually have mechanics or categories
        if not mechanics and not categories:
            continue
            
        csv_writer.writerow([game_id, title, description, str(mechanics), str(categories)])
        count += 1
        
    return count

def run_scraper(start_id: int, max_id: int):
    """Run the massive BGG scraping pipeline."""
    os.makedirs(DATA_DIR, exist_ok=True)
    file_exists = os.path.isfile(OUTPUT_CSV)
    
    with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["game_id", "title", "description", "mechanics", "categories"])
            
        logger.info(f"Starting BGG Scraper from ID {start_id} to {max_id}...")
        total_saved = 0
        
        for batch_start in range(start_id, max_id + 1, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE - 1, max_id)
            logger.info(f"Fetching batch {batch_start} to {batch_end}...")
            
            xml_data = fetch_batch(batch_start, batch_end)
            if xml_data:
                saved = parse_xml_and_save(xml_data, writer)
                total_saved += saved
                logger.info(f"Saved {saved} valid games. Total saved: {total_saved}")
            else:
                logger.error(f"Completely failed batch {batch_start}-{batch_end}. Skipping.")
                
            # Respect rate limit
            time.sleep(DELAY_BETWEEN_BATCHES)
            
    logger.info(f"Scraping complete! Saved {total_saved} games to {OUTPUT_CSV}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BGG Massive Dataset Scraper")
    parser.add_argument("--start", type=int, default=1, help="Starting Board Game ID")
    parser.add_argument("--end", type=int, default=5000, help="Ending Board Game ID")
    args = parser.parse_args()
    
    run_scraper(args.start, args.end)
