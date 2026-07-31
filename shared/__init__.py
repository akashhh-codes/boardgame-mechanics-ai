"""
Shared Utilities Module
=======================
Common helper functions used across all project modules.
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime

# ─── Logging Setup ──────────────────────────────────────────────
def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Create a configured logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─── Data Loading ───────────────────────────────────────────────
def load_board_game_data(csv_path: str) -> pd.DataFrame:
    """
    Load and validate the board games CSV dataset.
    Performs initial data quality checks and type coercion.
    """
    logger = setup_logger("DataLoader")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset: {df.shape[0]} games, {df.shape[1]} columns")
    
    # Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    
    return df


def parse_mechanics_list(mechanics_str: str) -> list:
    """
    Parse a mechanics string into a clean list.
    Handles comma-separated, bracket-enclosed, and other common formats.
    """
    if pd.isna(mechanics_str) or not isinstance(mechanics_str, str):
        return []
    
    # Remove brackets and quotes
    cleaned = mechanics_str.strip("[](){}")
    # Split by comma and clean each item
    mechanics = [m.strip().strip("'\"") for m in cleaned.split(",")]
    # Remove empty strings
    return [m for m in mechanics if m]


# ─── File I/O Helpers ──────────────────────────────────────────
def save_json(data: dict, filepath: str, indent: int = 2):
    """Save data as formatted JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text: str, filepath: str):
    """Save text to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


# ─── Metadata / Provenance ────────────────────────────────────
def create_provenance_record(source: str, operation: str, details: dict = None) -> dict:
    """Create a provenance metadata record for tracking pipeline operations."""
    return {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "operation": operation,
        "details": details or {}
    }
