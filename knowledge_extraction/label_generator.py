"""
Module 1: LLM-Assisted Label Generator
=======================================
Manages the weakly supervised prompt engineering pipeline to
assign accurate, MDA-compliant labels to the extracted clusters.

Uses the LUDES Game Mechanics Ontology MDA classification:
- Algorithm
  - Action (Action Programming, Auction, Pool Building, etc.)
  - Ruleset (Game Balance, Asymmetry, Phase segmentation, etc.)
- Data Representation (Tokens, Cards, Dice, Boards, etc.)

The module supports:
1. Google Gemini API (free tier) — primary
2. Lightweight keyword-based fallback labeling (no API needed)

Responsibilities (per proposal):
- Command local LLMs to assign accurate, MDA-compliant labels
- Ensure outputs align with formal game studies terminology
"""

import os
import json
import re
from typing import Dict, List, Optional
from collections import Counter

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GEMINI_API_KEY, GEMINI_MODEL, LLM_TEMPERATURE, ONTOLOGY_OUTPUT_DIR
from shared import setup_logger, save_json

logger = setup_logger("LabelGenerator")


# ─── MDA Framework Category Definitions ──────────────────────
MDA_CATEGORIES = {
    "Algorithm": {
        "description": "Processes and player interactions — the operable rules",
        "subcategories": {
            "Action": "Discrete player actions (Auction, Worker Placement, Drafting, etc.)",
            "Ruleset": "Systemic rules (Game Balance, Asymmetry, Phase Structure, etc.)",
            "Turn Structure": "How turns/rounds are organized (Action Points, Phases, etc.)",
            "Conflict Resolution": "How conflicts are resolved (Dice Rolling, Card Comparison, etc.)",
        }
    },
    "Data Representation": {
        "description": "System data structures — the game's components and state",
        "subcategories": {
            "Component": "Physical game components (Cards, Dice, Tokens, Board, etc.)",
            "State": "Game state trackers (Victory Points, Resources, Health, etc.)",
            "Spatial": "Spatial representations (Grid, Hex Map, Network, Area, etc.)",
        }
    },
    "Player Interaction": {
        "description": "How players interact with each other through mechanics",
        "subcategories": {
            "Cooperative": "Players work together (Cooperative Play, Trading, Alliances)",
            "Competitive": "Players compete (Bidding, Combat, Racing, Blocking)",
            "Negotiation": "Player-to-player negotiation and deal-making",
        }
    }
}


# ─── Gemini API Label Generation ─────────────────────────────
def generate_labels_gemini(
    clusters: Dict,
    api_key: str = None
) -> Dict[int, Dict]:
    """
    Generate MDA-compliant labels for clusters using Google Gemini API.
    
    For each cluster, the LLM analyzes the member mechanics and generates:
    - A formal mechanical label aligned with game studies terminology
    - MDA category classification (Algorithm/Data Representation)
    - A brief definition
    
    Args:
        clusters: Dict mapping cluster_id to cluster info with 'members' list
        api_key: Gemini API key (uses env var if not provided)
        
    Returns:
        Dict mapping cluster_id to label info
    """
    api_key = api_key or GEMINI_API_KEY
    
    if not api_key:
        logger.warning("No Gemini API key — falling back to lightweight keyword labeling")
        return generate_labels_tfidf(clusters)
    
    try:
        try:
            import google.genai as genai
            legacy_gemini = False
        except ImportError:
            import google.generativeai as genai
            legacy_gemini = True

        if legacy_gemini:
            genai.configure(api_key=api_key)
            chat = None
        else:
            client = genai.Client(api_key=api_key)
            chat = client.chats.create(model=GEMINI_MODEL)
    except ImportError:
        logger.warning("Gemini client not installed — using TF-IDF fallback")
        return generate_labels_tfidf(clusters)
    except Exception as e:
        logger.warning(f"Gemini setup failed: {e} — using TF-IDF fallback")
        return generate_labels_tfidf(clusters)

    labeled_clusters = {}
    
    for cluster_id, cluster_info in clusters.items():
        members = cluster_info.get("members", [])
        if not members:
            continue
        
        prompt = f"""You are an expert in formal game studies and the MDA (Mechanics, Dynamics, Aesthetics) framework.

Analyze these board game mechanics that were grouped together by hierarchical clustering:
{json.dumps(members[:20], indent=2)}

Based on the LUDES Game Mechanics Ontology, classify this cluster:

1. LABEL: Give a concise, formal mechanical category name (e.g., "Resource Management", "Area Control", "Action Selection")
2. MDA_CATEGORY: Classify as one of: "Algorithm/Action", "Algorithm/Ruleset", "Algorithm/Turn Structure", "Algorithm/Conflict Resolution", "Data Representation/Component", "Data Representation/State", "Data Representation/Spatial", "Player Interaction/Cooperative", "Player Interaction/Competitive", "Player Interaction/Negotiation"
3. DEFINITION: A 1-2 sentence formal definition of this mechanic category
4. PARENT: The broader parent category this falls under

Respond in this exact JSON format:
{{"label": "...", "mda_category": "...", "definition": "...", "parent": "..."}}"""

        try:
            if chat is not None:
                response = chat.send_message(
                    prompt,
                    config={
                        "temperature": LLM_TEMPERATURE,
                        "max_output_tokens": 300,
                    }
                )
            else:
                response = genai.generate_content(prompt)

            response_text = getattr(response, 'text', '')
            if not response_text and hasattr(response, 'json'):
                response_text = response.json().get('text', '')

            # Parse JSON from response
            response_text = response_text.strip()
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                label_data = json.loads(json_match.group())
            else:
                label_data = json.loads(response_text)
            
            labeled_clusters[int(cluster_id)] = {
                "label": label_data.get("label", f"Cluster_{cluster_id}"),
                "mda_category": label_data.get("mda_category", "Unknown"),
                "definition": label_data.get("definition", ""),
                "parent": label_data.get("parent", ""),
                "members": members,
                "source": "gemini",
            }
            
            logger.info(f"Cluster {cluster_id} → {label_data.get('label', 'Unknown')}")
            
        except Exception as e:
            logger.warning(f"Gemini labeling failed for cluster {cluster_id}: {e}")
            # Fallback for this cluster
            fallback = generate_single_label_tfidf(members)
            labeled_clusters[int(cluster_id)] = {
                **fallback,
                "members": members,
                "source": "keyword_based_fallback",
            }
    
    return labeled_clusters


# ─── TF-IDF Fallback Label Generation ────────────────────────
def generate_single_label_tfidf(members: List[str]) -> Dict:
    """
    Generate a label for a single cluster using TF-IDF keyword extraction.
    Identifies the most representative term(s) from the cluster members.
    """
    # Simple word frequency analysis
    all_words = []
    for member in members:
        words = re.findall(r'\b[A-Za-z]{3,}\b', member)
        all_words.extend([w.title() for w in words])
    
    # Remove common stop words
    stop_words = {
        "The", "And", "For", "With", "From", "That", "This", "Are",
        "Was", "Were", "Has", "Have", "Game", "Games", "Board",
        "Player", "Players", "Card", "Cards", "Play", "Playing",
    }
    filtered = [w for w in all_words if w not in stop_words]
    
    if not filtered:
        return {
            "label": f"Miscellaneous Mechanics",
            "mda_category": "Algorithm/Action",
            "definition": "A group of related game mechanics",
            "parent": "Algorithm",
        }
    
    # Most common words form the label
    counter = Counter(filtered)
    top_words = [word for word, _ in counter.most_common(3)]
    label = " ".join(top_words[:2]) if len(top_words) >= 2 else top_words[0]
    
    # Guess MDA category based on keywords
    mda_category = guess_mda_category(members)
    
    return {
        "label": label,
        "mda_category": mda_category,
        "definition": f"Mechanic category characterized by: {', '.join(top_words)}",
        "parent": mda_category.split("/")[0] if "/" in mda_category else "Algorithm",
    }


def guess_mda_category(members: List[str]) -> str:
    """Heuristic MDA category classification based on keyword matching."""
    text = " ".join(members).lower()
    
    # Data Representation indicators
    if any(kw in text for kw in ["card", "dice", "token", "board", "tile", "component"]):
        return "Data Representation/Component"
    if any(kw in text for kw in ["point", "score", "resource", "track", "health"]):
        return "Data Representation/State"
    if any(kw in text for kw in ["hex", "grid", "map", "area", "territory"]):
        return "Data Representation/Spatial"
    
    # Player Interaction indicators
    if any(kw in text for kw in ["cooperative", "team", "alliance", "together"]):
        return "Player Interaction/Cooperative"
    if any(kw in text for kw in ["combat", "attack", "battle", "war", "fight"]):
        return "Player Interaction/Competitive"
    if any(kw in text for kw in ["trade", "negotiate", "deal", "bargain"]):
        return "Player Interaction/Negotiation"
    
    # Algorithm indicators
    if any(kw in text for kw in ["auction", "bid", "draft", "select", "choose", "pick"]):
        return "Algorithm/Action"
    if any(kw in text for kw in ["phase", "round", "turn", "order", "sequence"]):
        return "Algorithm/Turn Structure"
    if any(kw in text for kw in ["balance", "asymmetr", "variable", "power"]):
        return "Algorithm/Ruleset"
    
    return "Algorithm/Action"


def generate_labels_tfidf(clusters: Dict) -> Dict[int, Dict]:
    """
    Generate labels for all clusters using TF-IDF keyword extraction.
    This is the no-API-key fallback method.
    """
    logger.info("Using TF-IDF keyword extraction for cluster labeling")
    
    labeled_clusters = {}
    for cluster_id, cluster_info in clusters.items():
        members = cluster_info.get("members", [])
        label_data = generate_single_label_tfidf(members)
        labeled_clusters[int(cluster_id)] = {
            **label_data,
            "members": members,
            "source": "tfidf",
        }
        logger.info(f"Cluster {cluster_id} → {label_data['label']}")
    
    return labeled_clusters


# ─── Main Label Generation Pipeline ─────────────────────────
def run_labeling_pipeline(
    clusters: Dict,
    api_key: str = None
) -> Dict[int, Dict]:
    """
    Execute the full label generation pipeline.
    
    Attempts Gemini API first, falls back to TF-IDF if unavailable.
    
    Args:
        clusters: Clustering results with member lists
        api_key: Optional Gemini API key
        
    Returns:
        Dict mapping cluster_id to labeled cluster info
    """
    labeled = generate_labels_gemini(clusters, api_key)
    
    # Save labeled results
    output_path = os.path.join(ONTOLOGY_OUTPUT_DIR, "labeled_clusters.json")
    serializable = {str(k): v for k, v in labeled.items()}
    save_json(serializable, output_path)
    logger.info(f"Labeled clusters saved to: {output_path}")
    
    return labeled
