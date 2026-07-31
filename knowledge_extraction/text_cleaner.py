"""
Module 1: NLP Text Cleaner & Rule Extractor
============================================
Implements linguistic rule-based filters and syntactic dependency
parsers to isolate actionable algorithmic rules from thematic game lore.

Per the MDA framework:
- MECHANICS: Foundational rules, algorithms, data representations
- DYNAMICS: Run-time behaviors emerging from mechanics (filter out)
- AESTHETICS: Emotional responses / thematic elements (filter out)

The module uses:
- spaCy dependency parsing to identify imperative verbs and rule structures
- Regex patterns for mechanical rule detection
- Sentence classification: RULE / FLAVOR / COMPONENT / EXAMPLE

Responsibilities (per proposal):
- Develop linguistic rule-based filters and syntactic dependency parsers
- Clean text corpus and isolate actionable algorithmic rules from lore
"""

import os
import re
from typing import List, Dict, Tuple
from enum import Enum

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PARSED_MARKDOWN_DIR, CLEANED_RULES_DIR
from shared import setup_logger, save_json

logger = setup_logger("TextCleaner")


# ─── Sentence Classification Categories (MDA-aligned) ─────────
class SentenceType(str, Enum):
    RULE = "RULE"               # Mechanical rule / algorithm
    COMPONENT = "COMPONENT"     # Game component / data representation
    EXAMPLE = "EXAMPLE"         # Gameplay example
    FLAVOR = "FLAVOR"           # Thematic lore / aesthetic description
    SETUP = "SETUP"             # Setup instructions
    UNKNOWN = "UNKNOWN"


# ─── Rule Detection Patterns ──────────────────────────────────
# Imperative verb patterns indicative of mechanical rules
RULE_INDICATORS = [
    r"\b(must|shall|cannot|may not|is not allowed)\b",
    r"\b(place|move|draw|discard|roll|pay|spend|gain|lose|take|give|choose|select|reveal|shuffle)\b",
    r"\b(each player|the active player|all players|starting player)\b",
    r"\b(at the (start|end|beginning) of)\b",
    r"\b(in (clockwise|turn) order)\b",
    r"\b(if .+ then)\b",
    r"\b(during (your|a|the|each) (turn|phase|round|action))\b",
    r"\b(victory point|score|win|winning condition)\b",
    r"\b(maximum|minimum|exactly|at least|at most|up to)\b",
    r"\b(per turn|per round|per player|once per)\b",
]

# Component / data representation patterns
COMPONENT_INDICATORS = [
    r"\b(\d+)\s*(card|tile|token|die|dice|board|piece|cube|marker|meeple)s?\b",
    r"\b(deck|hand|supply|reserve|pile|stack|bag|pool)\b",
    r"\b(resource|gold|coin|money|wood|stone|brick|wheat|ore|sheep)\b",
    r"\b(board|map|grid|hex|space|track|path)\b",
]

# Flavor text / aesthetic patterns (to be filtered)
FLAVOR_INDICATORS = [
    r"\b(once upon|long ago|in (a|the) (land|world|kingdom|realm))\b",
    r"\b(legend|story|tale|myth|ancient|mysterious)\b",
    r"\b(beautiful|magnificent|terrifying|glorious|dark|evil)\b",
    r"\b(hero|villain|dragon|wizard|warrior|quest|adventure)\b",
    r"^\".*\"$",  # Quoted flavor text
]

# Setup instruction patterns
SETUP_INDICATORS = [
    r"\b(setup|set up|preparation|before (the|you) (game|play|begin))\b",
    r"\b(place .+ (on|in|near|beside) the)\b",
    r"\b(give each player|distribute)\b",
    r"\b(shuffle .+ (deck|cards|tiles))\b",
]


# ─── Sentence Classification ─────────────────────────────────
def classify_sentence(sentence: str) -> SentenceType:
    """
    Classify a sentence based on MDA framework alignment.
    Uses regex pattern matching against mechanical rule indicators.
    
    Returns the most likely SentenceType for the sentence.
    """
    text_lower = sentence.lower().strip()
    
    if len(text_lower) < 5:
        return SentenceType.UNKNOWN
    
    # Score each category
    scores = {
        SentenceType.RULE: 0,
        SentenceType.COMPONENT: 0,
        SentenceType.FLAVOR: 0,
        SentenceType.SETUP: 0,
        SentenceType.EXAMPLE: 0,
    }
    
    for pattern in RULE_INDICATORS:
        if re.search(pattern, text_lower):
            scores[SentenceType.RULE] += 2
    
    for pattern in COMPONENT_INDICATORS:
        if re.search(pattern, text_lower):
            scores[SentenceType.COMPONENT] += 2
    
    for pattern in FLAVOR_INDICATORS:
        if re.search(pattern, text_lower):
            scores[SentenceType.FLAVOR] += 3  # Higher weight to filter flavor
    
    for pattern in SETUP_INDICATORS:
        if re.search(pattern, text_lower):
            scores[SentenceType.SETUP] += 2
    
    # Example detection
    if re.search(r"\b(example|e\.g\.|for instance|such as)\b", text_lower):
        scores[SentenceType.EXAMPLE] += 2
    
    # Imperative verb detection (strong rule indicator)
    if re.match(r"^(place|move|draw|roll|take|give|choose|play|add|remove|swap|flip)\b", text_lower):
        scores[SentenceType.RULE] += 3
    
    # Get highest scoring category
    max_score = max(scores.values())
    if max_score == 0:
        return SentenceType.UNKNOWN
    
    # Critical Fix: Prevent False Positives for Mixed Sentences
    # "If the Dragon enters the Castle, draw 2 cards" -> Rule (2) vs Flavor (3)
    # If the sentence contains ANY concrete rule indicators, it MUST be preserved.
    if scores[SentenceType.RULE] > 0 and scores[SentenceType.FLAVOR] > 0:
        return SentenceType.RULE
    
    return max(scores, key=scores.get)


# ─── spaCy-Based Dependency Analysis ─────────────────────────
def analyze_with_spacy(sentences: List[str]) -> List[Dict]:
    """
    Use spaCy dependency parsing for deeper syntactic analysis.
    Identifies imperative structures, logical conditionals,
    and verb-subject associations characteristic of formal rules.
    """
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Downloading spaCy model...")
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
    except ImportError:
        logger.warning("spaCy not available — using regex-only classification")
        return [{"text": s, "has_imperative": False, "has_conditional": False} for s in sentences]
    
    results = []
    for sent_text in sentences:
        doc = nlp(sent_text)
        
        has_imperative = False
        has_conditional = False
        action_verbs = []
        
        for token in doc:
            # Detect imperative mood (verb at start without subject)
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                # Check if there's no explicit subject before the verb
                subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                if not subjects and token.i < 3:  # Verb near beginning
                    has_imperative = True
            
            # Detect conditional structures
            if token.lower_ in ("if", "when", "whenever", "unless"):
                has_conditional = True
            
            # Collect action verbs
            if token.pos_ == "VERB" and token.dep_ in ("ROOT", "conj", "xcomp"):
                action_verbs.append(token.lemma_)
        
        results.append({
            "text": sent_text,
            "has_imperative": has_imperative,
            "has_conditional": has_conditional,
            "action_verbs": action_verbs,
            "token_count": len(doc),
        })
    
    return results


# ─── Text Cleaning Pipeline ──────────────────────────────────
def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex-based rules."""
    # Split on period, question mark, exclamation followed by space + capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Also split on newlines that look like sentence boundaries
    expanded = []
    for sent in sentences:
        sub_sents = sent.split("\n")
        expanded.extend(sub_sents)
    
    # Clean and filter
    cleaned = [s.strip() for s in expanded if len(s.strip()) > 10]
    return cleaned


def clean_text_corpus(
    raw_text: str,
    include_types: List[SentenceType] = None
) -> Dict:
    """
    Full text cleaning pipeline:
    1. Split into sentences
    2. Classify each sentence (MDA-aligned)
    3. Optional spaCy dependency analysis
    4. Filter to keep only mechanical rules and components
    
    Args:
        raw_text: Raw extracted text from PDF
        include_types: Which SentenceTypes to keep (default: RULE, COMPONENT, SETUP)
    
    Returns:
        Dict with classified sentences and statistics
    """
    if include_types is None:
        include_types = [SentenceType.RULE, SentenceType.COMPONENT, SentenceType.SETUP]
    
    # Split into sentences
    sentences = split_into_sentences(raw_text)
    logger.info(f"Split text into {len(sentences)} sentences")
    
    # 3. Optional spaCy dependency analysis for deeper syntactic context
    spacy_results = analyze_with_spacy(sentences)
    spacy_map = {res["text"]: res for res in spacy_results}
    
    # Classify each sentence
    classified = []
    for sent in sentences:
        sent_type = classify_sentence(sent)
        
        # Deep syntactic override: Use spaCy logic to upgrade classification cautiously.
        spacy_info = spacy_map.get(sent, {
            "has_imperative": False,
            "has_conditional": False,
            "action_verbs": []
        })
        has_imperative = spacy_info["has_imperative"]
        has_conditional = spacy_info["has_conditional"]
        action_verbs = spacy_info["action_verbs"]
        
        if sent_type == SentenceType.UNKNOWN:
            if has_imperative or (has_conditional and action_verbs):
                sent_type = SentenceType.RULE
        elif sent_type == SentenceType.FLAVOR:
            if has_imperative:
                sent_type = SentenceType.RULE
                
        classified.append({
            "text": sent,
            "type": sent_type.value,
            "included": sent_type in include_types,
            "spacy_imperative": has_imperative,
            "spacy_conditional": has_conditional,
            "action_verbs": spacy_info.get("action_verbs", [])
        })
    
    # Statistics
    type_counts = {}
    for item in classified:
        t = item["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    
    # Filter to included types
    included_sentences = [item["text"] for item in classified if item["included"]]
    
    logger.info(f"Classification breakdown: {type_counts}")
    logger.info(f"Retained {len(included_sentences)}/{len(sentences)} sentences as mechanical rules")
    
    return {
        "all_classified": classified,
        "included_sentences": included_sentences,
        "statistics": type_counts,
        "total_sentences": len(sentences),
        "retained_sentences": len(included_sentences),
    }


# ─── Process Board Game CSV Descriptions ─────────────────────
def clean_game_descriptions(df, text_column: str = "description") -> List[Dict]:
    """
    Clean and extract mechanical rules from board game descriptions
    in the CSV dataset. Processes each game's description to isolate
    rule-dense sentences.
    
    Args:
        df: DataFrame with game data
        text_column: Column containing text descriptions
        
    Returns:
        List of dicts with game info and cleaned rule sentences
    """
    results = []
    
    for idx, row in df.iterrows():
        text = str(row.get(text_column, ""))
        if not text or text == "nan":
            continue
        
        # Quick classification
        sentences = split_into_sentences(text)
        rule_sentences = []
        
        for sent in sentences:
            sent_type = classify_sentence(sent)
            if sent_type in (SentenceType.RULE, SentenceType.COMPONENT, SentenceType.SETUP):
                rule_sentences.append(sent)
        
        results.append({
            "game_index": idx,
            "total_sentences": len(sentences),
            "rule_sentences": rule_sentences,
            "rule_count": len(rule_sentences),
        })
    
    logger.info(f"Processed {len(results)} game descriptions")
    return results


# ─── Run Full Cleaning Pipeline ──────────────────────────────
def run_cleaning_pipeline(
    input_dir: str = PARSED_MARKDOWN_DIR,
    output_dir: str = CLEANED_RULES_DIR
) -> List[Dict]:
    """
    Process all parsed markdown files through the cleaning pipeline.
    
    Returns:
        List of cleaning results for each processed file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    md_files = [f for f in os.listdir(input_dir) if f.endswith(".md")] if os.path.exists(input_dir) else []
    
    if not md_files:
        logger.info("No parsed markdown files found to clean")
        return []
    
    results = []
    for md_file in md_files:
        filepath = os.path.join(input_dir, md_file)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        cleaning_result = clean_text_corpus(raw_text)
        
        # Save cleaned rules
        output_name = os.path.splitext(md_file)[0]
        output_path = os.path.join(output_dir, f"{output_name}_rules.json")
        save_json(cleaning_result, output_path)
        
        results.append({
            "file": md_file,
            "output": output_path,
            **cleaning_result["statistics"],
            "retained": cleaning_result["retained_sentences"],
        })
    
    return results


# ─── CLI Entry Point ──────────────────────────────────────────
if __name__ == "__main__":
    import json
    results = run_cleaning_pipeline()
    print(json.dumps(results, indent=2))
