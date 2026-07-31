"""
Module 2: System Evaluation Framework
======================================
Implements quantitative and qualitative evaluation metrics for
the RAG backend and overall system performance.

Quantitative Metrics (Information Retrieval):
- Hit Rate@K: Fraction of queries returning at least one relevant result in top K
- Mean Reciprocal Rank (MRR): Average of 1/rank of first relevant result

Qualitative Metrics (HCI):
- System Usability Scale (SUS) scoring framework
- Semantic Relevance Ratings (Likert scale processing)

Responsibilities (per proposal):
- Design and execute quantitative evaluation frameworks
- Calculate retrieval performance metrics (Hit Rate@K, MRR, SUS)
- Conduct user testing sessions and gather HCI data
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TOP_K_RESULTS, ONTOLOGY_OUTPUT_DIR
from shared import setup_logger, save_json

logger = setup_logger("Evaluator")


# ─── Mean Reciprocal Rank (MRR) ─────────────────────────────
def reciprocal_rank(
    retrieved: List[str],
    relevant: List[str]
) -> float:
    """
    Compute Reciprocal Rank for a single query.
    
    RR = 1 / rank of the first relevant result
    
    Answers: How fast does the user see a relevant mechanic?
    MRR close to 1.0 means the best result is consistently at the top.
    """
    relevant_set = set(r.lower() for r in relevant)
    
    for i, item in enumerate(retrieved):
        if item.lower() in relevant_set:
            return 1.0 / (i + 1)
    
    return 0.0


def mean_reciprocal_rank(
    all_retrieved: List[List[str]],
    all_relevant: List[List[str]]
) -> float:
    """
    Compute Mean Reciprocal Rank across all queries.
    
    MRR = (1/|Q|) * Σ(1/rank_i) for all queries Q
    """
    if not all_retrieved:
        return 0.0
    
    rr_scores = [
        reciprocal_rank(ret, rel)
        for ret, rel in zip(all_retrieved, all_relevant)
    ]
    
    return sum(rr_scores) / len(rr_scores)


# ─── Hit Rate @ K ────────────────────────────────────────────
def hit_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int = TOP_K_RESULTS
) -> int:
    """
    Compute Hit@K: returns 1 if at least one relevant item is in top K, else 0.
    """
    if k == 0:
        return 0
    top_k = retrieved[:k]
    relevant_set = set(r.lower() for r in relevant)
    return 1 if any(item.lower() in relevant_set for item in top_k) else 0





# ─── System Usability Scale (SUS) ───────────────────────────
def compute_sus_score(responses: List[int]) -> Dict:
    """
    Compute System Usability Scale (SUS) score from questionnaire responses.
    
    SUS is a 10-item questionnaire where each item is rated 1-5.
    Odd-numbered items are positive, even-numbered are negative.
    
    Scoring:
    - For odd items (1,3,5,7,9): score = response - 1
    - For even items (2,4,6,8,10): score = 5 - response
    - Final SUS = sum of scores * 2.5 (scales to 0-100)
    
    Args:
        responses: List of 10 integers (1-5 each)
        
    Returns:
        Dict with SUS score and interpretation
    """
    if len(responses) != 10:
        raise ValueError("SUS requires exactly 10 responses")
    
    if not all(1 <= r <= 5 for r in responses):
        raise ValueError("Each response must be between 1 and 5")
    
    scores = []
    for i, response in enumerate(responses):
        if i % 2 == 0:  # Odd items (0-indexed: 0,2,4,6,8)
            scores.append(response - 1)
        else:  # Even items (0-indexed: 1,3,5,7,9)
            scores.append(5 - response)
    
    sus_score = sum(scores) * 2.5
    
    # Interpretation
    if sus_score >= 80.3:
        grade = "A"
        adjective = "Excellent"
    elif sus_score >= 68:
        grade = "B" if sus_score >= 74 else "C"
        adjective = "Good" if sus_score >= 74 else "OK"
    elif sus_score >= 51:
        grade = "D"
        adjective = "Poor"
    else:
        grade = "F"
        adjective = "Awful"
    
    return {
        "sus_score": round(sus_score, 1),
        "grade": grade,
        "adjective": adjective,
        "item_scores": scores,
        "interpretation": f"SUS Score: {sus_score:.1f}/100 ({adjective}, Grade {grade})"
    }


# ─── Comprehensive Evaluation Pipeline ──────────────────────
def run_evaluation(
    retriever,
    test_queries: List[Dict]
) -> Dict:
    """
    Run comprehensive evaluation of the RAG system.
    
    Args:
        retriever: RAGRetriever instance
        test_queries: List of dicts with 'query' and 'relevant_mechanics'
        k_values: List of K values to evaluate at
        
    Returns:
        Complete evaluation results dict
    """
    all_retrieved = []
    all_relevant = []
    per_query_results = []
    
    for test in test_queries:
        query = test["query"]
        relevant = test.get("relevant_mechanics", [])
        
        # Run retrieval
        result = retriever.retrieve(query)
        retrieved = [r["mechanic"] for r in result["results"]]
        
        all_retrieved.append(retrieved)
        all_relevant.append(relevant)
        
        # Per-query metrics
        query_metrics = {"query": query}
        query_metrics["hit@5"] = hit_at_k(retrieved, relevant, 5)
        query_metrics["reciprocal_rank"] = round(reciprocal_rank(retrieved, relevant), 4)
        per_query_results.append(query_metrics)
    
    # Aggregate metrics
    aggregate = {}
    hits = [q["hit@5"] for q in per_query_results]
    aggregate["hit_rate@5"] = round(np.mean(hits), 4) if hits else 0
    aggregate["MRR"] = round(mean_reciprocal_rank(all_retrieved, all_relevant), 4)
    
    evaluation_results = {
        "aggregate_metrics": aggregate,
        "per_query_results": per_query_results,
        "n_queries": len(test_queries)
    }
    
    # Save evaluation results
    output_path = os.path.join(ONTOLOGY_OUTPUT_DIR, "evaluation_results.json")
    save_json(evaluation_results, output_path)
    logger.info(f"Evaluation results saved to: {output_path}")
    
    return evaluation_results


# ─── Built-in Test Queries ───────────────────────────────────
SAMPLE_TEST_QUERIES = [
    {
        "query": "I want a game where players bid on resources and manage an economy",
        "relevant_mechanics": [
            "Auction/Bidding", "Auction: English", "Resource Management",
            "Market", "Trading", "Economic", "Stock Holding"
        ]
    },
    {
        "query": "A cooperative game where players work together against the game itself",
        "relevant_mechanics": [
            "Cooperative Game", "Solo / Solitaire Game", "Team-Based Game",
            "Communication Limits", "Variable Player Powers"
        ]
    },
    {
        "query": "Players place workers on a board to claim actions and gather resources",
        "relevant_mechanics": [
            "Worker Placement", "Action Points", "Resource Management",
            "Area Majority / Influence", "Turn Order: Claim Action"
        ]
    },
    {
        "query": "A strategy game with territory control and military combat on a hex map",
        "relevant_mechanics": [
            "Area Control / Area Influence", "Hexagon Grid", "Combat",
            "Dice Rolling", "Area Movement", "Grid Movement"
        ]
    },
    {
        "query": "Card drafting game where you build a deck over time to score points",
        "relevant_mechanics": [
            "Card Drafting", "Deck Building", "Hand Management",
            "Set Collection", "Open Drafting", "Closed Drafting"
        ]
    },
]
