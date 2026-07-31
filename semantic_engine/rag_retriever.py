"""
Module 2: RAG Retriever & Semantic Matching
===========================================
Engineers the backend semantic matching algorithms for the
Retrieval-Augmented Generation (RAG) pipeline.

Pipeline:
1. KeyBERT extracts semantic keywords from user's natural language query
2. Keywords are embedded into the same SBERT vector space
3. Cosine similarity retrieves top-K matching mechanic nodes
4. Hierarchical context (parents/children) enriches the results

Responsibilities (per proposal):
- Implement KeyBERT for dynamic user prompt parsing
- Map prompts to the ontology vector space via cosine similarity
- Engineer the RAG retrieval module
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TOP_K_RESULTS, KEYBERT_TOP_N, SIMILARITY_THRESHOLD
from shared import setup_logger

logger = setup_logger("RAGRetriever")


# ─── KeyBERT Keyword Extraction ─────────────────────────────
def extract_keywords(
    text: str,
    top_n: int = KEYBERT_TOP_N,
    diversity: float = 0.5
) -> List[Tuple[str, float]]:
    """
    Extract semantically relevant keywords from user input using KeyBERT.
    
    KeyBERT leverages BERT embeddings to find sub-phrases that are
    most semantically relevant, filtering out conversational filler
    to isolate design-relevant concepts.
    
    Args:
        text: User's natural language input
        top_n: Number of keywords to extract
        diversity: Diversity of keywords (0=similar, 1=diverse)
        
    Returns:
        List of (keyword, relevance_score) tuples
    """
    try:
        from keybert import KeyBERT
        
        kw_model = KeyBERT(model="all-MiniLM-L6-v2")
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=top_n,
            use_mmr=True,
            diversity=diversity
        )
        
        logger.info(f"KeyBERT extracted {len(keywords)} keywords: {[k[0] for k in keywords]}")
        return keywords
        
    except ImportError:
        logger.warning("KeyBERT not available — using basic keyword extraction")
        return extract_keywords_basic(text, top_n)


def extract_keywords_basic(text: str, top_n: int = 5) -> List[Tuple[str, float]]:
    """
    Basic keyword extraction fallback using TF-IDF-style scoring.
    Used when KeyBERT is not installed.
    """
    import re
    from collections import Counter
    
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "must", "need", "want",
        "like", "with", "for", "from", "that", "this", "which", "who",
        "what", "where", "when", "how", "and", "or", "but", "not", "no",
        "in", "on", "at", "to", "of", "by", "it", "its", "my", "your",
        "i", "me", "we", "us", "they", "them", "game", "board", "player",
        "players", "also", "very", "really", "just", "some", "any",
    }
    
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    filtered = [w for w in words if w not in stop_words]
    
    # Also extract 2-gram phrases
    bigrams = [f"{filtered[i]} {filtered[i+1]}" for i in range(len(filtered)-1)]
    
    all_terms = filtered + bigrams
    counter = Counter(all_terms)
    
    top_keywords = counter.most_common(top_n)
    # Normalize scores
    max_count = max(c for _, c in top_keywords) if top_keywords else 1
    return [(kw, count / max_count) for kw, count in top_keywords]


# ─── Semantic Similarity Search ──────────────────────────────
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def batch_cosine_similarity(
    query_vec: np.ndarray,
    corpus_vecs: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and all corpus vectors.
    Vectorized for efficiency using numpy.
    """
    # Normalize vectors
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    corpus_norms = corpus_vecs / (np.linalg.norm(corpus_vecs, axis=1, keepdims=True) + 1e-10)
    
    # Batch dot product
    similarities = np.dot(corpus_norms, query_norm)
    return similarities


# ─── RAG Retrieval Engine ────────────────────────────────────
class RAGRetriever:
    """
    Retrieval-Augmented Generation engine for semantic mechanic search.
    
    Connects user's natural language design concepts to the formal
    mechanics stored in the vector database by:
    1. Extracting semantic keywords from input
    2. Embedding into shared semantic space
    3. Computing cosine similarity against indexed mechanics
    4. Retrieving top-K results with hierarchical context
    """
    
    def __init__(
        self,
        embedding_engine,
        mechanic_labels: List[str],
        mechanic_embeddings: np.ndarray,
        labeled_clusters: Dict = None,
        game_data: object = None
    ):
        """
        Initialize the RAG retriever.
        
        Args:
            embedding_engine: EmbeddingEngine instance for query encoding
            mechanic_labels: List of mechanic names/labels
            mechanic_embeddings: Pre-computed mechanic embeddings
            labeled_clusters: Cluster labeling info for hierarchical context
            game_data: DataFrame with game information for examples
        """
        self.engine = embedding_engine
        self.mechanic_labels = mechanic_labels
        self.mechanic_embeddings = mechanic_embeddings
        self.labeled_clusters = labeled_clusters or {}
        self.game_data = game_data
        
        # Initialize Cross-Encoder for Stage 2 Re-ranking
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("CrossEncoder initialized successfully for Stage 2 Re-ranking.")
        except Exception as e:
            logger.warning(f"Could not load CrossEncoder: {e}")
            self.cross_encoder = None
        
        logger.info(
            f"RAGRetriever initialized: {len(mechanic_labels)} mechanics, "
            f"embedding dim={mechanic_embeddings.shape[1]}"
        )
    
    def _expand_query_with_llm(self, query: str, api_key: str = None) -> str:
        """Use Gemini to translate user query into formal mechanic names."""
        self.last_llm_error = None
        
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            return query
            
        self.llm_attempted = True
        try:
            import time
            logger.info("Waiting 16 seconds to respect Gemini Free Tier rate limits (15 RPM)...")
            time.sleep(16)  # Avoid Free Tier 15 RPM rate limit

            try:
                import google.genai as genai
                legacy_gemini = False
            except ImportError:
                import google.generativeai as genai
                legacy_gemini = True

            prompt = f"You are an expert game designer. A user is looking for a board game with these characteristics: '{query}'. Translate this into a comma-separated list of formal board game mechanics (e.g. Worker Placement, Action Points). Only output the list."

            if not legacy_gemini and hasattr(genai, 'Client'):
                client = genai.Client(api_key=api_key)
                chat = client.chats.create(model="gemini-2.0-flash")
                response = chat.send_message(
                    prompt,
                    config={"temperature": 0.3, "max_output_tokens": 200}
                )
                response_text = getattr(response, 'text', '')
                if not response_text and hasattr(response, 'json'):
                    response_text = response.json().get('text', '')
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content(prompt)
                response_text = getattr(response, 'text', '')

            if response_text:
                logger.info(f"LLM Expanded Query: {response_text.strip()}")
                return response_text.strip()
            return query
        except Exception as e:
            self.last_llm_error = str(e)
            logger.warning(f"LLM Expansion failed: {e}")
            return query
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        min_similarity: float = SIMILARITY_THRESHOLD,
        include_context: bool = True,
        api_key: str = None
    ) -> Dict:
        """
        Retrieve the most relevant mechanics for a user query.
        
        Full RAG pipeline:
        1. Extract keywords with KeyBERT
        2. Embed query into semantic space
        3. Compute similarities against all mechanics
        4. Return top-K with hierarchical context
        
        Args:
            query: User's natural language design description
            top_k: Number of results to retrieve
            min_similarity: Minimum similarity threshold
            include_context: Whether to include hierarchical context
            api_key: Optional Gemini API key to force LLM expansion
            
        Returns:
            Dict with retrieved mechanics, keywords, similarities, and context
        """
        logger.info(f"RAG query: '{query[:80]}...'")
        
        # 1. LLM Query Expansion
        expanded_query = self._expand_query_with_llm(query, api_key=api_key)
        
        # 2. Embed the query
        if getattr(self, 'llm_attempted', False) and not getattr(self, 'last_llm_error', None):
            # LLM Success: Embed the perfect list directly
            query_embedding = self.engine.encode_single(expanded_query)
            # Generate dummy keywords for the UI visualization
            keywords = [(kw.strip(), 1.0) for kw in expanded_query.split(',') if kw.strip()][:5]
        else:
            # LLM Failed or not used: Fallback to KeyBERT Dense logic
            keywords = extract_keywords(query)
            if keywords:
                dense_query = " ".join([kw[0] for kw in keywords])
                query_embedding = self.engine.encode_single(dense_query)
            else:
                query_embedding = self.engine.encode_single(query)
        
        # 3. Compute similarities
        similarities = batch_cosine_similarity(query_embedding, self.mechanic_embeddings)
        
        # 3.5 Lexical / Keyword Boost (Smarter Match)
        query_lower = query.lower()
        import re
        
        GAME_STOP_WORDS = {'game', 'card', 'cards', 'board', 'boards', 'player', 'players', 
                           'play', 'turn', 'turns', 'point', 'points', 'and', 'or', 'the', 
                           'a', 'an', 'to', 'with', 'on', 'in', 'for', 'of', 'is', 'it'}
        
        query_tokens = set(re.findall(r'[a-z]+', query_lower))
        expanded_query_tokens = set()
        for qt in query_tokens:
            expanded_query_tokens.add(qt)
            if qt.endswith('s'): expanded_query_tokens.add(qt[:-1])
            if qt.endswith('es'): expanded_query_tokens.add(qt[:-2])
            if qt.endswith('ing'): expanded_query_tokens.add(qt[:-3])
            
        for idx, mech_name in enumerate(self.mechanic_labels):
            mech_lower = mech_name.lower()
            
            # 1. Exact phrase match
            if mech_lower in query_lower and len(mech_lower) > 4:
                similarities[idx] += 0.15
            else:
                # 2. Token overlap match
                mech_tokens = set(re.findall(r'[a-z]+', mech_lower))
                mech_tokens = mech_tokens - GAME_STOP_WORDS
                
                if mech_tokens:
                    overlap = mech_tokens.intersection(expanded_query_tokens)
                    if overlap:
                        # Boost based on how much of the mechanic name matched
                        similarities[idx] += 0.05 * (len(overlap) / len(mech_tokens))
        
        # 3.8 Ontology-Aware Re-Ranking
        if self.labeled_clusters:
            best_idx = int(np.argmax(similarities))
            best_mechanic = self.mechanic_labels[best_idx]
            
            # Find the cluster of the top mechanic
            for cluster_info in self.labeled_clusters.values():
                members = cluster_info.get("members", [])
                if best_mechanic in members:
                    # Apply a small boost to all sibling mechanics
                    for sibling in members:
                        if sibling != best_mechanic:
                            try:
                                sibling_idx = self.mechanic_labels.index(sibling)
                                similarities[sibling_idx] += 0.05
                            except ValueError:
                                pass
                    break
        
        # 4. Get top-K indices
        top_indices = np.argsort(similarities)[::-1][:25] # Grab Top 25 for re-ranking
        
        # ----- STAGE 2: CROSS ENCODER RE-RANKING -----
        if hasattr(self, 'cross_encoder') and self.cross_encoder:
            ce_candidates = [self.mechanic_labels[idx] for idx in top_indices]
            ce_pairs = [[query, cand] for cand in ce_candidates]
            ce_scores = self.cross_encoder.predict(ce_pairs)
            
            # Normalize logits to 0.5 - 1.0 to bypass min_similarity threshold
            ce_scores = (ce_scores - np.min(ce_scores)) / (np.max(ce_scores) - np.min(ce_scores) + 1e-9)
            ce_scores = ce_scores * 0.4 + 0.6
            
            # Sort the top 25 by cross-encoder score
            ce_sorted_indices = np.argsort(ce_scores)[::-1]
            
            # Update similarities with CE scores so they display nicely
            for local_idx, real_idx in enumerate(top_indices):
                similarities[real_idx] = float(ce_scores[local_idx])
                
            # Re-map the top indices
            top_indices = [top_indices[i] for i in ce_sorted_indices]
            
        # Truncate to final top_k
        top_indices = top_indices[:top_k]
        # 5. Build results
        results = []
        for idx in top_indices:
            sim_score = float(similarities[idx])
            if sim_score < min_similarity:
                continue
            
            mechanic_name = self.mechanic_labels[idx]
            
            result = {
                "mechanic": mechanic_name,
                "similarity": round(sim_score, 4),
                "rank": len(results) + 1,
            }
            
            # Add hierarchical context
            if include_context and self.labeled_clusters:
                context = self._get_hierarchical_context(mechanic_name)
                result["context"] = context
            
            # Add game examples
            if self.game_data is not None:
                examples = self._get_game_examples(mechanic_name)
                result["game_examples"] = examples
            
            results.append(result)
        
        retrieval_result = {
            "query": query,
            "expanded_query": locals().get('expanded_query', query),
            "used_llm": getattr(self, 'llm_attempted', False) and not getattr(self, 'last_llm_error', None),
            "llm_error": getattr(self, 'last_llm_error', None),
            "keywords": [{"keyword": k, "score": round(s, 4)} for k, s in keywords],
            "total_results": len(results),
            "results": results,
        }
        
        logger.info(f"Retrieved {len(results)} mechanics (top similarity: {results[0]['similarity'] if results else 0})")
        return retrieval_result
    
    def _get_hierarchical_context(self, mechanic_name: str) -> Dict:
        """
        Get the parent class, sibling mechanics, and sub-categories
        for a retrieved mechanic to provide taxonomic context.
        """
        context = {
            "parent_category": "",
            "mda_category": "",
            "cluster_label": "",
            "sibling_mechanics": [],
        }
        
        for cluster_id, cluster_info in self.labeled_clusters.items():
            members = cluster_info.get("members", [])
            if mechanic_name in members:
                context["cluster_label"] = cluster_info.get("label", "")
                context["mda_category"] = cluster_info.get("mda_category", "")
                context["parent_category"] = cluster_info.get("parent", "")
                context["sibling_mechanics"] = [m for m in members if m != mechanic_name][:5]
                break
        
        return context
    
    def _get_game_examples(self, mechanic_name: str, max_examples: int = 3) -> List[Dict]:
        """
        Retrieve specific game examples that use the given mechanic.
        """
        if self.game_data is None:
            return []
        
        examples = []
        mechanic_lower = mechanic_name.lower()
        
        for _, row in self.game_data.iterrows():
            mechanics_str = str(row.get("mechanics", "")).lower()
            if mechanic_lower in mechanics_str:
                examples.append({
                    "title": row.get("title", row.get("name", "Unknown")),
                    "year": str(row.get("year", row.get("yearpublished", ""))),
                    "rating": str(row.get("rating", row.get("average", ""))),
                })
                if len(examples) >= max_examples:
                    break
        
        return examples


# ─── Convenience Function ────────────────────────────────────
def create_retriever(
    embedding_engine,
    mechanic_labels: List[str],
    mechanic_embeddings: np.ndarray,
    labeled_clusters: Dict = None,
    game_data=None
) -> RAGRetriever:
    """Create and return a configured RAGRetriever instance."""
    return RAGRetriever(
        embedding_engine=embedding_engine,
        mechanic_labels=mechanic_labels,
        mechanic_embeddings=mechanic_embeddings,
        labeled_clusters=labeled_clusters,
        game_data=game_data
    )
