"""
Module 2: Embedding Architecture & Dimensionality Reduction
===========================================================
Deploys and manages Sentence-BERT models for generating high-fidelity
semantic vector spaces from parsed game rulesets. Implements UMAP for
dimensionality reduction to manage the curse of dimensionality.

Pipeline:
1. Generate sentence-level semantic embeddings via Sentence-BERT (SBERT)
2. Apply UMAP for nonlinear dimensionality reduction
3. Preserve both local neighbor relations and global topology

Responsibilities (per proposal):
- Deploy a pre-trained Sentence-BERT model
- Generate high-fidelity semantic vector spaces
- Implement UMAP for dimensionality reduction
"""

import os
import numpy as np
from typing import List, Tuple, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    SBERT_MODEL_NAME, UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS, UMAP_MIN_DIST, DATA_DIR
)
from shared import setup_logger

logger = setup_logger("Embeddings")


# ─── Sentence-BERT Embedding Generator ──────────────────────
class EmbeddingEngine:
    """
    Manages Sentence-BERT model for semantic embedding generation.
    
    SBERT uses a bi-directional transformer architecture to project
    textual data into a high-dimensional semantic vector space,
    capturing deep contextual meaning rather than surface word frequency.
    """
    
    def __init__(self, model_name: str = SBERT_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        logger.info(f"EmbeddingEngine initialized with model: {model_name}")
    
    @property
    def model(self):
        """Lazy-load the Sentence-BERT model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading Sentence-BERT model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
        return self._model
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate semantic embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            show_progress: Show progress bar during encoding
            normalize: L2-normalize the embeddings (important for cosine similarity)
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        logger.info(f"Encoding {len(texts)} texts with SBERT ({self.model_name})")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        logger.info(f"Generated embeddings: shape={embeddings.shape}")
        return embeddings
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text string."""
        return self.encode([text], show_progress=False, normalize=normalize)[0]
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimensionality."""
        return self.model.get_sentence_embedding_dimension()


# ─── UMAP Dimensionality Reduction ──────────────────────────
def apply_umap(
    embeddings: np.ndarray,
    n_components: int = UMAP_N_COMPONENTS,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
    metric: str = "cosine",
    random_state: int = 42
) -> Tuple[np.ndarray, object]:
    """
    Apply UMAP (Uniform Manifold Approximation and Projection) for
    nonlinear dimensionality reduction.
    
    UMAP preserves both local neighbor relations and global topological
    structure, making it the preferred choice for preparing complex
    text embeddings for downstream clustering.
    
    Args:
        embeddings: High-dimensional embedding vectors (n_samples x original_dim)
        n_components: Target number of dimensions
        n_neighbors: Number of nearest neighbors for manifold approximation
        min_dist: Minimum distance between embedded points
        metric: Distance metric for UMAP
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (reduced_embeddings, fitted_umap_model)
    """
    try:
        import umap
    except ImportError:
        logger.warning("UMAP not available — using PCA fallback for dimensionality reduction")
        return apply_pca_fallback(embeddings, n_components)
    
    logger.info(
        f"Applying UMAP: {embeddings.shape[1]}D → {n_components}D "
        f"(n_neighbors={n_neighbors}, min_dist={min_dist})"
    )
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False
    )
    
    reduced = reducer.fit_transform(embeddings)
    
    logger.info(f"UMAP reduction complete: {embeddings.shape} → {reduced.shape}")
    return reduced, reducer


def apply_pca_fallback(
    embeddings: np.ndarray,
    n_components: int = UMAP_N_COMPONENTS
) -> Tuple[np.ndarray, object]:
    """
    PCA fallback when UMAP is not available.
    Linear dimensionality reduction preserving maximum variance.
    """
    from sklearn.decomposition import PCA
    
    logger.info(f"Applying PCA fallback: {embeddings.shape[1]}D → {n_components}D")
    
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    
    explained_var = sum(pca.explained_variance_ratio_) * 100
    logger.info(f"PCA complete: {explained_var:.1f}% variance explained")
    
    return reduced, pca


# ─── UMAP for 2D Visualization ──────────────────────────────
def reduce_for_visualization(
    embeddings: np.ndarray,
    labels: List[str] = None
) -> np.ndarray:
    """
    Reduce embeddings to 2D for scatter plot visualization.
    
    Returns:
        numpy array of shape (n_samples, 2)
    """
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.3,
            spread=1.2,
            metric="cosine",
            random_state=42
        )
        coords_2d = reducer.fit_transform(embeddings)
    except ImportError:
        from sklearn.decomposition import PCA
        n_comp = min(2, embeddings.shape[0], embeddings.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        coords_2d = pca.fit_transform(embeddings)
    
    return coords_2d


# ─── Main Embedding Pipeline ────────────────────────────────
def run_embedding_pipeline(
    texts: List[str],
    model_name: str = SBERT_MODEL_NAME,
    reduce_dims: bool = True,
    n_components: int = UMAP_N_COMPONENTS
) -> dict:
    """
    Execute the full embedding pipeline:
    1. Generate SBERT embeddings
    2. Apply UMAP dimensionality reduction
    3. Generate 2D visualization coordinates
    
    Args:
        texts: List of text strings to embed
        model_name: SBERT model name
        reduce_dims: Whether to apply UMAP reduction
        n_components: Target dimensions for UMAP
        
    Returns:
        Dict with embeddings, reduced embeddings, and 2D coords
    """
    engine = EmbeddingEngine(model_name)
    
    # 1. Generate full embeddings
    full_embeddings = engine.encode(texts)
    
    results = {
        "full_embeddings": full_embeddings,
        "embedding_dim": full_embeddings.shape[1],
        "n_samples": full_embeddings.shape[0],
    }
    
    # 2. Reduce dimensions for clustering
    if reduce_dims and full_embeddings.shape[1] > n_components:
        reduced, reducer = apply_umap(full_embeddings, n_components=n_components)
        results["reduced_embeddings"] = reduced
        results["reduced_dim"] = reduced.shape[1]
        results["reducer"] = reducer
    else:
        results["reduced_embeddings"] = full_embeddings
        results["reduced_dim"] = full_embeddings.shape[1]
    
    # 3. 2D for visualization
    coords_2d = reduce_for_visualization(full_embeddings)
    results["coords_2d"] = coords_2d
    
    return results
