"""
Module 1: Hierarchical Agglomerative Clustering (HAC)
=====================================================
Implements the core ontology engineering pipeline using unsupervised
bottom-up hierarchical clustering to construct the mathematical
dendrogram of board game mechanics.

The HAC algorithm:
1. Starts with each mechanic term as its own cluster
2. Progressively merges nearest pairs using Ward's linkage
3. Generates a complete dendrogram (hierarchical taxonomy)
4. Cuts the dendrogram at optimal threshold for discrete clusters

Responsibilities (per proposal):
- Implement HAC algorithms with fine-tuned distance metrics
- Optimize dendrogram shape and depth via linkage criteria
- Generate visual dendrogram for the mechanics taxonomy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
from typing import Dict, List, Tuple, Optional

from config import (
    HAC_LINKAGE_METHOD, HAC_DISTANCE_METRIC,
    MAX_CLUSTERS, MIN_CLUSTERS, DATA_DIR, ONTOLOGY_OUTPUT_DIR
)
from shared import setup_logger, save_json

logger = setup_logger("Clustering")


# ─── Optimal Cluster Count Detection ─────────────────────────
def find_optimal_clusters(
    embeddings: np.ndarray,
    min_k: int = MIN_CLUSTERS,
    max_k: int = MAX_CLUSTERS
) -> int:
    """
    Determine optimal number of clusters using silhouette analysis.
    
    Evaluates silhouette scores across a range of k values and
    selects the k that maximizes average silhouette coefficient.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import AgglomerativeClustering
    
    best_k = min_k
    best_score = -1
    scores = {}
    
    for k in range(min_k, min(max_k + 1, len(embeddings))):
        try:
            clusterer = AgglomerativeClustering(
                n_clusters=k,
                linkage=HAC_LINKAGE_METHOD,
            )
            labels = clusterer.fit_predict(embeddings)
            
            if len(set(labels)) < 2:
                continue
            
            score = silhouette_score(embeddings, labels, metric="euclidean")
            scores[k] = score
            
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            logger.warning(f"Silhouette analysis failed for k={k}: {e}")
    
    logger.info(f"Optimal clusters: k={best_k} (silhouette={best_score:.4f})")
    return best_k


# ─── Hierarchical Agglomerative Clustering ───────────────────
def perform_hac(
    embeddings: np.ndarray,
    labels: List[str] = None,
    n_clusters: int = None
) -> Dict:
    """
    Perform Hierarchical Agglomerative Clustering on semantic embeddings.
    
    Uses Ward's minimum variance linkage which minimizes the total
    within-cluster variance, producing compact, well-separated clusters.
    
    Args:
        embeddings: Semantic embedding vectors (n_samples x n_dimensions)
        labels: Optional list of labels for each embedding
        n_clusters: Number of clusters (auto-detected if None)
        
    Returns:
        Dict containing cluster assignments, linkage matrix,
        dendrogram data, and cluster metadata
    """
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from scipy.spatial.distance import pdist
    from sklearn.cluster import AgglomerativeClustering
    
    n_samples = embeddings.shape[0]
    logger.info(f"Running HAC on {n_samples} samples, {embeddings.shape[1]} dimensions")
    
    if labels is None:
        labels = [f"item_{i}" for i in range(n_samples)]
    
    # Auto-detect optimal cluster count
    if n_clusters is None:
        n_clusters = find_optimal_clusters(embeddings)
    
    # Compute linkage matrix for dendrogram visualization
    linkage_matrix = linkage(
        embeddings,
        method=HAC_LINKAGE_METHOD,
        metric=HAC_DISTANCE_METRIC
    )
    
    # Perform clustering
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=HAC_LINKAGE_METHOD,
    )
    cluster_assignments = clusterer.fit_predict(embeddings)
    
    # Compute Advanced Clustering Metrics (Silhouette & CPCC)
    from sklearn.metrics import silhouette_score
    from scipy.cluster.hierarchy import cophenet
    
    try:
        sil_score = silhouette_score(embeddings, cluster_assignments, metric=HAC_DISTANCE_METRIC)
    except Exception as e:
        logger.warning(f"Failed to calculate Silhouette Score: {e}")
        sil_score = 0.0
        
    try:
        original_dists = pdist(embeddings, metric=HAC_DISTANCE_METRIC)
        cpcc_score, _ = cophenet(linkage_matrix, original_dists)
    except Exception as e:
        logger.warning(f"Failed to calculate CPCC: {e}")
        cpcc_score = 0.0
        
    logger.info(f"Clustering Metrics - Silhouette: {sil_score:.4f}, CPCC: {cpcc_score:.4f}")
    
    # Build cluster contents
    clusters = {}
    for idx, cluster_id in enumerate(cluster_assignments):
        cid = int(cluster_id)
        if cid not in clusters:
            clusters[cid] = {"members": [], "indices": []}
        clusters[cid]["members"].append(labels[idx])
        clusters[cid]["indices"].append(int(idx))
    
    # Compute cluster centroids and sizes
    for cid, cluster in clusters.items():
        indices = cluster["indices"]
        centroid = embeddings[indices].mean(axis=0)
        cluster["centroid"] = centroid.tolist()
        cluster["size"] = len(indices)
    
    logger.info(f"HAC complete: {n_clusters} clusters, sizes: {[c['size'] for c in clusters.values()]}")
    
    return {
        "n_clusters": n_clusters,
        "cluster_assignments": cluster_assignments.tolist(),
        "clusters": clusters,
        "linkage_matrix": linkage_matrix.tolist(),
        "labels": labels,
        "metrics": {
            "silhouette_score": float(sil_score),
            "cpcc": float(cpcc_score)
        }
    }


# ─── Dendrogram Visualization ────────────────────────────────
def generate_dendrogram(
    linkage_matrix: np.ndarray,
    labels: List[str],
    output_path: str = None,
    title: str = "Board Game Mechanics Taxonomy — Hierarchical Dendrogram",
    max_display_labels: int = 80,
    figsize: Tuple[int, int] = (24, 14)
) -> str:
    """
    Generate a publication-quality dendrogram visualization.
    
    Creates a visual representation of the hierarchical taxonomy
    showing how mechanics cluster from specific implementations
    up to abstract algorithmic categories.
    
    Args:
        linkage_matrix: Scipy linkage matrix from HAC
        labels: Labels for leaf nodes
        output_path: Where to save the image
        title: Plot title
        max_display_labels: Maximum number of labels to show
        figsize: Figure dimensions
        
    Returns:
        Path to the saved dendrogram image
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "mechanics_dendrogram.png")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Calculate dynamic color threshold for distinct cluster coloring
    Z_arr = np.array(linkage_matrix)
    max_d = np.max(Z_arr[:, 2]) if len(Z_arr) > 0 else 1.0
    color_thresh = 0.65 * max_d

    # Setup dendrogram kwargs
    kwargs = {
        "leaf_rotation": 90,
        "leaf_font_size": 7,
        "color_threshold": color_thresh,
        "above_threshold_color": "#7c83ff",
        "ax": ax,
    }
    
    # Truncate if too many labels or if dimensions somehow mismatch
    expected_labels_len = len(linkage_matrix) + 1
    if len(labels) > max_display_labels or len(labels) != expected_labels_len:
        if len(labels) != expected_labels_len:
            logger.warning(f"Dimension mismatch in dendrogram: Z length is {len(linkage_matrix)}, labels length is {len(labels)}. Forcing truncation.")
        kwargs["truncate_mode"] = "lastp"
        kwargs["p"] = min(max_display_labels, expected_labels_len)
        kwargs["show_leaf_counts"] = True
        truncated = True
    else:
        kwargs["labels"] = labels
        truncated = False
    
    # Generate dendrogram
    dendrogram(
        np.array(linkage_matrix),
        **kwargs
    )
    
    # Styling
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Game Mechanics", fontsize=12)
    ax.set_ylabel("Merge Distance (Ward's Linkage)", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    if truncated:
        ax.text(
            0.02, 0.98,
            f"Showing {max_display_labels} of {len(labels)} mechanics",
            transform=ax.transAxes, fontsize=9, color="gray",
            verticalalignment="top"
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    
    logger.info(f"Dendrogram saved to: {output_path}")
    return output_path


# ─── Build Hierarchical Taxonomy Tree ────────────────────────
def build_taxonomy_tree(
    linkage_matrix: np.ndarray,
    labels: List[str],
    cluster_assignments: List[int],
    cluster_labels: Dict[int, str] = None
) -> Dict:
    """
    Convert the flat linkage matrix into a nested hierarchical tree
    structure suitable for ontology export.
    
    The tree represents the full taxonomy with:
    - Root node (top-level concept)
    - Internal nodes (abstract mechanic categories)
    - Leaf nodes (specific mechanics)
    
    Returns:
        Nested dict representing the taxonomy tree
    """
    from scipy.cluster.hierarchy import to_tree
    
    linkage_np = np.array(linkage_matrix)
    root_node = to_tree(linkage_np)
    
    def build_subtree(node, depth=0):
        if node.is_leaf():
            idx = node.id
            label = labels[idx] if idx < len(labels) else f"mechanic_{idx}"
            cluster_id = cluster_assignments[idx] if idx < len(cluster_assignments) else -1
            return {
                "id": int(node.id),
                "label": label,
                "cluster_id": int(cluster_id),
                "type": "leaf",
                "depth": depth,
                "distance": 0.0,
            }
        else:
            left = build_subtree(node.get_left(), depth + 1)
            right = build_subtree(node.get_right(), depth + 1)
            
            # Determine label for internal node
            node_label = f"Category_L{depth}"
            if cluster_labels and depth in cluster_labels:
                node_label = cluster_labels[depth]
            
            return {
                "id": int(node.id),
                "label": node_label,
                "type": "internal",
                "depth": depth,
                "distance": float(node.dist),
                "count": int(node.count),
                "children": [left, right],
            }
    
    tree = build_subtree(root_node)
    logger.info(f"Built taxonomy tree: depth={root_node.count}, leaves={len(labels)}")
    return tree


# ─── Main Clustering Pipeline ────────────────────────────────
def run_clustering_pipeline(
    embeddings: np.ndarray,
    mechanic_labels: List[str],
    n_clusters: int = None
) -> Dict:
    """
    Execute the full clustering pipeline:
    1. Perform HAC
    2. Generate dendrogram visualization
    3. Build taxonomy tree
    4. Save all outputs
    
    Args:
        embeddings: Semantic embedding vectors
        mechanic_labels: Labels for each embedding
        n_clusters: Number of clusters (auto-detected if None)
        
    Returns:
        Complete clustering results dict
    """
    # 1. Perform HAC
    hac_results = perform_hac(embeddings, mechanic_labels, n_clusters)
    
    # 2. Generate dendrogram
    dendro_path = generate_dendrogram(
        linkage_matrix=hac_results["linkage_matrix"],
        labels=mechanic_labels,
    )
    
    # 3. Build taxonomy tree
    taxonomy_tree = build_taxonomy_tree(
        linkage_matrix=hac_results["linkage_matrix"],
        labels=mechanic_labels,
        cluster_assignments=hac_results["cluster_assignments"],
    )
    
    # 4. Save results
    results = {
        "n_clusters": hac_results["n_clusters"],
        "cluster_assignments": hac_results["cluster_assignments"],
        "clusters": {
            str(k): {
                "members": v["members"],
                "size": v["size"]
            } for k, v in hac_results["clusters"].items()
        },
        "dendrogram_path": dendro_path,
        "taxonomy_tree": taxonomy_tree,
    }
    
    output_path = os.path.join(ONTOLOGY_OUTPUT_DIR, "clustering_results.json")
    save_json(results, output_path)
    logger.info(f"Clustering results saved to: {output_path}")
    
    return results
