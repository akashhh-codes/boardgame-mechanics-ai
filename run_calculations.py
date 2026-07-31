import sys
import os
import ast
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from config import DATASET_CSV
from semantic_engine.embeddings import EmbeddingEngine
from semantic_engine.rag_retriever import RAGRetriever
from semantic_engine.evaluator import run_evaluation, SAMPLE_TEST_QUERIES
from knowledge_extraction.clustering import run_clustering_pipeline
from semantic_engine.embeddings import apply_umap

def main():
    print("1. Loading dataset...")
    df = pd.read_csv(DATASET_CSV)
    
    def parse_list(s):
        if pd.isna(s) or not isinstance(s, str): return []
        try:
            return ast.literal_eval(s)
        except:
            return [m.strip().strip("'\"") for m in s.strip("[]").split(",") if m.strip()]
            
    df["mechanics_list"] = df["mechanics"].apply(parse_list)
    
    all_mechs = []
    for mechs in df["mechanics_list"]:
        all_mechs.extend(mechs)
    all_mechanics = sorted(set(all_mechs))
    
    print(f"   Found {len(all_mechanics)} unique mechanics.")
    
    print("2. Initializing Embedding Engine...")
    engine = EmbeddingEngine()
    print("   Computing mechanics embeddings...")
    mechanic_embeddings = engine.encode(all_mechanics)
    
    print("3. Running Ontology Clustering Pipeline (Testing Silhouette & CPCC)...")
    reduced, _ = apply_umap(mechanic_embeddings, n_components=min(10, len(all_mechanics) - 2))
    n_clusters = min(15, max(3, len(all_mechanics) // 8))
    # This will trigger the logger to print Silhouette and CPCC
    cluster_results = run_clustering_pipeline(reduced, all_mechanics, n_clusters=n_clusters)
    
    print("\n4. Initializing RAG Retriever...")
    retriever = RAGRetriever(
        embedding_engine=engine,
        mechanic_labels=all_mechanics,
        mechanic_embeddings=mechanic_embeddings,
        labeled_clusters={},
        game_data=df
    )
    
    print("\n5. Running Search Evaluation Pipeline...")
    try:
        results = run_evaluation(retriever, SAMPLE_TEST_QUERIES)
        print("\n--- Search Evaluation Metrics ---")
        for key, value in results["aggregate_metrics"].items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error running evaluator: {e}")

if __name__ == "__main__":
    main()
