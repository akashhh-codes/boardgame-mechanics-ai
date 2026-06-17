import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_top_matches(user_vector, database_embeddings, top_k=5):
    """Calculates cosine similarity arrays and maps the top matching indices."""
    similarities = cosine_similarity(user_vector, database_embeddings)[0]
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    return top_k_indices, similarities
