from sklearn.cluster import KMeans

def train_kmeans_paradigm(embeddings, n_clusters=8, random_state=42):
    """Trains an unsupervised K-Means algorithm over the NLP vector space."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)
    return cluster_ids
