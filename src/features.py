from sentence_transformers import SentenceTransformer

def get_embedding_model(model_name='all-MiniLM-L6-v2'):
    """Initializes and returns the text encoder model."""
    return SentenceTransformer(model_name)

def generate_text_embeddings(df, model):
    """Encodes the clean description corpus into dense text vectors."""
    corpus = df['description'].fillna("").tolist()
    embeddings = model.encode(corpus, show_progress_bar=False)
    return embeddings
