import streamlit as st
import os
from collections import Counter
from PIL import Image

# Import custom modular framework components
from src.preprocessing import load_and_preprocess_data
from src.features import get_embedding_model, generate_text_embeddings
from src.train import train_kmeans_paradigm
from src.evaluate import calculate_top_matches
from src.utils import get_cluster_profiles

st.set_page_config(page_title="AI Board Game Mechanics Generator", layout="wide")

st.title("🧩 AI Tool for Board Game Mechanics Idea Generation")
st.write("Welcome! This modular, enterprise data science tool uses NLP and machine learning to anchor game design rules without hallucinations.")

# --- CORE DATA SCIENCE PIPELINE (Cached) ---
@st.cache_resource
def run_cached_pipeline():
    # 1. Preprocessing
    df = load_and_preprocess_data("data/top_1000_board_games.csv")
    # 2. Vectorization Features
    model = get_embedding_model()
    embeddings = generate_text_embeddings(df, model)
    # 3. Clustering Engine Training
    df['cluster_id'] = train_kmeans_paradigm(embeddings)
    return df, model, embeddings

df, model, embeddings = run_cached_pipeline()

# --- SIDEBAR: DENDROGRAM ---
st.sidebar.header("📊 Mapped Mechanics Taxonomy")
dendrogram_path = "data/mechanics_dendrogram.png"
if os.path.exists(dendrogram_path):
    img = Image.open(dendrogram_path)
    st.sidebar.image(img, caption="Discovered Mechanics Taxonomy", use_container_width=True)

# --- USER CONCEPT PROCESSING ---
st.header("💡 Step 1: Input Your Game Concept Idea")
user_concept = st.text_area("Describe your game rules, theme, or player interactions:")

if user_concept:
    st.header("🤖 Step 2: NLP Similarity Analysis & Rule Structuring")
    
    # Run evaluation script modules
    user_vector = model.encode([user_concept])
    top_k_indices, similarities = calculate_top_matches(user_vector, embeddings)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🔍 Grounded Real-World Benchmarks")
        retrieved_mechanics = []
        for idx in top_k_indices:
            title = df['title'].iloc[idx]
            score = similarities[idx]
            game_mechs = df['mechanics_list'].iloc[idx]
            retrieved_mechanics.extend(game_mechs)
            st.markdown(f"**{title}** *(Similarity: {score:.2%})*")
            st.caption(f"Mechanics: {', '.join(game_mechs[:4])}")
            
    with col2:
        st.subheader("🛠️ Recommended Core Mechanics")
        mech_counts = Counter(retrieved_mechanics)
        for mech, count in mech_counts.most_common(4):
            st.info(f"👉 **{mech}** (Present in {count} baseline titles)")
            
    # Step 3: Broad Rule Paradigm Matching
    st.markdown("---")
    st.header("🗂️ Step 3: Taxonomy & Broad Rule Set Alignment")
    assigned_clusters = [df['cluster_id'].iloc[idx] for idx in top_k_indices]
    dominant_cluster = Counter(assigned_clusters).most_common(1)[0][0]
    
    profiles = get_cluster_profiles()
    profile = profiles.get(dominant_cluster, {"name": "General Strategy", "desc": "Balanced layout."})
    
    st.info(f"🧬 **Assigned Rule Family Paradigm:** **Cluster #{dominant_cluster} — {profile['name']}**")
    st.write(f"**Structural Rule Blueprint:** {profile['desc']}")
    st.success("🎉 Processing complete! Application is running on highly decoupled infrastructure parameters.")
