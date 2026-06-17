
import streamlit as st
import pandas as pd
import numpy as np
import os
import ast
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image  # Added to safely process image objects

# Set up page layout
st.set_page_config(page_title="AI Board Game Mechanics Generator", layout="wide")

st.title("🧩 AI Tool for Board Game Mechanics Idea Generation")
st.write("Welcome! This data-science-driven tool uses NLP and unsupervised machine learning to map your design ideas to verified board game taxonomies without AI hallucinations.")

# --- LOAD DATA & MODELS (Cached to run quickly) ---
@st.cache_resource
def load_resources():
    # Load dataset
    df = pd.read_csv("top_1000_board_games.csv")
    # Re-parse lists
    df['mechanics_list'] = df['mechanics'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    df['categories_list'] = df['categories'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    
    # Load encoder model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate/cache embeddings matrix locally for speed
    embeddings = model.encode(df['description'].fillna("").tolist(), show_progress_bar=False)
    return df, model, embeddings

df, model, embeddings = load_resources()

# --- SIDEBAR: ACCESS THE DENDROGRAM ---
st.sidebar.header("📊 Mapped Mechanics Taxonomy")
st.sidebar.write("This tree structure shows how game mechanics statistically group together based on 1,000 top BGG games.")

dendrogram_path = "mechanics_dendrogram.png"
if os.path.exists(dendrogram_path):
    try:
        # FIX: Open the image via PIL first instead of passing a raw path string
        img = Image.open(dendrogram_path)
        st.sidebar.image(img, caption="Discovered Mechanics Hierarchy (Dendrogram)", use_container_width=True)
    except Exception as e:
        st.sidebar.error(f"Error rendering image array: {e}")
else:
    st.sidebar.warning("Dendrogram image file not found in project directory.")

# --- MAIN PANEL: USER INPUT ---
st.header("💡 Step 1: Input Your Game Concept Idea")
user_concept = st.text_area(
    "Describe your game rules, theme, or how players interact:",
    placeholder="Example: A sci-fi survival game where players work together to manage limited resource grids on a shifting modular space station board..."
)

if user_concept:
    st.header("🤖 Step 2: NLP Similarity Analysis & Rule Structuring")
    
    # 1. Vectorize input & process similarity
    user_vector = model.encode([user_concept])
    similarities = cosine_similarity(user_vector, embeddings)[0]
    top_k_indices = np.argsort(similarities)[::-1][:5]
    
    # 2. Display Grounded Real-World Evidence
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔍 Grounded Real-World Benchmarks")
        st.write("Your concept is mathematically closest to these verified titles:")
        
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
        st.write("Incorporate these high-probability mechanics to build out your rules system:")
        
        mech_counts = Counter(retrieved_mechanics)
        for mech, count in mech_counts.most_common(4):
            st.info(f"👉 **{mech}** (Present in {count} related baseline games)")

    st.success("🎉 Process Complete! Your ideas are fully anchored to the baseline ontology to prevent design hallucinations.")

    # 3. Dynamic Assignment to the Professor's "Broad Rule Sets"
    st.markdown("---")
    st.header("🗂️ Step 3: Taxonomy & Broad Rule Set Alignment")
    
    # Calculate the average cluster assignment of the top matches to find the dominant paradigm
    assigned_clusters = [df['cluster_id'].iloc[idx] for idx in top_k_indices]
    dominant_cluster = Counter(assigned_clusters).most_common(1)[0][0]
    
    # Structure text profiles for the clusters we discovered in K-Means
    cluster_profiles = {
        0: {"name": "Card-Driven Tactical Hand Management", "desc": "Rules focus heavy emphasis on playing hand combinations, scaling player powers over time, and resolving actions using dice metrics."},
        1: {"name": "Heavy Political & Territory Strategy", "desc": "Rules dictate map-control boundaries, military or economic area majority, and long-term civilization resource engine management."},
        2: {"name": "Engine Building & Tactical Card Play", "desc": "Rules emphasize card synergy, private board expansion, set-collection optimization, and high-efficiency engine loops."},
        3: {"name": "Drafting & Closed-Economy Optimization", "desc": "Rules center on tight drafting pools, direct resource conversion, building placement, and contract fulfillment constraints."},
        4: {"name": "Heavy Macro-Economic Logistics", "desc": "Complex financial management simulation rules. Features network construction, supply-demand curves, and worker optimization layouts."},
        5: {"name": "Narrative Campaign & Cooperative Survival", "desc": "Rules enforce complete player cooperation against automated game AI, scenario-driven legacy maps, and custom player asymmetric roles."},
        6: {"name": "Spatial Optimization & Worker Placement", "desc": "Rules dictate alternate engine building using spatial tile placement layouts, shared worker pools, and drafting resource spaces."},
        7: {"name": "Grand Asymmetric Conflicts & Dice Resolution", "desc": "High-stakes grand strategy rules featuring asymmetrical starting conditions, hidden deployment, and tactical dice combat grids."}
    }
    
    profile = cluster_profiles.get(dominant_cluster, {"name": "General Strategy Paradigm", "desc": "Standard balance of tactical choices and mechanics."})
    
    st.info(f"🧬 **Assigned Rule Family Paradigm:** **Cluster #{dominant_cluster} — {profile['name']}**")
    st.write(f"**Structural Rule Blueprint:** {profile['desc']}")
