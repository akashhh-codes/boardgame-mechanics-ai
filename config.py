"""
Shared Configuration for AI Board Game Ontology Project
========================================================
Central configuration constants used across all modules.
"""

import os

# Load .env file if present (keeps API keys out of source code)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass  # python-dotenv not installed; user must set env vars manually

# ─── Project Root ───────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── Data Paths ─────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_PDFS_DIR = os.path.join(DATA_DIR, "raw_pdfs")
PARSED_MARKDOWN_DIR = os.path.join(DATA_DIR, "parsed_markdown")
CLEANED_RULES_DIR = os.path.join(DATA_DIR, "cleaned_rules")
ONTOLOGY_OUTPUT_DIR = os.path.join(DATA_DIR, "ontology_output")
DATASET_CSV = os.path.join(DATA_DIR, "top_1000_board_games.csv")

# ─── Model Configuration ───────────────────────────────────────
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"       # Sentence-BERT model
UMAP_N_COMPONENTS = 10                       # UMAP target dimensions
UMAP_N_NEIGHBORS = 15                        # UMAP neighbors parameter
UMAP_MIN_DIST = 0.1                          # UMAP min distance

# ─── Clustering Configuration ──────────────────────────────────
HAC_LINKAGE_METHOD = "ward"                  # Hierarchical clustering linkage
HAC_DISTANCE_METRIC = "euclidean"            # Distance metric (ward requires euclidean)
HAC_OPTIMAL_THRESHOLD = None                 # Auto-detect if None
MAX_CLUSTERS = 25                            # Maximum number of leaf clusters
MIN_CLUSTERS = 5                             # Minimum number of leaf clusters

# ─── LLM Configuration ─────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 512

# ─── RAG Configuration ─────────────────────────────────────────
TOP_K_RESULTS = 10                           # Number of top results to retrieve
KEYBERT_TOP_N = 5                            # Number of keywords to extract
SIMILARITY_THRESHOLD = 0.15                  # Minimum cosine similarity threshold

# ─── OntoClean Metaproperties ──────────────────────────────────
ONTOCLEAN_PROPERTIES = ["identity", "rigidity", "unity", "dependence"]

# ─── Streamlit UI Configuration ────────────────────────────────
APP_TITLE = "AI Board Game Ontology & Semantic Design Tool"
APP_ICON = "🎲"
APP_LAYOUT = "wide"

# ─── Ensure directories exist ──────────────────────────────────
for d in [DATA_DIR, RAW_PDFS_DIR, PARSED_MARKDOWN_DIR, CLEANED_RULES_DIR, ONTOLOGY_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)
