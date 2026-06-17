# 🧩 AI Tool for Board Game Mechanics Idea Generation

A production-grade, full-stack data science application that maps abstract game concepts to grounded, real-world mechanics and taxonomies using advanced Natural Language Processing (NLP) and unsupervised machine learning. Built entirely with a decoupled modular architecture and deployed seamlessly to the cloud.

---

## 🛠️ Project Architecture & Directory Layout

Unlike standard monolithic notebooks, this system is engineered following professional software development patterns. The codebase is entirely decoupled into separate functional modules:

```text
project/
│
├── data/
│   ├── top_1000_board_games.csv     # Curated BGG truth dataset
│   └── mechanics_dendrogram.png     # Agglomerative statistical hierarchy
│
├── src/
│   ├── preprocessing.py             # Data ingestion & structural string cleaning
│   ├── features.py                  # Sentence-Transformer dense vector embedding pipeline
│   ├── train.py                     # Unsupervised K-Means model training
│   ├── evaluate.py                  # Cosine Similarity array calculation
│   └── utils.py                     # Static structural data profiles & configurations
│
├── app.py                           # Decoupled Streamlit front-end configuration
├── requirements.txt                 # Target environment dependencies
└── README.md                        # Documentation
```
## 🚀 Core Methodology & AI Guardrails

Large Language Models (LLMs) frequently hallucinate fictional game mechanics when asked to generate rules out of thin air. This tool implements a **Closed-Loop Retrieval-Augmented Strategy** to ensure absolute data validity:

1. **Semantic Feature Mapping:** Raw human-written text descriptions are transformed into a dense, 384-dimensional vector coordinate space using a state-of-the-art Transformer encoder (`all-MiniLM-L6-v2`).
2. **Mathematical Grounding:** Input concepts are verified against the Top 1,000 board games in the database using **Cosine Similarity** arrays. This anchors user recommendations strictly onto verified real-world evidence.
3. **Statistical Taxonomy (Ontology):** Using Agglomerative Hierarchical Clustering with Ward's linkage, the system maps out an objective taxonomy of game mechanics, visualized directly on the dashboard via an analytical **Dendrogram**.
4. **Macro Rule Paradigms:** An unsupervised **K-Means Clustering** engine groups the database into 8 distinct strategic rule profiles (e.g., *Heavy Macro-Economic Logistics*, *Narrative Campaign Systems*), allowing input prompts to instantly align with a broader structural rule family blueprint.

---

## 📦 Local Installation & Setup

To run the application or the background modules locally on your workstation:

### 1. Clone the Repository
```bash
git clone [https://github.com/akashhh-codes/boardgame-mechanics-ai.git](https://github.com/akashhh-codes/boardgame-mechanics-ai.git)
cd boardgame-mechanics-ai
