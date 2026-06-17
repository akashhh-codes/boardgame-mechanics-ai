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
