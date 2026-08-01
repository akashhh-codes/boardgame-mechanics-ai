<div align="center">
  <h1>🧠 AI Driven Ontology Extraction and Semantic Tooling for Board Game Design</h1>
  <p><strong>Ontology-Guided Hybrid Retrieval & Semantic Design Tool</strong></p>
  
  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ontology-extraction-and-semantic-tooling-for-board-game-design.streamlit.app/)
  [![Framework: MDA](https://img.shields.io/badge/Framework-MDA-ff69b4.svg)](#-theoretical-framework)
  [![Validation: OntoClean](https://img.shields.io/badge/Validation-OntoClean-success.svg)](#-theoretical-framework)
</div>

---

An AI-powered research system for extracting hierarchical knowledge structures from board game mechanics. It bridges the gap between raw unstructured rulebook text and formal game studies by using **Natural Language Processing (NLP)**, **Hierarchical Agglomerative Clustering (HAC)**, and **Retrieval-Augmented Generation (RAG)**.

This end-to-end pipeline ingests complex board game rulebooks, extracts mechanical instructions, clusters them into a mathematically validated ontology, and exposes the knowledge via an interactive semantic search dashboard for game designers.

---

## ✨ Core Features

- **📄 Layout-Aware PDF Parsing**: Parses complex, multi-column board game rulebooks using `PyMuPDF` with spatial coordinate mapping.
- **🧹 MDA-Aligned NLP Classification**: Analyzes sentences using `spaCy` dependency parsing to isolate operable rules (Algorithms) from thematic lore (Flavor).
- **🌳 Automated Ontology Generation**: Uses `Sentence-BERT` embeddings and `HAC` (Ward's Linkage) to group mechanics into a hierarchical taxonomy.
- **🤖 LLM-Assisted Labeling**: Leverages Google Gemini Flash to assign formal, MDA-compliant labels to generated clusters.
- **🔍 Hybrid Semantic RAG**: Matches natural language queries to board game mechanics using Cosine Similarity and Cross-Encoder re-ranking.
- **📊 Interactive Dashboard**: A premium Streamlit interface to visualize 2D UMAP projections, interactive dendrograms, and IR evaluation metrics.

---

## 🛠️ Technical Architecture

| Component | Technology |
|-----------|------------|
| **Embeddings** | Sentence-BERT (`all-MiniLM-L6-v2`) — 384-dimensional semantic vectors |
| **Dimensionality Reduction** | UMAP (Uniform Manifold Approximation & Projection) |
| **Clustering** | Hierarchical Agglomerative Clustering (Ward's linkage) |
| **Label Generation** | Google Gemini 2.0 Flash (`google-genai` SDK) |
| **Ontology Validation** | OntoClean (Guarino & Welty metaproperties) |
| **Semantic Search** | Two-Stage RAG (Cosine Similarity + `ms-marco` Cross-Encoder) |
| **PDF Parsing** | PyMuPDF / pdfplumber (fallback) |
| **NLP Classification** | spaCy + regex-based MDA sentence classifier |
| **Ontology Export** | RDFLib (OWL/RDF) + structured JSON |
| **Frontend** | Streamlit (multi-page, dark-themed UI) |

---

## 🚀 Quick Start Guide

### 1. Clone & Install
```bash
git clone https://github.com/akashhh-codes/boardgame-mechanics-ai.git
cd boardgame-mechanics-ai
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure Environment
Create a `.env` file in the project root to enable LLM-assisted cluster labeling:
```
GEMINI_API_KEY=your-gemini-api-key-here
```
> Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey)

### 3. Launch the Dashboard
Start the interactive Streamlit application:
```bash
streamlit run app.py
```

---

## 📊 Dashboard Modules

1. **🏠 Dashboard (Home)**: The central RAG search engine. Describe a game concept in natural language to retrieve matching mechanics, similarity scores, and hierarchical ontology context.
2. **🌳 Ontology Explorer**: View the HAC dendrogram, MDA-labeled clusters, OntoClean validation results, 2D UMAP visualizations, and export the ontology (JSON/OWL).
3. **📈 Evaluation**: Review quantitative benchmark metrics (Hit Rate@K, MRR) and qualitative System Usability Scale (SUS) scores.
4. **📊 System / Debug**: Inspect the data processing pipeline, parser outputs, text-cleaning stages, configuration, and generated artifacts used by the system.

---

## 🔄 Data Pipeline Overview

The system operates on a continuous pipeline transforming unstructured documents into a queried knowledge base:

```text
[ PDF Rulebooks ]  →  PDF Parser (PyMuPDF)  →  Markdown Text
                                                     ↓
                                        NLP Text Cleaner (MDA-aligned)
                                                     ↓
                                        Classified Rule Sentences
                                                     ↓
Mechanics  →  SBERT Embeddings  →  UMAP Reduction
                                                     ↓
                                          HAC Clustering + Dendrogram
                                                     ↓
                                          Gemini LLM Cluster Labeling
                                                     ↓
                                          OntoClean Validation
                                                     ↓
                                          Ontology Export (JSON / OWL)
                                                     ↓
                                          Hybrid RAG Retrieval
                                                     ↓
                                          Evaluation Metrics
                                                     ↓
                                          Interactive Streamlit Dashboard
```

---

## 📚 Theoretical Framework

The project is grounded in established academic methodologies from game studies and knowledge engineering:

- **MDA Framework** *(Hunicke, LeBlanc, Zubek 2004)*: Mechanics → Dynamics → Aesthetics. The pipeline treats board game rules as *Algorithms* (action rulesets) and *Data Representations* (components), forming the foundational Mechanics layer.
- **OntoClean Methodology** *(Guarino & Welty 2004)*: Validates the generated taxonomy using philosophical metaproperties: Identity (+I), Rigidity (+R), Unity (+U), and Dependence (+D) to ensure structural integrity.
- **Ontology Learning Framework** *(Buitelaar et al.)*: Follows the four-stage pipeline: Term Extraction → Synonym Merging → Concept Formation → Taxonomy Extraction.

---

## 📂 Project Structure

```text
AI TOOL/
├── app.py                          # Phase 3 - Interactive Dashboard
├── config.py                       # Central configuration & environment loader
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project metadata
│
├── knowledge_extraction/           # Phase 1 - Knowledge Extraction
│   ├── scraper.py                  # Fallback rulebook scraper
│   ├── bgg_scraper.py              # BGG XML API dataset scraper
│   ├── pdf_parser.py               # Layout-aware PDF → Markdown parser
│   ├── text_cleaner.py             # MDA-aligned NLP sentence classifier
│   ├── clustering.py               # HAC & dendrogram generation
│   ├── label_generator.py          # LLM-assisted cluster labeling
│   ├── ontoclean_validator.py      # OntoClean metaproperty validation
│   └── ontology_exporter.py        # JSON & OWL/RDF ontology export
│
├── semantic_engine/                # Phase 2 - Semantic Retrieval
│   ├── embeddings.py               # SBERT encoding + UMAP reduction
│   ├── rag_retriever.py            # Hybrid RAG search logic
│   └── evaluator.py                # Hit Rate@K, MRR, SUS metrics
│
├── shared/                         # Shared utilities & structured logging
│
└── data/                           # Generated outputs & datasets
```

---

## 📄 License & Attribution

This project was developed for academic and research purposes as an internship project by **Akash Chauhan**. It utilizes data collected from BoardGameGeek.

Distributed under the **MIT License**. See `LICENSE` for more information.
