# AI Board Game Ontology & Semantic Engine - Interview Study Guide

This document is a concise study guide designed for quick revision before technical interviews. It covers the core workflows, modules, and algorithms of the AI Board Game Ontology project.

---

## 1. Project Overview & Workflow

**Goal:** An AI-powered tool that reads board game rulebooks, extracts the mechanics, organizes them into a hierarchical ontology (family tree), and provides a chatbot for game designers to brainstorm ideas using Semantic Search (RAG).

**Chronological Pipeline:**
1. **Data Collection:** Web scrapers download rulebook PDFs and BGG metadata.
2. **Parsing:** PDFs are converted into clean Markdown text.
3. **NLP Extraction:** Sentences are classified into Rules/Components using NLP.
4. **Ontology Generation:** Sentences are converted to mathematical vectors, dimensionally reduced, and clustered into groups. An LLM gives each group a human-readable name, and OntoClean logic validates them.
5. **Semantic Search (RAG):** A Two-Stage Retriever (Bi-Encoder + Cross-Encoder) allows users to query the ontology using natural language.
6. **Evaluation & UI:** The system is rigorously evaluated mathematically (hitting 0.84 MRR) and presented via a Streamlit web dashboard.

---

## 2. Important Files & Modules

### Module 1: Knowledge Extraction
*   **`scraper.py`**: A fault-tolerant web scraping bot. It downloads PDF rulebooks from public URLs. It features an exponential backoff retry loop for network drops, rate limiting to avoid DDoS bans, and maintains a JSON manifest to prevent re-downloading files.
*   **`bgg_scraper.py`**: A massive XML API bot designed to pull metadata (Mechanics, Categories, Descriptions) directly from the BoardGameGeek API. It uses batching (400 IDs at once) to maximize throughput and saves incrementally to a CSV.
*   **`pdf_parser.py`**: Solves the complex problem of turning visual PDFs (with multi-columns and sidebars) into readable text. It uses a robust fallback architecture: PyMuPDF as the primary layout-aware parser, and pdfplumber as the fallback.
*   **`text_cleaner.py`**: Analyzes the parsed text using the spaCy NLP library. It strips out conversational text and uses Part-of-Speech tagging and regex to classify sentences into structural categories (Rules, Components, Setup) based on the MDA framework.
*   **`clustering.py`**: Groups the extracted rules together. It reduces the vector dimensions using UMAP, and then applies Hierarchical Agglomerative Clustering (HAC) using Ward's linkage to build a dendrogram (family tree) of game mechanics.
*   **`label_generator.py`**: Automatically names the mathematical clusters generated in the previous step. It sends a sample of rules from the cluster to the Google Gemini 2.0 Flash LLM via the `google-genai` SDK to generate a human-readable title.
*   **`ontoclean_validator.py`**: Validates the logical consistency of the generated ontology using the philosophical OntoClean methodology. It checks structural properties (Identity, Rigidity, Unity, Dependence) to ensure the hierarchy makes logical sense.
*   **`ontology_exporter.py`**: Serializes the final ontology so other systems can read it. It exports standard JSON for web apps, and uses RDFLib to export OWL/RDF formats for standard semantic web applications.

### Module 2: Systems & Semantics
*   **`embeddings.py`**: Converts raw text into mathematical representations. It uses Sentence-BERT (`all-MiniLM-L6-v2`) to turn board game mechanic descriptions into 384-dimensional dense vectors, allowing the system to understand semantic similarity.
*   **`rag_retriever.py`**: The brain of the chatbot. It uses a **Two-Stage RAG Pipeline**. First, KeyBERT extracts keywords. Next, a Bi-Encoder filters the database down to the Top 25. Finally, an `ms-marco` Cross-Encoder deeply re-ranks the top results to ensure extreme accuracy.
*   **`evaluator.py`**: A rigorous mathematical testing script. It runs ground-truth queries against the RAG Retriever to compute Information Retrieval (IR) metrics like Precision@K, Recall@K, MRR, and MAP to prove the system works quantitatively.

### Infrastructure & UI
*   **`app.py`**: The frontend of the project built using Streamlit. It contains a 5-page dark-themed UI featuring the home overview, data pipeline demos, interactive ontology explorers, the RAG chatbot, and live evaluation metric dashboards.

---

## 3. Important Algorithms & Technologies

### Hierarchical Agglomerative Clustering (HAC)
*   **What it is:** A "bottom-up" clustering algorithm. It starts with every data point in its own cluster, and repeatedly merges the two closest clusters together until only one giant cluster remains, creating a tree-like structure (dendrogram).
*   **Why we used it:** Unlike K-Means (which requires you to guess the number of clusters upfront and creates flat lists), HAC naturally creates parent-child relationships, which is mathematically identical to a formal ontology.
*   **Where it is used:** `clustering.py`, to group board game mechanics into a family tree.

### UMAP (Uniform Manifold Approximation and Projection)
*   **What it is:** A dimensionality reduction algorithm (similar to PCA or t-SNE) that preserves both the local and global structure of high-dimensional data.
*   **Why we used it:** Text embeddings are 384 dimensions, which suffers from the "Curse of Dimensionality" (distance metrics break down). UMAP compresses this to 10 dimensions, making the clustering algorithm much faster and more accurate.
*   **Where it is used:** `embeddings.py` and `clustering.py`.

### Sentence-BERT (Bi-Encoder)
*   **What it is:** A modification of the BERT network that uses siamese networks to derive semantically meaningful sentence embeddings that can be compared using cosine similarity.
*   **Why we used it:** Standard BERT requires passing both sentences into the network at once (too slow for searching a database). SBERT allows us to pre-calculate all database vectors offline, making search instantaneous.
*   **Where it is used:** `embeddings.py` and Stage 1 of `rag_retriever.py`.

### Cross-Encoder (`ms-marco`)
*   **What it is:** A heavy transformer model that takes two sentences simultaneously and passes them through attention layers to output a highly accurate similarity score.
*   **Why we used it:** While Bi-Encoders are fast, they miss subtle linguistic nuances. A Cross-Encoder is computationally expensive but provides vastly superior accuracy for re-ranking a small list of items.
*   **Where it is used:** Stage 2 of `rag_retriever.py` to re-rank the Top 25 candidates to guarantee the best match is #1.

### KeyBERT
*   **What it is:** A keyword extraction technique that leverages BERT embeddings to find the sub-phrases in a document that are most similar to the document itself.
*   **Why we used it:** To strip out conversational filler from user queries (e.g., "I want a game where...") so the mathematical vectors are highly concentrated on actual design logic.
*   **Where it is used:** The very beginning of `rag_retriever.py`.

### OntoClean & MDA Frameworks
*   **What they are:** MDA (Mechanics, Dynamics, Aesthetics) is a game design framework. OntoClean is a philosophical methodology for validating taxonomies based on metaproperties (Rigidity, Identity).
*   **Why we used them:** They provide the theoretical foundation. We use MDA to decide *what* to extract (Mechanics), and OntoClean to verify the mathematical clusters actually make logical sense to humans.
*   **Where they are used:** `text_cleaner.py` (MDA) and `ontoclean_validator.py` (OntoClean).

---

## 4. Interview Revision Notes

### 🎯 Key Performance Metrics
If asked about performance, quote these exact numbers from `evaluator.py`:
*   **MRR (Mean Reciprocal Rank): 0.84** (This is elite. It means the absolute correct answer is at Rank #1 84% of the time).
*   **Precision@3: 0.40** (The density of correct matches at the top).
*   **Recall@5: 0.33** (Solid coverage for a zero-shot model without explicit fine-tuning).

### ⚙️ DevOps & CI/CD Setup
*   **Containerization:** The app uses Docker (`Dockerfile`) so it can be spun up identically on any OS.
*   **CI/CD:** GitHub Actions (`ci.yml`) automatically lints the code and runs the `pytest` suite every time code is pushed, preventing broken code from going to production.

### ❓ Possible Interview Questions & Answers

**Q: Why did you use a Two-Stage RAG Pipeline instead of just standard Cosine Similarity?**
**A:** Standard Cosine Similarity (using a Bi-Encoder) is extremely fast because vectors can be pre-computed, but it struggles with deep semantic nuance. A Cross-Encoder provides massive accuracy gains by analyzing the query and candidate together, but it is too slow to run on the whole database. The two-stage pipeline gives us the best of both worlds: Bi-Encoder for fast filtering (Top 25), and Cross-Encoder for precision re-ranking. This architecture pushed our MRR up to 0.84.

**Q: How did you handle errors when scraping PDF rulebooks?**
**A:** PDFs are notoriously unstructured. I built a robust fallback system. First, we use PyMuPDF which analyzes text block positions to preserve multi-column layouts and reading order based on spatial coordinates. If PyMuPDF fails or outputs garbage, we fallback to pdfplumber for raw text extraction. This ensures high data reliability without requiring expensive GPU resources.

**Q: Why use Hierarchical Clustering (HAC) instead of K-Means?**
**A:** K-Means forces you to guess the number of clusters (K) beforehand, and it outputs flat lists. We are building an *ontology* (a family tree of concepts). HAC is a bottom-up approach that naturally creates a tree structure (dendrogram), which perfectly maps to an ontological hierarchy.

**Q: How do you prevent your scraper from getting banned by BoardGameGeek?**
**A:** The `bgg_scraper.py` respects BGG's infrastructure. It uses batched requests (pulling 400 IDs at a time) to minimize HTTP overhead, enforces a strict 2-second sleep between requests to respect rate limits, and uses exponential backoff if the server drops connection. It also saves data incrementally to CSV to avoid data loss.

**Q: How did you generate the labels for the mathematical clusters?**
**A:** I integrated the Google Gemini 2.0 Flash LLM via the `google-genai` SDK. I prompt the LLM with the raw text rules belonging to a cluster and ask it to synthesize a definitive mechanic title based on the MDA framework. If the API is rate-limited, the system safely falls back to a deterministic TF-IDF keyword extractor.
