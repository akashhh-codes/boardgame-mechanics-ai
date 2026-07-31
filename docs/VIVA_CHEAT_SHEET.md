# 🚀 ULTIMATE VIVA CHEAT SHEET (1-Page)

Read this carefully. If you know these 7 points, you know 80% of the project and will pass the Viva.

### 1. What was the problem you were trying to solve?   
Board game designers lack a search engine to brainstorm ideas because game rules are locked inside unstructured PDF rulebooks. Standard keyword search (TF-IDF) doesn't understand the *meaning* of a rule (e.g., "trading resources" = "Set Collection").

### 2. How did you parse the PDFs? (Spatial Coordinate Extraction)
Normal text extractors fail on rulebooks because they read straight across the page, mixing columns and sidebars together. I used **PyMuPDF**, which calculates the exact spatial X/Y bounding boxes of every text block. This allows the system to read the left column completely before reading the right column, perfectly preserving the semantic layout without needing an expensive GPU-based Vision AI.

### 3. How did you filter the text? (NLP vs Dumb Chunking)
Standard RAG pipelines blindly slice text into 500-token chunks, which cuts game rules in half and destroys context. Instead, I used **spaCy NLP** to extract complete grammatical sentences. I applied the **MDA framework** to write regex rules that keep *Mechanics* and delete *Flavor*. 
**Critical Fix:** I discovered a "False Positive" edge case where thematic keywords (e.g., "Dragon") were causing the system to delete important mechanical sentences. I updated the heuristic scoring algorithm so that if a sentence contains both Rule and Flavor semantic flags, the Rule classification forcefully overrides to ensure zero loss of mechanical context.

### 4. How did you build the Ontology (Family Tree)? (UMAP + HAC)
1.  **Sentence-BERT:** I converted the text rules into 384-dimensional mathematical vectors.
2.  **UMAP:** 384 dimensions is too high (distance math breaks down—the *Curse of Dimensionality*). I used UMAP to reduce it to 10 dimensions.
3.  **HAC (Hierarchical Agglomerative Clustering):** I used HAC instead of K-Means. K-Means creates flat lists. HAC is a "bottom-up" algorithm that creates a tree (dendrogram), which perfectly matches the parent-child structure of an Ontology.

### 5. How did you name the Clusters? (Generative AI & XAI)
Clustering algorithms just output mathematical blobs. 
1. **Explainable AI (XAI):** First, I calculate the **Medoid** (the single sentence or mechanic closest to the exact mathematical center of the cluster). This acts as the "Representative Mechanic" so users understand *why* a cluster formed.
2. **LLM Generation:** I pass this Medoid and the cluster members to the **Google Gemini 2.0 Flash LLM** to generate formal taxonomic labels.
3. **Semantic vs Canonical Flaw:** High MRR proves our vectors map perfectly semantically, but it does *not* prove Gemini picked the exact industry-standard name (e.g., Gemini might invent "Board Resource Allocation" instead of "Worker Placement"). To fix this in the future, we would force the LLM to select from a Golden Taxonomy list.

### 6. How does the Semantic Search work? (The Two-Stage Pipeline)
If a user searches for *"a game with military combat on a map"*:
*   **Stage 1 (Bi-Encoder):** Uses Cosine Similarity to compare the query vector to the database. It is extremely fast (O(1)) but misses subtle linguistic nuance. It filters the database to the **Top 25**.
*   **Stage 2 (Cross-Encoder):** Uses an `ms-marco` transformer. It analyzes the query and the candidate *together* using cross-attention. It is mathematically perfect but incredibly slow (O(N^2)). By only running it on the Top 25, we get maximum speed and maximum accuracy.

### 7. How did you prove it works? (The Metrics)
I didn't just guess. I used four critical Information Retrieval (IR) metrics to prove the Two-Stage Pipeline works perfectly:
*   **MRR @ 5 = 0.84**: The absolute perfect answer is pushed to Rank #1 (by the Cross-Encoder) 84% of the time.
*   **Hit Rate @ 5 = 1.00 (100%)**: The fast Bi-Encoder filter never misses; it captures the right answer in its top 5 candidates every single time.
*   **Taxonomical Precision (CPCC / Silhouette)**: Proves the underlying clustering tree is mathematically valid before the LLM even looks at it.

### 8. What is the weakest point of the project? (Future Scope)
Honestly, **Regex**. Regex-based sentence classification in `text_cleaner.py` is fragile because human language is infinitely variable. In Version 2.0, I would completely deprecate Regex and replace it with a **fine-tuned Transformer model (like RoBERTa)** for binary sequence classification (Rule vs. Flavor). This would provide much higher robustness and eliminate heuristic edge cases.
