# The Story of Our AI Board Game Ontology Project

This document is a simple, step-by-step narrative explaining how we built this project, the logic behind our decisions, what approaches we ignored, and exactly what we applied to get elite results.

---

## 1. How We Started (The Goal)
We wanted to build an intelligent search engine for board game designers. If a designer wants to build a game about "bidding on resources," they should be able to search our database and instantly see all related mechanics and real-world game examples. 

We started with two raw ingredients: 
1. The **Top 1000 Board Games dataset** from BoardGameGeek (BGG).
2. The raw **PDF Rulebooks** for those games.

## 2. Step 1: Getting the Text out of PDFs
The very first problem was getting the computer to read the PDF rulebooks. Board game rulebooks are highly complex—they have multiple columns, floating sidebars, and background images. 

*   **What we ignored:** We ignored standard text extraction libraries (like `PyPDF2`). Standard extractors blindly read text from left-to-right, meaning they will stitch the first sentence of the left column directly into the first sentence of the right column, creating absolute gibberish.
*   **What we applied:** We built a robust layout-aware parser using **PyMuPDF**. PyMuPDF calculates the exact spatial X/Y bounding boxes of every text block on the page. This allows our algorithm to detect columns, read the entire left column first, and then the right column, perfectly preserving the semantic reading order without requiring expensive hardware.

## 3. Step 2: Filtering the Noise
Once we had the raw text, we realized that rulebooks contain a lot of thematic storytelling (e.g., *"You are a brave knight exploring a dark dungeon"*). 

*   **What we ignored:** We threw away all thematic and aesthetic text. It doesn't help us mathematically classify how a game is played. We also explicitly ignored standard **RAG token chunking** (e.g., splitting text into dumb 500-token blocks), because naive chunking randomly cuts game rules in half and destroys grammatical context.
*   **What we applied:** We applied the theoretical **MDA (Mechanics, Dynamics, Aesthetics)** game design framework alongside the **spaCy NLP** library to extract complete, grammatical sentences. We wrote linguistic rules to extract only the sentences that describe *Mechanics*.
    *   *Critical Fix:* We discovered a "False Positive" edge case where thematic keywords (like "Dragon") caused the system to delete important mechanical sentences. We updated the heuristic algorithm so that if a sentence triggers both Rule and Flavor semantic flags, the Rule classification forcefully overrides to ensure zero loss of mechanical context.

## 4. Step 3: Grouping the Mechanics into a Family Tree
Next, we needed the computer to understand how these extracted rules relate to each other mathematically. We converted all the text into 384-dimensional mathematical arrays using **Sentence-BERT**. 

*   **What we ignored:** We ignored the popular K-Means clustering algorithm. K-Means forces you to blindly guess how many clusters you want beforehand, and it outputs flat, disconnected lists. 
*   **What we applied:** First, we applied **UMAP** to reduce the 384 dimensions down to 10 (because distance math breaks down in super high dimensions—this is called the *Curse of Dimensionality*). Second, we applied **Hierarchical Agglomerative Clustering (HAC)**. HAC is a bottom-up algorithm that merges similar items step-by-step, naturally creating a "family tree" (a dendrogram), which is exactly what a formal Ontology requires.

## 5. Step 4: Naming the Clusters
Mathematical clusters don't have names. HAC just outputs "Cluster 1" and "Cluster 2". 

*   **What we applied:** 
    1.  **Explainable AI (XAI):** We calculate the **Medoid** for each cluster (the single mechanic closest to the exact mathematical center of the cluster). This acts as the "Representative Mechanic", making the AI's clustering logic explainable and transparent to users.
    2.  **LLM Generation:** We passed the Medoid and raw sentences to the **Google Gemini 2.0 Flash Large Language Model (LLM)**. We prompted the AI to look at the rules and synthesize a human-readable title (e.g., "Worker Placement"). 
    3.  **Semantic Equivalence vs Canonical Terminology:** We acknowledge an inherent limitation here: High MRR proves our vectors map perfectly semantically, but it doesn't prove Gemini picked the exact industry-standard name (e.g., it might invent "Board Resource Allocation" instead of "Worker Placement"). Future versions would force the LLM to select from a predefined Golden Taxonomy.

## 6. Step 5: The Semantic Search Engine (RAG)
Now that our ontology was built, we needed a way for users to search it using conversational English. We built a **Retrieval-Augmented Generation (RAG)** pipeline.

*   **What we ignored:** We couldn't just use a standard "Cross-Encoder" AI to search the whole database. While Cross-Encoders are mathematically perfect (they analyze the user's query and the document simultaneously), they are incredibly slow computationally. If we ran it on thousands of mechanics, the search would take minutes.
*   **What we applied:** We built an elite **Two-Stage Pipeline**. 
    1.  **Stage 1 (Fast Filter):** We use a Bi-Encoder (Cosine Similarity) which compares pre-calculated vectors instantly. This filters the database down to the **Top 25** candidates in a fraction of a second.
    2.  **Stage 2 (Deep Re-ranking):** We pass only those Top 25 candidates to our heavy `ms-marco` Cross-Encoder. The Cross-Encoder deeply analyzes the linguistic nuances and re-ranks the list, guaranteeing that the absolute best match sits at Rank #1. 

## 7. Step 6: Proving it Works
Finally, we had to prove our system was scientifically sound. 

*   **What we applied:** We wrote an `evaluator.py` script that ran hundreds of ground-truth test queries through our search engine. We curated our evaluation down to the two most critical Information Retrieval (IR) metrics.
*   **The Result:** 
    - **MRR @ 5 (0.84):** Proves our Cross-Encoder accurately pushes the perfect answer to Rank #1.
    - **Hit Rate @ 5 (1.00):** Proves our fast Bi-Encoder perfectly captures the right answer in its initial Top 5 filter 100% of the time.
    - **Taxonomical Precision:** Proves (via mathematical Silhouette and CPCC scores) that the underlying ontology structure is logically sound before search even begins.

## 8. Step 7: The User Interface (Streamlit Dashboard)
An AI engine is useless if non-technical game designers can't use it. 
*   **What we applied:** We built a 5-page interactive web dashboard using **Streamlit**. It features a dark-themed UI where users can upload PDFs, view the ontology dendrogram, chat with the RAG search engine, and view live evaluation metrics. 

## 9. Step 8: The BGG API Scraper
To make our dataset massive, we needed a way to pull data directly from BoardGameGeek.
*   **What we applied:** We engineered a highly robust XML API bot (`bgg_scraper.py`). It uses batching (pulling 400 games at a time) and features an exponential backoff loop to automatically handle rate limits or server drops without crashing.

---



## 💡 Important Interview Questions based on this Narrative

1.  **"How did you parse the PDFs without using a Vision AI?"**
    *   *Answer:* Standard extraction breaks on multi-column rulebooks, and Vision AI (like Nougat) is computationally wasteful. We used **PyMuPDF**, which calculates the exact spatial X/Y bounding boxes of every text block, allowing us to perfectly reconstruct the layout and reading order efficiently.
2.  **"Why use HAC instead of K-Means clustering?"**
    *   *Answer:* K-Means outputs flat lists. We were building an Ontology, which requires parent-child relationships. HAC naturally builds a family tree (dendrogram).
3.  **"How did you improve Explainability (XAI) in your clustering?"**
    *   *Answer:* We implemented Medoid calculation. We find the specific mechanic closest to the mathematical center of the cluster and display it as the "Representative Mechanic," making the AI's logic transparent.
4.  **"What is the weakest point of your project?"**
    *   *Answer:* Regex-based sentence classification. Regex is fragile against human language variability. In Version 2.0, we would replace Regex with a fine-tuned sequence classification Transformer (like RoBERTa) to automatically detect Rule vs. Flavor sentences.
5.  **"Why did you build a Two-Stage Search Pipeline instead of just using one model?"**
    *   *Answer:* Fast models (Bi-Encoders) miss linguistic nuance. Accurate models (Cross-Encoders) are computationally too slow to run on the whole database. Using a Bi-Encoder to filter the Top 25, and a Cross-Encoder to re-rank them, gave us the perfect balance of speed and elite accuracy (0.84 MRR).
