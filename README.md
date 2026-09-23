**🎬 Global AI Movie Vibe Recommender**

An intelligent, semantic movie recommendation engine that searches across a catalog of 60,000+ movies. By combining an offline vector database for high-speed semantic retrieval with Large Language Models for synthesis, the system translates abstract natural language vibe descriptions (e.g., "intense space survival" or "mind-bending heist thriller") into top film recommendations with live ratings and streaming availability.

---

**⚡ The Architecture: Hybrid RAG**

Traditional recommender systems rely on rigid keyword/genre filters or suffer from context window bloat by trying to pass raw data tables straight to an LLM. This project implements a Hybrid Retrieval-Augmented Generation (RAG) pipeline:

User Input ("intense space survival")
                │
                ▼
[ all-MiniLM-L6-v2 Embeddings ] (Edge / Local Inference)
                │
                ▼
[ ChromaDB Vector Index ] (Local Index of 60,000 Movie Vectors)
                │
                ▼
Top 4 Document Matches (Title, Rating, Plot Overview)
                │
                ▼
[ Google Gemini API ] (gemini-3.5-flash / gemini-3.5-flash-lite)
                │
                ▼
 Streamlit Web UI (Markdown Output with Streaming Availability)


 1. Local Vector Search: Converts user prompts into high-dimensional semantic coordinates using the local sentence-transformers/all-MiniLM-L6-v2 model. It queries a local ChromaDB index to fetch the 4 closest matching films in milliseconds without internet latency or token costs.
 2.  Context Block Assembly: Pulls factual metadata (titles, database ratings, plot summaries) straight from the local vector database.
 3.  LLM Synthesis & Streaming Discovery: Gemini digests the localized context block, validates why the picks match the requested vibe, and draws upon its vast parametric memory to determine where each title is actively streaming (e.g., Netflix, Prime Video, Disney+, Apple TV, JioCinema).

---

**FeaturesNatural Language Semantic Search**: 

Query by feelings, complex themes, or plot vibes rather than rigid genre tags
1. Zero Token Waste: Searches 60,000+ titles locally; the LLM processes only the top candidate matches rather than the entire database.
2. Dynamic Streaming Insights: Employs Gemini to identify major streaming platforms without relying on brittle, outdated static CSV files or rate-limited third-party APIs.
3. Automated Index Unpacker: Built-in auto-extraction utility for bundled vector database archives (chroma_storage.zip or multi-part split binaries in chroma_storage_parts/) for simple zero-friction deployment.
4. Resource Caching: Leverages @st.cache_resource so heavy transformer models and database clients load into system memory only once.

---

**Engineering Takeaways**

1. Avoid Prompt-As-A-Database Anti-Patterns: Passing entire catalogs directly into system instructions creates severe context window bloat, runs up token costs, and hits hard token limits. RAG decouples storage from inference.
2. Edge Embeddings Over Cloud Batching: Generating vectors on your local machine using lightweight open-source models (all-MiniLM-L6-v2) eliminates API rate limits (429 RESOURCE_EXHAUSTED) and network bottlenecks during heavy catalog ingestion.
3. Hybrid Data Layering: Using deterministic local databases for static factual records (titles, overviews, ratings) combined with generative models for volatile real-world knowledge (streaming distribution rights) avoids stale database joins and broken external scrapers.

---
