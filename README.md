# Mini RAG – Construction Marketplace Assistant (Technical Focus)

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for a construction marketplace AI assistant.  
The system answers user questions **strictly using internal documents** such as policies, FAQs, and specifications.

The focus of this implementation is **groundedness, determinism, and explainability** rather than creative generation.

---

## Architecture Summary
1. **Document Chunking**
   - Markdown documents are split into semantically meaningful chunks (headings + bullet groups).
   - Each chunk preserves source metadata for transparency.

2. **Embeddings**
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Reason: Lightweight, fast, strong semantic similarity performance on CPU.

3. **Vector Store**
   - FAISS (CPU)
   - Cosine similarity used for top-k retrieval.

4. **Retrieval**
   - For each query, top-k relevant chunks are retrieved.
   - Retrieved chunks are printed verbatim before answer generation.

5. **LLM Answer Generation**
   - Model: FLAN-T5 (local, CPU)
   - Role strictly limited to **rephrasing retrieved facts**.
   - If facts are insufficient → explicit fallback response.

---

## Grounding Strategy
- No free-form generation.
- Answers are assembled directly from retrieved chunks.
- LLM is only used when multiple facts need coherent phrasing.
- Numeric and list-based queries bypass the LLM entirely.

---

## Example
**Query:** What factors affect construction project delays?

**Retrieved Context:**
- Integrated project management system  
- Daily project tracking  
- Instant deviation flagging  

**Answer:**
Integrated project management, daily tracking, and instant deviation flagging are used to manage construction delays.

---
## Environment Configuration

This project uses a `.env` file for configurable runtime parameters such as:
- Embedding model selection
- LLM model selection
- Retrieval depth (TOP_K)
- Generation limits
- Strict grounding mode

The `.env` file is intentionally excluded from version control.

### Example `.env` (template only)

Create a `.env` file in the project root with the following structure:

```env
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=google/flan-t5-base
TOP_K=3
MAX_NEW_TOKENS=150
STRICT_GROUNDING=true

## I've still included .env to make it simpler, but in production i would never do it .(it is the biggest mistake an ml engineer would commit)



## How to Run
```bash
1. Create Virtual Environment
python -m venv venv
.\venv\Scripts\Activate.ps1

2. Install Requirements
pip install -r requirements.txt

3. Run the Application
python app/main.py
```

---

## Evaluation Notes
- Zero hallucinations observed in test queries.
- Some answers are intentionally concise to preserve factual accuracy.

---

## Author
**Shashank KS**
