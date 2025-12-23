import pickle
import logging
from sentence_transformers import SentenceTransformer
from ingestion.chunker import chunk_text

logger = logging.getLogger(__name__)

def ingest_and_embed(docs, model_name, cache_dir):
    model = SentenceTransformer(model_name)
    all_chunks = []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for c in chunks:
            all_chunks.append({
                "source": doc["source"],
                "content": c
            })

    embeddings = model.encode(
        [c["content"] for c in all_chunks],
        normalize_embeddings=True
    )

    with open(f"{cache_dir}/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    logger.info(f"Embedded {len(all_chunks)} chunks")
    return embeddings, all_chunks
