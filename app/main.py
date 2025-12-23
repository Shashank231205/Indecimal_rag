import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import os

from ingestion.loader import load_documents
from ingestion.embedder import ingest_and_embed
from vector_store.faiss_store import FaissVectorStore
from rag.retriever import Retriever
from rag.generator import Generator
from rag.pipeline import RAGPipeline

load_dotenv()

def main():
    docs = load_documents("data/raw", ["md"])

    embeddings, chunks = ingest_and_embed(
        docs,
        os.getenv("EMBEDDING_MODEL"),
        "cache"
    )

    store = FaissVectorStore("cache/faiss.index", "cache/chunks.pkl")
    store.build(embeddings, chunks)
    store.load()

    rag = RAGPipeline(
        Retriever(store, os.getenv("EMBEDDING_MODEL")),
        Generator(os.getenv("LLM_MODEL"))
    )

    print("System ready ✅ (type exit to quit)\n")

    while True:
        q = input("Ask: ").strip()
        if q.lower() == "exit":
            break

        ctx, ans = rag.answer(q)

        print("\n--- Retrieved Context ---")
        for c in ctx[:2]:
            print(f"[{c['source']}] {c['content'][:250]}")

        print("\n--- Answer ---")
        print(ans)
        print()

if __name__ == "__main__":
    main()
