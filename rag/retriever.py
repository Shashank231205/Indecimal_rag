from sentence_transformers import SentenceTransformer

STOPWORDS = {
    "what","how","does","do","is","are","the","a","an",
    "and","or","to","of","during","when","that"
}

class Retriever:
    def __init__(self, store, model_name):
        self.store = store
        self.embedder = SentenceTransformer(model_name)

    def retrieve(self, question):
        q_emb = self.embedder.encode(question, normalize_embeddings=True)
        candidates = self.store.search(q_emb, top_k=5)

        q_terms = {
            w for w in question.lower().split()
            if w not in STOPWORDS
        }

        filtered = []
        for c in candidates:
            c_terms = set(c["content"].lower().split())
            if len(q_terms & c_terms) >= 2:
                filtered.append(c)

        return filtered[:2]
