import faiss
import numpy as np
import pickle

class FaissVectorStore:
    def __init__(self, index_path, meta_path):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.chunks = None

    def build(self, embeddings, chunks):
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.array(embeddings))
        faiss.write_index(self.index, self.index_path)

        with open(self.meta_path, "wb") as f:
            pickle.dump(chunks, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.chunks = pickle.load(f)

    def search(self, q_emb, top_k):
        scores, ids = self.index.search(q_emb.reshape(1, -1), top_k)
        return [self.chunks[i] for i in ids[0]]
