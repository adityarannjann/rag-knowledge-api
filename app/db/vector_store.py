import os
import faiss
import pickle
import numpy as np

class VectorStore:
    def __init__(self, dim=384, index_path="data/faiss.index", meta_path="data/meta.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.metadata = []

    def add(self, embeddings, documents):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)

        for doc in documents:
            self.metadata.append(doc.page_content)

        self._save()

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def search(self, query_embedding, k=3):
        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding, k)

        results = [self.metadata[i] for i in indices[0] if i < len(self.metadata)]
        return results