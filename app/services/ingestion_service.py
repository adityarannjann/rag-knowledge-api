import os
from app.utils.document_loader import load_document
from app.utils.text_chunker import chunk_documents
from app.adapters.embedding_adapter import EmbeddingAdapter
from app.db.vector_store import VectorStore

class IngestionService:

    def __init__(self):
        self.embedder = EmbeddingAdapter()
        self.vector_store = VectorStore()

    def ingest(self, file_path: str):
        # 1. Load
        documents = load_document(file_path)

        # 2. Chunk
        chunks = chunk_documents(documents)

        # 3. Extract text
        texts = [chunk.page_content for chunk in chunks]

        # 4. Embed
        embeddings = self.embedder.embed_documents(texts)

        # 5. Store
        self.vector_store.add(embeddings, chunks)

        return {
            "message": "Document ingested successfully",
            "chunks": len(chunks)
        }
