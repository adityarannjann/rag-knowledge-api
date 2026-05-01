from app.adapters.embedding_adapter import EmbeddingAdapter
from app.db.vector_store import VectorStore
from app.adapters.llm_adapter import get_llm


class QueryService:

    def __init__(self):
        self.embedder = EmbeddingAdapter()
        self.vector_store = VectorStore()
        self.llm = get_llm()

    def query(self, question: str):
        # 1. Embed query
        query_embedding = self.embedder.embed_query(question)

        # 2. Retrieve relevant docs
        docs = self.vector_store.search(query_embedding, k=3)

        # 3. Build context
        context = "\n\n".join(docs)

        # 4. Prompt
        prompt = f"""
You are a helpful assistant. Answer ONLY from the context below.
If answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""

        # 5. Generate answer
        answer = self.llm.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "context_used": docs
        }