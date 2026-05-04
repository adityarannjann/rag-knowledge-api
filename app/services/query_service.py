from app.adapters.embedding_adapter import EmbeddingAdapter
from app.db.vector_store import VectorStore
from app.adapters.llm_adapter import get_llm
from app.core.logger import logger


class QueryService:

    def __init__(self):
        self.embedder = EmbeddingAdapter()
        self.vector_store = VectorStore()
        self.llm = get_llm()

    def build_prompt(self, question, context, history):
        history_text = ""
        for msg in history:
            role = msg.role.capitalize()
            history_text += f"{role}: {msg.content}\n"

        prompt = f"""
You are an intelligent assistant for answering questions from documents.

Rules:
- Answer ONLY from the provided context
- If unsure, say "I don't know"
- Keep answers concise and clear

Conversation History:
{history_text}

Context:
{context}

User Question:
{question}

Answer:
"""
        return prompt

    def query(self, question: str, history=[]):
        # 1. Embed query
        query_embedding = self.embedder.embed_query(question)

        # 2. Retrieve docs
        docs = self.vector_store.search(query_embedding, k=5)

        logger.info(f"Query received: {question}")
        logger.info(f"Retrieved {len(docs)} documents")

        context = "\n\n".join(docs)

        # 3. Build prompt
        prompt = self.build_prompt(question, context, history)

        # 4. Generate
        try:
            answer = self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"LLM Error: {str(e)}")
            answer = "Something went wrong while generating the answer."

        return {
            "answer": answer,
            "sources": docs
        }
    