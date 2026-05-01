from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="RAG API", version="1.0")

app.include_router(router)

@app.get("/")
def root():
    return {"message": "RAG API is running"}