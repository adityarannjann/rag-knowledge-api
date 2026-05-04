from fastapi import APIRouter, UploadFile, File
import shutil
import os

from app.services.ingestion_service import IngestionService
from app.services.query_service import QueryService
from app.models.chat_models import ChatRequest, ChatResponse

router = APIRouter()
ingestion_service = IngestionService()
query_service = QueryService()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = ingestion_service.ingest(file_path)
    return result


@router.post("/query")
def query_rag(question: str):
    return query_service.query(question)

@router.post("/chat", response_model=ChatResponse)
def chat_rag(request: ChatRequest):
    result = query_service.query(
        question=request.question,
        history=request.history
    )
    return result
