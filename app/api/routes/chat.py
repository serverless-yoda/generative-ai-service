# app/api/routes/chat.py

from fastapi import APIRouter
from app.api.core.services import GenerationService

router = APIRouter()
service = GenerationService()

@router.get("/generate/text")
def generate_text_endpoint(prompt: str):
    return service.generate_text(prompt)
