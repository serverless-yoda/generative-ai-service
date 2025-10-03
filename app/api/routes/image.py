# generative-ai-service/app/api/routes/image.py

from fastapi import APIRouter, Response
from app.api.core.services import GenerationService
from app.api.core.utils import img_to_bytes

router = APIRouter()
service = GenerationService()



@router.get("/generate/image")
def generate_image_endpoint(prompt: str):
    image = service.generate_image(prompt)
    return Response(content=img_to_bytes(image), media_type="image/png")
