# generative-ai-service/app/api/routes/huggingface/image_async.py
import asyncio
from fastapi import APIRouter, HTTPException, Depends, Query, Body,status
from fastapi.responses import StreamingResponse
from app.api.core.schemas import ImageModelRequest, ImageModelResponse, ImageSize

from app.api.core.huggingface_services import GenerationService
from app.api.core.utils import export_to_image_buffer

router = APIRouter()

@router.get("/image")
async def generate_image_endpoint(prompt: str, svc: GenerationService = Depends()) -> ImageModelResponse:
    try:
       
        loop = asyncio.get_running_loop()
        image = await loop.run_in_executor(None,svc.generate_image, prompt)
        buffer = export_to_image_buffer(image)
        return StreamingResponse(buffer, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
