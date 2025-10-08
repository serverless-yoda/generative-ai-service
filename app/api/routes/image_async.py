# generative-ai-service/app/api/routes/image.py
import asyncio
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse

from app.api.core.services import GenerationService
from app.api.core.utils import export_to_image_buffer

router = APIRouter()

@router.get("/image")
async def generate_image_endpoint(prompt: str, svc: GenerationService = Depends()):
    try:
        loop = asyncio.get_running_loop()
        image = await loop.run_in_executor(None,svc.generate_image, prompt)
        buffer = export_to_image_buffer(image)
        return StreamingResponse(buffer, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
