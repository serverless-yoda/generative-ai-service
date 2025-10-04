# generative-ai-service/app/api/routes/image.py

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse

from app.api.core.services import GenerationService

from app.api.core.utils import export_to_image_buffer

router = APIRouter()
service = GenerationService()
executor = ThreadPoolExecutor(max_workers=4)

async def run_blocking(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, func, *args)

@router.get("/image")
async def generate_image_endpoint(prompt: str):
    try:
        image = await run_blocking(service.generate_image, prompt)
        buffer = export_to_image_buffer(image)
        return StreamingResponse(buffer, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
