# generative-ai-service/app/api/routes/video.py

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, Query, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from app.api.core.services import GenerationService
from app.api.core.utils import export_to_video_buffer

router = APIRouter()
service = GenerationService()
executor = ThreadPoolExecutor(max_workers=4)

async def run_blocking(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, func, *args)

@router.post("/video")
async def generate_video_endpoint(image: UploadFile = File(...), num_frames: int = Query(25, ge=1)):
    try:
        image_bytes = await image.read()
        frames = await run_blocking(service.generate_video, image_bytes, num_frames)
        video_buffer = export_to_video_buffer(frames)
        return StreamingResponse(video_buffer, media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

