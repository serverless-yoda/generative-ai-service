# app/api/routes/video.py
from fastapi import APIRouter, File, Query
from fastapi.responses import StreamingResponse
from app.api.core.services import GenerationService
from app.api.core.utils import export_to_video_buffer

router = APIRouter()
service = GenerationService()

@router.post("/generate/video")
def generate_video_endpoint(image: bytes = File(...), num_frames: int = Query(25)):
    frames = service.generate_video(image, num_frames)
    return StreamingResponse(export_to_video_buffer(frames), media_type="video/mp4")
