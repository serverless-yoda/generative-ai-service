# generative-ai-service/app/api/core/lifespan.py
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI


from app.api.core.models import (
    load_text_model,
    load_audio_model,
    load_image_model,
    load_video_model,
    load_3d_model,
)


models = {}

@asynccontextmanager
async def ai_lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.models = {        
        "text":  load_text_model(),   # cached TinyLlama pipeline
        "audio": load_audio_model(),
        "image": load_image_model(),
        "video": load_video_model(),
        "3d":    load_3d_model(),
    }
    try:
        yield
    finally:
       app.state.models.clear()
    