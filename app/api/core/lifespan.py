# generative-ai-service/app/api/core/lifespan.py
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI


from app.api.models.huggingface_models import (
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
        "HF_text":  load_text_model(),   # cached TinyLlama pipeline
        "HF_audio": load_audio_model(),
        "HF_image": load_image_model(),
        "HF_video": load_video_model(),
        "HF_3d":    load_3d_model(),
    }
    try:
        yield
    finally:
       app.state.models.clear()
    