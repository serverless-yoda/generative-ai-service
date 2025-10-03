# generative-ai-service/app/main.py

from fastapi import FastAPI
from app.api.core.config import settings
from app.api.core.lifespan import ai_lifespan

# Import router modules directly to avoid package __init__ side-effects
from app.api.routes.chat import router as chat_router
from app.api.routes.audio import router as audio_router
from app.api.routes.image import router as image_router
from app.api.routes.video import router as video_router
from app.api.routes.threeD import router as three_d_router

# Attach lifespan to manage (optional) preload + cleanup
app = FastAPI(title=settings.app_name, lifespan=ai_lifespan)

# Include all feature routers
app.include_router(chat_router,   prefix="/chat",  tags=["chat"])
app.include_router(audio_router,  prefix="/audio", tags=["audio"])
app.include_router(image_router,  prefix="/image", tags=["image"])
app.include_router(video_router,  prefix="/video", tags=["video"])
app.include_router(three_d_router, prefix="/3d",   tags=["3d"])

@app.get("/")
def health_check():
    return {"status": "healthy"}