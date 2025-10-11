# generative-ai-service/app/main.py

from fastapi import FastAPI
from app.api.routes.huggingface.chat_async import router as chat_router
from app.api.routes.huggingface.audio_async import router as audio_router
from app.api.routes.huggingface.image_async import router as image_router
from app.api.routes.huggingface.three_d_async import router as three_d_router
from app.api.routes.huggingface.video_async import router as video_router
from app.api.routes.huggingface.text_async import router as text_router


from app.api.core.lifespan import ai_lifespan

# middleware
from app.api.middleware.monitor_service import monitor_service

app = FastAPI(title="Generative AI Service",lifespan = ai_lifespan)

# register middleware
app.middleware("http")(monitor_service)

app.include_router(chat_router,    prefix="/generate", tags=['huggingface'])
app.include_router(audio_router,   prefix="/generate", tags=['huggingface'])
app.include_router(image_router,   prefix="/generate", tags=['huggingface'])
app.include_router(three_d_router, prefix="/generate", tags=['huggingface'])
app.include_router(video_router,   prefix="/generate", tags=['huggingface'])
app.include_router(text_router,    prefix="/generate", tags=['huggingface'])


@app.get("/")
async def root():
    return {"status": "healthy"}
