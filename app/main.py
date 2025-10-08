# generative-ai-service/app/main.py

from fastapi import FastAPI
from app.api.routes.chat_async import router as chat_router
from app.api.routes.audio_async import router as audio_router
from app.api.routes.image_async import router as image_router
from app.api.routes.three_d_async import router as three_d_router
from app.api.routes.video_async import router as video_router

from app.api.core.lifespan import ai_lifespan

app = FastAPI(title="Generative AI Service",lifespan = ai_lifespan)


app.include_router(chat_router,    prefix="/generate", tags=['async'])
app.include_router(audio_router,   prefix="/generate", tags=['async'])
app.include_router(image_router,   prefix="/generate", tags=['async'])
app.include_router(three_d_router, prefix="/generate", tags=['async'])
app.include_router(video_router,   prefix="/generate", tags=['async'])

@app.get("/")
async def root():
    return {"status": "healthy"}
