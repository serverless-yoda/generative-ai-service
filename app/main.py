# generative-ai-service/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.huggingface.chat_async import router as chat_router
from app.api.routes.huggingface.audio_async import router as audio_router
from app.api.routes.huggingface.image_async import router as image_router
from app.api.routes.huggingface.three_d_async import router as three_d_router
from app.api.routes.huggingface.video_async import router as video_router
from app.api.routes.huggingface.text_async import router as text_router

# rag
from app.api.routes.rag.fileupload_async import router as upload_router
from app.api.routes.rag.rag_text_async import router as rag_text_router

# aoai
from app.api.routes.aoai.text_stream import router as stream_router
# postgres
from app.api.routes.postgres.conversation import router as conversation_router

from app.api.core.lifespan import ai_lifespan

# middleware
from app.api.middleware.monitor_service import monitor_service

app = FastAPI(title="Generative AI Service",lifespan = ai_lifespan)

# register middleware
app.middleware("http")(monitor_service)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router,           prefix="/generate", tags=['huggingface'])
app.include_router(audio_router,          prefix="/generate", tags=['huggingface'])
app.include_router(image_router,          prefix="/generate", tags=['huggingface'])
app.include_router(three_d_router,        prefix="/generate", tags=['huggingface'])
app.include_router(video_router,          prefix="/generate", tags=['huggingface'])
app.include_router(text_router,           prefix="/generate", tags=['huggingface'])
app.include_router(upload_router,         prefix="/file",     tags=['rag'])
app.include_router(rag_text_router,       prefix="/rag",      tags=['rag'])
app.include_router(stream_router,         prefix="/generate", tags=['azure openai'])
app.include_router(conversation_router,   prefix="/postgres", tags=['database'])


@app.get("/")
async def root():
    return {"status": "healthy"}
