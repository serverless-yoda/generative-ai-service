from fastapi import FastAPI
from app.api.routes import chat, audio, image, video
from app.api.core.config import settings

app = FastAPI(title=settings.app_name)

app.include_router(chat.router)
app.include_router(audio.router)
app.include_router(image.router)
app.include_router(video.router)

@app.get("/")
def health_check():
    return {"status": "healthy"}
