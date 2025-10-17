# app/api/routes/aoai/text_streaming.py

from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, Depends

from app.api.core.aoai.service import azure_chat_client

router = APIRouter()

@router.get('/text/stream')
async def serve_text_streaming(prompt: str) -> StreamingResponse:
    return StreamingResponse(
        azure_chat_client.chat_stream(prompt), media_type='text/event-stream'
    )

