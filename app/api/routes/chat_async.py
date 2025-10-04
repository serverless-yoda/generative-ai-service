# generative-ai-service/app/api/routes/chat.py

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException
from app.api.core.services import GenerationService

router = APIRouter()
service = GenerationService()
executor = ThreadPoolExecutor(max_workers=4)

async def run_blocking(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, func, *args)

@router.get("/chat")
async def chat_endpoint(prompt: str):
    try:
        response = await run_blocking(service.generate_text, prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


