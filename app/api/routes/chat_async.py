# generative-ai-service/app/api/routes/chat_async.py
import asyncio
from fastapi import Depends,APIRouter, HTTPException

from app.api.core.services import GenerationService

router = APIRouter()

@router.get("/chat")
async def chat_endpoint(prompt: str,svc: GenerationService = Depends()):
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None,svc.generate_text, prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


