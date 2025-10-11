# generative-ai-service/app/api/routes/huggingface/text_async.py
import asyncio
from fastapi import Depends,APIRouter, HTTPException, Body, status

from app.api.core.huggingface_services import GenerationService
from app.api.core.schemas import TextModelRequest,TextModelResponse
router = APIRouter()

@router.post("/text")
async def chat_endpoint(body: TextModelRequest = Body(...),svc: GenerationService = Depends()) -> TextModelResponse:
    try:
        if body.model not in ['tinyLlama','gemma2b']:
            raise HTTPException(
                detail=f"Model {body.model} is not supported",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None,svc.generate_text, body.prompt, body.temperature)
        return TextModelResponse(
            content=response,
            model = body.model,
            temperature = body.temperature
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


