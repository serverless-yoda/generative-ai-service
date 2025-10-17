# generative-ai-service/app/api/routes/huggingface/text_async.py
import asyncio
from fastapi import (
        Depends,
        APIRouter, 
        HTTPException, 
        Body, 
        status,
        Request
    )

from app.api.core.huggingface.service import GenerationService
from app.api.core.huggingface.schemas import TextModelRequest,TextModelResponse
from app.api.rag.rag_dependencies import  get_rag_content
from app.api.dependencies import get_urls_contents
router = APIRouter()


@router.post("/text", response_model_exclude_defaults=True)
async def chat_endpoint(req: Request,
                        body: TextModelRequest = Body(...), 
                        svc: GenerationService = Depends(), 
                        urls_content = Depends(get_urls_contents),
                        rag_content: str = Depends(get_rag_content)) -> TextModelResponse:
    try:
        if body.model not in ['tinyLlama','gemma2b']:
            raise HTTPException(
                detail=f"Model {body.model} is not supported",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        loop = asyncio.get_running_loop()
        prompt = body.prompt + " " + urls_content + " " + rag_content
        response = await loop.run_in_executor(None,svc.generate_text, prompt, body.temperature)
        return TextModelResponse(
            content=response,
            model = body.model,
            temperature = body.temperature,
            ip = req.client.host
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


