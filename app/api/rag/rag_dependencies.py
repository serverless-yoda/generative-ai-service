# app/api/rag/rag_deppendencies.py
import asyncio
from fastapi import Body
from loguru import logger


from app.api.core.huggingface.schemas import TextModelRequest
from app.api.rag.data_transformation import embed
from app.api.rag.rag_services import vector_service


async def get_rag_content(body: TextModelRequest = Body(...)) -> str:
    rag_content = await vector_service.search(
            "knowledgebase",
            embed(body.prompt),
            3,
            0.7
        )
    rag_content_str = "\n".join([c.payload['original_text'] for c in rag_content])
    return rag_content_str
