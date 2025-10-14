# generative-ai-service/app/api/deppendencies.py
import asyncio
from fastapi import Body
from loguru import logger

from app.api.common.web_scraping import extract_url, fetch_all_urls
from app.api.core.schemas import TextModelRequest, TextModelResponse
from app.api.common.data_transformation import embed, chunk_text
from app.api.core.rag_services import vector_service

async def get_urls_contents(body: TextModelRequest = Body(...)) -> str:
    urls = extract_url(body.prompt)
    if urls:
        try:
            urls_content = await fetch_all_urls(urls)
            return urls_content
        except Exception as e:
            logger.warning(f"Failed to fetch of several URL. Error: {e}")
    return ""


async def get_rag_content(body: TextModelRequest = Body(...)) -> str:
    rag_content = await vector_service.search(
            "knowledgebase",
            embed(body.prompt),
            3,
            0.7
        )
    rag_content_str = "\n".join([c.payload['original_text'] for c in rag_content])
    return rag_content_str
