# generative-ai-service/app/api/deppendencies.py
import asyncio
from fastapi import Body
from loguru import logger

from app.api.common.web_scraping import extract_url, fetch_all_urls
from app.api.core.schemas import TextModelRequest

async def get_urls_contents(body: TextModelRequest = Body(...)) -> str:
    urls = extract_url(body.prompt)
    if urls:
        try:
            urls_content = await fetch_all_urls(urls)
            return urls_content
        except Exception as e:
            logger.warning(f"Failed to fetch of several URL. Error: {e}")
    return ""
