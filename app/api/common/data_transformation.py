# app/api/common/data_transformation.py

import re, aiofiles

from transformers import AutoModel
from typing import Any, AsyncGenerator

DEFAULT_CHUNK_SIZE = 1024 * 1024 * 50 # 50 megabytes

embedder = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
async def load(filepath: str) -> AsyncGenerator[str, Any]:
    async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
        while chunk := await f.read(DEFAULT_CHUNK_SIZE):
            yield chunk

def clean_text(content: str) -> str:
    t = content.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\. ,", "", t)
    t = t.replace("..", ".")
    t = t.replace(". .", ".")
    return t.replace("\n", " ").strip()

def embed(text: str) -> list[float]:
    return embedder.encode(text).tolist()