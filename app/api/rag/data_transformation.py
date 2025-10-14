# app/api/rag/data_transformation.py

import re, aiofiles

from transformers import AutoModel
from transformers import AutoTokenizer
from typing import Any, AsyncGenerator

#DEFAULT_CHUNK_SIZE = 1024 * 1024 * 50 # 50 megabytes
DEFAULT_CHUNK_SIZE = 1024 * 4  # 4 KB

embedder = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
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
    if len(text) > 2000:  # or use token count
        raise ValueError("Text too long for embedding model")
    return embedder.encode(text).tolist()

def chunk_text(text: str, max_tokens: int = 512) -> list[str]:
    tokens = tokenizer.encode(text, truncation=False)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]
