# app/api/core/rag_services.py
import os
from loguru import logger
from app.api.repository.vector import VectorRepository
from app.api.common.data_transformation import clean_text,embed,load, chunk_text

class VectorService(VectorRepository):
    def __init__(self):
        super().__init__()

    async def store_content_in_db(
        self,
        filepath: str,
        chunk_size:int = 512,
        collection_name: str="knowledgebase",
        collection_size: int = 768
    )-> None:
        await self.create_collection(
            collection_name,
            collection_size
        )
        logger.debug(f'Inserting {filepath} content to database')
        async for chunk in load(filepath):
            logger.debug(f'Inserting {chunk[0:20]}.. into database')
            
            for sub_chunk in chunk_text(clean_text(chunk)):
                embedding_vector = embed(sub_chunk)
                filename = os.path.basename(filepath)
                await self.create(collection_name, embedding_vector, sub_chunk, filename)


vector_service = VectorService()