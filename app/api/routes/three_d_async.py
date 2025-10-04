# generative-ai-service/app/api/routes/threeD.py

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, Query

from fastapi.responses import StreamingResponse
from app.api.core.services import GenerationService
from app.api.core.utils import mesh_to_obj_buffer 


router = APIRouter()
service = GenerationService()
executor = ThreadPoolExecutor(max_workers=4)

async def run_blocking(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, func, *args)

@router.get("/3d")
async def generate_3d_endpoint(prompt: str, num_steps: int = Query(25, ge=1)):
    try:
        mesh = await run_blocking(service.generate_3d, prompt, num_steps)
        buffer = mesh_to_obj_buffer(mesh)
        return StreamingResponse(
            buffer, media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={prompt}.obj"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
