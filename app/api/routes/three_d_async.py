# generative-ai-service/app/api/routes/threeD.py
import asyncio
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse

from app.api.core.services import GenerationService
from app.api.core.utils import mesh_to_obj_buffer 


router = APIRouter()

@router.get("/3d")
async def generate_3d_endpoint(prompt: str, num_steps: int = Query(25, ge=1), svc: GenerationService = Depends()):
    try:
        loop = asyncio.get_running_loop()
        mesh = await loop.run_in_executor(None, svc.generate_3d, prompt, num_steps)
        buffer = mesh_to_obj_buffer(mesh)
        return StreamingResponse(
            buffer, media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={prompt}.obj"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
