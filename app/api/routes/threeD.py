# generative-ai-service/app/api/routes/video.py

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse
from app.api.core.services import GenerationService
from app.api.core.utils import mesh_to_obj_buffer 

router = APIRouter()
service = GenerationService()

@router.get(
    "/generate/3d",
    responses={status.HTTP_200_OK: {"content": {"model/obj": {}}}}, 
    response_class=StreamingResponse,
)
def generate_3d_endpoint(prompt: str,num_inference_steps: int):
    mesh = service.generate_3d(prompt,num_inference_steps)
    response = StreamingResponse(mesh_to_obj_buffer(mesh), media_type="model/obj")
    response.headers['Content=Disposition'] = (f"attachment; filename={prompt}.obj")
    return response