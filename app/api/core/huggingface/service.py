# generative-ai-service/app/api/core/huggingface/services.py
from fastapi.responses import StreamingResponse
from fastapi import Request,Depends

from app.api.models.huggingface.models import (
    load_text_model,  generate_text,
    load_audio_model, generate_audio,
    load_image_model, generate_image,
    load_video_model, generate_video,
    load_3d_model,    generate_3d_geometry
)

from PIL import Image
from io import BytesIO


def get_models(request: Request):
    return request.app.state.models

class GenerationService:
    def __init__(self, models: dict = Depends(get_models)):        
        self.text_pipe     = models["HF_text"]  # TinyLlama pipeline
        self.audio_pipe    = models["HF_audio"]
        self.image_pipe    = models["HF_image"]
        self.video_pipe    = models["HF_video"]
        self.geometry_pipe = models["HF_3d"]


    def generate_text(self, prompt: str,  
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        top_k: int = 50,
        top_p: float = 0.95) -> str:
        return generate_text(self.text_pipe, prompt,temperature,max_new_tokens,top_k,top_p)

    def generate_audio(self, prompt: str, preset):
        self.audio_processor, self.audio_model = self.audio_pipe
        audio_data, sample_rate = generate_audio(self.audio_processor, self.audio_model, prompt, preset)
        return audio_data, sample_rate

    def generate_image(self, prompt: str):
        self.image_pipe = self.image_pipe        
        return generate_image(self.image_pipe, prompt)

    def generate_video(self, image_bytes: bytes, num_frames: int):
        self.video_pipe = self.video_pipe       
        image = Image.open(BytesIO(image_bytes))
        frames = generate_video(self.video_pipe, image, num_frames)
        return frames

    def generate_3d(self,prompt: str, num_inference_steps: int = 25):
        self.threeD_pipe = self.geometry_pipe
        return  generate_3d_geometry(self.threeD_pipe,prompt=prompt,num_inference_steps=num_inference_steps)
        

