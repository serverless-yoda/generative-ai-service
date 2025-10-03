# generative-ai-service/app/api/core/services.py

from app.api.core.models import (
    load_text_model,  generate_text,
    load_audio_model, generate_audio,
    load_image_model, generate_image,
    load_video_model, generate_video,
    load_3d_model,    generate_3d_geometry
)
from app.api.core.utils import audio_array_to_buffer, img_to_bytes, export_to_video_buffer
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO

class GenerationService:
    def __init__(self):
        self.text_pipe = load_text_model()

    def generate_text(self, prompt: str) -> str:
        return generate_text(self.text_pipe, prompt)

    def generate_audio(self, prompt: str, preset):
        self.audio_processor, self.audio_model = load_audio_model()
        audio_data, sample_rate = generate_audio(self.audio_processor, self.audio_model, prompt, preset)
        return audio_data, sample_rate

    def generate_image(self, prompt: str):
        self.image_pipe = load_image_model()        
        return generate_image(self.image_pipe, prompt)

    def generate_video(self, image_bytes: bytes, num_frames: int):
        self.video_pipe = load_video_model()        
        image = Image.open(BytesIO(image_bytes))
        frames = generate_video(self.video_pipe, image, num_frames)
        return frames

    def generate_3d(self,prompt: str, num_inference_steps: int = 25):
        self.threeD_pipe = load_3d_model()
        return  generate_3d_geometry(self.threeD_pipe,prompt=prompt,num_inference_steps=num_inference_steps)
        
