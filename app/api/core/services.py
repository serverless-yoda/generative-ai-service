# app/core/services.py
from app.api.core.models import (
    load_text_model, generate_text,
    load_audio_model, generate_audio,
    load_image_model, generate_image,
    load_video_model, generate_video
)
from app.api.core.utils import audio_array_to_buffer, img_to_bytes, export_to_video_buffer
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO

class GenerationService:
    def __init__(self):
        self.text_pipe = load_text_model()
        self.audio_processor, self.audio_model = load_audio_model()
        self.image_pipe = load_image_model()
        self.video_pipe = load_video_model()

    def generate_text(self, prompt: str) -> str:
        return generate_text(self.text_pipe, prompt)

    def generate_audio(self, prompt: str, preset):
        audio_data, sample_rate = generate_audio(self.audio_processor, self.audio_model, prompt, preset)
        return audio_data, sample_rate

    def generate_image(self, prompt: str):
        return generate_image(self.image_pipe, prompt)

    def generate_video(self, image_bytes: bytes, num_frames: int):
        image = Image.open(BytesIO(image_bytes))
        frames = generate_video(self.video_pipe, image, num_frames)
        return frames
