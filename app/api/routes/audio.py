# app/api/routes/audio.py

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.api.core.schemas import VoicePresets
from app.api.core.services import GenerationService
from app.api.core.utils import audio_array_to_buffer

router = APIRouter()
service = GenerationService()

@router.get("/generate/audio")
def generate_audio_endpoint(prompt: str, preset: VoicePresets = 'v2/en_speaker_1'):
    audio_data, sample_rate = service.generate_audio(prompt, preset)
    return StreamingResponse(audio_array_to_buffer(audio_data, sample_rate), media_type="audio/wav")
