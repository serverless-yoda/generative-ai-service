# generative-ai-service/app/api/routes/audio.py
import asyncio
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from app.api.core.services import GenerationService
from app.api.core.schemas import VoicePresets
from app.api.core.services import GenerationService
from app.api.core.utils import audio_array_to_buffer

router = APIRouter()



@router.get("/audio")
async def generate_audio_endpoint(prompt: str, preset: VoicePresets = 'v2/en_speaker_1', svc: GenerationService = Depends()):
    try:
        loop = asyncio.get_running_loop()
        audio_data, sample_rate = await loop.run_in_executor(None,svc.generate_audio, prompt, preset)
        audio_buffer = audio_array_to_buffer(audio_data, sample_rate)  # Adjust helper if needed
        return StreamingResponse(audio_buffer, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
