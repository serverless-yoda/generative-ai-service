# generative-ai-service/app/api/routes/audio.py

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.api.core.schemas import VoicePresets
from app.api.core.services import GenerationService
from app.api.core.utils import audio_array_to_buffer

router = APIRouter()
service = GenerationService()
executor = ThreadPoolExecutor(max_workers=4)

async def run_blocking(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, func, *args)


@router.get("/audio")
async def generate_audio_endpoint(prompt: str, preset: VoicePresets = 'v2/en_speaker_1'):
    try:
        audio_data, sample_rate = await run_blocking(service.generate_audio, prompt, preset)
        audio_buffer = audio_array_to_buffer(audio_data, sample_rate)  # Adjust helper if needed
        return StreamingResponse(audio_buffer, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
