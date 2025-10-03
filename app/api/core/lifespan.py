# generative-ai-service/app/api/core/lifespan.py
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import anyio
import torch
from fastapi import FastAPI

from app.api.core import models


def _env_flag(name: str) -> bool:
    return os.getenv(name, "0") in ("1", "true", "True", "YES", "yes")


@asynccontextmanager
async def ai_lifespan(app: FastAPI):
    """
    App-level lifespan that can optionally preload selected models and ensures cleanup.
    """
    PRELOAD_TEXT = _env_flag("PRELOAD_TEXT")
    PRELOAD_AUDIO = _env_flag("PRELOAD_AUDIO")
    PRELOAD_IMAGE = _env_flag("PRELOAD_IMAGE")
    PRELOAD_VIDEO = _env_flag("PRELOAD_VIDEO")
    PRELOAD_3D = _env_flag("PRELOAD_3D")

    # ---- Preload (in worker threads to avoid blocking the event loop) ----
    if PRELOAD_TEXT:
        app.state.text_pipe = await anyio.to_thread.run_sync(models.load_text_model)
    if PRELOAD_AUDIO:
        app.state.audio = await anyio.to_thread.run_sync(models.load_audio_model)
    if PRELOAD_IMAGE:
        app.state.image_pipe = await anyio.to_thread.run_sync(models.load_image_model)
    if PRELOAD_VIDEO:
        app.state.video_pipe = await anyio.to_thread.run_sync(models.load_video_model)
    if PRELOAD_3D:
        app.state.shape_pipe = await anyio.to_thread.run_sync(models.load_3d_model)

    try:
        yield
    finally:
        # ---- Cleanup (move to CPU, drop refs, clear CUDA cache) ----
        for attr in ("text_pipe", "image_pipe", "video_pipe", "shape_pipe"):
            pipe = getattr(app.state, attr, None)
            if pipe is not None and hasattr(pipe, "to"):
                try:
                    pipe.to("cpu")
                except Exception:
                    pass
                setattr(app.state, attr, None)

        audio = getattr(app.state, "audio", None)
        if audio is not None and isinstance(audio, tuple):
            processor, model = audio
            try:
                model.to("cpu")
            except Exception:
                pass
            app.state.audio = None

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass