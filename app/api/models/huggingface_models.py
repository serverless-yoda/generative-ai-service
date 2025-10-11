# app/api/models/huggingface_models.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple, List

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from app.api.core.schemas import VoicePresets

# ----- Global runtime config (lightweight) -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

SYSTEM_PROMPT = """
Your name is FastAPI bot and you are a helpful
chatbot responsible for teaching FastAPI to your users.
Always respond in markdown.
""".strip()


# -------------------------
# TEXT
# -------------------------
@lru_cache(maxsize=1)
def load_text_model():
    from transformers import pipeline  # lazy import
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1b-Chat-v1.0",
        dtype=dtype,
        device=device,
    )
    return pipe


def generate_text(pipe, prompt: str, 
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
    tok = pipe.tokenizer
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    
    # transformers uses tokenize kw; ensure it's the correct import
    prompt_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    predictions = pipe(
        prompt_text,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,        
        #token_id=getattr(tok, "eos_token_id", None),
        #eos_token_id=getattr(tok, "eos_token_id", None),
        return_full_text=False,  # only generated completion

    )
    text = predictions[0]["generated_text"]    
    # Cleanup in case any tags survived decoding
    for tag in ("<|assistant|>", "</s>", "<|eot_id|>"):
        text = text.replace(tag, "")
    return text.strip()



# -------------------------
# AUDIO (Bark small)
# -------------------------
@lru_cache(maxsize=1)
def load_audio_model() -> Tuple["BarkProcessor", "BarkModel"]:
    from transformers import BarkProcessor, BarkModel  # lazy import
    processor = BarkProcessor.from_pretrained("suno/bark-small")
    model = BarkModel.from_pretrained("suno/bark-small", dtype=dtype)
    model.to(device).eval()

    # Avoid pad-id warnings
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return processor, model


def generate_audio(
    processor, model, prompt: str, preset: VoicePresets, *, do_sample: bool = True
) -> Tuple[NDArray[np.float32], int]:
    inputs = processor(text=[prompt], return_tensors="pt", voice_preset=preset)
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        audio_tensor = model.generate(
            **inputs,
            do_sample=do_sample,
            pad_token_id=model.generation_config.pad_token_id,
        )

    audio = audio_tensor[0].detach().cpu().numpy().astype(np.float32)
    sample_rate = model.generation_config.sample_rate
    return audio, sample_rate


# -------------------------
# IMAGE (Tiny SD)
# -------------------------
@lru_cache(maxsize=1)
def load_image_model():
    from diffusers import DiffusionPipeline  # lazy import
    pipe = DiffusionPipeline.from_pretrained(
        "segmind/tiny-sd", dtype=dtype
    )
    # Move after construction (not in from_pretrained)
    pipe.to(device)
    return pipe


def generate_image(pipe, prompt: str) -> Image.Image:
    output = pipe(prompt, num_inference_steps=10).images[0]
    return output


# -------------------------
# VIDEO (SVD)
# -------------------------
@lru_cache(maxsize=1)
def load_video_model():
    from diffusers import StableVideoDiffusionPipeline  # lazy import
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe.to(device)
    return pipe


def generate_video(pipe, image: Image.Image, num_frames: int = 25) -> List[Image.Image]:
    image = image.resize((1024, 576))
    generator = torch.manual_seed(42)
    frames = pipe(
        image, decode_chunk_size=8, generator=generator, num_frames=num_frames
    ).frames[0]
    return frames


# -------------------------
# 3D (Shap-E)
# -------------------------
@lru_cache(maxsize=1)
def load_3d_model():
    from diffusers import ShapEPipeline  # lazy import
    pipe = ShapEPipeline.from_pretrained("openai/shap-e")
    pipe.to(device)
    return pipe


def generate_3d_geometry(pipe, prompt: str, num_inference_steps: int):
    result = pipe(
        prompt,
        guidance_scale=15.0,
        num_inference_steps=num_inference_steps,  # NOTE: is it "num_inference_steps"?
        output_type="mesh",
    )
   
    #print("Result keys:", result.keys())
    #print("Images:", result.images)
    return result.images[0]