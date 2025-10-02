# app/core/models.py

import tokenize
import torch
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from PIL import Image

from app.api.core.schemas import VoicePresets
#from app.api.core.utils import  float32_to_wav_bytes, audio_array_to_buffer


from transformers import Pipeline, pipeline, AutoProcessor, AutoModel, BarkProcessor, BarkModel
from diffusers import DiffusionPipeline, StableDiffusionInpaintPipelineLegacy, StableVideoDiffusionPipeline

prompt = "How to create,setup and run a FastAPI application?"
system_prompt = """
Your name is FastAPI bot and you are a helpful
chatbot responsible for teaching FastAPI to your users.
Always respond in markdown.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

def load_text_model():
    return (
        pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1b-Chat-v1.0",
            dtype=dtype,
            device=device
        )
    )

def generate_text(pipe: Pipeline, prompt : str, temperature: float=0.7) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    predictions = pipe(
        prompt, temperature=temperature,max_new_tokens=256,do_sample=True,top_k=50, top_p=0.95
    )

    return predictions[0]["generated_text"].split("</s>\n<|assistant|>\n")[-1]

def load_audio_model() -> Tuple[BarkProcessor, BarkModel]:
    """
    Load Bark 'small' with efficient dtype, move to device, and set pad_token_id explicitly.
    """
    processor = BarkProcessor.from_pretrained("suno/bark-small")

    #dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = BarkModel.from_pretrained("suno/bark-small",dtype=dtype)
    model.to(device).eval()

    # Avoid pad-id warnings: set PAD to EOS if not present (doc-friendly pattern)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return processor, model

def generate_audio(
    processor: BarkProcessor,
    model: BarkModel,
    prompt: str,
    preset: VoicePresets,
    *,
    do_sample: bool = True,
) -> Tuple[NDArray[np.float32], int]:
    """
    Generate audio from text using Bark and a given voice preset.
    Returns (waveform float32 in [-1, 1], sample_rate).
    """
    # 1) Encode text *and* voice preset via the processor (doc pattern)
    #    NOTE: Do NOT pass padding here; some versions double-forward padding.
    inputs = processor(text=[prompt], return_tensors="pt", voice_preset=preset)

    # 2) Ensure attention_mask exists (single prompt => all ones)
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    # 3) Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 4) Generate without passing voice_preset; keep pad_token explicit
    with torch.inference_mode():
        audio_tensor = model.generate(
            **inputs,
            do_sample=do_sample,
            pad_token_id=model.generation_config.pad_token_id,
        )

    # Bark returns a mono waveform tensor in [-1, 1], shape (1, T)
    audio = audio_tensor[0].detach().cpu().numpy().astype(np.float32)
    sample_rate = model.generation_config.sample_rate
    return audio, sample_rate


def load_image_model() -> StableDiffusionInpaintPipelineLegacy:
    return (
        DiffusionPipeline.from_pretrained("segmind/tiny-sd",dtype=dtype, device=device )
    )

def generate_image(
    pipe: StableDiffusionInpaintPipelineLegacy, prompt: str
) -> Image.Image:
    output = pipe(prompt, num_inference_steps=10).images[0]
    return output

def load_video_model() -> StableVideoDiffusionPipeline:

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        dtype=dtype ,
        variant="fp16",
        device=device,
    )
    return pipe


def generate_video(
    pipe: StableVideoDiffusionPipeline, image: Image.Image, num_frames: int = 25
) -> list[Image.Image]:
    image = image.resize((1024, 576))
    generator = torch.manual_seed(42)
    frames = pipe(
        image, decode_chunk_size=8, generator=generator, num_frames=num_frames
    ).frames[0]
    return frames