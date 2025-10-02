import os
from fastapi import FastAPI, status, Response
from fastapi.responses import StreamingResponse
from openai import AzureOpenAI

from app.config import settings
from app.models import load_text_model, generate_text
from app.models import load_audio_model, generate_audio
from app.models import load_image_model, generate_image
from app.schemas import VoicePresets
from app.utils import audio_array_to_buffer, img_to_bytes

app = FastAPI()

AZURE_OPENAI_ENDPOINT   = settings.azure_endpoint_url
AZURE_OPENAI_DEPLOYMENT = settings.azure_deployment_name
AZURE_OPENAI_API_KEY    = settings.azure_openai_api_key
AZURE_OPENAI_API_VERSION= settings.azure_openai_version


@app.get("/")
def root_controller():
    return "status - healthy"

@app.get("/chat")
def chat_controller():
    if not AZURE_OPENAI_ENDPOINT or ".openai.azure.com" not in AZURE_OPENAI_ENDPOINT:
        return {"Error": "Invalid AZURE_OPENAI_ENDPOINT (expected https://<resource>.openai.azure.com)"}
    if not AZURE_OPENAI_API_KEY:
        return {"Error": "AZURE_OPENAI_API_KEY not set"}
    if not AZURE_OPENAI_DEPLOYMENT:
        return {"Error": "AZURE_OPENAI_DEPLOYMENT not set (must be Azure deployment name)"}
    if not AZURE_OPENAI_API_VERSION:
        return {"Error": "AZURE_OPENAI_API_VERSION not set"}
  
    try:
        client = AzureOpenAI(
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
        )

        SYSTEM_PROMPT = (
        "You are a product ideation assistant. Respond ONLY in valid GitHub-Flavored Markdown with this structure:\n\n"
        "## Idea Name\n\n"
        "## One-Liner\n\n"
        "### Problem\n"
        "- Bullet 1\n"
        "- Bullet 2\n\n"
        "### AI Agent Design\n"
        "- Bullet 1\n"
        "- Bullet 2\n\n"
        "### Target Users\n"
        "- Bullet 1\n\n"
        "### Monetization\n"
        "- Bullet 1\n\n"
        "Rules:\n"
        "- Every bullet MUST start with '- ' (dash + space).\n"
        "- Insert a blank line after each heading and before lists.\n"
        "- Do NOT include phrases like 'A blank line'.\n"
        "- Do NOT wrap the response in code fences.\n"
        "- Keep the entire response under 150 words.\n"
        "- Do NOT insert extra spaces inside words. Keep words intact.\n"
    )


        # Put format rules in SYSTEM, keep USER short/goal-focused
        system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }

        user_message = {
            "role": "user",
            "content": "Generate one concise business idea for AI Agents.",
        }

        
        response = client.chat.completions.create(
            model =AZURE_OPENAI_DEPLOYMENT,
            messages=[system_message, user_message]
        )

        return {"Data": response.choices[0].message.content}
    except Exception as e:
        return {"Error": str(e)}

@app.get('/generate/text')
def serve_language_model_controller(prompt: str) -> str:
    pipe = load_text_model()
    output = generate_text(pipe,prompt)
    return output


@app.get('/generate/audio',responses={status.HTTP_200_OK: {'content': {'audio/wav':{}}}},response_class=StreamingResponse,)
def serve_text_to_audio_moder_controller(prompt: str, preset: VoicePresets = 'v2/en_speaker_1',):
    processor, model = load_audio_model()
    output, sample_rate = generate_audio(processor, model, prompt, preset)
    return StreamingResponse(
        audio_array_to_buffer(output, sample_rate), media_type="audio/wav"
    )

@app.get(
    "/generate/image",
    responses={status.HTTP_200_OK: {"content": {"image/png": {}}}},
    response_class=Response,
)
def serve_text_to_image_model_controller(prompt: str):
    pipe = load_image_model()
    output = generate_image(pipe, prompt)
    return Response(content=img_to_bytes(output), media_type="image/png")
