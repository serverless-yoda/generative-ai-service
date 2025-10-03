import requests
import streamlit as st

st.set_page_config(page_title="FastAPI Services", page_icon="🤖", layout="centered")
st.title("FastAPI Services")

# -----------------------------
# Session state: chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.markdown("### Settings")
    base_url = st.text_input("FastAPI base URL", value="http://localhost:8000")
    mode = st.radio(
        "Response type",
        options=["Chat (Text)", "Generate Audio", "Generate Image", "Generate 3D"],
        help=(
            "Select what you want the assistant to return for your prompt:\n"
            "• Chat (Text): calls GET /generate/text\n"
            "• Generate Audio: calls GET /generate/audio\n"
            "• Generate Image: calls GET /generate/image\n"
            "• Generate 3D: calls GET /generate/3d"
        ),
    )
    st.divider()
    if st.button("🧹 Clear chat"):
        st.session_state.messages.clear()
        st.rerun()

# -----------------------------
# Helpers
# -----------------------------
def append_message(role: str, msg_type: str, content, mime: str | None = None):
    st.session_state.messages.append(
        {"role": role, "type": msg_type, "content": content, "mime": mime}
    )

def render_message(message: dict):
    with st.chat_message(message["role"]):
        msg_type = message.get("type", "text")
        content = message.get("content")
        mime = message.get("mime")

        if msg_type == "text":
            st.markdown(content)
        elif msg_type == "audio":
            st.text("🎵 Generated audio")
            st.audio(content, format=mime)
            st.download_button(
                "Download audio",
                data=content,
                file_name="audio_output" + _ext_from_mime(mime, ".wav"),
                mime=mime or "application/octet-stream",
                use_container_width=True,
            )
        elif msg_type == "image":
            st.text("🖼️ Generated image")
            st.image(content, caption="Generated image", use_column_width=True)
            st.download_button(
                "Download image",
                data=content,
                file_name="image_output" + _ext_from_mime(mime, ".png"),
                mime=mime or "application/octet-stream",
                use_container_width=True,
            )
        elif msg_type == "3d":
            st.text("📦 Generated 3D model (.obj)")
            st.download_button(
                "Download 3D Model",
                data=content,
                file_name="model_output.obj",
                mime=mime or "application/octet-stream",
                use_container_width=True,
            )
        else:
            st.markdown(f"*Unsupported message type:* `{msg_type}`")

def _ext_from_mime(mime: str | None, default_ext: str) -> str:
    if not mime:
        return default_ext
    if "wav" in mime:
        return ".wav"
    if "mpeg" in mime or "mp3" in mime:
        return ".mp3"
    if "ogg" in mime:
        return ".ogg"
    if "flac" in mime:
        return ".flac"
    if "png" in mime:
        return ".png"
    if "jpeg" in mime or "jpg" in mime:
        return ".jpg"
    if "webp" in mime:
        return ".webp"
    if "obj" in mime:
        return ".obj"
    return default_ext

def call_text_api(prompt: str) -> str:
    url = f"{base_url.rstrip('/')}/generate/text"
    resp = requests.get(url, params={"prompt": prompt}, timeout=240)
    resp.raise_for_status()
    return resp.text

def call_audio_api(prompt: str) -> tuple[bytes, str | None]:
    url = f"{base_url.rstrip('/')}/generate/audio"
    resp = requests.get(url, params={"prompt": prompt}, timeout=5000)
    resp.raise_for_status()
    mime = resp.headers.get("Content-Type")
    return resp.content, mime

def call_image_api(prompt: str) -> tuple[bytes, str | None]:
    url = f"{base_url.rstrip('/')}/generate/image"
    resp = requests.get(url, params={"prompt": prompt}, timeout=120)
    resp.raise_for_status()
    mime = resp.headers.get("Content-Type")
    return resp.content, mime

def call_3d_api(prompt: str) -> tuple[bytes, str | None]:
    url = f"{base_url.rstrip('/')}/generate/3d"
    resp = requests.get(url, params={"prompt": prompt, "num_inference_steps": 25}, timeout=300)
    resp.raise_for_status()
    mime = resp.headers.get("Content-Type")
    return resp.content, mime

# -----------------------------
# Replay history
# -----------------------------
for message in st.session_state.messages:
    render_message(message)

# -----------------------------
# Single chat input (routes by mode)
# -----------------------------
placeholder = {
    "Chat (Text)": "Ask anything…",
    "Generate Audio": "Describe the audio/voice you want…",
    "Generate Image": "Describe the image you want…",
    "Generate 3D": "Describe the 3D object you want…",
}[mode]

if prompt := st.chat_input(placeholder):
    append_message("user", "text", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if mode == "Chat (Text)":
                    assistant_text = call_text_api(prompt)
                    st.markdown(assistant_text)
                    append_message("assistant", "text", assistant_text)

                elif mode == "Generate Audio":
                    audio_bytes, mime = call_audio_api(prompt)
                    st.text("Here is your generated audio")
                    st.audio(audio_bytes, format=mime)
                    append_message("assistant", "audio", audio_bytes, mime)

                elif mode == "Generate Image":
                    image_bytes, mime = call_image_api(prompt)
                    st.text("Here is your generated image")
                    st.image(image_bytes)
                    append_message("assistant", "image", image_bytes, mime)

                elif mode == "Generate 3D":
                    obj_bytes, mime = call_3d_api(prompt)
                    st.text("Here is your generated 3D model")
                    st.download_button(
                        "Download 3D Model",
                        data=obj_bytes,
                        file_name=f"{prompt}.obj",
                        mime=mime or "application/octet-stream",
                        use_container_width=True,
                    )
                    append_message("assistant", "3d", obj_bytes, mime)

    except requests.exceptions.RequestException as e:
        with st.chat_message("assistant"):
            st.error(f"Request failed: {e}")
            append_message("assistant", "text", f"Request failed: {e}")