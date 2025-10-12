import io
import csv
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
        options=[
            "Chat (Text)",
            "Generate Audio",
            "Generate Image",
            "Generate 3D",
            "Generate Video",
            "Generate Text",
            "Upload PDFs",   # <-- plural
        ],
        help=(
            "Select what you want the assistant to return for your prompt:\n"
            "• Chat (Text): calls GET /generate/chat\n"
            "• Generate Audio: calls GET /generate/audio\n"
            "• Generate Image: calls GET /generate/image\n"
            "• Generate 3D: calls GET /generate/3d\n"
            "• Generate Video: calls POST /generate/video\n"
            "• Generate Text: calls POST /text (falls back to /generate/text)\n"
            "• Upload PDFs: calls POST /upload (falls back to /rag/upload)"
        ),
    )
    st.divider()
    if st.button("🧹 Clear chat"):
        st.session_state.messages.clear()
        st.rerun()

    # Extra controls for "Generate Text" only
    if mode == "Generate Text":
        st.markdown("### Generate Text options")
        text_model = st.selectbox(
            "Model",
            options=["tinyLlama", "gemma2b"],
            help="Your FastAPI endpoint only supports tinyLlama and gemma2b."
        )
        text_temperature = st.slider(
            "Temperature",
            min_value=0.0, max_value=1.0, value=0.7, step=0.05,
            help="Higher values = more creative; lower = more deterministic."
        )
        urls_raw = st.text_area(
            "Optional URLs (one per line)",
            value="",
            placeholder="https://example.com/doc1\nhttps://example.com/doc2",
            help="If provided, these will be sent as a list to the backend to build URL context."
        )

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
        elif msg_type == "video":
            st.text("🎥 Generated video")
            st.video(content)
            st.download_button(
                "Download video",
                data=content,
                file_name="video_output.mp4",
                mime=mime or "video/mp4",
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
    if "mp4" in mime:
        return ".mp4"
    return default_ext


def _looks_like_pdf(name: str, data: bytes) -> bool:
    """Basic client-side validation to avoid obvious non-PDF uploads."""
    if not name.lower().endswith(".pdf"):
        return False
    # PDF files typically start with %PDF-
    return data.startswith(b"%PDF")


# -----------------------------
# Backend calls
# -----------------------------
def call_text_api(prompt: str) -> str:
    """
    Legacy text API used by 'Chat (Text)' mode (GET).
    """
    url = f"{base_url.rstrip('/')}/generate/text"
    resp = requests.get(url, params={"prompt": prompt}, timeout=240)
    resp.raise_for_status()
    return resp.text


def call_generate_text_api(prompt: str, model: str, temperature: float, urls: list[str] | None = None) -> dict:
    """
    Client for your FastAPI /text endpoint.

    Tries POST {base_url}/text with JSON:
      {"prompt": str, "model": str, "temperature": float, "urls": [str, ...]}
    Falls back to POST {base_url}/generate/text for compatibility.

    Returns dict: {"content", "model", "temperature", "ip"}
    """
    payload = {
        "prompt": prompt,
        "model": model,
        "temperature": float(temperature),
    }
    if urls:
        payload["urls"] = urls

    def _post(url: str) -> requests.Response:
        return requests.post(url, json=payload, timeout=240)

    # Try /text first (matches your router.post("/text"))
    try:
        resp = _post(f"{base_url.rstrip('/')}/text")
        resp.raise_for_status()
    except requests.RequestException:
        # Fallback to /generate/text
        resp = _post(f"{base_url.rstrip('/')}/generate/text")
        resp.raise_for_status()

    # Parse JSON (TextModelResponse)
    try:
        data = resp.json()
        if isinstance(data, dict) and "content" in data:
            return {
                "content": data.get("content", ""),
                "model": data.get("model"),
                "temperature": data.get("temperature"),
                "ip": data.get("ip"),
            }
    except ValueError:
        pass

    return {"content": resp.text, "model": model, "temperature": temperature, "ip": None}


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


def call_video_api(image_file, num_frames: int) -> tuple[bytes, str | None]:
    url = f"{base_url.rstrip('/')}/generate/video"
    files = {"image": ("uploaded_img.png", image_file, "image/png")}
    params = {"num_frames": num_frames}
    resp = requests.post(url, files=files, params=params, timeout=600)
    resp.raise_for_status()
    mime = resp.headers.get("Content-Type")
    return resp.content, mime


def call_upload_api(pdf_name: str, pdf_bytes: bytes) -> dict:
    """
    Upload a single PDF to your FastAPI file upload endpoint.

    Primary: POST {base_url}/upload
    Fallback: POST {base_url}/rag/upload (if router mounted with a /rag prefix)
    """
    files = {
        "file": (pdf_name, pdf_bytes, "application/pdf"),
    }

    def _post(url: str) -> requests.Response:
        # FastAPI expects multipart/form-data with field name "file"
        return requests.post(url, files=files, timeout=600)

    # Try /upload first
    try:
        resp = _post(f"{base_url.rstrip('/')}/upload")
        resp.raise_for_status()
    except requests.RequestException:
        # Fallback: /rag/upload if router is mounted with a prefix
        resp = _post(f"{base_url.rstrip('/')}/rag/upload")
        resp.raise_for_status()

    try:
        return resp.json()  # {"filename": "...", "message": "..."}
    except ValueError:
        # Non-JSON response (unexpected)
        return {"filename": pdf_name, "message": resp.text or "Upload completed."}


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
    "Generate Video": "Upload an image to use as the video seed…",
    "Generate Text": "Describe the text you want generated…",
    "Upload PDFs": "Select PDFs below and click Upload…",
}[mode]

# Extra inputs for video mode
if mode == "Generate Video":
    uploaded_file = st.file_uploader("Upload image to generate video", type=["png", "jpg", "jpeg"])
    num_frames = st.number_input("Number of frames", min_value=1, max_value=100, value=25)

# -----------------------------
# Multi PDF Upload UI (no chat input)
# -----------------------------
if mode == "Upload PDFs":
    pdf_files = st.file_uploader(
        "Select one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )
    num_selected = len(pdf_files) if pdf_files else 0
    st.caption(f"{num_selected} file(s) selected")

    col1, col2 = st.columns([1, 3])
    with col1:
        do_upload = st.button(
            f"⬆️ Upload {num_selected} PDF{'s' if num_selected != 1 else ''}",
            type="primary",
            disabled=(num_selected == 0)
        )

    if do_upload and pdf_files:
        results = []
        progress = st.progress(0)
        status_area = st.container()

        for idx, f in enumerate(pdf_files, start=1):
            name = f.name
            data = f.getvalue()
            size_mb = len(data) / (1024 * 1024)

            with status_area.expander(f"Uploading: {name} ({size_mb:.2f} MB)", expanded=False):
                # Client-side quick validation
                if not _looks_like_pdf(name, data):
                    msg = "Client-side validation failed (not a PDF)."
                    st.error(msg)
                    results.append({
                        "filename": name,
                        "size_mb": round(size_mb, 2),
                        "status": "error",
                        "message": msg,
                    })
                else:
                    try:
                        resp = call_upload_api(name, data)
                        msg = resp.get("message", "File uploaded successfully")
                        st.success(f"✅ {msg}")
                        results.append({
                            "filename": resp.get("filename", name),
                            "size_mb": round(size_mb, 2),
                            "status": "ok",
                            "message": msg,
                        })
                    except requests.exceptions.HTTPError as e:
                        try:
                            detail = e.response.json().get("detail")
                        except Exception:
                            detail = str(e)
                        st.error(f"❌ Upload failed: {detail}")
                        results.append({
                            "filename": name,
                            "size_mb": round(size_mb, 2),
                            "status": "error",
                            "message": detail,
                        })
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Upload failed: {e}")
                        results.append({
                            "filename": name,
                            "size_mb": round(size_mb, 2),
                            "status": "error",
                            "message": str(e),
                        })

            progress.progress(int(idx / len(pdf_files) * 100))

        # Summary
        ok_count = sum(1 for r in results if r["status"] == "ok")
        err_count = sum(1 for r in results if r["status"] == "error")
        st.success(f"Finished: {ok_count} success, {err_count} failed.")

        # Results table
        st.subheader("Upload results")
        st.dataframe(
            {
                "Filename": [r["filename"] for r in results],
                "Size (MB)": [r["size_mb"] for r in results],
                "Status": [r["status"] for r in results],
                "Message": [r["message"] for r in results],
            },
            use_container_width=True,
        )

        # CSV download of results
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["filename", "size_mb", "status", "message"])
        for r in results:
            writer.writerow([r["filename"], r["size_mb"], r["status"], r["message"]])
        st.download_button(
            "Download results CSV",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name="upload_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Add brief history entry
        append_message("assistant", "text", f"Uploaded PDFs — {ok_count} succeeded, {err_count} failed.")

# -----------------------------
# Chat-like input for other modes
# -----------------------------
elif prompt := st.chat_input(placeholder):
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

                elif mode == "Generate Text":
                    # Parse URLs textarea into a list (trim empty lines)
                    urls_list = None
                    try:
                        urls_list = [u.strip() for u in urls_raw.splitlines() if u.strip()]
                        if len(urls_list) == 0:
                            urls_list = None
                    except Exception:
                        urls_list = None

                    result = call_generate_text_api(prompt, text_model, text_temperature, urls_list)
                    st.markdown(result["content"])

                    with st.expander("Response details"):
                        st.markdown(f"- **Model**: `{result.get('model', text_model)}`")
                        st.markdown(f"- **Temperature**: `{result.get('temperature', text_temperature)}`")
                        if result.get("ip"):
                            st.markdown(f"- **Server IP**: `{result['ip']}`")

                    append_message("assistant", "text", result["content"])

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

                elif mode == "Generate Video":
                    if uploaded_file is not None:
                        video_bytes, mime = call_video_api(uploaded_file.getvalue(), num_frames)
                        st.video(video_bytes)
                        st.download_button(
                            "Download generated video",
                            data=video_bytes,
                            file_name="generated_video.mp4",
                            mime=mime or "video/mp4",
                            use_container_width=True,
                        )
                        append_message("assistant", "video", video_bytes, mime)

    except requests.exceptions.HTTPError as e:
        # Surface backend-provided details (e.g., 400 when model not supported, or non-PDF upload)
        try:
            detail = e.response.json().get("detail")
        except Exception:
            detail = str(e)
        with st.chat_message("assistant"):
            st.error(f"Request failed: {detail}")
            append_message("assistant", "text", f"Request failed: {detail}")

    except requests.exceptions.RequestException as e:
        with st.chat_message("assistant"):
            st.error(f"Request failed: {e}")
            append_message("assistant", "text", f"Request failed: {e}")