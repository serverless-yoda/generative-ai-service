# generative-ai-service/app/api/core/utils.py

import soundfile, wave, av, os, tempfile
import numpy as np
import open3d as o3d
import torch

from PIL import Image
from typing import Literal
from io import BytesIO
from numpy.typing import NDArray
from pathlib import Path

from diffusers.pipelines.shap_e.renderer import MeshDecoderOutput



def export_to_image_buffer(image: Image.Image) -> BytesIO:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def mesh_to_obj_buffer(mesh):
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.verts.cpu().detach().numpy())
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces.cpu().detach().numpy())

    # Handle vertex colors if present
    if hasattr(mesh, "vertex_channels") and len(mesh.vertex_channels) == 3:
        vert_color = torch.stack([mesh.vertex_channels[channel] for channel in "RGB"], dim=1)
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vert_color.cpu().detach().numpy())

    # Create a temporary file and close its handle immediately after getting the name
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".obj")
    tmp_path = tmp.name
    tmp.close()  # Close so Open3D can write to it on Windows

    try:
        o3d.io.write_triangle_mesh(tmp_path, mesh_o3d, write_ascii=True)
        with open(tmp_path, "rb") as f:
            buffer = BytesIO(f.read())
    finally:
        os.remove(tmp_path)

    return buffer


def export_to_video_buffer(images: list[Image.Image]) -> BytesIO:
    buffer = BytesIO()
    output = av.open(buffer, "w", format="mp4")
    stream = output.add_stream("h264", 30)
    stream.width = images[0].width
    stream.height = images[0].height
    stream.pix_fmt = "yuv444p"
    stream.options = {"crf": "17"}
    for image in images:
        frame = av.VideoFrame.from_image(image)
        packet = stream.encode(frame)
        output.mux(packet)
    packet = stream.encode(None)
    output.mux(packet)
    buffer.seek(0)
    return buffer


def img_to_bytes(image: Image.Image, img_format: Literal["PNG", "JPEG"] = "PNG") -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=img_format)
    return buffer.getvalue()


def audio_array_to_buffer(audio_array: np.array, sample_rate: int) -> BytesIO:
    buffer = BytesIO()
    soundfile.write(buffer, audio_array, sample_rate, format="wav")
    buffer.seek(0)
    return buffer


def float32_to_wav_bytes(audio: NDArray[np.float32], sample_rate: int) -> bytes:
    audio_int16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    buf.seek(0)
    return buf.read()
