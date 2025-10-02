import soundfile, wave, av
import numpy as np
from PIL import Image
from typing import Literal
from io import BytesIO
from numpy.typing import NDArray


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
