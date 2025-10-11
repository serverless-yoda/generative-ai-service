# app/api/common/file_upload.py
import os, aiofiles
from aiofiles.os import makedirs
from fastapi import UploadFile

DEFAULT_CHUNK_SIZE = 1024 * 1024 * 50

async def save_file(file: UploadFile) -> str:
    await makedirs('upload', exist_ok=True)
    file_path = os.path.join('upload', file.filename)
    async with aiofiles.open(file_path, 'wb') as f:
        while chunk := await file.read(DEFAULT_CHUNK_SIZE):
            await f.write(chunk)
    return file_path