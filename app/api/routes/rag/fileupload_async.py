# app/api/routes/rag/fileupload_async.py
from typing import Annotated
from app.api.common.file_upload import save_file
from fastapi import APIRouter, HTTPException, status, File, UploadFile

router = APIRouter()

@router.post('/upload')
async def file_upload(file: Annotated[UploadFile, File(description="Uploaded PDF Documents")]):
    if file.content_type != 'application/pdf':
        raise HTTPException(
            detail = "Only PDF are supported",
            status_code = status.HTTP_400_BAD_REQUEST
        )

    try:
        await save_file(file)
    except Exception as e:
        raise HTTPException(
            detail = f"Failed to load PDF: {e}",
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    return {"filename": file.filename, "message": "File uploaded successfully"}
