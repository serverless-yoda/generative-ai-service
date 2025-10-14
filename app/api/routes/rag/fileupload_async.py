# app/api/routes/rag/fileupload_async.py
from typing import Annotated
from app.api.common.file_upload import save_file
from fastapi import (
        APIRouter, 
        HTTPException, 
        status, 
        File, 
        UploadFile,
        BackgroundTasks
        )
from pathlib import Path
from app.api.common.pdf_extractor import pdf_to_text_extractor
from app.api.rag.rag_services import vector_service


router = APIRouter()

@router.post('/upload')
async def file_upload(file: Annotated[UploadFile, File(description="Uploaded PDF Documents")],
                     bg_text_processor: BackgroundTasks):
    if file.content_type != 'application/pdf':
        raise HTTPException(
            detail = "Only PDF are supported",
            status_code = status.HTTP_400_BAD_REQUEST
        )

    try:
        filepath = await save_file(file)
        
        # Derive the text path safely
        text_path = str(Path(filepath).with_suffix('.txt'))

        # Schedule background work (PASS CALLABLE + ARGS)
        bg_text_processor.add_task(pdf_to_text_extractor, filepath)
        bg_text_processor.add_task(vector_service.store_content_in_db, text_path, 512, "knowledgebase", 768)

        
    except Exception as e:
        raise HTTPException(
            detail = f"Failed to load PDF: {e}",
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    return {"filename": file.filename, "message": "File uploaded successfully"}
