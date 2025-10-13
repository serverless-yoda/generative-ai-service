# app/api/common/pdf_extractor.py

from app.api.common import file_upload
from pypdf import PdfReader

def pdf_to_text_extractor(filepath: str) -> str:
    content = ""
    pdf_reader = PdfReader(filepath, strict=True)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            content += f'{page_text}\n\n'
    with open(filepath.replace('pdf','txt'),'w',encoding='utf-8') as file:
        file.write(content)
