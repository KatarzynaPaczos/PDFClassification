from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from src.functions import classify_with_clip, detect_form_type
from src.functions import extract_year, extract_text
app = FastAPI()
    
@app.post("/classify")
async def schedule_classify_task(file: Optional[UploadFile] = File(None)):
    """Endpoint to classify a document"""

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        file_bytes = await file.read()

        # first classification
        doc_type = classify_with_clip(file_bytes)
        if doc_type != "form":
            text = extract_text(file_bytes)
            return {"document_type": doc_type, "year": extract_year(text)}
        
        # now only forms left (or OTHER)
        text = extract_text(file_bytes)
        doc_type = detect_form_type(text)
        return {"document_type": doc_type, "year": extract_year(text)}
    
    except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")