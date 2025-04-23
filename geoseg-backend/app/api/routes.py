from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.services.processor import process_segmentation
from app.schemas.response import SegmentationResponse
import tempfile
import os

router = APIRouter()

@router.post("/segment/", response_model=SegmentationResponse)
async def segment_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = process_segmentation(tmp_path)
    finally:
        os.unlink(tmp_path)

    return result
