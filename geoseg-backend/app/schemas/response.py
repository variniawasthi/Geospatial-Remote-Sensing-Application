from pydantic import BaseModel
from typing import Dict

class SegmentationResponse(BaseModel):
    mask_path: str
    area_stats: Dict[str, float]
