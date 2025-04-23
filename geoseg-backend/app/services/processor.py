from app.models.segmentor import Segmentor
from app.utils.image_tools import save_mask, compute_area_stats
from app.schemas.response import SegmentationResponse

segmentor = Segmentor()

def process_segmentation(image_path: str) -> SegmentationResponse:
    mask = segmentor.segment(image_path)
    mask_path = save_mask(mask)
    stats = compute_area_stats(mask)

    return SegmentationResponse(mask_path=mask_path, area_stats=stats)
