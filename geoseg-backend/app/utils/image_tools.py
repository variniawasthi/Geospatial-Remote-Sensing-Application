import numpy as np
from PIL import Image
import os
import uuid
from app.core.config import settings

LABELS = {
    0: "background",
    2: "building",
    3: "road",
    5: "vegetation",
    6: "water"
}

COLORS = {
    "background": [0, 0, 0],
    "building": [128, 0, 0],
    "road": [128, 64, 128],
    "vegetation": [0, 128, 0],
    "water": [0, 0, 255]
}

def save_mask(mask_array: np.ndarray) -> str:
    height, width = mask_array.shape
    mask_color = np.zeros((height, width, 3), dtype=np.uint8)

    for label_id, label_name in LABELS.items():
        color = COLORS.get(label_name, [255, 255, 255])
        mask_color[mask_array == label_id] = color

    os.makedirs(settings.MASK_OUTPUT_DIR, exist_ok=True)
    mask_path = os.path.join(settings.MASK_OUTPUT_DIR, f"{uuid.uuid4()}.png")
    Image.fromarray(mask_color).save(mask_path)

    return mask_path

def compute_area_stats(mask_array: np.ndarray) -> dict:
    total_pixels = mask_array.size
    stats = {}
    for label_id, label_name in LABELS.items():
        count = (mask_array == label_id).sum()
        stats[label_name] = round((count / total_pixels) * 100, 2)
    return stats
