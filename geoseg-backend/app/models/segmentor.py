import torch
from torchvision import models
import torchvision.transforms as T
from PIL import Image
import numpy as np

class Segmentor:
    def __init__(self):
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def segment(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)["out"]
        return output.squeeze(0).argmax(0).cpu().numpy()
