from typing import Optional

import cv2
import numpy as np
import torch
from controlnet_aux.lineart import Generator
from controlnet_aux.open_pose.util import HWC3, resize_image
from einops import rearrange
from huggingface_hub import hf_hub_download
from PIL import Image


class LineartDetector:
    def __init__(
        self,
        model: torch.nn.Module,
        # coarse_model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model.eval()
        # self.model_coarse = coarse_model.eval()

        self.model.to(device)
        # self.model_coarse.to(device)

    @classmethod
    def from_pretrained(  # type: ignore
        cls,
        pretrained_model_or_path: str,
        filename: Optional[str] = None,
        coarse_filename: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ):  # type: ignore
        filename = filename or "sk_model.pth"
        # coarse_filename = coarse_filename or "sk_model2.pth"

        model_path = hf_hub_download(
            pretrained_model_or_path, filename, cache_dir=cache_dir
        )
        # coarse_model_path = hf_hub_download(
        #     pretrained_model_or_path, coarse_filename, cache_dir=cache_dir
        # )

        model = Generator(3, 1, 3)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

        # coarse_model = Generator(3, 1, 3)
        # coarse_model.load_state_dict(
        #     torch.load(coarse_model_path, map_location=torch.device("cpu"))
        # )

        return cls(
            model,
            # coarse_model,
            device,
        )

    def __call__(
        self,
        input_image,
        coarse=False,
        detect_resolution=512,
        image_resolution=512,
        return_pil=True,
    ):
        device = next(iter(self.model.parameters())).device
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        # model = self.model_coarse if coarse else self.model
        assert input_image.ndim == 3
        image = input_image
        with torch.no_grad():
            image = torch.from_numpy(image).float().to(device)
            image = image / 255.0
            image = rearrange(image, "h w c -> 1 c h w")
            # line = model(image)[0][0]
            line = self.model(image)[0][0]

            line = line.cpu().numpy()
            line = (line * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = line

        detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        detected_map = 255 - detected_map

        if return_pil:
            detected_map = Image.fromarray(detected_map)

        return detected_map
