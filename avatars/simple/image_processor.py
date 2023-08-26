import os
import random
from typing import List, Optional

import numpy as np
import torch
from cv2 import cv2
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
)
from numpy.typing import NDArray
from PIL import Image

from avatars.cond_creators.conditional_creators import BaseConditionalCreator


class ImageProcessorControlNet:
    def __init__(
        self,
        conditional_creators: List[BaseConditionalCreator],
        sd_model: StableDiffusionControlNetPipeline,
    ):
        """
        Processes each image with SD and ControlNet.
        Supports multiple ControlNet conditional images.
        But remember to pass ControlNet conditional creators in the same order as the ControlNet models.

        :param conditional_creators: list of conditional creators, can be only one for single ControlNet model
        :param sd_model: Stable Diffusion model
        """

        self._conditional_creators = conditional_creators
        self._sd_model = sd_model

    def process(
        self,
        image: NDArray,
        prompt: str,
        controlnet_conditioning_scale: List[float],
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
    ) -> NDArray:
        seed_everything(27)

        conditional_images_pil = []
        for curr_conditional_creator in self._conditional_creators:
            curr_conditional_image = curr_conditional_creator(image)
            curr_conditional_image_pil = Image.fromarray(curr_conditional_image)
            conditional_images_pil.append(curr_conditional_image_pil)

        out_image_pil = self._sd_model(
            prompt=prompt,
            image=conditional_images_pil,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images[0]
        out_image = np.array(out_image_pil)
        out_image = cv2.resize(out_image, (image.shape[1], image.shape[0]))

        return out_image


def seed_everything(seed: int) -> int:
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random In addition,
    sets the following environment variables:

    - `PL_GLOBAL_SEED`: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - `PL_SEED_WORKERS`: (optional) is set to 1 if ``workers=True``.

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
    """

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed
