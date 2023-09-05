"""Script for generating simple avatars using Stable Diffusion with ControlNet"""

import torch
from cv2 import cv2
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
)

from avatars.common_tools import (
    ConditionalType,
    get_all_conditional_creators,
    get_all_controlnet_models,
    load_pipeline,
)
from avatars.simple.image_processor import ImageProcessorControlNetImg2Img

if __name__ == "__main__":
    torch_dtype = torch.float16
    device = torch.device("cuda:2")

    _controlnet_model_names = [
        "lllyasviel/control_v11p_sd15_lineart",
    ]
    _sd_model_id = "XpucT/Deliberate"

    _conditional_types = [ConditionalType.LINEART]

    _controlnet_models = get_all_controlnet_models(
        _controlnet_model_names, torch_dtype=torch_dtype
    )
    _conditional_creators = get_all_conditional_creators(
        conditional_types=_conditional_types
    )
    sd_model = load_pipeline(
        pipeline_cls=StableDiffusionControlNetImg2ImgPipeline,
        controlnet=_controlnet_models,
        sd_model_id=_sd_model_id,
        torch_dtype=torch_dtype,
        disable_tqdm=True,
        device=torch.device(device),
    )

    if not isinstance(sd_model, StableDiffusionControlNetImg2ImgPipeline):
        raise ValueError("sd_model is not StableDiffusionControlNetImg2ImgPipeline")

    image_processor = ImageProcessorControlNetImg2Img(
        sd_model=sd_model,
        conditional_creators=_conditional_creators,
    )

    image_path = "images/face1.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    _prompt = "a man in barbie style, perfect face, nice skeen, pink clothing, green eyes, cinematic composition, cinematic lighting, focused"
    _negative_prompt = "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"
    _controlnet_conditioning_scale = [1.0]
    _num_inference_steps = 30
    _guidance_scale = 7.5
    _image_strength = 0.6

    original_height, original_width = image.shape[:2]
    max_height = 1024
    max_width = 1024

    scale = min(max_width / original_width, max_height / original_height)
    width = int(original_width * scale)
    height = int(original_height * scale)

    image = cv2.resize(image, (width, height))

    output_image = image_processor.process(
        image=image,
        prompt=_prompt,
        negative_prompt=_negative_prompt,
        controlnet_conditioning_scale=_controlnet_conditioning_scale,
        num_inference_steps=_num_inference_steps,
        guidance_scale=_guidance_scale,
        image_strength=_image_strength,
    )

    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("images/face1_res.jpg", output_image)
