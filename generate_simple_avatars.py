"""Script for generating simple avatars using Stable Diffusion with ControlNet"""

import torch
from cv2 import cv2
from diffusers import StableDiffusionControlNetPipeline

from avatars.common_tools import (
    ConditionalType,
    get_all_conditional_creators,
    get_all_controlnet_models,
    load_pipeline,
)
from avatars.simple.image_processor import ImageProcessorControlNet

if __name__ == "__main__":
    torch_dtype = torch.float16
    device = torch.device("cuda:2")

    _controlnet_model_names = [
        "lllyasviel/sd-controlnet-canny",
    ]
    _sd_model_id = "XpucT/Deliberate"

    _conditional_types = [ConditionalType.CANNY]

    _controlnet_models = get_all_controlnet_models(_controlnet_model_names)
    _conditional_creators = get_all_conditional_creators(
        conditional_types=_conditional_types
    )
    sd_model = load_pipeline(
        pipeline_cls=StableDiffusionControlNetPipeline,
        controlnet=_controlnet_models,
        sd_model_id=_sd_model_id,
        torch_dtype=torch_dtype,
        disable_tqdm=True,
        device=torch.device(device),
    )

    if not isinstance(sd_model, StableDiffusionControlNetPipeline):
        raise ValueError("sd_model is not StableDiffusionControlNetPipeline")

    image_processor = ImageProcessorControlNet(
        sd_model=sd_model,
        conditional_creators=_conditional_creators,
    )

    image_path = "images/face1.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    _prompt = "a male, (stone skin cracked:1.4), (intricate details:1.22), hdr, (intricate details, hyperdetailed:1.2), whole body, cinematic, intense, cinematic composition, cinematic lighting, (rim lighting:1.3), color grading, focused"
    _negative_prompt = "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"
    _controlnet_conditioning_scale = [0.65]
    _num_inference_steps = 30
    _guidance_scale = 7.5

    output_image = image_processor.process(
        image=image,
        prompt=_prompt,
        negative_prompt=_negative_prompt,
        controlnet_conditioning_scale=_controlnet_conditioning_scale,
        num_inference_steps=_num_inference_steps,
        guidance_scale=_guidance_scale,
    )

    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("images/face1_out.jpg", output_image)
