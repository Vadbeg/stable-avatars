"""Module with coomon tools for CLIs"""

import enum
from pathlib import Path
from typing import List, Type

import torch
from cv2 import cv2
from diffusers import ControlNetModel, DiffusionPipeline, StableDiffusionImg2ImgPipeline
from numpy.typing import NDArray

from avatars.cond_creators.conditional_creators import (
    BaseConditionalCreator,
    CannyConditionalCreator,
    DepthConditionalDetector,
    FaceMaskConditionalCreator,
    FacialLandmarksConditionalCreator,
    HEDConditionalDetector,
    LineArtConditionalCreator,
    MediaPipeFaceConditionalCreator,
    NormalBaeConditionalCreator,
    OpenPoseConditionalDetector,
    SoftEdgeConditionalDetector,
)


class ConditionalType(enum.Enum):
    CANNY = "canny"
    HED = "hed"
    DEPTH = "depth"
    OPENPOSE = "openpose"
    SCRIBBLE = "scribble"
    SOFTEDGE = "softedge"
    FACE_LANDMARK = "face_landmark"
    MEDIAPIPE_FACE_LANDMARK = "mediapipe_face_landmark"
    LINEART = "lineart"
    FACE_MASK = "face_mask"
    ONLY_FACE_MASK = "only_face_mask"
    ONLY_FACE_MASK_NO_SKIN = "only_face_mask_no_skin"
    NORMAL_BAE = "normal_bae"


def _get_conditional_creator(
    conditional_type: ConditionalType,
) -> BaseConditionalCreator:
    if conditional_type == ConditionalType.CANNY:
        return CannyConditionalCreator(
            low_threshold=100,
            high_threshold=200,
        )
    elif conditional_type == ConditionalType.HED:
        return HEDConditionalDetector(
            pretrained_model_or_path="lllyasviel/Annotators",
        )
    elif conditional_type == ConditionalType.DEPTH:
        return DepthConditionalDetector()
    elif conditional_type == ConditionalType.OPENPOSE:
        return OpenPoseConditionalDetector(
            pretrained_model_or_path="lllyasviel/Annotators",
        )
    elif conditional_type == ConditionalType.SCRIBBLE:
        return HEDConditionalDetector(
            pretrained_model_or_path="lllyasviel/Annotators",
            scribble=True,
        )
    elif conditional_type == ConditionalType.SOFTEDGE:
        return SoftEdgeConditionalDetector(
            pretrained_model_or_path="lllyasviel/Annotators",
        )
    elif conditional_type == ConditionalType.LINEART:
        return LineArtConditionalCreator(
            pretrained_model_or_path="lllyasviel/Annotators",
        )
    elif conditional_type == ConditionalType.NORMAL_BAE:
        return NormalBaeConditionalCreator(
            pretrained_model_or_path="lllyasviel/Annotators",
        )
    elif conditional_type == ConditionalType.FACE_LANDMARK:
        return FacialLandmarksConditionalCreator()
    elif conditional_type == ConditionalType.MEDIAPIPE_FACE_LANDMARK:
        return MediaPipeFaceConditionalCreator()
    elif conditional_type == ConditionalType.FACE_MASK:
        return FaceMaskConditionalCreator(
            pretrained_model_or_path=Path("checkpoints/fpn.resnet50.1024x1024.jit.pth"),
            inference_on_crop=True,
        )
    elif conditional_type == ConditionalType.ONLY_FACE_MASK:
        return FaceMaskConditionalCreator(
            pretrained_model_or_path=Path("checkpoints/fpn.resnet50.1024x1024.jit.pth"),
            only_face=True,
            inference_on_crop=True,
        )
    elif conditional_type == ConditionalType.ONLY_FACE_MASK_NO_SKIN:
        return FaceMaskConditionalCreator(
            pretrained_model_or_path=Path("checkpoints/fpn.resnet50.1024x1024.jit.pth"),
            only_face=True,
            no_skin=True,
            inference_on_crop=True,
        )
    else:
        raise ValueError(f"Conditional type {conditional_type} is not supported yet")


def get_all_conditional_creators(
    conditional_types: List[ConditionalType],
) -> List[BaseConditionalCreator]:
    return [
        _get_conditional_creator(curr_conditional_type)
        for curr_conditional_type in conditional_types
    ]


def get_all_controlnet_models(
    controlnet_model_ids: List[str],
    torch_dtype: torch.dtype = torch.float32,
) -> List[ControlNetModel]:
    all_controlnet_models = []

    for curr_controlnet_model_id in controlnet_model_ids:
        if Path(curr_controlnet_model_id).exists():
            curr_controlnet_model = ControlNetModel.from_pretrained(
                Path(curr_controlnet_model_id),
                torch_dtype=torch_dtype,
            )
        elif curr_controlnet_model_id == "CrucibleAI/ControlNetMediaPipeFace":
            curr_controlnet_model = ControlNetModel.from_pretrained(
                curr_controlnet_model_id,
                torch_dtype=torch_dtype,
                subfolder="diffusion_sd15",
            )
        else:
            curr_controlnet_model = ControlNetModel.from_pretrained(
                curr_controlnet_model_id,
                torch_dtype=torch_dtype,
            )

        all_controlnet_models.append(curr_controlnet_model)

    return all_controlnet_models


def check_controlnet_model_and_conditional_type_match(
    controlnet_model_ids: List[str], conditional_types: List[ConditionalType]
) -> None:
    for curr_controlnet_model_id, curr_conditional_type in zip(
        controlnet_model_ids, conditional_types
    ):
        if curr_conditional_type.value not in curr_controlnet_model_id:
            raise ValueError(
                f"Conditional type {curr_conditional_type} does "
                f"not match ControlNet model ID {curr_controlnet_model_id}"
            )


def resize_image_with_proportions(
    image: NDArray, max_height: int, max_width: int
) -> NDArray:
    """Resize image with proportions"""
    height, width = image.shape[:2]
    if height > width:
        new_height = max_height
        new_width = int(width * (new_height / height))
    else:
        new_width = max_width
        new_height = int(height * (new_width / width))

    return cv2.resize(image, (new_width, new_height))


def load_pipeline(
    pipeline_cls: Type[DiffusionPipeline],
    sd_model_id: str,
    controlnet: List[ControlNetModel],
    torch_dtype: torch.dtype = torch.float32,
    disable_tqdm: bool = True,
    device: torch.device = torch.device("cpu"),
) -> DiffusionPipeline:

    if sd_model_id.endswith(".safetensors") or sd_model_id.endswith(".ckpt"):
        if pipeline_cls == StableDiffusionImg2ImgPipeline:
            sd_model = pipeline_cls.from_ckpt(
                sd_model_id,
                torch_dtype=torch_dtype,
                disable_tqdm=disable_tqdm,
                use_safetensors=True,
            )
        else:
            pipeline_cls.__name__ = "StableDiffusionControlNetPipeline"
            sd_model = pipeline_cls.from_ckpt(
                sd_model_id,
                controlnet=controlnet,
                torch_dtype=torch_dtype,
                disable_tqdm=disable_tqdm,
                use_safetensors=True,
            )
    else:
        if pipeline_cls == StableDiffusionImg2ImgPipeline:
            sd_model = pipeline_cls.from_pretrained(
                sd_model_id,
                torch_dtype=torch_dtype,
                disable_tqdm=disable_tqdm,
            )
        else:
            sd_model = pipeline_cls.from_pretrained(
                sd_model_id,
                controlnet=controlnet,
                torch_dtype=torch_dtype,
                disable_tqdm=disable_tqdm,
            )

    sd_model.safety_checker = None
    sd_model.to(device)

    return sd_model
