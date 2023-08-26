"""Module with code for creating conditional images"""

from abc import ABC, abstractmethod
from pathlib import Path

import albumentations as albu
import face_recognition
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from controlnet_aux import (
    HEDdetector,
    NormalBaeDetector,
    OpenposeDetector,
    PidiNetDetector,
)
from cv2 import cv2
from numpy.typing import NDArray
from PIL import Image
from transformers import pipeline

from avatars.cond_creators.lineart_detector import LineartDetector
from avatars.face_detection import MediaPipeFaceAnnotator


class BaseConditionalCreator(ABC):
    """Base class for conditional creators"""

    @abstractmethod
    def __call__(self, image: NDArray) -> NDArray:
        """
        Create a conditional image from an original image
        :param image: *RGB* image with shape (height, width, 3) from cv2.imread
        :return: conditional image
        """

        pass


class CannyConditionalCreator(BaseConditionalCreator):
    def __init__(self, low_threshold: int = 100, high_threshold: int = 200):
        self._low_threshold = low_threshold
        self._high_threshold = high_threshold

    def __call__(self, image: NDArray) -> NDArray:
        conditional_image = cv2.Canny(
            image=image,
            threshold1=self._low_threshold,
            threshold2=self._high_threshold,
        )
        conditional_image = conditional_image[:, :, None]
        conditional_image = np.concatenate(
            [conditional_image, conditional_image, conditional_image], axis=2
        )

        return conditional_image


class HEDConditionalDetector(BaseConditionalCreator):
    def __init__(
        self,
        pretrained_model_or_path: str = "lllyasviel/Annotators",
        scribble: bool = False,
    ):
        self._hed_detector = HEDdetector.from_pretrained(
            pretrained_model_or_path=pretrained_model_or_path,
        )

        self._scribble = scribble

    def __call__(self, image: NDArray) -> NDArray:
        conditional_image = self._hed_detector(
            image, return_pil=False, scribble=self._scribble
        )
        conditional_image = cv2.resize(
            src=conditional_image,
            dsize=(image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        return conditional_image


class SoftEdgeConditionalDetector(BaseConditionalCreator):
    def __init__(self, pretrained_model_or_path: str = "lllyasviel/Annotators"):
        self._pidi_detector = PidiNetDetector.from_pretrained(
            pretrained_model_or_path=pretrained_model_or_path,
        )

    def __call__(self, image: NDArray) -> NDArray:
        conditional_image = self._pidi_detector(image, return_pil=False, safe=True)
        conditional_image = cv2.resize(
            src=conditional_image,
            dsize=(image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        return conditional_image


class OpenPoseConditionalDetector(BaseConditionalCreator):
    def __init__(self, pretrained_model_or_path: str = "lllyasviel/Annotators"):
        self._openpose_detector = OpenposeDetector.from_pretrained(
            pretrained_model_or_path=pretrained_model_or_path,
        )

    def __call__(self, image: NDArray) -> NDArray:
        conditional_image = self._openpose_detector(
            image, hand_and_face=True, return_pil=False
        )
        conditional_image = cv2.resize(
            src=conditional_image,
            dsize=(image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        return conditional_image


class DepthConditionalDetector(BaseConditionalCreator):
    def __init__(self):
        self._depth_estimator = pipeline("depth-estimation")

    def __call__(self, image: NDArray) -> NDArray:
        image_pil = Image.fromarray(image)
        depth_pil = self._depth_estimator(image_pil)["depth"]
        depth = np.array(depth_pil)
        depth = np.array(depth)
        depth = depth[:, :, None]
        depth = np.concatenate([depth, depth, depth], axis=2)

        return depth


class FacialLandmarksConditionalCreator(BaseConditionalCreator):
    def __init__(self):
        pass

    def __call__(self, image: NDArray) -> NDArray:
        face_landmarks = face_recognition.face_landmarks(
            face_image=image, model="large"
        )
        landmarks_conditional_image = self._draw_landmarks_on_empty(
            image=image,
            landmarks=face_landmarks,
        )

        return landmarks_conditional_image

    @staticmethod
    def _draw_landmarks_on_empty(
        image: NDArray,
        landmarks: list[dict[str, list[tuple[int, int]]]],
        radius: int = 3,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> NDArray:
        landmarks_image = np.zeros_like(image)

        for curr_face_landmarks in landmarks:
            for curr_part, curr_landmarks in curr_face_landmarks.items():
                for curr_landmark in curr_landmarks:
                    cv2.circle(
                        img=landmarks_image,
                        center=curr_landmark,
                        radius=radius,
                        color=color,
                        thickness=-1,
                    )

        return landmarks_image


class MediaPipeFaceConditionalCreator(BaseConditionalCreator):
    def __init__(self):
        self._face_detector = MediaPipeFaceAnnotator()

    def __call__(self, image: NDArray) -> NDArray:
        image_pil = Image.fromarray(image)
        image_pil = image_pil.convert(mode="RGB")

        face_landmarks_image = self._face_detector.generate(
            image=image_pil,
            max_faces=1,
        )
        face_landmarks_image = face_landmarks_image.convert(mode="RGB")

        face_landmarks_image = np.array(face_landmarks_image)

        return face_landmarks_image


class MockedConditionalCreator(BaseConditionalCreator):
    """Base class for conditional creators"""

    def __call__(self, image: NDArray) -> NDArray:

        return image


class LineArtConditionalCreator(BaseConditionalCreator):
    def __init__(
        self,
        pretrained_model_or_path: str = "lllyasviel/Annotators",
    ):
        self._lineart_detector = LineartDetector.from_pretrained(
            pretrained_model_or_path=pretrained_model_or_path,
            device=torch.device("cuda"),
        )

    def __call__(self, image: NDArray) -> NDArray:
        conditional_image = self._lineart_detector(
            image,
            return_pil=False,
        )
        conditional_image = cv2.resize(
            src=conditional_image,
            dsize=(image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        return conditional_image


class FaceMaskConditionalCreator(BaseConditionalCreator):
    def __init__(
        self,
        pretrained_model_or_path: Path = Path("checkpoints/face_seg.pt"),
        only_face: bool = False,
        no_skin: bool = False,
        inference_on_crop: bool = False,
    ):
        self._model = torch.jit.load(pretrained_model_or_path)

        self._image_size = (1024, 1024)
        self._prepocess_image = albu.Compose(
            [
                albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        self._device = torch.device("cuda")
        self._model = self._model.to(self._device)

        self._infer_on_crop = inference_on_crop

        self._class_to_color_mappings: dict[int, tuple[int, int, int]] = {
            0: (0, 0, 0),
            1: (255, 255, 255),
            2: (255, 0, 0),
            3: (0, 255, 0),
            4: (0, 0, 255),
            5: (255, 255, 0),
            6: (255, 0, 255),
            7: (0, 255, 255),
            8: (128, 128, 128),
        }

        if only_face:
            self._class_to_color_mappings.pop(1)
        if no_skin:
            self._class_to_color_mappings.pop(2)

    def __call__(self, image: NDArray) -> NDArray:
        output_size = image.shape[:2]

        face_location = None
        original_image_size = image.shape[:2]

        image, pad_x, pad_y, new_width, new_height = self.resize_with_padding(
            image=image,
            size=1024,
        )

        image_tensor = self._prepocess_image(image=image)["image"]
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self._device)

        mask = self._model(image_tensor)
        mask = mask.argmax(dim=1)
        mask_numpy = mask.cpu().numpy()
        mask_numpy = mask_numpy[0]

        mask_numpy = mask_numpy[pad_y : pad_y + new_height, pad_x : pad_x + new_width]
        mask_numpy = cv2.resize(
            src=mask_numpy,
            dsize=(original_image_size[1], original_image_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        conditional_image = self._reformat_on_mask(mask=mask_numpy)

        return conditional_image

    def _reformat_on_mask(self, mask: NDArray) -> NDArray:
        """Reformat mask from the original dataset to the format used by the controlnet model"""
        mask = mask.astype(np.uint8)

        new_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for class_id, color in self._class_to_color_mappings.items():
            mask_class = mask == class_id
            new_mask[mask_class] = np.array(color)

        return new_mask

    @staticmethod
    def resize_with_padding(
        image: NDArray, size: int
    ) -> tuple[NDArray, int, int, int, int]:
        # Calculate the aspect ratio
        height, width, _ = image.shape
        aspect_ratio = width / height

        # Determine the new dimensions
        if aspect_ratio >= 1:
            new_width = size
            new_height = int(size / aspect_ratio)
        else:
            new_height = size
            new_width = int(size * aspect_ratio)

        # Resize the image while maintaining aspect ratio
        resized_img = cv2.resize(image, (new_width, new_height))

        # Create a black background
        background = np.zeros((size, size, 3), dtype=np.uint8)

        # Calculate the padding for the black stripes
        pad_x = (size - new_width) // 2
        pad_y = (size - new_height) // 2

        # Add the resized image to the black background
        background[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized_img

        return background, pad_x, pad_y, new_width, new_height


class NormalBaeConditionalCreator(BaseConditionalCreator):
    def __init__(
        self,
        pretrained_model_or_path: str = "lllyasviel/Annotators",
    ):
        self._normalbae_detector = NormalBaeDetector.from_pretrained(
            pretrained_model_or_path=pretrained_model_or_path,
        )

    def __call__(self, image: NDArray) -> NDArray:
        conditional_image = self._normalbae_detector(
            image,
            return_pil=False,
        )
        conditional_image = cv2.resize(
            src=conditional_image,
            dsize=(image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        return conditional_image
