"""Module with mediapipe face annotator class"""


from typing import Mapping

import mediapipe as mp
import numpy
from PIL import Image


class MediaPipeFaceAnnotator:
    def __init__(self):
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_face_mesh = mp.solutions.face_mesh

        self._drawing_spec = mp.solutions.drawing_styles.DrawingSpec

        f_thick = 2
        f_rad = 1
        right_iris_draw = self._drawing_spec(
            color=(10, 200, 250), thickness=f_thick, circle_radius=f_rad
        )
        right_eye_draw = self._drawing_spec(
            color=(10, 200, 180), thickness=f_thick, circle_radius=f_rad
        )
        right_eyebrow_draw = self._drawing_spec(
            color=(10, 220, 180), thickness=f_thick, circle_radius=f_rad
        )
        left_iris_draw = self._drawing_spec(
            color=(250, 200, 10), thickness=f_thick, circle_radius=f_rad
        )
        left_eye_draw = self._drawing_spec(
            color=(180, 200, 10), thickness=f_thick, circle_radius=f_rad
        )
        left_eyebrow_draw = self._drawing_spec(
            color=(180, 220, 10), thickness=f_thick, circle_radius=f_rad
        )
        mouth_draw = self._drawing_spec(
            color=(10, 180, 10), thickness=f_thick, circle_radius=f_rad
        )
        head_draw = self._drawing_spec(
            color=(10, 200, 10), thickness=f_thick, circle_radius=f_rad
        )

        # mp_face_mesh.FACEMESH_CONTOURS has all the items we care about.
        self._face_connection_spec = {}
        for edge in self._mp_face_mesh.FACEMESH_FACE_OVAL:
            self._face_connection_spec[edge] = head_draw
        for edge in self._mp_face_mesh.FACEMESH_LEFT_EYE:
            self._face_connection_spec[edge] = left_eye_draw
        for edge in self._mp_face_mesh.FACEMESH_LEFT_EYEBROW:
            self._face_connection_spec[edge] = left_eyebrow_draw
        for edge in self._mp_face_mesh.FACEMESH_RIGHT_EYE:
            self._face_connection_spec[edge] = right_eye_draw
        for edge in self._mp_face_mesh.FACEMESH_RIGHT_EYEBROW:
            self._face_connection_spec[edge] = right_eyebrow_draw
        for edge in self._mp_face_mesh.FACEMESH_LIPS:
            self._face_connection_spec[edge] = mouth_draw
        self._iris_landmark_spec = {468: right_iris_draw, 473: left_iris_draw}

    def generate(
        self,
        image: Image.Image,
        max_faces: int,
        min_face_size_pixels: int = 0,
    ) -> Image.Image:
        with self._mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            img_rgb = numpy.asarray(image)
            results = face_mesh.process(img_rgb).multi_face_landmarks

            # Filter faces that are too small
            filtered_landmarks = []
            for lm in results:
                landmarks = lm.landmark
                face_rect = [
                    landmarks[0].x,
                    landmarks[0].y,
                    landmarks[0].x,
                    landmarks[0].y,
                ]  # Left, up, right, down.
                for i in range(len(landmarks)):
                    face_rect[0] = min(face_rect[0], landmarks[i].x)
                    face_rect[1] = min(face_rect[1], landmarks[i].y)
                    face_rect[2] = max(face_rect[2], landmarks[i].x)
                    face_rect[3] = max(face_rect[3], landmarks[i].y)
                if min_face_size_pixels > 0:
                    face_width = abs(face_rect[2] - face_rect[0])
                    face_height = abs(face_rect[3] - face_rect[1])
                    face_width_pixels = face_width * image.size[0]
                    face_height_pixels = face_height * image.size[1]
                    face_size = min(face_width_pixels, face_height_pixels)
                    if face_size >= min_face_size_pixels:
                        filtered_landmarks.append(lm)
                else:
                    filtered_landmarks.append(lm)

            # Annotations are drawn in BGR for some reason, but we don't need to flip a zero-filled image at the start.
            empty = numpy.zeros_like(img_rgb)
            # Draw detected faces:
            for face_landmarks in filtered_landmarks:
                self._mp_drawing.draw_landmarks(
                    empty,
                    face_landmarks,
                    connections=self._face_connection_spec.keys(),
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._face_connection_spec,
                )
                self._draw_pupils(empty, face_landmarks, self._iris_landmark_spec, 2)

            # Flip BGR back to RGB.
            empty = self._reverse_channels(empty)
            empty_pil = Image.fromarray(empty)

            return empty_pil

    def _draw_pupils(self, image, landmark_list, drawing_spec, halfwidth: int = 2):  # type: ignore
        """We have a custom function to draw the pupils because the mp.draw_landmarks method requires a parameter for all
        landmarks.  Until our PR is merged into mediapipe, we need this separate method."""
        if len(image.shape) != 3:
            raise ValueError("Input image must be H,W,C.")
        image_rows, image_cols, image_channels = image.shape
        if image_channels != 3:  # BGR channels
            raise ValueError("Input image must contain three channel bgr data.")
        for idx, landmark in enumerate(landmark_list.landmark):
            if (landmark.HasField("visibility") and landmark.visibility < 0.9) or (
                landmark.HasField("presence") and landmark.presence < 0.5
            ):
                continue
            if (
                landmark.x >= 1.0
                or landmark.x < 0
                or landmark.y >= 1.0
                or landmark.y < 0
            ):
                continue
            image_x = int(image_cols * landmark.x)
            image_y = int(image_rows * landmark.y)
            draw_color = None
            if isinstance(drawing_spec, Mapping):
                if drawing_spec.get(idx) is None:
                    continue
                else:
                    draw_color = drawing_spec[idx].color
            elif isinstance(drawing_spec, self._drawing_spec):
                draw_color = drawing_spec.color
            image[
                image_y - halfwidth : image_y + halfwidth,
                image_x - halfwidth : image_x + halfwidth,
                :,
            ] = draw_color

    @staticmethod
    def _reverse_channels(image):
        """Given a numpy array in RGB form, convert to BGR.  Will also convert from BGR to RGB."""
        # im[:,:,::-1] is a neat hack to convert BGR to RGB by reversing the indexing order.
        # im[:,:,::[2,1,0]] would also work but makes a copy of the data.
        return image[:, :, ::-1]
