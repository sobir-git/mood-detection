from typing import Sequence, List, Optional

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import transforms as AT
from cv2 import cv2

from face_crop import FaceCropper, FaceNotFoundError
from utils import MoodModel


class ImagePipeline:
    fc = FaceCropper()

    def __init__(self):
        pass

    def _crop_face(self, img: np.ndarray) -> Optional[np.ndarray]:
        try:
            img = self.fc.crop_face(img)
        except FaceNotFoundError:
            img = None
        return img

    def crop_faces(self, images: Sequence[np.ndarray]) -> Sequence[Optional[np.ndarray]]:
        r = [self._crop_face(img) for img in images]
        return r

    def to_gray(self, images: Sequence[np.ndarray]) -> Sequence[Optional[np.ndarray]]:
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None else None
                  for img in images]
        return images

    def make_float(self, images: Sequence[np.ndarray]) -> Sequence[Optional[np.ndarray]]:
        images = [(img.astype(np.float32) / 255)[:, :, None] if img is not None else None
                  for img in images]
        return images

    def from_camera(self, images: Sequence[np.ndarray]) -> List[Optional[torch.Tensor]]:
        assert images[0].ndim == 3

        # crop faces
        images = self.crop_faces(images)

        # convert to grayscale
        images = self.to_gray(images)

        # make float
        images = self.make_float(images)

        # apply transforms, converts to tensor
        images = self.transform(images)

        return images

    def transform(self, images: Sequence[np.ndarray]) -> List[Optional[torch.Tensor]]:
        """
        Transform cropped, gray, float images.
        """
        for img in images:
            if img is not None:
                assert img.dtype == np.float32
                assert img.shape[2] == 1, img.shape

        T = A.Compose([
            A.Resize(48, 48),
            AT.ToTensor(),
        ])
        images = [T(image=img)['image'].expand(3, -1, -1) if img is not None else None
                  for img in images]
        return images


class NNPredictor:
    ip = ImagePipeline()

    def __init__(self, moodmodel: MoodModel):
        self.net = moodmodel

    @torch.no_grad()
    def compute_mood(self, images: Sequence[np.ndarray]) -> List[Optional[float]]:
        """
        Take raw images, crop the face, predict happiness.
        """
        assert images[0].ndim == 3

        self.net.eval()

        images = self.ip.from_camera(images)

        mask = [img is not None for img in images]
        images = [img for img in images if img is not None]
        if len(images) == 0:
            return [None] * len(mask)
        images = torch.stack(images, dim=0)

        predictions = self.net(images)
        predictions = predictions.flatten().tolist()

        result = []
        j = 0
        for i in range(len(mask)):
            if mask[i]:
                result.append(predictions[j])
                j += 1
            else:
                result.append(None)
        return result
