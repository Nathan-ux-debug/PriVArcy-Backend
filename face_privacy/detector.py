"""Face detection.

Uses OpenCV's built-in Haar cascade classifier — ships with opencv-python
(no model download needed), runs on CPU, and is fast enough for real-time
at the cost of being less accurate than a modern deep-learning face
detector, especially off-angle or low-light. See the README for the
upgrade path (e.g. cv2's DNN-based YuNet face detector) if accuracy here
turns out to be the bottleneck.
"""

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .exceptions import FacePrivacyError
from .models import FaceBox

logger = logging.getLogger(__name__)


class FaceDetector:
    """Finds face bounding boxes in a frame.

    Parameters:
        cascade_path: Path to a Haar cascade XML file. Defaults to
                       OpenCV's bundled frontal-face cascade.
        scale_factor: How much the image size is reduced at each scale
                       (smaller = more thorough but slower). Default 1.1.
        min_neighbors: How many overlapping detections are required to
                        keep a face (higher = fewer false positives, but
                        can miss real faces). Default 5.
        min_size_px: Smallest face (width, height) in pixels to detect —
                      filters out tiny false-positive blobs.
    """

    def __init__(
        self,
        cascade_path: str | Path | None = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size_px: tuple[int, int] = (30, 30),
    ) -> None:
        if scale_factor <= 1.0:
            raise FacePrivacyError(f"scale_factor must be > 1.0, got {scale_factor}")
        if min_neighbors < 1:
            raise FacePrivacyError(f"min_neighbors must be >= 1, got {min_neighbors}")

        path = str(cascade_path) if cascade_path else cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise FacePrivacyError(f"Could not load Haar cascade from '{path}'.")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size_px = min_size_px

    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        """Return every detected face in the frame.

        Raises:
            FacePrivacyError: if frame is invalid.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise FacePrivacyError("detect() received an empty or invalid frame array.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        # detectMultiScale3 also returns reject levels / weights, which we
        # use as a rough confidence proxy — plain detectMultiScale doesn't.
        boxes, _reject_levels, weights = self._cascade.detectMultiScale3(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size_px,
            outputRejectLevels=True,
        )

        faces: List[FaceBox] = []
        for (x, y, w, h), weight in zip(boxes, weights):
            # `weight` is an unbounded positive score, not a probability;
            # squash it into a rough [0,1] confidence proxy for consistency
            # with the rest of the pipeline (yolo_detection's confidences).
            confidence = float(min(1.0, weight / 10.0))
            faces.append(FaceBox(bbox=(float(x), float(y), float(x + w), float(y + h)), confidence=confidence))

        return faces
