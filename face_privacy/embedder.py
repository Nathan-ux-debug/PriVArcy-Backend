"""Face embedding.

Turns a cropped face image into a fixed-length feature vector so two
faces can be compared numerically. Uses OpenCV's built-in HOG (Histogram
of Oriented Gradients) descriptor — no extra dependency or model
download, unlike a deep embedding network (FaceNet/ArcFace).

Trade-off, stated plainly: HOG captures edge/gradient structure, not a
learned notion of "identity" the way a deep face-recognition network
does. It works reasonably for a small enrolled set under fairly
consistent conditions (similar angle/lighting to the enrollment photos),
but is meaningfully less robust than a deep embedding model to pose,
lighting, and expression changes. See the README for the upgrade path.
"""

import logging

import cv2
import numpy as np

from .exceptions import FacePrivacyError

logger = logging.getLogger(__name__)

# Fixed size every face crop is resized to before computing HOG features,
# so all embeddings are the same length and directly comparable.
_FACE_SIZE = (96, 96)
_HOG = cv2.HOGDescriptor(_winSize=_FACE_SIZE, _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)


class FaceEmbedder:
    """Computes a HOG-based embedding vector for a cropped face image."""

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        """Return a 1-D, L2-normalized feature vector for one face crop.

        Args:
            face_crop: BGR (or grayscale) NumPy array containing just the
                        face region — e.g. frame[y1:y2, x1:x2] from a
                        FaceBox.

        Raises:
            FacePrivacyError: if face_crop is empty/invalid.
        """
        if face_crop is None or not isinstance(face_crop, np.ndarray) or face_crop.size == 0:
            raise FacePrivacyError("embed() received an empty or invalid face crop.")

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
        resized = cv2.resize(gray, _FACE_SIZE)
        # Light histogram equalization helps normalize lighting differences
        # between the enrollment photo and a later live frame.
        equalized = cv2.equalizeHist(resized)

        vector = _HOG.compute(equalized).flatten().astype(np.float64)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
