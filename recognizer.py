"""
recognizer.py
=============
Face RECOGNITION for the attendance monitor, using OpenCV's LBPH
(Local Binary Patterns Histograms) recogniser.

How it fits together
--------------------
1. During registration, cv_pipeline detects the face (Haar Cascade) and this
   module saves ~30 cropped gray samples to  dataset/<student_id>/.
2. train() reads every sample and trains one LBPH model, saved as
   trainer.yml so it does not need retraining on the next start.
3. predict() compares a new face crop with the model and returns the
   student id, or None when the face is not close enough to anyone
   (= "unrecognised face").

Requires the contrib build of OpenCV (cv2.face):  opencv-contrib-python
"""

import os
import shutil
from typing import Optional, Tuple

import cv2
import numpy as np

from cv_pipeline import FACE_SIZE

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "dataset")
MODEL_PATH = os.path.join(BASE, "trainer.yml")

# LBPH returns a DISTANCE: lower = better match. A face is accepted only if
# its distance is at or below this value. Lower it (e.g. 60) to reduce wrong
# matches; raise it (e.g. 80) if registered students are not being recognised.
CONFIDENCE_THRESHOLD = 70.0


class RecognizerError(Exception):
    """Raised when the recogniser cannot be trained or loaded."""


def is_available() -> bool:
    """True when OpenCV was installed with the contrib `face` module."""
    return hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")


class FaceRecognizer:
    def __init__(self, model_path: str = MODEL_PATH,
                 dataset_dir: str = DATASET_DIR,
                 threshold: float = CONFIDENCE_THRESHOLD):
        if not is_available():
            raise RecognizerError(
                "cv2.face is missing. Install the contrib build:\n"
                "pip uninstall opencv-python\n"
                "pip install opencv-contrib-python")
        self.model_path = model_path
        self.dataset_dir = dataset_dir
        self.threshold = threshold
        self.model = None
        self.trained = False

    # ------------------------------------------------------------------
    def load(self) -> bool:
        """Load a previously trained model. Returns True if one was found."""
        if not os.path.isfile(self.model_path):
            return False
        try:
            model = cv2.face.LBPHFaceRecognizer_create()
            model.read(self.model_path)
        except cv2.error as exc:
            raise RecognizerError(f"Could not load {self.model_path}: {exc}")
        self.model, self.trained = model, True
        return True

    def save_sample(self, student_id: int, face_img: np.ndarray,
                    index: int) -> None:
        folder = os.path.join(self.dataset_dir, str(student_id))
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, f"{index:03d}.png"), face_img)

    def remove_student(self, student_id: int) -> None:
        shutil.rmtree(os.path.join(self.dataset_dir, str(student_id)),
                      ignore_errors=True)

    def train(self) -> int:
        """Train on every saved sample. Returns the number of images used."""
        images, labels = [], []
        if os.path.isdir(self.dataset_dir):
            for folder in sorted(os.listdir(self.dataset_dir)):
                folder_path = os.path.join(self.dataset_dir, folder)
                if not (folder.isdigit() and os.path.isdir(folder_path)):
                    continue
                for name in sorted(os.listdir(folder_path)):
                    img = cv2.imread(os.path.join(folder_path, name),
                                     cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    images.append(cv2.resize(img, FACE_SIZE))
                    labels.append(int(folder))

        if not images:                     # nobody left to recognise
            self.model, self.trained = None, False
            if os.path.isfile(self.model_path):
                os.remove(self.model_path)
            return 0

        try:
            model = cv2.face.LBPHFaceRecognizer_create()
            model.train(images, np.array(labels, dtype=np.int32))
            model.write(self.model_path)
        except cv2.error as exc:
            raise RecognizerError(f"Training failed: {exc}")
        self.model, self.trained = model, True
        return len(images)

    def predict(self, face_img: np.ndarray) -> Tuple[Optional[int], float]:
        """Return (student_id, distance); student_id is None if unknown."""
        if not self.trained:
            return None, 0.0
        label, distance = self.model.predict(face_img)
        if distance <= self.threshold:
            return int(label), float(distance)
        return None, float(distance)