"""
cv_pipeline.py
==============
All OpenCV image-processing logic for the
"Smart Classroom Face Detection and Attendance Monitor".

This module contains NO GUI code, so it can be tested on its own
(see test_pipeline.py). The GUI in main.py only calls these functions.

CSC 126 laboratory topics demonstrated here
-------------------------------------------
A. Image Operations
     - capture_frame()      : read a frame from the webcam
     - resize_frame()       : scale the frame to a fixed working width
     - preprocess()         : BGR -> grayscale, brightness measurement,
                              CLAHE contrast enhancement
C. Haar Cascade Classifier
     - load_face_cascade()  : load OpenCV's pre-trained frontal-face XML
     - detect_faces()       : cascade.detectMultiScale() on the gray image
D. Image Analysis and Transformation
     - analyze_faces()      : per-face analysis (ordering, size, % of frame)
     - CountSmoother        : temporal analysis of the face count
     - lighting_status()    : interprets the measured brightness
     - draw_results()       : bounding boxes, labels, and overlays
     - prepare_face()       : crop + resize each face for recognition

Recognition itself (LBPH) lives in recognizer.py; this module only calls it.
"""

import os
import sys
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import List, Optional, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
WORKING_WIDTH = 800          # frames are resized to this width (if larger)
SCALE_FACTOR = 1.1           # image-pyramid step used by detectMultiScale
MIN_NEIGHBORS = 6            # higher = fewer false positives, may miss faces
MIN_FACE_FRACTION = 0.06     # smallest face = 6% of frame width (min 40 px)
SMOOTHING_WINDOW = 7         # number of frames used to stabilise the count

LOW_LIGHT_THRESHOLD = 70     # mean gray value below this = too dark
HIGH_LIGHT_THRESHOLD = 195   # mean gray value above this = too bright

FACE_SIZE = (200, 200)       # every face crop is resized to this for recognition

GREEN = (0, 200, 0)          # BGR colours used for drawing
RED = (0, 0, 220)
WHITE = (255, 255, 255)
DARK = (30, 30, 30)


# --------------------------------------------------------------------------
# Custom exceptions (so the GUI can show friendly error messages)
# --------------------------------------------------------------------------
class CameraError(Exception):
    """Raised when the webcam cannot be opened or read."""


class CascadeError(Exception):
    """Raised when the Haar Cascade XML file is missing or invalid."""


# --------------------------------------------------------------------------
# Data structure for one detected face
# --------------------------------------------------------------------------
@dataclass
class Face:
    """Result of analysing one detected face region."""
    face_id: int        # 1-based label, ordered left to right
    x: int
    y: int
    w: int
    h: int
    area_percent: float  # face area as a percentage of the whole frame
    # Filled in by the recogniser (None / "" = unknown person)
    student_id: Optional[int] = None
    name: str = ""
    confidence: float = 0.0
    crop: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


# ==========================================================================
# A. IMAGE OPERATIONS
# ==========================================================================
def open_camera(index: int = 0) -> cv2.VideoCapture:
    """Open the webcam and verify that it really delivers frames."""
    # CAP_DSHOW opens faster and more reliably on Windows.
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    try:
        capture = cv2.VideoCapture(index, backend)
    except Exception as exc:  # unusual driver-level failure
        raise CameraError(f"Camera initialisation failed: {exc}") from exc

    if not capture.isOpened():
        capture.release()
        raise CameraError(
            "Webcam not available. Check that it is connected, not used by "
            "another application, and that camera permission is allowed."
        )

    # A camera can "open" but still fail to deliver frames, so test one read.
    ok, _ = capture.read()
    if not ok:
        capture.release()
        raise CameraError("The webcam opened but did not return any image.")
    return capture


def capture_frame(capture: cv2.VideoCapture) -> np.ndarray:
    """Grab one BGR frame from the webcam (raises CameraError on failure)."""
    ok, frame = capture.read()
    if not ok or frame is None:
        raise CameraError("Lost connection to the webcam (frame read failed).")
    return frame


def release_camera(capture: Optional[cv2.VideoCapture]) -> None:
    """Safely release the webcam (safe to call more than once)."""
    if capture is not None:
        try:
            capture.release()
        except Exception:
            pass


def resize_frame(frame: np.ndarray, width: int = WORKING_WIDTH) -> np.ndarray:
    """Shrink large frames to a fixed width, keeping the aspect ratio.

    Smaller frames are faster to process, which keeps the live feed smooth.
    """
    height, current_width = frame.shape[:2]
    if current_width <= width:
        return frame
    scale = width / current_width
    return cv2.resize(frame, (width, int(height * scale)),
                      interpolation=cv2.INTER_AREA)


_clahe = None  # created once and reused for every frame


def preprocess(frame: np.ndarray) -> Tuple[np.ndarray, float]:
    """Prepare a BGR frame for the Haar Cascade.

    Steps:
      1. Convert BGR -> grayscale (Haar features only use intensity).
      2. Measure the average brightness (used for the lighting indicator).
      3. Light Gaussian blur to suppress sensor noise.
      4. CLAHE (adaptive histogram equalisation) to improve contrast in
         uneven classroom lighting.

    Returns (enhanced_gray_image, mean_brightness).
    """
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    enhanced = _clahe.apply(blurred)
    return enhanced, brightness


# ==========================================================================
# C. HAAR CASCADE CLASSIFIER
# ==========================================================================
def load_face_cascade(path: Optional[str] = None) -> cv2.CascadeClassifier:
    """Load OpenCV's pre-trained frontal-face Haar Cascade.

    By default the XML file that ships with opencv-python is used.
    """
    if path is None:
        path = os.path.join(cv2.data.haarcascades, CASCADE_FILENAME)

    if not os.path.isfile(path):
        raise CascadeError(f"Haar Cascade file not found:\n{path}")

    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise CascadeError(f"Haar Cascade file could not be loaded:\n{path}")
    return cascade


def detect_faces(cascade: cv2.CascadeClassifier,
                 gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Run the Haar Cascade on a preprocessed gray image.

    Returns a list of (x, y, w, h) rectangles, one per detected face.
    """
    min_side = max(40, int(gray.shape[1] * MIN_FACE_FRACTION))
    detections = cascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=(min_side, min_side),
    )
    return [tuple(int(v) for v in rect) for rect in detections]


# ==========================================================================
# D. IMAGE ANALYSIS AND TRANSFORMATION
# ==========================================================================
def analyze_faces(rects: List[Tuple[int, int, int, int]],
                  frame_shape: Tuple[int, ...]) -> List[Face]:
    """Analyse each detected face region.

    - Orders faces left to right and gives each a label number.
    - Computes how much of the frame each face occupies (a rough hint of
      how close the person is to the camera).
    """
    frame_area = frame_shape[0] * frame_shape[1]
    faces = []
    for number, (x, y, w, h) in enumerate(sorted(rects, key=lambda r: r[0]), 1):
        faces.append(Face(number, x, y, w, h, 100.0 * (w * h) / frame_area))
    return faces


def prepare_face(gray: np.ndarray, face: Face) -> np.ndarray:
    """Cut the face out of the enhanced gray image and resize it."""
    crop = gray[face.y:face.y + face.h, face.x:face.x + face.w]
    return cv2.resize(crop, FACE_SIZE, interpolation=cv2.INTER_AREA)


def count_faces(faces: List[Face]) -> int:
    """Total number of faces detected in the current frame."""
    return len(faces)


class CountSmoother:
    """Stabilises the face count using the median of the last N frames.

    Haar Cascades sometimes miss a face for a single frame. Showing the raw
    count would make the number flicker, so the displayed count is the
    median of recent frames. It still follows people entering/leaving
    after a short delay (about half a second).
    """

    def __init__(self, window: int = SMOOTHING_WINDOW):
        self._history = deque(maxlen=window)

    def update(self, raw_count: int) -> int:
        self._history.append(raw_count)
        return int(round(median(self._history)))

    def reset(self) -> None:
        self._history.clear()


def lighting_status(brightness: float) -> str:
    """Translate the mean gray level into a human-readable lighting label."""
    if brightness < LOW_LIGHT_THRESHOLD:
        return "Too Dark"
    if brightness > HIGH_LIGHT_THRESHOLD:
        return "Too Bright"
    return "Good"


def draw_results(frame: np.ndarray, faces: List[Face],
                 stable_count: int, lighting: str,
                 recognition: bool = False) -> np.ndarray:
    """Draw bounding boxes, labels, and an information overlay on the frame.

    With recognition on, known students get a green box and their name;
    unrecognised faces get a red box and "Unknown".
    """
    for face in faces:
        if recognition:
            known = face.student_id is not None
            colour = GREEN if known else RED
            label = face.name if known else "Unknown - not registered"
        else:
            colour = GREEN
            label = f"Face Detected #{face.face_id}"

        # Bounding box around the face
        cv2.rectangle(frame, (face.x, face.y),
                      (face.x + face.w, face.y + face.h), colour, 2)

        # Label above the box (filled background keeps it readable)
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(face.y - text_h - baseline - 6, 0)
        cv2.rectangle(frame, (face.x, top),
                      (face.x + text_w + 8, top + text_h + baseline + 6),
                      colour, cv2.FILLED)
        cv2.putText(frame, label, (face.x + 4, top + text_h + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    WHITE if colour == RED else DARK, 1, cv2.LINE_AA)

    # Semi-transparent information bar at the top of the frame
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 36), DARK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    status = "Face Detected" if stable_count > 0 else "No Face Detected"
    info = f"Faces: {stable_count}   |   {status}   |   Lighting: {lighting}"
    cv2.putText(frame, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                WHITE, 1, cv2.LINE_AA)
    return frame


def process_frame(frame: np.ndarray, cascade: cv2.CascadeClassifier,
                  smoother: CountSmoother, recognizer=None, names=None):
    """Run the complete pipeline on one frame.

    recognizer : optional recognizer.FaceRecognizer (None = detection only)
    names      : optional {student_id: name} used for the box labels

    Returns (annotated_frame, raw_count, stable_count, lighting_label, faces).
    """
    frame = resize_frame(frame)                      # A. Image Operations
    gray, brightness = preprocess(frame)             # A. Image Operations
    rects = detect_faces(cascade, gray)              # C. Haar Cascade
    faces = analyze_faces(rects, frame.shape)        # D. Image Analysis
    raw_count = count_faces(faces)                   # D. Image Analysis
    stable_count = smoother.update(raw_count)        # D. Image Analysis
    lighting = lighting_status(brightness)           # D. Image Analysis

    use_recognition = recognizer is not None and recognizer.trained
    for face in faces:
        face.crop = prepare_face(gray, face)
        if use_recognition:
            student_id, confidence = recognizer.predict(face.crop)
            face.confidence = confidence
            if student_id is not None:
                face.student_id = student_id
                face.name = (names or {}).get(student_id, f"ID {student_id}")

    annotated = draw_results(frame, faces, stable_count, lighting,
                             recognition=use_recognition)
    return annotated, raw_count, stable_count, lighting, faces