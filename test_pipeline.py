"""
test_pipeline.py
================
Offline test of the OpenCV pipeline (no webcam and no GUI needed).

Runs the same pipeline used by the live app on every image found in
sample_inputs/ and saves the annotated results to screenshots/.
It is useful when no webcam is available and as extra evidence for the
documentation.

Run:  python test_pipeline.py
"""

import glob
import os

import cv2

import cv_pipeline as cvp

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE, "sample_inputs")
OUTPUT_DIR = os.path.join(BASE, "screenshots")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cascade = cvp.load_face_cascade()

    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")) +
                   glob.glob(os.path.join(INPUT_DIR, "*.png")))
    if not paths:
        print("No images found in sample_inputs/.")
        return

    print(f"{'Image':<28}{'Faces detected':>15}{'Lighting':>12}")
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"{os.path.basename(path):<28}  (could not be read)")
            continue
        smoother = cvp.CountSmoother()          # fresh smoother per image
        annotated, raw, _, lighting, _ = cvp.process_frame(image, cascade, smoother)
        out_path = os.path.join(OUTPUT_DIR,
                                "result_" + os.path.basename(path))
        cv2.imwrite(out_path, annotated)
        print(f"{os.path.basename(path):<28}{raw:>15}{lighting:>12}")

    print(f"\nAnnotated results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()