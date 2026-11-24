"""
test_database.py
================
Offline checks for database.py and recognizer.py (no webcam, no GUI).
Uses temporary files, so your real attendance.db is never touched.

Run:  python test_database.py
"""

import os
import tempfile

import cv2
import numpy as np

import cv_pipeline as cvp
import recognizer as rec
from database import Database, DuplicateStudentError


def check(label, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        raise SystemExit(1)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # ---------------- database ----------------
        db = Database(os.path.join(tmp, "test.db"))
        a = db.add_student("2024-0001", "Ana Cruz", "BSCS 2A")
        b = db.add_student("2024-0002", "Ben Reyes", "BSCS 2A")
        try:
            db.add_student("2024-0001", "Duplicate", "")
            check("duplicate student number rejected", False)
        except DuplicateStudentError:
            check("duplicate student number rejected", True)

        check("first mark_present saves a record", db.mark_present(a, 31.5))
        check("second mark_present same day is ignored",
              not db.mark_present(a, 28.0))
        sheet = {r["name"]: r["status"] for r in db.attendance_sheet(
            __import__("datetime").date.today().isoformat())}
        check("Ana Present, Ben Absent",
              sheet == {"Ana Cruz": "Present", "Ben Reyes": "Absent"})

        csv_path = os.path.join(tmp, "out.csv")
        today = __import__("datetime").date.today().isoformat()
        check("CSV export writes 2 rows", db.export_csv(csv_path, today) == 2)
        db.delete_student(b)
        check("delete_student works", len(db.list_students()) == 1)
        db.close()

        # ---------------- recogniser ----------------
        if not rec.is_available():
            print("[SKIP] cv2.face not installed - recogniser not tested")
            return
        rng = np.random.default_rng(1)
        def fake_face(sigma):
            """Textured test image; different sigma = different texture."""
            noise = rng.integers(0, 255, cvp.FACE_SIZE[::-1],
                                 dtype=np.uint8)
            blur = cv2.GaussianBlur(noise, (0, 0), sigma)
            return cv2.normalize(blur, None, 0, 255,
                                 cv2.NORM_MINMAX).astype(np.uint8)

        faces = {1: fake_face(2), 2: fake_face(6)}
        r = rec.FaceRecognizer(os.path.join(tmp, "t.yml"),
                               os.path.join(tmp, "ds"))
        for sid, img in faces.items():
            for i in range(5):
                shifted = np.clip(img.astype(int) + 3 * i - 6,   # lighting shift
                                  0, 255).astype(np.uint8)
                r.save_sample(sid, shifted, i)
        check("train uses all 10 samples", r.train() == 10)
        check("student 1 recognised", r.predict(faces[1])[0] == 1)
        check("student 2 recognised", r.predict(faces[2])[0] == 2)
        stranger = np.tile(np.linspace(0, 255, cvp.FACE_SIZE[0]).astype(np.uint8),
                           (cvp.FACE_SIZE[1], 1))
        check("stranger is unknown", r.predict(stranger)[0] is None)
        r2 = rec.FaceRecognizer(os.path.join(tmp, "t.yml"),
                                os.path.join(tmp, "ds"))
        check("saved model reloads", r2.load() and r2.predict(faces[1])[0] == 1)

        # whole pipeline on a blank frame (no faces) must not crash
        try:
            cascade = cvp.load_face_cascade()
        except cvp.CascadeError:
            print("[SKIP] Haar XML not found - pipeline check skipped")
            print("\nAll checks passed.")
            return
        out = cvp.process_frame(np.zeros((480, 640, 3), np.uint8), cascade,
                                cvp.CountSmoother(), r, {1: "Ana", 2: "Ben"})
        check("process_frame returns 5 values", len(out) == 5)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()