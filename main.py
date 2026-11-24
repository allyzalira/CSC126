"""
main.py
=======
Smart Classroom Face Detection and Attendance Monitor  (CSC 126 Midterm)

Tkinter desktop interface. Image processing is done by cv_pipeline.py,
recognition by recognizer.py, and storage by database.py (SQLite).
This file only handles the window, buttons, and the update loop.

Run:  python main.py

Workflow
--------
1. Register Student -> enter details, face the camera, ~30 samples are saved
   and the recogniser is retrained.
2. Start Camera     -> registered students are recognised and marked Present
   (once per day) in attendance.db. Unknown faces are shown in red.
3. View Records     -> see Present/Absent per date and export to CSV.
"""

import tkinter as tk
from datetime import date, datetime
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import cv_pipeline as cvp
import recognizer as rec
from database import Database, DuplicateStudentError

APP_TITLE = "Smart Classroom Face Detection and Attendance Monitor"
VIDEO_W, VIDEO_H = 800, 600      # fixed size of the video area
FRAME_DELAY_MS = 15              # delay between processed frames

CONFIRM_FRAMES = 8               # consecutive frames a face must be matched
                                 # before attendance is saved (avoids flukes)
SAMPLES_NEEDED = 30              # face images captured per student
SAMPLE_EVERY_N_FRAMES = 3        # spacing so the samples are not identical

# Colours
BG = "#1e2a38"
PANEL = "#2b3b4e"
TEXT = "#ffffff"
MUTED = "#aab7c4"
OK = "#2ecc71"
BAD = "#e74c3c"
WARN = "#f1c40f"


# ==========================================================================
# Register-student dialog
# ==========================================================================
class RegisterDialog(tk.Toplevel):
    """Small form asking for the student's details."""

    def __init__(self, master, on_submit):
        super().__init__(master)
        self.title("Register Student")
        self.configure(bg=PANEL, padx=20, pady=15)
        self.resizable(False, False)
        self.transient(master)
        self.on_submit = on_submit

        self.vars = {}
        for row, (key, caption) in enumerate(
                [("no", "Student No."), ("name", "Full Name"),
                 ("course", "Course / Section")]):
            tk.Label(self, text=caption, bg=PANEL, fg=TEXT,
                     font=("Segoe UI", 10)).grid(row=row, column=0,
                                                 sticky="w", pady=5)
            self.vars[key] = tk.StringVar()
            tk.Entry(self, textvariable=self.vars[key], width=28,
                     font=("Segoe UI", 10)).grid(row=row, column=1,
                                                 padx=(10, 0), pady=5)

        tk.Label(self, bg=PANEL, fg=MUTED, justify="left",
                 font=("Segoe UI", 9),
                 text="Only the student should be in front of the camera.\n"
                      "Look straight ahead and turn slightly left/right.\n"
                      "The student must consent to storing their face images."
                 ).grid(row=3, column=0, columnspan=2, pady=(8, 10), sticky="w")

        tk.Button(self, text="Start Capturing", command=self._submit,
                  bg=OK, fg="black", relief="flat", pady=6,
                  font=("Segoe UI", 10, "bold")).grid(row=4, column=0,
                                                      columnspan=2, sticky="ew")
        self.bind("<Return>", lambda _e: self._submit())
        self.grab_set()

    def _submit(self):
        student_no = self.vars["no"].get().strip()
        name = self.vars["name"].get().strip()
        course = self.vars["course"].get().strip()
        if not student_no or not name:
            messagebox.showwarning("Missing details",
                                   "Student No. and Full Name are required.",
                                   parent=self)
            return
        error = self.on_submit(student_no, name, course)
        if error:
            messagebox.showerror("Registration", error, parent=self)
        else:
            self.destroy()


# ==========================================================================
# Attendance records window
# ==========================================================================
class RecordsWindow(tk.Toplevel):
    """Shows Present/Absent for every registered student on one date."""

    def __init__(self, master, db):
        super().__init__(master)
        self.title("Attendance Records")
        self.db = db
        self.geometry("760x460")

        top = tk.Frame(self, padx=10, pady=8)
        top.pack(fill="x")
        tk.Label(top, text="Date (YYYY-MM-DD):").pack(side="left")
        self.date_var = tk.StringVar(value=date.today().isoformat())
        tk.Entry(top, textvariable=self.date_var, width=12).pack(
            side="left", padx=6)
        tk.Button(top, text="Show", command=self.refresh).pack(side="left")
        tk.Button(top, text="Export CSV", command=self.export).pack(
            side="right")
        self.summary = tk.Label(top, text="", font=("Segoe UI", 10, "bold"))
        self.summary.pack(side="right", padx=15)

        columns = ("student_no", "name", "course", "time_in", "status")
        self.tree = ttk.Treeview(self, columns=columns, show="headings")
        for col, text, width in [("student_no", "Student No.", 110),
                                 ("name", "Name", 220),
                                 ("course", "Course", 150),
                                 ("time_in", "Time In", 90),
                                 ("status", "Status", 90)]:
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor="w")
        self.tree.tag_configure("Present", foreground="#1e8e3e")
        self.tree.tag_configure("Absent", foreground="#c0392b")
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True,
                       padx=(10, 0), pady=(0, 10))
        scroll.pack(side="left", fill="y", pady=(0, 10), padx=(0, 10))
        self.refresh()

    def _valid_date(self):
        text = self.date_var.get().strip()
        try:
            datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            messagebox.showwarning("Date", "Use the format YYYY-MM-DD.",
                                   parent=self)
            return None
        return text

    def refresh(self):
        day = self._valid_date()
        if day is None:
            return
        self.tree.delete(*self.tree.get_children())
        rows = self.db.attendance_sheet(day)
        present = 0
        for row in rows:
            present += row["status"] == "Present"
            self.tree.insert("", "end", tags=(row["status"],), values=(
                row["student_no"], row["name"], row["course"] or "",
                row["time_in"] or "", row["status"]))
        self.summary.config(text=f"Present: {present} / {len(rows)}")

    def export(self):
        day = self._valid_date()
        if day is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".csv",
            initialfile=f"attendance_{day}.csv",
            filetypes=[("CSV file", "*.csv")])
        if path:
            count = self.db.export_csv(path, day)
            messagebox.showinfo("Export", f"Saved {count} rows to:\n{path}",
                                parent=self)


# ==========================================================================
# Main application
# ==========================================================================
class AttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.configure(bg=BG)
        self.resizable(False, False)

        # Application state
        self.capture = None          # cv2.VideoCapture (None when stopped)
        self.running = False
        self.loop_job = None         # id of the scheduled after() call
        self.peak_count = 0          # highest stable count in this session
        self.photo = None            # keeps a reference so Tk doesn't discard it
        self.smoother = cvp.CountSmoother()
        self.reg = None              # active registration, or None
        self.hits = {}               # student_id -> consecutive matched frames
        self.marked = set()          # student ids already saved today
        self.marked_date = None

        # Load the Haar Cascade once at start-up
        try:
            self.cascade = cvp.load_face_cascade()
        except cvp.CascadeError as exc:
            messagebox.showerror("Haar Cascade Error", str(exc))
            self.destroy()
            raise SystemExit(1)

        # Database
        self.db = Database()

        # Recogniser (needs opencv-contrib-python)
        self.recognizer = None
        self.names = {}
        if rec.is_available():
            try:
                self.recognizer = rec.FaceRecognizer()
                self.recognizer.load()
            except rec.RecognizerError as exc:
                messagebox.showwarning("Recogniser", str(exc))
        else:
            messagebox.showwarning(
                "Face recognition unavailable",
                "OpenCV was installed without the 'face' module.\n\n"
                "Run:\n  pip uninstall opencv-python\n"
                "  pip install opencv-contrib-python\n\n"
                "The app will run in detection-only mode.")
        self._refresh_names()

        self._build_interface()
        self._show_placeholder("Camera is off - press Start Camera")
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

    def _refresh_names(self):
        self.names = {s["id"]: s["name"] for s in self.db.list_students()}

    # ------------------------------------------------------------------
    # Interface construction
    # ------------------------------------------------------------------
    def _build_interface(self):
        # Title bar
        tk.Label(self, text=APP_TITLE, bg=BG, fg=TEXT,
                 font=("Segoe UI", 18, "bold"), pady=10).pack(fill="x")

        body = tk.Frame(self, bg=BG)
        body.pack(padx=15, pady=(0, 15))

        # Left: webcam display area (fixed size)
        video_box = tk.Frame(body, bg="black", width=VIDEO_W, height=VIDEO_H)
        video_box.pack(side="left")
        video_box.pack_propagate(False)
        self.video_label = tk.Label(video_box, bg="black")
        self.video_label.pack(expand=True)

        # Right: information + controls panel
        side = tk.Frame(body, bg=PANEL, width=280, height=VIDEO_H)
        side.pack(side="left", padx=(15, 0))
        side.pack_propagate(False)

        # Big face counter
        tk.Label(side, text="TOTAL FACES DETECTED", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 10, "bold")).pack(pady=(14, 0))
        self.count_var = tk.StringVar(value="0")
        tk.Label(side, textvariable=self.count_var, bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 44, "bold")).pack()

        # Status rows
        self.camera_var = tk.StringVar(value="Stopped")
        self.detect_var = tk.StringVar(value="-")
        self.recog_var = tk.StringVar(value="-")
        self.last_var = tk.StringVar(value="-")
        self.peak_var = tk.StringVar(value="0")
        self.light_var = tk.StringVar(value="-")
        self.camera_value = self._info_row(side, "Camera Status", self.camera_var, BAD)
        self.detect_value = self._info_row(side, "Detection Status", self.detect_var, MUTED)
        self.recog_value = self._info_row(side, "Recognition", self.recog_var, MUTED)
        self.last_value = self._info_row(side, "Last Attendance Saved", self.last_var, MUTED)
        self._info_row(side, "Peak Count (session)", self.peak_var, TEXT)
        self.light_value = self._info_row(side, "Lighting", self.light_var, MUTED)

        # Control buttons (2 per row to save space)
        buttons = tk.Frame(side, bg=PANEL)
        buttons.pack(side="bottom", fill="x", padx=14, pady=12)
        buttons.columnconfigure((0, 1), weight=1, uniform="b")
        self.start_btn = self._button(buttons, "Start Camera", OK, self.start_camera, 0, 0)
        self.stop_btn = self._button(buttons, "Stop Camera", WARN, self.stop_camera, 0, 1)
        self.reg_btn = self._button(buttons, "Register Student", "#5dade2",
                                    self.register_student, 1, 0)
        self._button(buttons, "View Records", "#af7ac5", self.view_records, 1, 1)
        self._button(buttons, "Exit Application", BAD, self.exit_app, 2, 0, span=2)
        self.stop_btn.config(state="disabled")
        if self.recognizer is None:
            self.reg_btn.config(state="disabled")

    def _info_row(self, parent, caption, variable, colour):
        tk.Label(parent, text=caption, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=20, pady=(9, 0))
        value = tk.Label(parent, textvariable=variable, bg=PANEL, fg=colour,
                         font=("Segoe UI", 12, "bold"), wraplength=240,
                         justify="left")
        value.pack(anchor="w", padx=20)
        return value

    def _button(self, parent, text, colour, command, row, col, span=1):
        button = tk.Button(parent, text=text, command=command, bg=colour,
                           fg="black", activebackground=colour,
                           font=("Segoe UI", 10, "bold"), relief="flat",
                           pady=6, cursor="hand2")
        button.grid(row=row, column=col, columnspan=span, sticky="ew",
                    padx=3, pady=3)
        return button

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    def _show_image(self, bgr_image):
        """Convert an OpenCV BGR image to a Tk image and show it."""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.config(image=self.photo)

    def _show_placeholder(self, message):
        """Show a black image with a message when the camera is off."""
        image = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        (w, _), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(image, message, ((VIDEO_W - w) // 2, VIDEO_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
        self._show_image(image)

    # ------------------------------------------------------------------
    # Camera controls
    # ------------------------------------------------------------------
    def start_camera(self):
        if self.running:
            return
        try:
            self.capture = cvp.open_camera(0)
        except cvp.CameraError as exc:
            messagebox.showerror("Camera Error", str(exc))
            self.camera_var.set("Unavailable")
            self.camera_value.config(fg=BAD)
            return

        self.running = True
        self.smoother.reset()
        self.hits.clear()
        self.camera_var.set("Running")
        self.camera_value.config(fg=OK)
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._update_frame()

    def stop_camera(self):
        """Stop the loop, release the webcam, and reset the display."""
        self.running = False
        if self.loop_job is not None:
            self.after_cancel(self.loop_job)
            self.loop_job = None
        cvp.release_camera(self.capture)
        self.capture = None
        if self.reg is not None:
            self._abort_registration()
        self.hits.clear()

        self.camera_var.set("Stopped")
        self.camera_value.config(fg=BAD)
        self.detect_var.set("-")
        self.detect_value.config(fg=MUTED)
        self.recog_var.set("-")
        self.recog_value.config(fg=MUTED)
        self.light_var.set("-")
        self.light_value.config(fg=MUTED)
        self.count_var.set("0")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._show_placeholder("Camera is off - press Start Camera")

    def exit_app(self):
        """Release the camera, close the database, and close the window."""
        self.stop_camera()
        self.db.close()
        self.destroy()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_student(self):
        if self.recognizer is None:
            messagebox.showerror("Unavailable",
                                 "Face recognition is not installed "
                                 "(see requirements.txt).")
            return
        if self.reg is not None:
            return
        RegisterDialog(self, self._begin_registration)

    def _begin_registration(self, student_no, name, course):
        """Called by the dialog. Returns an error string, or None on success."""
        try:
            student_id = self.db.add_student(student_no, name, course)
        except DuplicateStudentError as exc:
            return str(exc)

        if not self.running:
            self.start_camera()
            if not self.running:
                self.db.delete_student(student_id)
                return "The camera could not be started."

        self.reg = {"id": student_id, "name": name, "count": 0, "frame": 0}
        self.start_btn.config(state="disabled")
        self.reg_btn.config(state="disabled")
        return None

    def _registration_step(self, faces):
        """Save one sample every few frames until enough are captured."""
        reg = self.reg
        reg["frame"] += 1
        self.recog_value.config(fg=WARN)
        if not faces:
            self.recog_var.set(f"Registering {reg['name']}: face the camera")
            return

        face = max(faces, key=lambda f: f.w * f.h)   # largest = closest
        if reg["frame"] % SAMPLE_EVERY_N_FRAMES == 0:
            self.recognizer.save_sample(reg["id"], face.crop, reg["count"])
            reg["count"] += 1
        self.recog_var.set(
            f"Registering {reg['name']}: {reg['count']}/{SAMPLES_NEEDED}")
        if reg["count"] >= SAMPLES_NEEDED:
            self._finish_registration()

    def _finish_registration(self):
        reg, self.reg = self.reg, None
        self.reg_btn.config(state="normal")
        try:
            self.recognizer.train()
        except rec.RecognizerError as exc:
            self._discard_student(reg["id"])
            messagebox.showerror("Training Error", str(exc))
            return
        self._refresh_names()
        self.hits.clear()
        self.recog_var.set("-")
        messagebox.showinfo("Registered",
                            f"{reg['name']} was registered and the "
                            f"recogniser has been trained.")

    def _abort_registration(self):
        """Camera stopped mid-registration: remove the incomplete student."""
        reg, self.reg = self.reg, None
        self._discard_student(reg["id"])
        self.reg_btn.config(state="normal")

    def _discard_student(self, student_id):
        self.db.delete_student(student_id)
        self.recognizer.remove_student(student_id)
        try:
            self.recognizer.train()
        except rec.RecognizerError:
            pass
        self._refresh_names()

    # ------------------------------------------------------------------
    # Attendance
    # ------------------------------------------------------------------
    def _attendance_step(self, faces):
        """Match faces to students and save attendance in the database."""
        if self.recognizer is None or not self.recognizer.trained:
            self.recog_var.set("No students registered")
            self.recog_value.config(fg=MUTED)
            return

        today = date.today().isoformat()
        if today != self.marked_date:                # new day: reload
            self.marked = self.db.present_ids(today)
            self.marked_date = today

        seen = {f.student_id: f for f in faces if f.student_id is not None}
        unknown = any(f.student_id is None for f in faces)

        for student_id in list(self.hits):           # lost faces reset
            if student_id not in seen:
                del self.hits[student_id]

        for student_id, face in seen.items():
            self.hits[student_id] = self.hits.get(student_id, 0) + 1
            if (self.hits[student_id] >= CONFIRM_FRAMES
                    and student_id not in self.marked):
                saved = self.db.mark_present(student_id, face.confidence)
                self.marked.add(student_id)
                if saved:
                    self.last_var.set(
                        f"{face.name} at {datetime.now():%H:%M:%S}")
                    self.last_value.config(fg=OK)

        if unknown:
            self.recog_var.set("Unrecognized face!")
            self.recog_value.config(fg=BAD)
        elif seen:
            self.recog_var.set(", ".join(f.name for f in seen.values()))
            self.recog_value.config(fg=OK)
        else:
            self.recog_var.set("-")
            self.recog_value.config(fg=MUTED)

    # ------------------------------------------------------------------
    # Records window
    # ------------------------------------------------------------------
    def view_records(self):
        RecordsWindow(self, self.db)

    # ------------------------------------------------------------------
    # Main processing loop (runs repeatedly while the camera is on)
    # ------------------------------------------------------------------
    def _update_frame(self):
        if not self.running:
            return
        try:
            frame = cvp.capture_frame(self.capture)
            frame = cv2.flip(frame, 1)  # mirror view feels natural
            annotated, _, count, lighting, faces = cvp.process_frame(
                frame, self.cascade, self.smoother,
                self.recognizer, self.names)
        except cvp.CameraError as exc:
            self.stop_camera()
            messagebox.showerror("Camera Error", str(exc))
            return
        except cv2.error as exc:
            self.stop_camera()
            messagebox.showerror("Processing Error", f"OpenCV error:\n{exc}")
            return

        # Update the information panel
        self.count_var.set(str(count))
        self.peak_count = max(self.peak_count, count)
        self.peak_var.set(str(self.peak_count))
        if count > 0:
            self.detect_var.set("Face Detected")
            self.detect_value.config(fg=OK)
        else:
            self.detect_var.set("No Face Detected")
            self.detect_value.config(fg=WARN)
        self.light_var.set(lighting)
        self.light_value.config(fg=OK if lighting == "Good" else WARN)

        # Show the processed frame, then register or record attendance
        self._show_image(annotated)
        if self.reg is not None:
            self._registration_step(faces)
        else:
            self._attendance_step(faces)

        if self.running:   # registration dialogs may have stopped the camera
            self.loop_job = self.after(FRAME_DELAY_MS, self._update_frame)


def main():
    app = None
    try:
        app = AttendanceApp()
        app.mainloop()
    except SystemExit:
        pass
    except Exception as exc:  # unexpected crash: always free the webcam
        print(f"Unexpected error: {exc}")
        raise
    finally:
        if app is not None:
            try:
                cvp.release_camera(app.capture)
            except Exception:
                pass


if __name__ == "__main__":
    main()