"""
database.py
===========
SQLite storage for the Smart Classroom Attendance Monitor.

Uses only Python's built-in sqlite3 module (no extra install). The database
file `attendance.db` is created automatically next to this script.

Tables
------
students    one row per registered student
attendance  one row per student per day (UNIQUE(student_id, attendance_date)
            makes duplicate entries impossible, even if the same face is
            recognised thousands of times in one class)
"""

import csv
import os
import sqlite3
from datetime import datetime
from typing import List, Optional, Set

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "attendance.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_no    TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    course        TEXT,
    registered_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id      INTEGER NOT NULL
                    REFERENCES students(id) ON DELETE CASCADE,
    attendance_date TEXT NOT NULL,           -- YYYY-MM-DD
    time_in         TEXT NOT NULL,           -- HH:MM:SS
    status          TEXT NOT NULL DEFAULT 'Present',
    confidence      REAL,                    -- LBPH distance (lower = closer)
    UNIQUE (student_id, attendance_date)
);
"""


class DuplicateStudentError(Exception):
    """Raised when a student number is already registered."""


class Database:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Students
    # ------------------------------------------------------------------
    def add_student(self, student_no: str, name: str, course: str = "") -> int:
        """Register a student and return the new student id."""
        try:
            with self.conn:
                cur = self.conn.execute(
                    "INSERT INTO students (student_no, name, course, registered_at) "
                    "VALUES (?, ?, ?, ?)",
                    (student_no, name, course,
                     datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        except sqlite3.IntegrityError as exc:
            raise DuplicateStudentError(
                f"Student number {student_no} is already registered.") from exc
        return cur.lastrowid

    def delete_student(self, student_id: int) -> None:
        """Delete a student (their attendance rows are deleted too)."""
        with self.conn:
            self.conn.execute("DELETE FROM students WHERE id = ?", (student_id,))

    def list_students(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM students ORDER BY name").fetchall()

    def get_student(self, student_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()

    # ------------------------------------------------------------------
    # Attendance
    # ------------------------------------------------------------------
    def mark_present(self, student_id: int, confidence: float = 0.0) -> bool:
        """Record attendance for today.

        Returns True if a new record was saved, False if the student was
        already marked present today (no duplicate is created).
        """
        now = datetime.now()
        with self.conn:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO attendance "
                "(student_id, attendance_date, time_in, status, confidence) "
                "VALUES (?, ?, ?, 'Present', ?)",
                (student_id, now.strftime("%Y-%m-%d"),
                 now.strftime("%H:%M:%S"), confidence))
        return cur.rowcount == 1

    def present_ids(self, date_str: str) -> Set[int]:
        rows = self.conn.execute(
            "SELECT student_id FROM attendance WHERE attendance_date = ?",
            (date_str,)).fetchall()
        return {row["student_id"] for row in rows}

    def attendance_sheet(self, date_str: str) -> List[sqlite3.Row]:
        """Every registered student with Present/Absent for the given date."""
        return self.conn.execute(
            "SELECT s.id, s.student_no, s.name, s.course, a.time_in, "
            "       COALESCE(a.status, 'Absent') AS status "
            "FROM students s "
            "LEFT JOIN attendance a "
            "       ON a.student_id = s.id AND a.attendance_date = ? "
            "ORDER BY s.name", (date_str,)).fetchall()

    def export_csv(self, path: str, date_str: str) -> int:
        """Write the attendance sheet of one date to a CSV file."""
        rows = self.attendance_sheet(date_str)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Date", "Student No", "Name", "Course",
                             "Time In", "Status"])
            for row in rows:
                writer.writerow([date_str, row["student_no"], row["name"],
                                 row["course"], row["time_in"] or "",
                                 row["status"]])
        return len(rows)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass