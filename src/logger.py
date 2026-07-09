import csv
import os
from datetime import datetime
from src.config import LOG_FILE_PATH, LOG_DIR


class MotionLogger:
    def __init__(self, log_path=LOG_FILE_PATH):
        self.log_path = log_path
        self.log_count = 0

        # Buat direktori jika belum ada
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Tulis header jika file belum ada
        if not os.path.exists(log_path):
            self._write_header()

    def _write_header(self):
        with open(self.log_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "frame_number",
                "status",
                "method",
                "contour_count",
                "person_count",
            ])

    def log(self, frame_number, motion_detected, method_name, contour_count=0, person_count=0):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        status = "MOTION DETECTED" if motion_detected else "NO MOTION"

        with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                frame_number,
                status,
                method_name,
                contour_count,
                person_count,
            ])

        if motion_detected:
            self.log_count += 1

    def get_summary(self):
        return {
            "log_file": self.log_path,
            "total_motion_events": self.log_count,
        }
