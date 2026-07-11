"""
Benchmark deteksi (headless) untuk membandingkan metode di video sampel.

Menjalankan tiap metode pada N frame pertama video dan mencetak:
  - motion%      : persentase frame yang ter-flag MOTION
  - person_frames: jumlah frame yang mendeteksi minimal 1 orang
  - total_persons: total deteksi orang (indikasi false positive bila membengkak)
  - fps          : kecepatan proses

Berguna untuk memverifikasi tuning parameter deteksi tanpa GUI.

Jalankan:
    python benchmark.py [jumlah_frame]
"""

import os
import sys
import time

import cv2

# Pastikan root proyek ada di path agar package `src` bisa diimport.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.config as cfg
from src.preprocessing import preprocess_frame
from src.frame_difference import FrameDifference
from src.background_subtraction import BackgroundSubtraction
from src.human_detector import HumanDetector
from src.dnn_detector import DNNPersonDetector, ModelDownloadError

# Nonaktifkan self-learning agar benchmark deterministik.
cfg.ADAPTIVE_LEARNING_ENABLED = False


def run(name, detector, human=None, limit=400):
    cap = cv2.VideoCapture(cfg.VIDEO_INPUT_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Tidak bisa membuka video: {cfg.VIDEO_INPUT_PATH}")
        return

    n = motion = person_frames = total_persons = 0
    t0 = time.time()
    while n < limit:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        frame_resized, frame_pre = preprocess_frame(frame)
        motion_detected, annotated, contours = detector.detect(frame_pre, frame_resized)

        person_count = 0
        if human is not None and (motion_detected or contours):
            person_count, _, _ = human.detect_and_annotate(
                annotated, motion_contours=contours
            )

        if motion_detected:
            motion += 1
        if person_count > 0:
            person_frames += 1
            total_persons += person_count

    dt = time.time() - t0
    cap.release()
    print(
        f"{name:<28} frames={n} motion={motion} ({motion / max(n, 1):.0%}) "
        f"person_frames={person_frames} total_persons={total_persons} "
        f"fps={n / max(dt, 1e-9):.1f}"
    )


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 400

    run("3-Frame Diff", FrameDifference(), limit=limit)
    run("MOG2", BackgroundSubtraction(method="MOG2"), limit=limit)
    run("KNN", BackgroundSubtraction(method="KNN"), limit=limit)
    run("MOG2 + HOG", BackgroundSubtraction(method="MOG2"),
        human=HumanDetector(), limit=limit)
    try:
        run("MOG2 + DNN", BackgroundSubtraction(method="MOG2"),
            human=DNNPersonDetector(), limit=limit)
    except ModelDownloadError as e:
        print(f"MOG2 + DNN                   dilewati (model tidak tersedia): {e}")


if __name__ == "__main__":
    main()
