import cv2
import numpy as np
from collections import deque
from src.config import (
    FRAME_DIFF_THRESHOLD,
    MIN_CONTOUR_AREA,
    DILATE_ITERATIONS,
    BOUNDING_BOX_COLOR,
    BOUNDING_BOX_THICKNESS,
    MIN_HUMAN_ASPECT_RATIO,
    MAX_HUMAN_ASPECT_RATIO,
    MIN_HUMAN_HEIGHT,
    ADAPTIVE_THRESHOLD_ENABLED,
    ADAPTIVE_BLOCK_SIZE,
    ADAPTIVE_C,
    MOTION_PERSISTENCE_FRAMES,
    MOTION_HISTORY_WEIGHT,
    MOTION_MIN_TOTAL_AREA,
    MOTION_SINGLE_AREA_TRIGGER,
)


class FrameDifference:

    def __init__(self, threshold=FRAME_DIFF_THRESHOLD, min_area=MIN_CONTOUR_AREA):
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame = None
        self.prev_prev_frame = None  # Frame t-2 untuk 3-frame diff

        # Temporal smoothing: simpan mask history
        self.mask_history = deque(maxlen=MOTION_PERSISTENCE_FRAMES)
        self.accumulated_mask = None  # Akumulasi mask untuk temporal smoothing

    def detect(self, preprocessed_frame, display_frame):

        motion_detected = False
        contours_detected = []

        # Frame pertama: simpan dan return
        if self.prev_frame is None:
            self.prev_frame = preprocessed_frame.copy()
            self.accumulated_mask = np.zeros_like(preprocessed_frame)
            return False, display_frame.copy(), []

        # Frame kedua: simpan untuk 3-frame diff
        if self.prev_prev_frame is None:
            self.prev_prev_frame = self.prev_frame.copy()
            self.prev_frame = preprocessed_frame.copy()
            return False, display_frame.copy(), []

        # ====== 3-Frame Differencing ======
        # Diff antara frame t-2 dan t-1
        diff1 = cv2.absdiff(self.prev_prev_frame, self.prev_frame)
        # Diff antara frame t-1 dan t
        diff2 = cv2.absdiff(self.prev_frame, preprocessed_frame)
        # Diff tambahan: t-2 langsung ke t (menangkap gerakan lambat yang mungkin
        # terlewat oleh diff berturutan)
        diff3 = cv2.absdiff(self.prev_prev_frame, preprocessed_frame)

        # ====== Thresholding (Adaptive atau Fixed) ======
        if ADAPTIVE_THRESHOLD_ENABLED:
            # Adaptive threshold: menyesuaikan kondisi cahaya lokal
            # Lebih baik untuk scene dengan pencahayaan tidak merata
            thresh1 = cv2.adaptiveThreshold(
                diff1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
            )
            thresh2 = cv2.adaptiveThreshold(
                diff2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
            )
            thresh3 = cv2.adaptiveThreshold(
                diff3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
            )
        else:
            # Fixed threshold (original method)
            _, thresh1 = cv2.threshold(diff1, self.threshold, 255, cv2.THRESH_BINARY)
            _, thresh2 = cv2.threshold(diff2, self.threshold, 255, cv2.THRESH_BINARY)
            _, thresh3 = cv2.threshold(diff3, self.threshold, 255, cv2.THRESH_BINARY)

        # ====== Weighted OR + AND Combination ======
        # bitwise_AND: hanya area yang bergerak konsisten (akurat, tapi bisa miss gerakan kecil)
        mask_and = cv2.bitwise_and(thresh1, thresh2)
        # bitwise_OR: semua area yang bergerak (sensitif, tapi lebih noisy)
        mask_or = cv2.bitwise_or(thresh1, thresh2)
        # Tambahkan diff t-2→t sebagai boost untuk gerakan yang konsisten
        mask_or = cv2.bitwise_or(mask_or, thresh3)

        # Gabungkan: 60% OR (sensitif) + 40% AND (akurat)
        # Ini memberikan keseimbangan — gerakan kecil dari OR tetap tertangkap,
        # sementara AND membantu membuang noise
        combined = cv2.addWeighted(mask_or, 0.6, mask_and, 0.4, 0)
        # Re-threshold setelah blending
        _, combined = cv2.threshold(combined, 80, 255, cv2.THRESH_BINARY)

        # ====== Temporal Smoothing (Motion Persistence) ======
        # Simpan mask saat ini ke history
        self.mask_history.append(combined.copy())

        # Gabungkan mask history: gerakan dari beberapa frame terakhir
        # tetap dianggap aktif
        if len(self.mask_history) > 1:
            temporal_mask = np.zeros_like(combined, dtype=np.float32)
            for i, hist_mask in enumerate(self.mask_history):
                # Frame terbaru diberi bobot lebih tinggi
                weight = (i + 1) / len(self.mask_history)
                temporal_mask += hist_mask.astype(np.float32) * weight

            # Normalisasi dan threshold
            temporal_mask = np.clip(temporal_mask, 0, 255).astype(np.uint8)
            _, temporal_mask = cv2.threshold(temporal_mask, 50, 255, cv2.THRESH_BINARY)

            # Blend: current mask + temporal history
            combined = cv2.addWeighted(
                combined, 1.0 - MOTION_HISTORY_WEIGHT,
                temporal_mask, MOTION_HISTORY_WEIGHT,
                0
            )
            _, combined = cv2.threshold(combined, 60, 255, cv2.THRESH_BINARY)

        # ====== Morphological Operations (Optimized) ======
        # Opening: hilangkan noise kecil — kernel kecil agar detail tetap
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)

        # Closing: tutup lubang — kernel 5x5 (dari 7x7), agar siluet tetap utuh
        # tapi detail kecil tidak hilang
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)

        # Dilasi bertahap: iterasi dari config (default 3)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.dilate(combined, kernel_dilate, iterations=DILATE_ITERATIONS)

        # Temukan kontur pada binary image
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        annotated = display_frame.copy()

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            # ====== Aspect Ratio Filtering (Diperlonggar) ======
            (x, y, w, h) = cv2.boundingRect(contour)
            aspect_ratio = h / max(w, 1)

            # Filter: prioritaskan bentuk menyerupai manusia
            # Diperlonggar: aspect ratio 0.3-6.0, min height 20px
            is_human_shape = (
                MIN_HUMAN_ASPECT_RATIO <= aspect_ratio <= MAX_HUMAN_ASPECT_RATIO
                and h >= MIN_HUMAN_HEIGHT
            )
            # Terima kontur besar (mungkin grup orang atau gerakan besar)
            is_large_motion = area > self.min_area * 2
            # Terima gerakan kecil yang memenuhi minimal dimensi
            is_small_motion = area >= self.min_area and (w >= 10 and h >= 10)

            if not is_human_shape and not is_large_motion and not is_small_motion:
                continue

            motion_detected = True
            contours_detected.append(contour)

            # Gambar bounding box
            if is_human_shape:
                box_color = BOUNDING_BOX_COLOR  # Hijau
            elif is_large_motion:
                box_color = (0, 255, 255)  # Kuning untuk gerakan besar
            else:
                box_color = (255, 200, 0)  # Cyan untuk gerakan kecil

            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                box_color,
                BOUNDING_BOX_THICKNESS,
            )

            # Tampilkan area
            cv2.putText(
                annotated,
                f"A:{area:.0f}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                box_color,
                1,
            )

        # ====== Frame-level Motion Gate ======
        # Cegah "selalu MOTION": frame dianggap bergerak hanya jika total luas
        # kontur valid cukup besar, atau ada satu kontur yang dominan.
        total_area = sum(cv2.contourArea(c) for c in contours_detected)
        max_area = max((cv2.contourArea(c) for c in contours_detected), default=0.0)
        motion_detected = (
            total_area >= MOTION_MIN_TOTAL_AREA
            or max_area >= MOTION_SINGLE_AREA_TRIGGER
        )

        # Update frame history (geser)
        self.prev_prev_frame = self.prev_frame.copy()
        self.prev_frame = preprocessed_frame.copy()

        return motion_detected, annotated, contours_detected

    def reset(self):
        """Reset state (frame sebelumnya dan history)."""
        self.prev_frame = None
        self.prev_prev_frame = None
        self.mask_history.clear()
        self.accumulated_mask = None

    def get_method_name(self):
        return "3-Frame Differencing"
