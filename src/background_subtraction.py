import cv2
import numpy as np
from collections import deque
from src.config import (
    BG_SUB_METHOD,
    MOG2_HISTORY,
    MOG2_VAR_THRESHOLD,
    MOG2_DETECT_SHADOWS,
    MOG2_LEARNING_RATE_INIT,
    MOG2_LEARNING_RATE_STABLE,
    MOG2_WARMUP_FRAMES,
    KNN_HISTORY,
    KNN_DIST2_THRESHOLD,
    KNN_DETECT_SHADOWS,
    KNN_LEARNING_RATE_INIT,
    KNN_LEARNING_RATE_STABLE,
    KNN_WARMUP_FRAMES,
    MIN_CONTOUR_AREA,
    DILATE_ITERATIONS,
    BOUNDING_BOX_COLOR,
    BOUNDING_BOX_THICKNESS,
    MIN_HUMAN_ASPECT_RATIO,
    MAX_HUMAN_ASPECT_RATIO,
    MIN_HUMAN_HEIGHT,
    MOTION_PERSISTENCE_FRAMES,
    MOTION_HISTORY_WEIGHT,
    SHADOW_THRESHOLD,
)


class BackgroundSubtraction:
    def __init__(self, method=BG_SUB_METHOD, min_area=MIN_CONTOUR_AREA):
        self.method = method.upper()
        self.min_area = min_area
        self.frame_count = 0  # Counter untuk adaptive learning rate

        # Temporal smoothing: simpan mask history
        self.mask_history = deque(maxlen=MOTION_PERSISTENCE_FRAMES)

        if self.method == "MOG2":
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=MOG2_HISTORY,
                varThreshold=MOG2_VAR_THRESHOLD,
                detectShadows=MOG2_DETECT_SHADOWS,
            )
            self.lr_init = MOG2_LEARNING_RATE_INIT
            self.lr_stable = MOG2_LEARNING_RATE_STABLE
            self.warmup_frames = MOG2_WARMUP_FRAMES
        elif self.method == "KNN":
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=KNN_HISTORY,
                dist2Threshold=KNN_DIST2_THRESHOLD,
                detectShadows=KNN_DETECT_SHADOWS,
            )
            self.lr_init = KNN_LEARNING_RATE_INIT
            self.lr_stable = KNN_LEARNING_RATE_STABLE
            self.warmup_frames = KNN_WARMUP_FRAMES
        else:
            raise ValueError(f"Metode tidak dikenal: {method}. Gunakan 'MOG2' atau 'KNN'.")

    def _get_learning_rate(self):
        if self.frame_count < self.warmup_frames:
            # Transisi halus dari lr_init ke lr_stable
            progress = self.frame_count / self.warmup_frames
            return self.lr_init * (1 - progress) + self.lr_stable * progress
        return self.lr_stable

    def detect(self, preprocessed_frame, display_frame):
        motion_detected = False
        contours_detected = []
        self.frame_count += 1

        # ====== Background Subtraction dengan Adaptive Learning Rate ======
        learning_rate = self._get_learning_rate()
        fg_mask = self.bg_subtractor.apply(preprocessed_frame, learningRate=learning_rate)

        # ====== Shadow Removal (Ditingkatkan) ======
        # MOG2/KNN menandai shadow sebagai pixel 127
        # Threshold 250 (dari 200) → lebih agresif menghapus shadow
        _, fg_mask = cv2.threshold(fg_mask, SHADOW_THRESHOLD, 255, cv2.THRESH_BINARY)

        # ====== Temporal Smoothing (Motion Persistence) ======
        # Simpan mask saat ini ke history
        self.mask_history.append(fg_mask.copy())

        # Gabungkan mask history: gerakan dari beberapa frame terakhir
        # tetap dianggap aktif → menghindari deteksi flicker
        if len(self.mask_history) > 1:
            temporal_mask = np.zeros_like(fg_mask, dtype=np.float32)
            for i, hist_mask in enumerate(self.mask_history):
                # Frame terbaru diberi bobot lebih tinggi
                weight = (i + 1) / len(self.mask_history)
                temporal_mask += hist_mask.astype(np.float32) * weight

            # Normalisasi dan threshold
            temporal_mask = np.clip(temporal_mask, 0, 255).astype(np.uint8)
            _, temporal_mask = cv2.threshold(temporal_mask, 50, 255, cv2.THRESH_BINARY)

            # Blend: current mask + temporal history
            fg_mask = cv2.addWeighted(
                fg_mask, 1.0 - MOTION_HISTORY_WEIGHT,
                temporal_mask, MOTION_HISTORY_WEIGHT,
                0
            )
            _, fg_mask = cv2.threshold(fg_mask, 60, 255, cv2.THRESH_BINARY)

        # ====== Morphological Operations (Optimized) ======
        # Kernel kecil untuk opening (hilangkan noise kecil)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)

        # Kernel lebih kecil untuk closing: 5x5 (dari 7x7)
        # Tetap menutup lubang tapi detail kecil tidak hilang
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

        # Dilasi dengan kernel ellipse untuk area deteksi lebih natural
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=DILATE_ITERATIONS)

        # ====== Two-Pass Contour Detection ======
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        annotated = display_frame.copy()

        # Pass 1: Kumpulkan bounding box kontur utama (untuk referensi Pass 2)
        main_boxes = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            (x, y, w, h) = cv2.boundingRect(contour)
            aspect_ratio = h / max(w, 1)

            # Filter: prioritaskan bentuk menyerupai manusia tegak
            is_human_shape = (
                MIN_HUMAN_ASPECT_RATIO <= aspect_ratio <= MAX_HUMAN_ASPECT_RATIO
                and h >= MIN_HUMAN_HEIGHT
            )
            is_large_motion = area > self.min_area * 2
            is_small_motion = area >= self.min_area and (w >= 10 and h >= 10)

            if not is_human_shape and not is_large_motion and not is_small_motion:
                continue

            motion_detected = True
            contours_detected.append(contour)
            main_boxes.append((x, y, w, h))

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

            # Tampilkan area untuk debugging
            cv2.putText(
                annotated,
                f"A:{area:.0f}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                box_color,
                1,
            )

        # Pass 2: Deteksi kontur kecil di sekitar area gerakan utama
        # Ini menangkap detail halus (gerakan tangan, jari, kepala)
        # yang terlalu kecil untuk pass 1 tapi relevan karena dekat gerakan utama
        if main_boxes:
            small_area_threshold = self.min_area * 0.4  # 40% dari min normal
            for contour in contours:
                area = cv2.contourArea(contour)
                # Hanya kontur yang terlalu kecil untuk pass 1 tapi cukup besar
                if area < small_area_threshold or area >= self.min_area:
                    continue

                (x, y, w, h) = cv2.boundingRect(contour)

                # Cek apakah kontur kecil ini dekat dengan gerakan utama
                is_near_main = False
                for (mx, my, mw, mh) in main_boxes:
                    # Expand bounding box utama 50% sebagai zone
                    expand = 0.5
                    zx = mx - int(mw * expand)
                    zy = my - int(mh * expand)
                    zw = mw + int(mw * expand * 2)
                    zh = mh + int(mh * expand * 2)

                    # Cek overlap
                    if (x < zx + zw and x + w > zx and
                            y < zy + zh and y + h > zy):
                        is_near_main = True
                        break

                if is_near_main:
                    contours_detected.append(contour)
                    # Gambar bounding box tipis untuk detail kecil
                    cv2.rectangle(
                        annotated,
                        (x, y),
                        (x + w, y + h),
                        (200, 200, 200),  # Abu-abu
                        1,  # Tipis
                    )

        return motion_detected, annotated, contours_detected

    def reset(self):
        self.__init__(method=self.method, min_area=self.min_area)

    def get_method_name(self):
        return f"Background Subtraction ({self.method})"
