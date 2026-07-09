import cv2
import numpy as np
from src.config import (
    HOG_WIN_STRIDE,
    HOG_PADDING,
    HOG_SCALE,
    HOG_HIT_THRESHOLD,
    NMS_OVERLAP_THRESHOLD,
    MIN_HUMAN_ASPECT_RATIO,
    MAX_HUMAN_ASPECT_RATIO,
    MIN_HUMAN_HEIGHT,
    BOUNDING_BOX_THICKNESS,
)

# Warna bounding box untuk manusia terdeteksi (BGR - hijau terang)
HUMAN_BOX_COLOR = (0, 255, 0)
# Warna bounding box untuk motion umum (BGR - biru)
MOTION_BOX_COLOR = (255, 165, 0)


class HumanDetector:
    def __init__(
        self,
        win_stride=HOG_WIN_STRIDE,
        padding=HOG_PADDING,
        scale=HOG_SCALE,
        hit_threshold=HOG_HIT_THRESHOLD,
        nms_threshold=NMS_OVERLAP_THRESHOLD,
    ):

        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        self.hit_threshold = hit_threshold
        self.nms_threshold = nms_threshold

        # Inisialisasi HOG descriptor dengan SVM pre-trained
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect_humans(self, frame):
        # Deteksi menggunakan HOG
        rects, weights = self.hog.detectMultiScale(
            frame,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
            hitThreshold=self.hit_threshold,
        )

        if len(rects) == 0:
            return [], []

        # Konversi ke format (x1, y1, x2, y2) untuk NMS
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        scores = np.array(weights).flatten()

        # Terapkan Non-Maximum Suppression untuk menghilangkan duplikat
        picked_indices = self._non_max_suppression(boxes, scores)

        # Filter berdasarkan aspect ratio dan tinggi minimum
        final_boxes = []
        final_weights = []

        for i in picked_indices:
            x1, y1, x2, y2 = boxes[i]
            w = x2 - x1
            h = y2 - y1
            aspect_ratio = h / max(w, 1)

            # Filter: hanya terima bentuk menyerupai manusia tegak
            if (
                MIN_HUMAN_ASPECT_RATIO <= aspect_ratio <= MAX_HUMAN_ASPECT_RATIO
                and h >= MIN_HUMAN_HEIGHT
            ):
                final_boxes.append((int(x1), int(y1), int(w), int(h)))
                final_weights.append(float(scores[i]))

        return final_boxes, final_weights

    def _non_max_suppression(self, boxes, scores):
        if len(boxes) == 0:
            return []

        # Konversi ke float
        boxes = boxes.astype("float")

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # Hitung area setiap box
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Urutkan berdasarkan score (descending)
        order = scores.argsort()[::-1]

        picked = []

        while len(order) > 0:
            # Ambil box dengan score tertinggi
            i = order[0]
            picked.append(i)

            # Hitung overlap dengan box lainnya
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / areas[order[1:]]

            # Hapus box yang overlap melebihi threshold
            remaining = np.where(overlap <= self.nms_threshold)[0]
            order = order[remaining + 1]

        return picked

    def detect_and_annotate(self, frame, motion_contours=None):
        annotated = frame.copy()

        # Deteksi manusia dengan HOG
        human_boxes, weights = self.detect_humans(frame)
        person_count = len(human_boxes)

        # Gambar bounding box motion (biru) jika ada kontur motion
        if motion_contours is not None:
            for contour in motion_contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(
                    annotated,
                    (x, y),
                    (x + w, y + h),
                    MOTION_BOX_COLOR,
                    1,  # Garis tipis untuk motion umum
                )

        # Gambar bounding box manusia (hijau tebal)
        for i, (x, y, w, h) in enumerate(human_boxes):
            # Bounding box hijau tebal
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                HUMAN_BOX_COLOR,
                BOUNDING_BOX_THICKNESS + 1,
            )

            # Label dengan confidence score
            if i < len(weights):
                label = f"Person {i+1} ({weights[i]:.1f})"
            else:
                label = f"Person {i+1}"

            # Background untuk label text
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x, y - text_h - 8),
                (x + text_w + 4, y),
                HUMAN_BOX_COLOR,
                -1,  # Filled rectangle
            )
            cv2.putText(
                annotated,
                label,
                (x + 2, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Teks hitam di atas background hijau
                1,
            )

        return person_count, annotated, human_boxes

    def validate_motion_as_human(self, frame, motion_contours):
        human_boxes, _ = self.detect_humans(frame)
        human_contours = []
        non_human_contours = []

        for contour in motion_contours:
            (cx, cy, cw, ch) = cv2.boundingRect(contour)
            is_human = False

            # Cek apakah kontur overlap dengan salah satu deteksi HOG
            for (hx, hy, hw, hh) in human_boxes:
                # Hitung overlap
                ox1 = max(cx, hx)
                oy1 = max(cy, hy)
                ox2 = min(cx + cw, hx + hw)
                oy2 = min(cy + ch, hy + hh)

                if ox1 < ox2 and oy1 < oy2:
                    overlap_area = (ox2 - ox1) * (oy2 - oy1)
                    contour_area = cw * ch
                    if overlap_area > 0.2 * contour_area:  # 20% overlap minimum
                        is_human = True
                        break

            # Fallback: cek aspect ratio jika HOG tidak mendeteksi
            if not is_human:
                aspect_ratio = ch / max(cw, 1)
                if (
                    MIN_HUMAN_ASPECT_RATIO <= aspect_ratio <= MAX_HUMAN_ASPECT_RATIO
                    and ch >= MIN_HUMAN_HEIGHT
                ):
                    is_human = True

            if is_human:
                human_contours.append(contour)
            else:
                non_human_contours.append(contour)

        return human_contours, non_human_contours

    def get_method_name(self):
        return "HOG + SVM Pedestrian Detection"
