"""
Deteksi manusia berbasis Deep Learning menggunakan OpenCV DNN (MobileNet-SSD).

Modul ini memakai jaringan MobileNet-SSD (Caffe) yang dijalankan lewat
`cv2.dnn` — tanpa dependensi tambahan (torch/tensorflow). Model akan otomatis
di-download ke folder `models/` saat pertama kali dipakai.

Antarmuka `detect_and_annotate()` dibuat kompatibel dengan `HumanDetector`
(HOG) sehingga bisa dipertukarkan di `main.py`.

Catatan: MobileNet-SSD dilatih untuk objek yang menghadap/berdiri (dataset
VOC). Pada sudut kamera dari atas (mis. CCTV kelas), deteksi full-body memang
terbatas — untuk kasus itu, deteksi berbasis MOTION lebih andal.
"""

import os
import ssl
import urllib.request

import cv2
import numpy as np

from src.config import (
    DNN_PROTOTXT_PATH,
    DNN_MODEL_PATH,
    DNN_PROTOTXT_URL,
    DNN_MODEL_URL,
    DNN_CONFIDENCE_THRESHOLD,
    DNN_NMS_THRESHOLD,
    DNN_INPUT_SIZE,
    DNN_PERSON_CLASS_ID,
    DNN_USE_MOTION_ROI,
    MODELS_DIR,
    BOUNDING_BOX_THICKNESS,
)

# Warna bounding box manusia (BGR - hijau terang)
HUMAN_BOX_COLOR = (0, 255, 0)
# Warna bounding box motion umum (BGR - biru)
MOTION_BOX_COLOR = (255, 165, 0)


class ModelDownloadError(Exception):
    """Dilempar jika file model DNN tidak tersedia dan gagal di-download."""


def _build_ssl_opener():
    """Bangun opener urllib dengan CA bundle certifi jika tersedia."""
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


def ensure_model(prototxt_path=DNN_PROTOTXT_PATH, model_path=DNN_MODEL_PATH,
                 prototxt_url=DNN_PROTOTXT_URL, model_url=DNN_MODEL_URL):
    """Pastikan file prototxt & caffemodel tersedia; download jika belum ada.

    Returns:
        (prototxt_path, model_path)

    Raises:
        ModelDownloadError: jika file belum ada dan gagal di-download.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    opener = _build_ssl_opener()

    for path, url, label in (
        (prototxt_path, prototxt_url, "prototxt"),
        (model_path, model_url, "caffemodel"),
    ):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        print(f"[DNN] Mengunduh {label} model ke {path} ...")
        try:
            with opener.open(url, timeout=60) as resp, open(path, "wb") as f:
                f.write(resp.read())
        except Exception as e:
            raise ModelDownloadError(
                f"Gagal mengunduh {label} dari {url}: {e}"
            ) from e
        print(f"[DNN] Selesai mengunduh {label}.")

    return prototxt_path, model_path


class DNNPersonDetector:
    """Detektor manusia berbasis MobileNet-SSD (OpenCV DNN)."""

    def __init__(
        self,
        confidence_threshold=DNN_CONFIDENCE_THRESHOLD,
        nms_threshold=DNN_NMS_THRESHOLD,
        input_size=DNN_INPUT_SIZE,
        use_motion_roi=DNN_USE_MOTION_ROI,
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.use_motion_roi = use_motion_roi

        prototxt_path, model_path = ensure_model()
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    def _detect_in_image(self, image, offset=(0, 0)):
        """Jalankan jaringan pada satu citra, kembalikan (boxes, confidences).

        boxes dalam format (x, y, w, h) pada koordinat frame asli (memakai offset).
        """
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return [], []

        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (self.input_size, self.input_size)),
            0.007843,
            (self.input_size, self.input_size),
            127.5,
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        ox, oy = offset
        boxes = []
        confidences = []
        for i in range(detections.shape[2]):
            class_id = int(detections[0, 0, i, 1])
            confidence = float(detections[0, 0, i, 2])
            if class_id != DNN_PERSON_CLASS_ID:
                continue
            if confidence < self.confidence_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            boxes.append([int(x1 + ox), int(y1 + oy), int(x2 - x1), int(y2 - y1)])
            confidences.append(confidence)

        return boxes, confidences

    def detect_persons(self, frame, motion_contours=None):
        """Deteksi manusia pada frame.

        Jika `use_motion_roi` aktif dan ada kontur motion, jaringan hanya
        dijalankan pada region gerakan (di-upscale) sehingga lebih cepat dan
        fokus. Jika tidak, jaringan dijalankan pada seluruh frame.
        """
        all_boxes = []
        all_confidences = []

        rois = self._build_rois(frame, motion_contours)
        if rois:
            for (rx, ry, rw, rh) in rois:
                roi = frame[ry:ry + rh, rx:rx + rw]
                # Upscale ROI kecil agar orang jauh lebih terdeteksi
                scale = 1
                if max(rw, rh) < 200:
                    scale = 2
                    roi = cv2.resize(roi, None, fx=scale, fy=scale)
                boxes, confs = self._detect_in_image(roi, offset=(0, 0))
                for (bx, by, bw, bh), c in zip(boxes, confs):
                    all_boxes.append([
                        rx + bx // scale, ry + by // scale,
                        bw // scale, bh // scale,
                    ])
                    all_confidences.append(c)
        else:
            all_boxes, all_confidences = self._detect_in_image(frame)

        return self._apply_nms(all_boxes, all_confidences)

    def _build_rois(self, frame, motion_contours):
        """Bangun daftar ROI (x, y, w, h) dari kontur motion, dengan padding."""
        if not self.use_motion_roi or not motion_contours:
            return []

        h, w = frame.shape[:2]
        rois = []
        for contour in motion_contours:
            (x, y, cw, ch) = cv2.boundingRect(contour)
            # Padding 40% agar seluruh tubuh tercakup
            pad_x = int(cw * 0.4)
            pad_y = int(ch * 0.4)
            rx = max(0, x - pad_x)
            ry = max(0, y - pad_y)
            rw = min(w - rx, cw + 2 * pad_x)
            rh = min(h - ry, ch + 2 * pad_y)
            if rw > 0 and rh > 0:
                rois.append((rx, ry, rw, rh))
        return rois

    def _apply_nms(self, boxes, confidences):
        """Terapkan Non-Maximum Suppression pada gabungan deteksi."""
        if not boxes:
            return [], []

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confidence_threshold, self.nms_threshold
        )
        if len(indices) == 0:
            return [], []

        indices = np.array(indices).flatten()
        final_boxes = [tuple(boxes[i]) for i in indices]
        final_confidences = [confidences[i] for i in indices]
        return final_boxes, final_confidences

    def detect_and_annotate(self, frame, motion_contours=None):
        """Deteksi manusia dan gambar anotasi (kompatibel dengan HumanDetector).

        Returns:
            (person_count, annotated_frame, human_boxes)
        """
        annotated = frame.copy()
        human_boxes, confidences = self.detect_persons(frame, motion_contours)
        person_count = len(human_boxes)

        # Gambar kontur motion (garis tipis biru) untuk konteks
        if motion_contours is not None:
            for contour in motion_contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), MOTION_BOX_COLOR, 1)

        # Gambar bounding box manusia (hijau tebal) + confidence
        for i, (x, y, w, h) in enumerate(human_boxes):
            cv2.rectangle(
                annotated, (x, y), (x + w, y + h),
                HUMAN_BOX_COLOR, BOUNDING_BOX_THICKNESS + 1,
            )
            label = f"Person {i+1} ({confidences[i]:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated, (x, y - th - 8), (x + tw + 4, y),
                HUMAN_BOX_COLOR, -1,
            )
            cv2.putText(
                annotated, label, (x + 2, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            )

        return person_count, annotated, human_boxes

    def get_method_name(self):
        return "MobileNet-SSD Deep Learning Detection"
