import cv2
import numpy as np


class TrackedBox:
    def __init__(self, bbox, box_id, smoothing=0.5):
        self.id = box_id
        self.smoothing = smoothing

        # Posisi aktual dan smoothed
        x, y, w, h = bbox
        self.raw_bbox = bbox
        self.smooth_x = float(x)
        self.smooth_y = float(y)
        self.smooth_w = float(w)
        self.smooth_h = float(h)

        # State tracking
        self.frames_alive = 1      # Berapa frame sudah di-track
        self.frames_missing = 0    # Berapa frame tidak terdeteksi
        self.is_active = True      # Masih aktif (terdeteksi frame ini)
        self.area = 0.0            # Area kontur terakhir

    def update(self, bbox, area=0.0):
        x, y, w, h = bbox
        self.raw_bbox = bbox
        self.area = area

        # Exponential Moving Average untuk smoothing
        alpha = self.smoothing
        self.smooth_x = alpha * x + (1 - alpha) * self.smooth_x
        self.smooth_y = alpha * y + (1 - alpha) * self.smooth_y
        self.smooth_w = alpha * w + (1 - alpha) * self.smooth_w
        self.smooth_h = alpha * h + (1 - alpha) * self.smooth_h

        self.frames_alive += 1
        self.frames_missing = 0
        self.is_active = True

    def mark_missing(self):
        self.frames_missing += 1
        self.is_active = False

    def get_smooth_bbox(self):
        return (
            int(round(self.smooth_x)),
            int(round(self.smooth_y)),
            int(round(self.smooth_w)),
            int(round(self.smooth_h)),
        )


def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Koordinat intersection
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = w1 * h1 + w2 * h2 - intersection

    return intersection / max(union, 1)


class BBoxStabilizer:

    def __init__(self, smoothing=0.45, max_missing=5, min_frames_to_show=2, iou_threshold=0.2):
        self.smoothing = smoothing
        self.max_missing = max_missing
        self.min_frames_to_show = min_frames_to_show
        self.iou_threshold = iou_threshold

        self.tracked_boxes = []
        self.next_id = 0

    def update(self, detections):
        # Tandai semua box sebagai belum di-match
        matched_tracked = set()
        matched_detected = set()

        # ====== Matching: IoU antara tracked boxes dan deteksi baru ======
        if self.tracked_boxes and detections:
            # Hitung IoU matrix
            iou_matrix = np.zeros((len(self.tracked_boxes), len(detections)))
            for i, tracked in enumerate(self.tracked_boxes):
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = compute_iou(tracked.get_smooth_bbox(), det["bbox"])

            # Greedy matching: pasangkan berdasarkan IoU tertinggi
            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = iou_matrix.max()
                if max_iou < self.iou_threshold:
                    break

                i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)

                # Match found
                self.tracked_boxes[i].update(
                    detections[j]["bbox"],
                    detections[j].get("area", 0)
                )
                matched_tracked.add(i)
                matched_detected.add(j)

                # Hapus baris dan kolom dari pertimbangan
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

        # ====== Handle unmatched tracked boxes (hilang frame ini) ======
        for i, tracked in enumerate(self.tracked_boxes):
            if i not in matched_tracked:
                tracked.mark_missing()

        # ====== Handle unmatched detections (box baru) ======
        for j, det in enumerate(detections):
            if j not in matched_detected:
                new_box = TrackedBox(
                    det["bbox"],
                    self.next_id,
                    smoothing=self.smoothing
                )
                new_box.area = det.get("area", 0)
                self.tracked_boxes.append(new_box)
                self.next_id += 1

        # ====== Hapus box yang sudah terlalu lama hilang ======
        self.tracked_boxes = [
            tb for tb in self.tracked_boxes
            if tb.frames_missing <= self.max_missing
        ]

        # ====== Return stable boxes untuk ditampilkan ======
        stable_boxes = []
        for tb in self.tracked_boxes:
            # Hanya tampilkan jika sudah cukup lama terdeteksi
            if tb.frames_alive >= self.min_frames_to_show:
                stable_boxes.append({
                    "bbox": tb.get_smooth_bbox(),
                    "area": tb.area,
                    "is_active": tb.is_active,
                    "frames_alive": tb.frames_alive,
                })

        return stable_boxes

    def reset(self):
        self.tracked_boxes.clear()
        self.next_id = 0
