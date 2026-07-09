import cv2
from src.config import (
    RESIZE_WIDTH,
    RESIZE_HEIGHT,
    GAUSSIAN_BLUR_KERNEL,
    USE_CLAHE,
    CLAHE_CLIP_LIMIT,
    USE_BILATERAL_FILTER,
)


def resize_frame(frame, width=RESIZE_WIDTH, height=RESIZE_HEIGHT):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def convert_to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def apply_clahe(frame, clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(frame)


def apply_blur(frame, kernel_size=GAUSSIAN_BLUR_KERNEL):
    if USE_BILATERAL_FILTER:
        # Bilateral filter: d=9 (diameter), sigmaColor=75, sigmaSpace=75
        # Mempertahankan edge sehingga gerakan kecil lebih jelas
        return cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
    else:
        return cv2.GaussianBlur(frame, kernel_size, 0)


def preprocess_frame(frame):

    frame_resized = resize_frame(frame)
    gray = convert_to_grayscale(frame_resized)
    
    # Terapkan CLAHE untuk meningkatkan kontras (membantu HOG detector)
    if USE_CLAHE:
        gray = apply_clahe(gray)
    
    blurred = apply_blur(gray)
    return frame_resized, blurred
