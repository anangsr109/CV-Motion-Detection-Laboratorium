import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input video
VIDEO_INPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "video_kelas.mp4")

# Output paths
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
FRAMES_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "frames_terdeteksi")
VIDEO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "video_output.avi")

# Results paths
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
LOG_FILE_PATH = os.path.join(LOG_DIR, "motion_log.csv")
SCREENSHOTS_DIR = os.path.join(RESULTS_DIR, "screenshots")
GRAFIK_DIR = os.path.join(RESULTS_DIR, "grafik")

# ============================================================
# PREPROCESSING PARAMETER
# ============================================================
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480
GAUSSIAN_BLUR_KERNEL = (5, 5)    # Ukuran kernel Gaussian Blur — (5,5) optimal: filter noise tanpa hilangkan detail
USE_BILATERAL_FILTER = False     # True = bilateral filter (edge-preserving), False = Gaussian blur

# ============================================================
# FRAME DIFFERENCING PARAMETER
# ============================================================
FRAME_DIFF_THRESHOLD = 8           # Threshold binarisasi (0-255) — diturunkan dari 10 untuk menangkap gerakan halus
MIN_CONTOUR_AREA = 150             # Luas minimum kontur — diturunkan dari 300 untuk menangkap gerakan kecil/jauh
DILATE_ITERATIONS = 3              # Jumlah iterasi dilasi — dinaikkan dari 2 untuk menyambung area terfragmentasi
ADAPTIVE_THRESHOLD_ENABLED = True  # Gunakan adaptive threshold (menyesuaikan cahaya lokal)
ADAPTIVE_BLOCK_SIZE = 15           # Block size untuk adaptive threshold (harus ganjil)
ADAPTIVE_C = 5                     # Konstanta dikurangi dari threshold

# ============================================================
# BACKGROUND SUBTRACTION PARAMETER
# ============================================================
BG_SUB_METHOD = "MOG2"             # Pilihan: "MOG2" atau "KNN"
MOG2_HISTORY = 200                 # Jumlah frame untuk history MOG2 — diturunkan dari 300 untuk adaptasi lebih cepat
MOG2_VAR_THRESHOLD = 8             # Variance threshold MOG2 — diturunkan dari 10, lebih sensitif terhadap perubahan kecil
MOG2_DETECT_SHADOWS = True         # Deteksi bayangan
MOG2_LEARNING_RATE_INIT = 0.01     # Learning rate awal MOG2 (fase belajar background)
MOG2_LEARNING_RATE_STABLE = 0.003  # Learning rate stabil MOG2 (lebih sensitif terhadap foreground)
MOG2_WARMUP_FRAMES = 60            # Jumlah frame warmup sebelum beralih ke learning rate stabil
KNN_HISTORY = 200                  # Jumlah frame untuk history KNN — diturunkan dari 300
KNN_DIST2_THRESHOLD = 150.0        # Distance threshold KNN — diturunkan dari 200 untuk sensitifitas tinggi
KNN_DETECT_SHADOWS = True          # Deteksi bayangan
KNN_LEARNING_RATE_INIT = 0.01      # Learning rate awal KNN
KNN_LEARNING_RATE_STABLE = 0.003   # Learning rate stabil KNN
KNN_WARMUP_FRAMES = 60             # Jumlah frame warmup KNN

# ============================================================
# HUMAN DETECTION PARAMETER (HOG + SVM)
# ============================================================
HUMAN_DETECTION_ENABLED = True     # Aktifkan deteksi manusia (HOG+SVM)
HOG_WIN_STRIDE = (4, 4)            # Stride deteksi (lebih kecil = lebih presisi, lebih lambat)
HOG_PADDING = (16, 16)             # Padding di sekitar deteksi — diperbesar agar deteksi di tepi frame lebih baik
HOG_SCALE = 1.02                   # Skala piramida — dikecilkan agar lebih teliti mendeteksi objek kecil
HOG_HIT_THRESHOLD = -0.5           # Threshold confidence HOG — diturunkan agar lebih sensitif
NMS_OVERLAP_THRESHOLD = 0.5        # Threshold overlap untuk Non-Maximum Suppression
MIN_HUMAN_ASPECT_RATIO = 0.3       # Minimum rasio tinggi/lebar — dari 0.5, menangkap duduk/membungkuk/gerakan tangan
MAX_HUMAN_ASPECT_RATIO = 6.0       # Maximum rasio tinggi/lebar — diperlebar
MIN_HUMAN_HEIGHT = 20              # Tinggi minimum deteksi manusia (pixel) — dari 30, tangkap gerakan kecil/jauh
USE_CLAHE = True                   # Gunakan CLAHE untuk histogram equalization
CLAHE_CLIP_LIMIT = 2.5            # Clip limit CLAHE — sedikit dinaikkan dari 2.0 untuk kontras lebih baik

# ============================================================
# TEMPORAL SMOOTHING & MOTION PERSISTENCE
# ============================================================
MOTION_PERSISTENCE_FRAMES = 3      # Berapa frame motion tetap "aktif" setelah terdeteksi terakhir
MOTION_HISTORY_WEIGHT = 0.4        # Bobot history mask dalam temporal blending (0.0-1.0)

# ============================================================
# SHADOW REMOVAL
# ============================================================
SHADOW_THRESHOLD = 250             # Threshold untuk menghapus shadow (MOG2/KNN: shadow=127)

# ============================================================
# ALARM PARAMETER
# ============================================================
ALARM_ENABLED = True               # Aktifkan/nonaktifkan alarm
ALARM_SOUND_PATH = os.path.join(BASE_DIR, "assets", "alarm.wav")  # Path file suara alarm
ALARM_COOLDOWN = 3                 # Cooldown alarm dalam detik (hindari spam)

# ============================================================
# DISPLAY PARAMETER
# ============================================================
SHOW_PREVIEW = True                # Tampilkan preview window saat proses
BOUNDING_BOX_COLOR = (0, 255, 0)   # Warna bounding box (BGR - hijau)
BOUNDING_BOX_THICKNESS = 2         # Ketebalan garis bounding box
TEXT_COLOR = (0, 0, 255)           # Warna teks status (BGR - merah)
TEXT_FONT_SCALE = 0.7              # Ukuran font teks
TEXT_THICKNESS = 2                 # Ketebalan font teks

# ============================================================
# WEBCAM PARAMETER
# ============================================================
USE_WEBCAM = False                 # True = pakai webcam, False = pakai video file
WEBCAM_INDEX = 0                   # Index webcam (0 = default camera)

# ============================================================
# ADAPTIVE LEARNING PARAMETER
# ============================================================
ADAPTIVE_LEARNING_ENABLED = True       # Aktifkan self-learning
LEARNING_EPOCH_FRAMES = 100            # Evaluasi setiap N frame
LEARNED_PARAMS_PATH = os.path.join(BASE_DIR, "data", "learned_params.json")
MAX_DETECTION_RATE = 0.80              # Batas atas: terlalu sensitif jika di atas ini
MIN_DETECTION_RATE = 0.05              # Batas bawah: kurang sensitif jika di bawah ini
MAX_FALSE_POSITIVE_RATE = 0.40         # Batas FP yang bisa ditoleransi
FEEDBACK_LEARNING_RATE = 0.08          # Seberapa cepat belajar dari feedback operator
AUTO_TUNE_LEARNING_RATE = 0.05         # Seberapa cepat auto-tune adjust parameter

# ============================================================
# NAMA FOLDER PER METODE DETEKSI
# ============================================================
METHOD_DISPLAY_NAMES = {
    "1": "3-Frame Differencing",
    "2": "MOG2",
    "3": "KNN",
    "4": "BG Subtraction + HOG Human Detection",
}


def get_method_output_paths(method_key):

    method_name = METHOD_DISPLAY_NAMES.get(method_key, "unknown")

    # data/output/frames_terdeteksi_<metode>/
    method_frames_dir = os.path.join(OUTPUT_DIR, f"frames_terdeteksi_{method_name}")
    # data/output/video_output_<metode>.avi
    method_video_path = os.path.join(OUTPUT_DIR, f"video_output_{method_name}.avi")

    # results/logs/motion_log_<metode>.csv
    method_log_dir = os.path.join(RESULTS_DIR, "logs")
    method_log_file = os.path.join(method_log_dir, f"motion_log_{method_name}.csv")
    # results/screenshots/screenshots_<metode>/
    method_screenshots_dir = os.path.join(RESULTS_DIR, "screenshots", f"screenshots_{method_name}")

    return {
        "output_dir": OUTPUT_DIR,
        "frames_dir": method_frames_dir,
        "video_path": method_video_path,
        "log_dir": method_log_dir,
        "log_file": method_log_file,
        "screenshots_dir": method_screenshots_dir,
    }


# ============================================================
# BUAT DIREKTORI OTOMATIS
# ============================================================
def create_directories(method_key=None):
    # Direktori dasar (selalu dibuat)
    dirs = [
        os.path.join(BASE_DIR, "data", "raw"),
        OUTPUT_DIR,
        RESULTS_DIR,
        GRAFIK_DIR,
        os.path.join(RESULTS_DIR, "logs"),
        os.path.join(RESULTS_DIR, "screenshots"),
        os.path.join(BASE_DIR, "laporan", "assets"),
    ]

    # Tambahkan direktori spesifik metode jika ada
    if method_key:
        paths = get_method_output_paths(method_key)
        dirs.extend([
            paths["frames_dir"],
            paths["screenshots_dir"],
        ])

    for d in dirs:
        os.makedirs(d, exist_ok=True)

