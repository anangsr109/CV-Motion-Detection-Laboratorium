import cv2
import os
import sys
import time

# Tambahkan parent directory ke path agar bisa import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as config_module
from src.config import (
    VIDEO_INPUT_PATH,
    SHOW_PREVIEW,
    USE_WEBCAM,
    WEBCAM_INDEX,
    TEXT_COLOR,
    TEXT_FONT_SCALE,
    TEXT_THICKNESS,
    BOUNDING_BOX_COLOR,
    BOUNDING_BOX_THICKNESS,
    HUMAN_DETECTION_ENABLED,
    ADAPTIVE_LEARNING_ENABLED,
    LEARNING_EPOCH_FRAMES,
    create_directories,
    get_method_output_paths,
)
from src.preprocessing import preprocess_frame
from src.frame_difference import FrameDifference
from src.background_subtraction import BackgroundSubtraction
from src.human_detector import HumanDetector
from src.alarm import Alarm
from src.logger import MotionLogger
from src.bbox_stabilizer import BBoxStabilizer
from src.adaptive_learner import AdaptiveLearner


def select_method():
    print("\n" + "=" * 60)
    print("  MOTION DETECTION - LABORATORIUM (Enhanced)")
    print("=" * 60)
    print("\nPilih metode deteksi:")
    print("  [1] 3-Frame Differencing")
    print("  [2] Background Subtraction (MOG2)")
    print("  [3] Background Subtraction (KNN)")
    print("  [4] BG Subtraction + HOG Human Detection (Akurasi Tinggi) ★")
    print()

    while True:
        choice = input("Masukkan pilihan (1/2/3/4): ").strip()
        if choice == "1":
            return FrameDifference(), False, "1"
        elif choice == "2":
            return BackgroundSubtraction(method="MOG2"), False, "2"
        elif choice == "3":
            return BackgroundSubtraction(method="KNN"), False, "3"
        elif choice == "4":
            return BackgroundSubtraction(method="MOG2"), True, "4"
        else:
            print("Pilihan tidak valid. Silakan coba lagi.")


def run_detection():

    # Pilih metode
    detector, use_human_detection, method_key = select_method()
    method_name = detector.get_method_name()

    # Buat semua direktori yang dibutuhkan (termasuk folder spesifik metode)
    create_directories(method_key)

    # Dapatkan path output spesifik per metode
    method_paths = get_method_output_paths(method_key)
    method_video_path = method_paths["video_path"]
    method_frames_dir = method_paths["frames_dir"]
    method_screenshots_dir = method_paths["screenshots_dir"]
    method_log_file = method_paths["log_file"]

    # Inisialisasi Human Detector jika dipilih
    human_detector = None
    if use_human_detection:
        human_detector = HumanDetector()
        method_name = f"{method_name} + HOG Human Detection"
        print(f"\n★ Mode Akurasi Tinggi: HOG + SVM Pedestrian Detection aktif")

    print(f"\nMetode terpilih: {method_name}")
    print(f"Output folder : {method_paths['output_dir']}")
    print(f"Results folder: {method_paths['log_dir']}")
    print("-" * 60)

    # ====== Inisialisasi Adaptive Learner (Self-Learning) ======
    learner = None
    if ADAPTIVE_LEARNING_ENABLED:
        learner = AdaptiveLearner(
            config_module=config_module,
            epoch_frames=LEARNING_EPOCH_FRAMES,
        )
        # Muat knowledge dari sesi sebelumnya (jika ada)
        learner.load_knowledge()
        # Terapkan parameter yang sudah dipelajari
        learner.apply_to_config()
        print(f"\n[BRAIN] Self-Learning: AKTIF (epoch setiap {LEARNING_EPOCH_FRAMES} frame)")
        print(f"   Tekan [Y]=Deteksi Benar  [N]=Deteksi Salah  [R]=Reset Learning")
        print("-" * 60)

    # Inisialisasi komponen
    alarm = Alarm()
    logger = MotionLogger(log_path=method_log_file)
    stabilizer = BBoxStabilizer(
        smoothing=0.45,        # EMA factor: 0=sangat smooth, 1=tanpa smoothing
        max_missing=5,         # Box tetap tampil 5 frame setelah hilang
        min_frames_to_show=2,  # Box baru harus muncul 2 frame sebelum ditampilkan
        iou_threshold=0.2,     # Minimum IoU untuk dianggap box yang sama
    )

    # Buka video source
    if USE_WEBCAM:
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        print(f"Menggunakan webcam (index: {WEBCAM_INDEX})")
    else:
        if not os.path.exists(VIDEO_INPUT_PATH):
            print(f"\n[ERROR] Video tidak ditemukan: {VIDEO_INPUT_PATH}")
            print("Pastikan file video sudah ada di folder data/raw/")
            print("Atau set USE_WEBCAM = True di config.py untuk menggunakan webcam.")
            return
        cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
        print(f"Membaca video: {VIDEO_INPUT_PATH}")

    if not cap.isOpened():
        print("[ERROR] Gagal membuka video source!")
        return

    # Ambil info video
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps} | Resolusi: {frame_width}x{frame_height} | Total frame: {total_frames}")
    print(f"Tekan 'q' untuk keluar\n")

    # Inisialisasi video writer untuk output
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(method_video_path, fourcc, fps, (640, 480))

    frame_count = 0
    motion_frame_count = 0
    total_person_detected = 0
    max_persons_in_frame = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n[INFO] Video selesai / tidak ada frame lagi.")
                break

            frame_count += 1

            # Preprocessing
            frame_resized, frame_preprocessed = preprocess_frame(frame)

            # Deteksi motion (Stage 1)
            motion_detected, annotated_frame, contours = detector.detect(
                frame_preprocessed, frame_resized
            )

            # ====== Stabilisasi Bounding Box ======
            # Konversi contours ke format deteksi untuk stabilizer
            raw_detections = []
            for c in contours:
                area = cv2.contourArea(c)
                x, y, w, h = cv2.boundingRect(c)
                raw_detections.append({
                    "bbox": (x, y, w, h),
                    "area": area,
                    "contour": c,
                })

            # Update stabilizer dan dapatkan bounding box yang stabil
            stable_boxes = stabilizer.update(raw_detections)

            # Gambar ulang bounding box yang stabil pada frame bersih
            annotated_frame = frame_resized.copy()
            for sb in stable_boxes:
                x, y, w, h = sb["bbox"]
                # Warna: hijau jika aktif, abu-abu transparan jika persistence
                if sb["is_active"]:
                    box_color = BOUNDING_BOX_COLOR  # Hijau
                    thickness = BOUNDING_BOX_THICKNESS
                else:
                    box_color = (100, 200, 100)  # Hijau pudar (persistence)
                    thickness = 1

                cv2.rectangle(
                    annotated_frame,
                    (x, y),
                    (x + w, y + h),
                    box_color,
                    thickness,
                )
                # Label area
                cv2.putText(
                    annotated_frame,
                    f"A:{sb['area']:.0f}",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    box_color,
                    1,
                )

            # Update motion status berdasarkan stable boxes
            motion_detected = len(stable_boxes) > 0

            # Deteksi manusia (Stage 2) — jika mode akurasi tinggi
            person_count = 0
            if human_detector is not None and motion_detected:
                # Verifikasi apakah motion berasal dari manusia
                person_count, annotated_frame, human_boxes = (
                    human_detector.detect_and_annotate(
                        annotated_frame, motion_contours=contours
                    )
                )

                # Jika tidak ada manusia terdeteksi oleh HOG,
                # masih hitung motion tapi tandai sebagai unverified
                if person_count == 0:
                    # Cek kontur yang memenuhi aspect ratio (fallback)
                    human_contours, _ = human_detector.validate_motion_as_human(
                        frame_resized, contours
                    )
                    person_count = len(human_contours)

            elif human_detector is None and motion_detected:
                # Mode tanpa HOG: estimasi person count dari stable boxes
                person_count = len([sb for sb in stable_boxes if sb["is_active"]])

            # Update statistik
            if person_count > max_persons_in_frame:
                max_persons_in_frame = person_count

            # Tambahkan teks status pada frame
            if motion_detected and person_count > 0:
                status_text = f"STATUS: {person_count} ORANG TERDETEKSI!"
                color = (0, 0, 255)  # Merah
            elif motion_detected:
                status_text = "STATUS: MOTION DETECTED (non-human)"
                color = (0, 165, 255)  # Oranye
            else:
                status_text = "STATUS: No Motion"
                color = (0, 255, 0)  # Hijau

            cv2.putText(
                annotated_frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_FONT_SCALE,
                color,
                TEXT_THICKNESS,
            )

            # Tambahkan info metode
            cv2.putText(
                annotated_frame,
                f"Metode: {method_name}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            # Tambahkan info frame dan person count
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count} | Persons: {person_count}",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Tambahkan info kontur
            cv2.putText(
                annotated_frame,
                f"Contours: {len(contours)}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )

            if motion_detected:
                motion_frame_count += 1
                total_person_detected += person_count

                # Trigger alarm hanya jika ada manusia terdeteksi
                if person_count > 0:
                    alarm.trigger()

                # Simpan frame yang terdeteksi motion
                frame_filename = f"motion_frame_{frame_count:06d}.jpg"
                frame_path = os.path.join(method_frames_dir, frame_filename)
                cv2.imwrite(frame_path, annotated_frame)

                # Simpan screenshot untuk bukti
                if motion_frame_count <= 10:  # Batasi 10 screenshot pertama
                    screenshot_path = os.path.join(
                        method_screenshots_dir, f"screenshot_{frame_count:06d}.jpg"
                    )
                    cv2.imwrite(screenshot_path, annotated_frame)

                # Print status ke console
                print(
                    f"[Frame {frame_count:>6}] ⚠ MOTION | "
                    f"Persons: {person_count} | Kontur: {len(contours)} | "
                    f"Metode: {method_name}"
                )

            # ====== Self-Learning: Rekam statistik & auto-tune ======
            if learner is not None:
                learner.record_frame(
                    motion_detected=motion_detected,
                    person_count=person_count,
                    contour_count=len(contours),
                )

                # Jalankan auto-tune setiap epoch
                if learner.should_tune():
                    applied = learner.run_auto_tune()
                    if applied:
                        # Re-inisialisasi detector dengan parameter baru jika perlu
                        # (parameter sudah di-apply ke config_module)
                        pass

                # Gambar overlay learning status di frame
                overlay_lines = learner.get_overlay_text()
                overlay_y = annotated_frame.shape[0] - 20 * len(overlay_lines) - 10
                for i, line in enumerate(overlay_lines):
                    y_pos = overlay_y + i * 20
                    # Background gelap untuk teks
                    (tw, th), _ = cv2.getTextSize(
                        line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )
                    cv2.rectangle(
                        annotated_frame,
                        (8, y_pos - th - 4),
                        (14 + tw, y_pos + 4),
                        (0, 0, 0),
                        -1,
                    )
                    # Teks kuning untuk learning info
                    cv2.putText(
                        annotated_frame,
                        line,
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),  # Kuning
                        1,
                    )

            # Log setiap frame (dengan person_count)
            logger.log(
                frame_count,
                motion_detected,
                method_name,
                len(contours),
                person_count,
            )

            # Tulis ke video output
            out.write(annotated_frame)

            # Tampilkan preview
            if SHOW_PREVIEW:
                cv2.imshow("Motion Detection - Laboratorium", annotated_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                # Tekan 'q' untuk keluar
                if key == ord("q"):
                    print("\n[INFO] Dihentikan oleh pengguna (tombol 'q').")
                    break

                # ====== Self-Learning: Feedback dari operator ======
                if learner is not None:
                    if key == ord("y") or key == ord("Y"):
                        learner.submit_feedback(is_correct=True)
                    elif key == ord("n") or key == ord("N"):
                        learner.submit_feedback(is_correct=False)
                    elif key == ord("r") or key == ord("R"):
                        learner.reset_learning()
                        print("[LEARNING] [RESET] Learning direset! Parameter kembali ke default.")

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh pengguna (Ctrl+C).")

    finally:
        # ====== Self-Learning: Simpan knowledge sebelum exit ======
        if learner is not None:
            learner.save_knowledge()

        # Hitung durasi
        elapsed = time.time() - start_time

        # Tampilkan ringkasan
        print("\n" + "=" * 60)
        print("  RINGKASAN HASIL DETEKSI")
        print("=" * 60)
        print(f"  Metode             : {method_name}")
        print(f"  Total frame        : {frame_count}")
        print(f"  Frame motion       : {motion_frame_count}")
        print(f"  Persentase motion  : {(motion_frame_count/max(frame_count,1))*100:.1f}%")
        print(f"  Max orang/frame    : {max_persons_in_frame}")
        print(f"  Total deteksi orang: {total_person_detected}")
        print(f"  Durasi proses      : {elapsed:.1f} detik")
        print(f"  Alarm dipicu       : {alarm.get_status()['alarm_count']} kali")
        print(f"  Log tersimpan      : {logger.get_summary()['log_file']}")
        print(f"  Video output       : {method_video_path}")
        print(f"  Frames output      : {method_frames_dir}")
        print(f"  Screenshots        : {method_screenshots_dir}")
        if human_detector:
            print(f"  Human Detection    : ★ Aktif (HOG + SVM)")

        # ====== Self-Learning Ringkasan ======
        if learner is not None:
            status = learner.get_status()
            print(f"\n  --- SELF-LEARNING ---")
            print(f"  Learning Status    : {'[BRAIN] Aktif' if status['enabled'] else '[PAUSE] Nonaktif'}")
            print(f"  Learning Score     : {status['learning_score']}/100")
            print(f"  Total Epoch        : {status['epoch']}")
            print(f"  Adjustments        : {status['session_adjustments']} parameter disesuaikan")
            fb = status['feedback_stats']
            if fb['accuracy'] is not None:
                print(f"  Feedback Accuracy  : {fb['accuracy']:.1%} ({fb['total_correct']}v / {fb['total_incorrect']}x)")
            print(f"  Knowledge File     : {learner.knowledge_store.filepath}")

        print("=" * 60)

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection()

