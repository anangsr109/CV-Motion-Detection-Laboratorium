"""
Adaptive Self-Learning Engine untuk Motion Detection.

Modul ini memungkinkan sistem deteksi motion belajar dari:
1. Statistik runtime (detection rate, false positive rate)
2. Feedback operator (tekan Y/N saat deteksi)

Parameter yang dipelajari disimpan ke JSON dan dimuat kembali
saat sesi berikutnya, sehingga model semakin pintar seiring waktu.
"""

import json
import os
import copy
from datetime import datetime
from collections import deque


class ParameterState:
    """Menyimpan parameter saat ini beserta batas min/max yang diperbolehkan."""

    # Definisi parameter yang bisa di-learn beserta batas dan step
    LEARNABLE_PARAMS = {
        "FRAME_DIFF_THRESHOLD": {
            "min": 3, "max": 30, "step": 1, "type": int,
            "description": "Threshold binarisasi frame differencing",
        },
        "MIN_CONTOUR_AREA": {
            "min": 50, "max": 1000, "step": 25, "type": int,
            "description": "Luas minimum kontur untuk dianggap motion",
        },
        "MOG2_VAR_THRESHOLD": {
            "min": 4, "max": 25, "step": 1, "type": int,
            "description": "Variance threshold MOG2 background subtractor",
        },
        "KNN_DIST2_THRESHOLD": {
            "min": 50.0, "max": 400.0, "step": 10.0, "type": float,
            "description": "Distance threshold KNN background subtractor",
        },
        "HOG_HIT_THRESHOLD": {
            "min": -1.5, "max": 0.5, "step": 0.1, "type": float,
            "description": "Confidence threshold HOG pedestrian detector",
        },
        "MIN_HUMAN_HEIGHT": {
            "min": 10, "max": 80, "step": 5, "type": int,
            "description": "Tinggi minimum bounding box manusia (pixel)",
        },
        "NMS_OVERLAP_THRESHOLD": {
            "min": 0.2, "max": 0.8, "step": 0.05, "type": float,
            "description": "Overlap threshold untuk Non-Maximum Suppression",
        },
        "DILATE_ITERATIONS": {
            "min": 1, "max": 6, "step": 1, "type": int,
            "description": "Jumlah iterasi dilasi morfologi",
        },
    }

    def __init__(self):
        self.current_values = {}
        self.history = []  # List of {"timestamp", "param", "old", "new", "reason"}

    def initialize_from_config(self, config_module):
        """Load nilai awal dari config module."""
        for param_name in self.LEARNABLE_PARAMS:
            if hasattr(config_module, param_name):
                self.current_values[param_name] = getattr(config_module, param_name)

    def get(self, param_name):
        """Ambil nilai parameter saat ini."""
        return self.current_values.get(param_name)

    def set(self, param_name, value, reason="manual"):
        """Set nilai parameter dengan validasi batas."""
        if param_name not in self.LEARNABLE_PARAMS:
            return False

        spec = self.LEARNABLE_PARAMS[param_name]
        old_value = self.current_values.get(param_name)

        # Clamp ke batas min/max
        value = spec["type"](value)
        value = max(spec["min"], min(spec["max"], value))

        if old_value == value:
            return False  # Tidak ada perubahan

        self.current_values[param_name] = value
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "param": param_name,
            "old": old_value,
            "new": value,
            "reason": reason,
        })
        return True

    def adjust(self, param_name, direction, reason="auto_tune"):
        """Adjust parameter satu step ke arah tertentu.
        
        Args:
            param_name: Nama parameter
            direction: +1 (naikkan) atau -1 (turunkan)
            reason: Alasan perubahan
        """
        if param_name not in self.LEARNABLE_PARAMS:
            return False

        spec = self.LEARNABLE_PARAMS[param_name]
        current = self.current_values.get(param_name, spec["min"])
        delta = spec["step"] * direction

        new_value = current + delta
        return self.set(param_name, new_value, reason)

    def apply_to_config(self, config_module):
        """Terapkan parameter yang sudah dipelajari ke config module."""
        for param_name, value in self.current_values.items():
            if hasattr(config_module, param_name):
                setattr(config_module, param_name, value)

    def get_recent_adjustments(self, n=5):
        """Ambil N perubahan terakhir."""
        return self.history[-n:] if self.history else []


class FeedbackCollector:
    """Mengumpulkan feedback dari operator (benar/salah) untuk setiap deteksi."""

    def __init__(self, window_size=50):
        self.feedback_history = deque(maxlen=window_size)
        self.total_correct = 0
        self.total_incorrect = 0
        self.pending_feedback = False  # Apakah ada deteksi yang menunggu feedback
        self.last_detection_info = None  # Info deteksi terakhir

    def mark_detection(self, detection_info):
        """Tandai bahwa ada deteksi baru yang bisa diberi feedback."""
        self.pending_feedback = True
        self.last_detection_info = detection_info

    def submit_feedback(self, is_correct):
        """Submit feedback: True = deteksi benar, False = deteksi salah."""
        if not self.pending_feedback:
            return None

        entry = {
            "timestamp": datetime.now().isoformat(),
            "is_correct": is_correct,
            "detection_info": self.last_detection_info,
        }
        self.feedback_history.append(entry)

        if is_correct:
            self.total_correct += 1
        else:
            self.total_incorrect += 1

        self.pending_feedback = False
        return entry

    def get_accuracy(self):
        """Hitung akurasi berdasarkan feedback terkini."""
        if not self.feedback_history:
            return None

        correct = sum(1 for f in self.feedback_history if f["is_correct"])
        return correct / len(self.feedback_history)

    def get_recent_error_type(self, n=10):
        """Analisis tipe error terbanyak dari feedback terakhir.
        
        Returns:
            "false_positive" jika kebanyakan salah deteksi
            "false_negative" jika kebanyakan miss deteksi  
            "balanced" jika seimbang
            None jika belum ada data
        """
        recent = list(self.feedback_history)[-n:]
        if not recent:
            return None

        incorrect = [f for f in recent if not f["is_correct"]]
        if not incorrect:
            return "balanced"

        error_rate = len(incorrect) / len(recent)
        if error_rate > 0.3:
            return "false_positive"
        return "balanced"

    def get_stats(self):
        return {
            "total_correct": self.total_correct,
            "total_incorrect": self.total_incorrect,
            "accuracy": self.get_accuracy(),
            "recent_count": len(self.feedback_history),
        }


class AutoTuner:
    """Otomatis menyesuaikan parameter berdasarkan statistik runtime."""

    def __init__(
        self,
        max_detection_rate=0.80,
        min_detection_rate=0.05,
        max_false_positive_rate=0.40,
        feedback_lr=0.08,
        auto_tune_lr=0.05,
    ):
        self.max_detection_rate = max_detection_rate
        self.min_detection_rate = min_detection_rate
        self.max_false_positive_rate = max_false_positive_rate
        self.feedback_lr = feedback_lr
        self.auto_tune_lr = auto_tune_lr

        # Statistik per epoch
        self.epoch_frames = 0
        self.epoch_motion_frames = 0
        self.epoch_person_frames = 0
        self.epoch_contour_counts = []
        self.epoch_hog_scores = []
        self.epoch_non_human_motion = 0  # Motion tanpa manusia (possible FP)
        self.total_epochs = 0

    def reset_epoch(self):
        """Reset statistik untuk epoch baru."""
        self.epoch_frames = 0
        self.epoch_motion_frames = 0
        self.epoch_person_frames = 0
        self.epoch_contour_counts = []
        self.epoch_hog_scores = []
        self.epoch_non_human_motion = 0

    def record_frame(self, motion_detected, person_count, contour_count, hog_scores=None):
        """Rekam statistik satu frame."""
        self.epoch_frames += 1

        if motion_detected:
            self.epoch_motion_frames += 1
            if person_count > 0:
                self.epoch_person_frames += 1
            else:
                self.epoch_non_human_motion += 1

        self.epoch_contour_counts.append(contour_count)

        if hog_scores:
            self.epoch_hog_scores.extend(hog_scores)

    def compute_epoch_stats(self):
        """Hitung statistik epoch saat ini."""
        if self.epoch_frames == 0:
            return None

        detection_rate = self.epoch_motion_frames / self.epoch_frames
        person_rate = self.epoch_person_frames / max(self.epoch_motion_frames, 1)
        fp_rate = self.epoch_non_human_motion / max(self.epoch_motion_frames, 1)
        avg_contours = (
            sum(self.epoch_contour_counts) / len(self.epoch_contour_counts)
            if self.epoch_contour_counts else 0
        )
        avg_hog_score = (
            sum(self.epoch_hog_scores) / len(self.epoch_hog_scores)
            if self.epoch_hog_scores else 0
        )

        return {
            "detection_rate": detection_rate,
            "person_rate": person_rate,
            "false_positive_rate": fp_rate,
            "avg_contours": avg_contours,
            "avg_hog_score": avg_hog_score,
            "total_frames": self.epoch_frames,
            "motion_frames": self.epoch_motion_frames,
            "person_frames": self.epoch_person_frames,
        }

    def suggest_adjustments(self, stats, feedback_accuracy=None):
        """Berdasarkan statistik epoch, sarankan perubahan parameter.
        
        Returns:
            List of {"param": str, "direction": int, "reason": str}
        """
        if stats is None:
            return []

        adjustments = []

        # ====== Rule 1: Detection rate terlalu tinggi (terlalu sensitif) ======
        if stats["detection_rate"] > self.max_detection_rate:
            adjustments.append({
                "param": "FRAME_DIFF_THRESHOLD",
                "direction": +1,
                "reason": f"detection_rate={stats['detection_rate']:.1%} > {self.max_detection_rate:.0%} (terlalu sensitif)",
            })
            adjustments.append({
                "param": "MIN_CONTOUR_AREA",
                "direction": +1,
                "reason": f"detection_rate={stats['detection_rate']:.1%} > {self.max_detection_rate:.0%} (filter noise)",
            })
            adjustments.append({
                "param": "MOG2_VAR_THRESHOLD",
                "direction": +1,
                "reason": f"detection_rate={stats['detection_rate']:.1%} (kurangi sensitifitas BG subtractor)",
            })

        # ====== Rule 2: Detection rate terlalu rendah (kurang sensitif) ======
        elif stats["detection_rate"] < self.min_detection_rate:
            adjustments.append({
                "param": "FRAME_DIFF_THRESHOLD",
                "direction": -1,
                "reason": f"detection_rate={stats['detection_rate']:.1%} < {self.min_detection_rate:.0%} (kurang sensitif)",
            })
            adjustments.append({
                "param": "MIN_CONTOUR_AREA",
                "direction": -1,
                "reason": f"detection_rate={stats['detection_rate']:.1%} < {self.min_detection_rate:.0%} (tangkap gerakan kecil)",
            })
            adjustments.append({
                "param": "MOG2_VAR_THRESHOLD",
                "direction": -1,
                "reason": f"detection_rate={stats['detection_rate']:.1%} (tingkatkan sensitifitas BG subtractor)",
            })

        # ====== Rule 3: Terlalu banyak false positive ======
        if stats["false_positive_rate"] > self.max_false_positive_rate:
            adjustments.append({
                "param": "MIN_CONTOUR_AREA",
                "direction": +1,
                "reason": f"FP_rate={stats['false_positive_rate']:.1%} > {self.max_false_positive_rate:.0%}",
            })
            adjustments.append({
                "param": "HOG_HIT_THRESHOLD",
                "direction": +1,
                "reason": f"FP_rate={stats['false_positive_rate']:.1%} (perketat HOG)",
            })
            adjustments.append({
                "param": "MIN_HUMAN_HEIGHT",
                "direction": +1,
                "reason": f"FP_rate={stats['false_positive_rate']:.1%} (filter objek kecil)",
            })

        # ====== Rule 4: Terlalu banyak kontur (noisy) ======
        if stats["avg_contours"] > 20:
            adjustments.append({
                "param": "DILATE_ITERATIONS",
                "direction": -1,
                "reason": f"avg_contours={stats['avg_contours']:.0f} (terlalu banyak, kurangi dilasi)",
            })

        elif stats["avg_contours"] < 1 and stats["detection_rate"] < 0.20:
            adjustments.append({
                "param": "DILATE_ITERATIONS",
                "direction": +1,
                "reason": f"avg_contours={stats['avg_contours']:.0f} (terlalu sedikit, tambah dilasi)",
            })

        # ====== Rule 5: Feedback dari operator ======
        if feedback_accuracy is not None and feedback_accuracy < 0.6:
            # Akurasi rendah → model banyak salah → perketat parameter
            adjustments.append({
                "param": "MIN_CONTOUR_AREA",
                "direction": +1,
                "reason": f"feedback_accuracy={feedback_accuracy:.1%} (operator menilai banyak salah)",
            })
            adjustments.append({
                "param": "HOG_HIT_THRESHOLD",
                "direction": +1,
                "reason": f"feedback_accuracy={feedback_accuracy:.1%} (perketat deteksi)",
            })

        return adjustments


class KnowledgeStore:
    """Menyimpan dan memuat parameter yang dipelajari ke/dari file JSON."""

    def __init__(self, filepath):
        self.filepath = filepath
        self._ensure_directory()

    def _ensure_directory(self):
        """Pastikan direktori untuk file JSON ada."""
        directory = os.path.dirname(self.filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def save(self, param_state, auto_tuner, feedback_collector):
        """Simpan semua knowledge ke file JSON."""
        stats = auto_tuner.compute_epoch_stats()
        fb_stats = feedback_collector.get_stats()

        data = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "total_training_epochs": auto_tuner.total_epochs,
            "parameters": copy.deepcopy(param_state.current_values),
            "learning_history": param_state.get_recent_adjustments(20),
            "performance_stats": {
                "last_epoch_stats": stats,
                "feedback_accuracy": fb_stats["accuracy"],
                "total_feedback_correct": fb_stats["total_correct"],
                "total_feedback_incorrect": fb_stats["total_incorrect"],
            },
        }

        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            print(f"[LEARNING] WARNING: Gagal menyimpan knowledge: {e}")
            return False

    def load(self):
        """Muat knowledge dari file JSON. Return None jika tidak ada."""
        if not os.path.exists(self.filepath):
            return None

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, Exception) as e:
            print(f"[LEARNING] WARNING: Gagal memuat knowledge: {e}")
            return None

    def exists(self):
        return os.path.exists(self.filepath)


class AdaptiveLearner:
    """Kelas utama yang mengorkestrasi seluruh proses learning.
    
    Usage:
        learner = AdaptiveLearner(config_module)
        learner.load_knowledge()       # Load dari sesi sebelumnya
        learner.apply_to_config()      # Apply ke detectors
        
        # Dalam loop deteksi:
        learner.record_frame(motion, persons, contours)
        if learner.should_tune():
            learner.run_auto_tune()
        
        # Feedback dari operator:
        learner.submit_feedback(is_correct=True)
        
        # Saat selesai:
        learner.save_knowledge()
    """

    def __init__(self, config_module, epoch_frames=100):
        self.config = config_module
        self.epoch_frames = epoch_frames

        # Sub-components
        self.param_state = ParameterState()
        self.feedback = FeedbackCollector(window_size=50)
        self.auto_tuner = AutoTuner(
            max_detection_rate=getattr(config_module, "MAX_DETECTION_RATE", 0.80),
            min_detection_rate=getattr(config_module, "MIN_DETECTION_RATE", 0.05),
            max_false_positive_rate=getattr(config_module, "MAX_FALSE_POSITIVE_RATE", 0.40),
            feedback_lr=getattr(config_module, "FEEDBACK_LEARNING_RATE", 0.08),
            auto_tune_lr=getattr(config_module, "AUTO_TUNE_LEARNING_RATE", 0.05),
        )
        self.knowledge_store = KnowledgeStore(
            getattr(config_module, "LEARNED_PARAMS_PATH", "data/learned_params.json")
        )

        # State
        self.frame_count = 0
        self.session_adjustments = 0
        self.is_enabled = getattr(config_module, "ADAPTIVE_LEARNING_ENABLED", True)
        self.learning_score = 50  # Skor learning 0-100 (mulai dari 50)

        # Initialize parameter state dari config
        self.param_state.initialize_from_config(config_module)

    def load_knowledge(self):
        """Muat knowledge dari sesi sebelumnya."""
        if not self.is_enabled:
            return False

        data = self.knowledge_store.load()
        if data is None:
            print("[LEARNING] [NEW] Sesi pertama -- mulai dari parameter default")
            return False

        # Restore learned parameters
        if "parameters" in data:
            for param_name, value in data["parameters"].items():
                self.param_state.set(param_name, value, reason="loaded_from_knowledge")

        # Restore epoch count
        self.auto_tuner.total_epochs = data.get("total_training_epochs", 0)

        # Calculate learning score from history
        perf = data.get("performance_stats", {})
        fb_accuracy = perf.get("feedback_accuracy")
        if fb_accuracy is not None:
            self.learning_score = int(fb_accuracy * 100)
        else:
            # Score based on number of epochs trained
            epochs = data.get("total_training_epochs", 0)
            self.learning_score = min(50 + epochs * 2, 95)

        param_count = len(data.get("parameters", {}))
        epochs = data.get("total_training_epochs", 0)
        print(f"[LEARNING] [OK] Knowledge dimuat -- {param_count} parameter, {epochs} epoch sebelumnya")
        print(f"[LEARNING] [BRAIN] Learning Score: {self.learning_score}/100")

        return True

    def apply_to_config(self):
        """Terapkan parameter yang sudah dipelajari ke config module."""
        self.param_state.apply_to_config(self.config)

    def record_frame(self, motion_detected, person_count, contour_count, hog_scores=None):
        """Rekam statistik satu frame."""
        if not self.is_enabled:
            return

        self.frame_count += 1
        self.auto_tuner.record_frame(motion_detected, person_count, contour_count, hog_scores)

        # Tandai deteksi untuk feedback
        if motion_detected:
            self.feedback.mark_detection({
                "frame": self.frame_count,
                "person_count": person_count,
                "contour_count": contour_count,
            })

    def should_tune(self):
        """Apakah sudah waktunya menjalankan auto-tune?"""
        return (
            self.is_enabled
            and self.auto_tuner.epoch_frames >= self.epoch_frames
        )

    def run_auto_tune(self):
        """Jalankan auto-tune berdasarkan statistik epoch saat ini."""
        if not self.is_enabled:
            return []

        stats = self.auto_tuner.compute_epoch_stats()
        feedback_accuracy = self.feedback.get_accuracy()

        # Dapatkan saran adjustment
        adjustments = self.auto_tuner.suggest_adjustments(stats, feedback_accuracy)

        # Terapkan adjustments
        applied = []
        for adj in adjustments:
            success = self.param_state.adjust(
                adj["param"],
                adj["direction"],
                reason=adj["reason"],
            )
            if success:
                applied.append(adj)
                self.session_adjustments += 1

        # Terapkan ke config
        if applied:
            self.apply_to_config()
            self._update_learning_score(stats, feedback_accuracy)

        # Log hasil
        self.auto_tuner.total_epochs += 1
        epoch_num = self.auto_tuner.total_epochs

        if applied:
            print(f"\n[LEARNING] [BRAIN] Epoch {epoch_num} -- {len(applied)} parameter disesuaikan:")
            for adj in applied:
                param = adj["param"]
                new_val = self.param_state.get(param)
                print(f"  -> {param} = {new_val} ({adj['reason']})")
            print(f"[LEARNING] Score: {self.learning_score}/100")
        else:
            print(f"[LEARNING] Epoch {epoch_num} -- parameter sudah optimal [OK]")

        if stats:
            print(
                f"[LEARNING] Stats: detection={stats['detection_rate']:.1%} | "
                f"FP={stats['false_positive_rate']:.1%} | "
                f"contours={stats['avg_contours']:.1f}"
            )

        # Reset epoch
        self.auto_tuner.reset_epoch()

        return applied

    def submit_feedback(self, is_correct):
        """Submit feedback dari operator: True = benar, False = salah."""
        if not self.is_enabled:
            return

        entry = self.feedback.submit_feedback(is_correct)
        if entry is None:
            return

        if is_correct:
            print(f"[LEARNING] [OK] Feedback: BENAR -- parameter diperkuat")
            self._update_learning_score_from_feedback(True)
        else:
            print(f"[LEARNING] [X] Feedback: SALAH -- parameter akan disesuaikan")
            self._update_learning_score_from_feedback(False)

            # Langsung adjust parameter berdasarkan feedback negatif
            error_type = self.feedback.get_recent_error_type()
            if error_type == "false_positive":
                # Terlalu banyak salah deteksi → perketat
                self.param_state.adjust("MIN_CONTOUR_AREA", +1, "feedback: false positive")
                self.param_state.adjust("FRAME_DIFF_THRESHOLD", +1, "feedback: false positive")
                self.apply_to_config()

    def _update_learning_score(self, stats, feedback_accuracy):
        """Update learning score berdasarkan performa."""
        if stats is None:
            return

        # Score naik jika detection rate dalam range ideal (10-70%)
        ideal_low, ideal_high = 0.10, 0.70
        if ideal_low <= stats["detection_rate"] <= ideal_high:
            self.learning_score = min(self.learning_score + 2, 100)
        else:
            self.learning_score = max(self.learning_score - 1, 0)

        # Score naik jika FP rate rendah
        if stats["false_positive_rate"] < 0.20:
            self.learning_score = min(self.learning_score + 1, 100)

        # Score dari feedback accuracy
        if feedback_accuracy is not None:
            if feedback_accuracy > 0.8:
                self.learning_score = min(self.learning_score + 3, 100)
            elif feedback_accuracy < 0.5:
                self.learning_score = max(self.learning_score - 2, 0)

    def _update_learning_score_from_feedback(self, is_correct):
        """Update score dari satu feedback."""
        if is_correct:
            self.learning_score = min(self.learning_score + 1, 100)
        else:
            self.learning_score = max(self.learning_score - 2, 0)

    def save_knowledge(self):
        """Simpan semua knowledge yang dipelajari ke file JSON."""
        if not self.is_enabled:
            return False

        success = self.knowledge_store.save(
            self.param_state,
            self.auto_tuner,
            self.feedback,
        )

        if success:
            print(f"\n[LEARNING] [SAVE] Knowledge tersimpan ke: {self.knowledge_store.filepath}")
            print(f"[LEARNING] Total epoch: {self.auto_tuner.total_epochs}")
            print(f"[LEARNING] Adjustments sesi ini: {self.session_adjustments}")
            print(f"[LEARNING] Learning Score: {self.learning_score}/100")
            fb_stats = self.feedback.get_stats()
            if fb_stats["accuracy"] is not None:
                print(f"[LEARNING] Feedback accuracy: {fb_stats['accuracy']:.1%}")
        return success

    def reset_learning(self):
        """Reset semua learning ke default config."""
        self.param_state = ParameterState()
        self.param_state.initialize_from_config(self.config)
        self.feedback = FeedbackCollector(window_size=50)
        self.auto_tuner.reset_epoch()
        self.auto_tuner.total_epochs = 0
        self.session_adjustments = 0
        self.learning_score = 50
        self.frame_count = 0

        # Hapus file knowledge jika ada
        if os.path.exists(self.knowledge_store.filepath):
            os.remove(self.knowledge_store.filepath)

        self.apply_to_config()
        print("[LEARNING] [RESET] Learning direset ke parameter default")

    def get_status(self):
        """Dapatkan status learning saat ini untuk overlay."""
        return {
            "enabled": self.is_enabled,
            "epoch": self.auto_tuner.total_epochs,
            "frame_in_epoch": self.auto_tuner.epoch_frames,
            "epoch_size": self.epoch_frames,
            "learning_score": self.learning_score,
            "session_adjustments": self.session_adjustments,
            "params_learned": len(self.param_state.current_values),
            "feedback_stats": self.feedback.get_stats(),
            "has_knowledge_file": self.knowledge_store.exists(),
        }

    def get_overlay_text(self):
        """Hasilkan teks untuk on-screen overlay."""
        status = self.get_status()
        stats = self.auto_tuner.compute_epoch_stats()

        lines = []
        lines.append(
            f"LEARNING: Epoch {status['epoch']} | "
            f"Frame {status['frame_in_epoch']}/{status['epoch_size']}"
        )

        if stats and stats["total_frames"] > 0:
            lines.append(
                f"Det: {stats['detection_rate']:.0%} | "
                f"FP: {stats['false_positive_rate']:.0%} | "
                f"Score: {status['learning_score']}/100"
            )
        else:
            lines.append(f"Score: {status['learning_score']}/100")

        lines.append("[Y]=Benar [N]=Salah [R]=Reset")

        return lines
