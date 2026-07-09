import time
import os
import threading
import winsound
from src.config import ALARM_ENABLED, ALARM_COOLDOWN, ALARM_SOUND_PATH


class Alarm:
    def __init__(self, enabled=ALARM_ENABLED, cooldown=ALARM_COOLDOWN):
        self.enabled = enabled
        self.cooldown = cooldown
        self.last_alarm_time = 0
        self.alarm_count = 0

    def trigger(self):
        if not self.enabled:
            return False

        current_time = time.time()
        elapsed = current_time - self.last_alarm_time

        if elapsed < self.cooldown:
            return False

        self.last_alarm_time = current_time
        self.alarm_count += 1

        # Jalankan alarm di thread terpisah agar tidak blocking
        alarm_thread = threading.Thread(target=self._play_alarm, daemon=True)
        alarm_thread.start()

        return True

    def _play_alarm(self):
        try:
            # Cek apakah file suara custom tersedia
            if os.path.exists(ALARM_SOUND_PATH):
                winsound.PlaySound(ALARM_SOUND_PATH, winsound.SND_FILENAME)
            else:
                # Fallback: gunakan system beep
                winsound.Beep(1000, 500)  # Frekuensi 1000Hz, durasi 500ms
        except Exception:
            # Fallback terakhir: print ke console
            print("\a")  # Bell character

    def get_status(self):
        return {
            "enabled": self.enabled,
            "alarm_count": self.alarm_count,
            "cooldown": self.cooldown,
            "last_triggered": self.last_alarm_time,
        }

    def reset(self):
        self.last_alarm_time = 0
        self.alarm_count = 0
