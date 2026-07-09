# Motion Detection - Laboratorium 🎥

Mini project deteksi aktivitas (motion) di ruang kelas atau laboratorium menggunakan teknik pengolahan citra digital.

## 📋 Deskripsi

Program ini mendeteksi gerakan pada video rekaman ruang kelas/laboratorium menggunakan dua metode:

1. **Frame Differencing** — membandingkan frame saat ini dengan frame sebelumnya
2. **Background Subtraction (MOG2/KNN)** — membangun model background adaptif untuk mendeteksi foreground

Ketika gerakan terdeteksi, program akan:

- Menampilkan status **"MOTION DETECTED"** pada frame
- Menggambar **bounding box** di area gerakan
- Membunyikan **alarm** (beep)
- Menyimpan **log** ke file CSV
- Menyimpan **screenshot** sebagai bukti

## 📁 Struktur Proyek

```
mini project/
│
├── data/
│   ├── raw/                        # video input asli (mp4/avi)
│   │   └── video_kelas.mp4
│   └── output/                     # hasil video/frame setelah diproses
│       ├── frames_terdeteksi/      # frame saat motion terdeteksi
│       └── video_output.avi        # video hasil anotasi bounding box
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # semua parameter konfigurasi
│   ├── preprocessing.py            # grayscale, Gaussian blur, resize
│   ├── frame_difference.py         # metode 1: Frame Differencing
│   ├── background_subtraction.py   # metode 2: MOG2 / KNN
│   ├── alarm.py                    # logika alarm
│   ├── logger.py                   # pencatatan log ke CSV
│   └── main.py                     # entry point program
│
├── notebooks/
│   └── eksperimen_analisis.ipynb   # eksperimen & visualisasi
│
├── results/
│   ├── logs/
│   │   └── motion_log.csv          # log motion detection
│   ├── screenshots/                # bukti frame saat alarm aktif
│   └── grafik/                     # grafik perbandingan metode
│
├── laporan/
│   ├── laporan_mini_project.docx
│   └── assets/
│
├── requirements.txt
└── README.md
```

## ⚙️ Instalasi

1. **Clone atau download** proyek ini

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Siapkan video input:**
   - Letakkan file video (`.mp4` / `.avi`) di folder `data/raw/`
   - Rename menjadi `video_kelas.mp4` atau ubah path di `src/config.py`
   - Alternatif: set `USE_WEBCAM = True` di `config.py` untuk menggunakan webcam

## 🚀 Cara Menjalankan

```bash
python -m src.main
```

Atau:

```bash
python src/main.py
```

Program akan menampilkan menu pilihan metode:

```
==================================================
  MOTION DETECTION - LABORATORIUM
==================================================

Pilih metode deteksi:
  [1] Frame Differencing
  [2] Background Subtraction (MOG2)
  [3] Background Subtraction (KNN)
```

Tekan `q` pada window preview untuk menghentikan program.

## 🔧 Konfigurasi

Semua parameter dapat diatur di `src/config.py`:

| Parameter              | Default  | Keterangan                              |
| ---------------------- | -------- | --------------------------------------- |
| `FRAME_DIFF_THRESHOLD` | 30       | Threshold binarisasi frame differencing |
| `MIN_CONTOUR_AREA`     | 500      | Luas minimum kontur (pixel²)            |
| `GAUSSIAN_BLUR_KERNEL` | (21, 21) | Ukuran kernel Gaussian Blur             |
| `BG_SUB_METHOD`        | "MOG2"   | Metode background subtraction           |
| `ALARM_COOLDOWN`       | 3        | Cooldown alarm (detik)                  |
| `USE_WEBCAM`           | False    | Gunakan webcam sebagai input            |

## 📊 Output

- **Video output:** `data/output/video_output.avi` — video dengan anotasi bounding box
- **Frame terdeteksi:** `data/output/frames_terdeteksi/` — frame individual saat motion
- **Log CSV:** `results/logs/motion_log.csv` — catatan timestamp & status
- **Screenshots:** `results/screenshots/` — bukti visual alarm aktif

## 🛠️ Teknologi

- Python 3.8+
- OpenCV (cv2)
- NumPy
- Matplotlib

## 📄 Lisensi

Proyek ini dibuat untuk keperluan mini project mata kuliah Computer Vision.
