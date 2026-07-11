# Motion Detection - Laboratorium 🎥

Mini project deteksi aktivitas (motion) di ruang kelas atau laboratorium menggunakan teknik pengolahan citra digital.

## 📋 Deskripsi

Program ini mendeteksi gerakan pada video rekaman ruang kelas/laboratorium menggunakan beberapa metode:

1. **Frame Differencing** — membandingkan frame saat ini dengan frame sebelumnya
2. **Background Subtraction (MOG2/KNN)** — membangun model background adaptif untuk mendeteksi foreground
3. **HOG + SVM Human Detection** — verifikasi apakah gerakan berasal dari manusia
4. **Deep Learning Human Detection (OpenCV DNN / MobileNet-SSD)** — deteksi manusia berbasis jaringan saraf, jauh lebih presisi daripada HOG untuk orang yang berdiri/menghadap kamera

### Peningkatan kualitas deteksi

Pipeline deteksi telah ditingkatkan agar tidak lagi "selalu MOTION":

- **Motion gate tingkat-frame** — sebuah frame hanya dianggap MOTION bila total luas area bergerak melewati ambang, sehingga noise sensor tidak lagi memicu alarm palsu.
- **Parameter di-tuning ulang** agar tahan noise (lihat `benchmark.py`): pada video sampel, persentase frame "MOTION" turun dari **~100%** menjadi **~50-58%** (MOG2/Frame Diff).
- **Deteksi manusia deep-learning (DNN)** menggantikan HOG yang rawan false positive: pada video sampel, deteksi orang turun dari **~9-10 orang/frame (banyak palsu)** pada HOG menjadi **~1-2 orang/frame ber-confidence tinggi** pada DNN, dengan kecepatan ~2-3x lebih cepat.

> **Catatan jujur:** MobileNet-SSD dilatih untuk orang berdiri/menghadap kamera. Pada video CCTV kelas dengan sudut dari atas dan siswa duduk membelakangi kamera, deteksi full-body tetap terbatas (recall rendah) — untuk kondisi ini, deteksi berbasis **MOTION** adalah yang paling andal. DNN paling optimal untuk webcam/pintu/lab dengan orang menghadap kamera.

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
├── models/                         # model DNN (auto-download; .caffemodel di-gitignore)
│   └── MobileNetSSD_deploy.prototxt
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # semua parameter konfigurasi
│   ├── preprocessing.py            # grayscale, Gaussian blur, resize
│   ├── frame_difference.py         # metode 1: Frame Differencing
│   ├── background_subtraction.py   # metode 2: MOG2 / KNN
│   ├── human_detector.py           # metode 4: HOG + SVM
│   ├── dnn_detector.py             # metode 5: MobileNet-SSD (OpenCV DNN)
│   ├── bbox_stabilizer.py          # stabilisasi/tracking bounding box
│   ├── adaptive_learner.py         # self-learning auto-tune parameter
│   ├── alarm.py                    # logika alarm
│   ├── logger.py                   # pencatatan log ke CSV
│   └── main.py                     # entry point program
│
├── benchmark.py                    # benchmark headless semua metode
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
  MOTION DETECTION - LABORATORIUM (Enhanced)
==================================================

Pilih metode deteksi:
  [1] 3-Frame Differencing
  [2] Background Subtraction (MOG2)
  [3] Background Subtraction (KNN)
  [4] BG Subtraction + HOG Human Detection ★
  [5] BG Subtraction + DNN Human Detection (Deep Learning) ★★
```

Tekan `q` pada window preview untuk menghentikan program.

Saat pertama kali memilih metode **[5]**, model MobileNet-SSD (~23 MB) akan
otomatis di-download ke folder `models/`. Jika download gagal (mis. tanpa
internet), program otomatis fallback ke deteksi HOG.

### Benchmark (headless, tanpa GUI)

Untuk membandingkan semua metode di video sampel tanpa membuka window:

```bash
python benchmark.py 400     # proses 400 frame pertama
```

## 🔧 Konfigurasi

Semua parameter dapat diatur di `src/config.py`:

| Parameter              | Default  | Keterangan                              |
| ---------------------- | -------- | --------------------------------------- |
| `FRAME_DIFF_THRESHOLD`       | 25       | Threshold binarisasi frame differencing         |
| `MIN_CONTOUR_AREA`          | 800      | Luas minimum kontur (pixel²)                    |
| `MOTION_MIN_TOTAL_AREA`     | 4000     | Total luas gerakan minimum agar frame = MOTION  |
| `MOTION_SINGLE_AREA_TRIGGER`| 6000     | Satu kontur ≥ nilai ini langsung memicu MOTION  |
| `MOG2_VAR_THRESHOLD`        | 25       | Variance threshold MOG2 (makin besar=makin tahan noise) |
| `BG_SUB_METHOD`             | "MOG2"   | Metode background subtraction                   |
| `DNN_CONFIDENCE_THRESHOLD`  | 0.35     | Confidence minimum deteksi manusia (DNN)        |
| `DNN_USE_MOTION_ROI`        | True     | Jalankan DNN hanya pada ROI gerakan             |
| `ALARM_COOLDOWN`            | 3        | Cooldown alarm (detik)                          |
| `USE_WEBCAM`                | False    | Gunakan webcam sebagai input                    |

## 📊 Output

- **Video output:** `data/output/video_output.avi` — video dengan anotasi bounding box
- **Frame terdeteksi:** `data/output/frames_terdeteksi/` — frame individual saat motion
- **Log CSV:** `results/logs/motion_log.csv` — catatan timestamp & status
- **Screenshots:** `results/screenshots/` — bukti visual alarm aktif

## 🛠️ Teknologi

- Python 3.8+
- OpenCV (cv2) — termasuk modul `cv2.dnn` untuk deteksi deep-learning
- NumPy
- Matplotlib
- MobileNet-SSD (Caffe) — model deteksi manusia, auto-download ke `models/`

## 📄 Lisensi

Proyek ini dibuat untuk keperluan mini project mata kuliah Computer Vision.
