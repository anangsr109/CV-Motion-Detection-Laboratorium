---
name: testing-detection
description: Test the motion/human-detection pipeline end-to-end for this OpenCV CLI app. Use when verifying changes to motion detection, background subtraction, or the DNN/HOG human detectors.
---

# Testing the detection pipeline

This is a CLI OpenCV app (`python -m src.main`) whose only GUI is an interactive preview
window that waits for a `q` keypress. **Test headlessly via shell — do NOT record** (there is
no GUI activity worth recording). Collect command output + an annotated frame as evidence.

## Setup
- Deps: `python -m pip install -r requirements.txt` (opencv-python, numpy, matplotlib, certifi).
- DNN model (MobileNet-SSD, ~23MB) auto-downloads to `models/` on first use; the environment
  blueprint also pre-downloads it. The `.caffemodel` is gitignored; the `.prototxt` is committed.
- No secrets/login required. Sample video: `data/raw/video_kelas.mp4` (overhead classroom CCTV).
- Note: importing the `src` package from a root-level script may need the repo root on
  `sys.path` — `benchmark.py` already does `sys.path.insert(0, <repo root>)`.

## Test 1 — Benchmark (primary)
`python benchmark.py 400` prints motion% + person counts for all 5 methods.
Expected on the sample video (approx; parameters are tunable in `src/config.py`):
- 3-Frame Diff / MOG2 motion% around 50-58% (the key fix — was ~100% before the motion gate).
- KNN is more sensitive (~85%).
- MOG2 + HOG total_persons is huge (~3500, mostly false positives).
- MOG2 + DNN total_persons is far lower (~200, ~1-2/motion-frame) and ~2-3x faster than HOG.
If motion% is ~100% again, the frame-level gate (`MOTION_MIN_TOTAL_AREA`/`MOTION_SINGLE_AREA_TRIGGER`)
may be broken. If DNN's person count matches HOG's, the DNN may be silently falling back to HOG.

## Test 2 — Integration smoke
Run ~60 frames through main.py's exact chain: `preprocess_frame` -> `BackgroundSubtraction(MOG2)`
-> `BBoxStabilizer.update` -> `DNNPersonDetector.detect_and_annotate` -> `MotionLogger.log`
(+ `AdaptiveLearner`). Assert: zero exceptions, DNN loads without fallback, CSV columns =
`timestamp,frame_number,status,method,contour_count,person_count`. The final motion gate in
`main.py` must consider only active stabilizer boxes (`if sb["is_active"]`) for both total and max area.

## Test 3 — Visual proof
Run MOG2 + DNN over 400 frames, keep the frame with the most detections, save an annotated JPG.
Expect a small number of green `Person (conf)` boxes on actual students — NOT ~10 boxes blanketing
the frame (that would be HOG-style false positives).

## Known limitations (expected, not bugs)
- MobileNet-SSD recall is intentionally low on this overhead/seated CCTV angle; it detects a subset
  of clearly-visible students. MOTION detection is the reliable signal for this footage.
- Interactive preview, Y/N/R feedback keys, and the winsound alarm are Windows/GUI-only and are not
  exercised headlessly.

## Devin Secrets Needed
- None.
