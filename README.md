# TrueVision — Real-Time Deepfake Detection

> A desktop overlay app that detects deepfake videos in real-time using an ensemble of deep learning models (ResNeXt-50 + EfficientNet-B4), with a built-in feedback loop for continuous improvement.

---

## Quick Start (Windows)

```
1. Clone the repo
2. Download model weights  ← see section below
3. Double-click install.bat
4. Double-click start.bat
```

That's it.

---

## Prerequisites

| Tool | Version | Download |
|------|---------|----------|
| Python | 3.8+ | https://python.org/downloads |
| Node.js | 16+ | https://nodejs.org |
| pip | latest | comes with Python |

> Make sure both `python` and `node` are added to your system PATH during installation.

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/Susovan12/TrueVision.git
cd TrueVision
```

### Step 2 — Download model weights

These files are not included in the repo (too large for GitHub). Download and place them manually:

| File | Where to place it |
|------|-------------------|
| `FaceForensics.pth` | `backend/FaceForensics.pth` |
| `FaceForensics_PP.pth` | `backend/FaceDetector_PP/FaceDetector_PP/pth_fiels/FaceForensics_PP.pth` |
| `efficientnet_b4.pth` *(optional)* | `backend/efficientnet_b4.pth` |

> **Note:** Without `efficientnet_b4.pth`, the app runs in single-model mode (ResNeXt-50 only). Detection still works, just slightly lower accuracy.

### Step 3 — Install dependencies

**Option A — One-click (recommended for Windows):**

```
Double-click install.bat
```

**Option B — Manual:**

```bash
# Backend
pip install -r backend/requirements.txt

# Frontend
cd frontend
npm install
```

### Step 4 — Run the app

**Option A — One-click:**

```
Double-click start.bat
```

**Option B — Manual (two terminals):**

```bash
# Terminal 1 — Backend
cd backend
python api.py

# Terminal 2 — Frontend
cd frontend
npm start
```

The app launches as a floating overlay in the bottom-right corner of your screen.

---

## Project Structure

```
TrueVision/
│
├── backend/
│   ├── api.py                      # Flask API (predict, feedback, stats)
│   ├── retrain.py                  # Fine-tune model on feedback data
│   ├── sort_feedback_videos.py     # Sort videos by feedback label
│   ├── requirements.txt            # Python dependencies
│   ├── FaceForensics.pth           # ← place model weights here (not in repo)
│   └── FaceDetector_PP/
│       └── FaceDetector_PP/
│           ├── face_utils.py       # RetinaFace face detector wrapper
│           ├── pth_fiels/
│           │   └── FaceForensics_PP.pth  # ← place face detector weights here
│           └── external/
│               └── Pytorch_Retinaface/   # RetinaFace source
│
├── frontend/
│   ├── deepfake_detector_ui.html   # Main overlay UI
│   ├── fab.html                    # Floating action button
│   ├── main.js                     # Electron main process
│   ├── preload.js                  # Electron preload bridge
│   └── package.json
│
├── eval_results/                   # Training charts & metrics
├── install.bat                     # One-click installer (Windows)
├── start.bat                       # One-click launcher (Windows)
└── README.md
```

---

## How It Works

```
┌─────────────────────────┐         ┌──────────────────────────────┐
│   Electron Frontend     │  HTTP   │      Flask Backend           │
│                         │ ──────► │                              │
│  • Screen capture       │         │  • Face detection (Retina)   │
│  • Always-on-top UI     │ ◄────── │  • ResNeXt-50 classifier     │
│  • Result display       │         │  • EfficientNet-B4 (ensemble)│
│  • Feedback buttons     │         │  • SQLite feedback DB        │
└─────────────────────────┘         └──────────────────────────────┘
```

1. User points the overlay at a video (screen capture or file upload)
2. Backend extracts faces using RetinaFace
3. Each face is classified by ResNeXt-50 (and EfficientNet-B4 if available)
4. Scores are averaged → FAKE / REAL + confidence shown in overlay
5. User can mark the result as correct/wrong → stored for retraining

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Upload video → get FAKE/REAL + confidence |
| `POST` | `/feedback` | Submit correct/wrong feedback |
| `GET` | `/stats` | View feedback statistics |
| `GET` | `/health` | Backend health check |

### Example — predict

```bash
curl -X POST http://localhost:5000/predict \
  -F "video=@your_video.mp4"
```

Response:
```json
{
  "result": "FAKE",
  "confidence": 0.8731,
  "p_resnext": 0.9012,
  "p_efficientnet": 0.8321,
  "video_path": "..."
}
```

---

## Feedback & Retraining

User feedback is stored selectively:

| Feedback | Condition | Stored? |
|----------|-----------|---------|
| Wrong | always | ✅ Yes |
| Correct | confidence > 0.85 or < 0.15 | ✅ Yes |
| Correct | 0.15 ≤ confidence ≤ 0.85 | ❌ No (ambiguous) |

Once 50+ samples are collected, retraining is available:

```bash
cd backend
python retrain.py
```

After retraining, rename `resnext_finetuned.pth` → `FaceForensics.pth` and restart `api.py`.

---

## Evaluation Results

Training metrics are in `eval_results/`:

| Chart | Description |
|-------|-------------|
| `accuracy_curve.png` | Train vs validation accuracy |
| `loss_curve.png` | Train vs validation loss |
| `confusion_matrix.png` | TP / TN / FP / FN breakdown |
| `classification_report.png` | Precision, recall, F1 |
| `score_distribution.png` | Score histogram for real vs fake |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Electron, HTML/CSS/JS |
| Backend | Python 3.8+, Flask |
| Models | PyTorch (ResNeXt-50, EfficientNet-B4) |
| Face Detection | RetinaFace |
| Database | SQLite |

---

## Troubleshooting

**`python` not recognized**
→ Reinstall Python and check "Add to PATH" during setup.

**`npm` not recognized**
→ Reinstall Node.js and check "Add to PATH" during setup.

**"No faces detected" error**
→ The video may have no clear face frames. Try a different video or reduce `frame_skip` in `api.py`.

**App launches but no overlay visible**
→ Look for the small circular button in the bottom-right corner of your screen. Click it to open the overlay.

**EfficientNet not loading**
→ This is non-fatal. The app falls back to ResNeXt-50 only mode automatically.

---

## Author

**Susovan Patra**
- GitHub: [@Susovan12](https://github.com/Susovan12)
