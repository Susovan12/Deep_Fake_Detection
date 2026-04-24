"""
retrain.py — Fine-tune ResNeXt on feedback samples collected in feedback.db

Usage:
    python retrain.py
    python retrain.py --epochs 5 --lr 1e-5 --min_samples 10
"""

import os, sys, sqlite3, argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Normalize
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "feedback.db")
CKPT_OUT = os.path.join(BASE_DIR, "resnext_finetuned.pth")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
normalize = Normalize(mean, std)


# ── Load feedback samples from DB ─────────────────────────
def load_samples(db_path):
    con = sqlite3.connect(db_path)
    rows = con.execute("""
        SELECT video_path, true_label, confidence, feedback
        FROM feedback
        ORDER BY timestamp DESC
    """).fetchall()
    con.close()
    samples = []
    for video_path, true_label, confidence, feedback in rows:

        label = 1 if true_label == "FAKE" else 0
        samples.append({"video_path": video_path, "label": label,
                         "confidence": confidence, "feedback": feedback})
    return samples


# ── Dataset ───────────────────────────────────────────────
def extract_frames(video_path, num_frames=16, size=224):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (size, size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        return None
    return np.stack(frames)  # (T, H, W, 3)

class FeedbackDataset(Dataset):
    def __init__(self, samples, num_frames=16, size=224):
        self.samples    = samples
        self.num_frames = num_frames
        self.size       = size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s      = self.samples[idx]
        frames = extract_frames(s["video_path"], self.num_frames, self.size)
        if frames is None:
            # return zeros if video unreadable
            frames = np.zeros((self.num_frames, self.size, self.size, 3), dtype=np.uint8)

        # average frames → single image representation
        img = frames.mean(axis=0).astype(np.float32)
        t   = torch.from_numpy(img).permute(2, 0, 1) / 255.0
        t   = normalize(t)
        return t, torch.tensor(s["label"], dtype=torch.float32)


# ── Model ─────────────────────────────────────────────────
class MyResNeXt(models.resnet.ResNet):
    def __init__(self):
        super().__init__(block=models.resnet.Bottleneck,
                         layers=[3, 4, 6, 3], groups=32, width_per_group=4)
        self.fc = nn.Linear(2048, 1)


# ── Training ──────────────────────────────────────────────
def retrain(epochs=5, lr=1e-5, batch_size=8, min_samples=5):
    if not os.path.exists(DB_PATH):
        print("No feedback.db found. Run predict.py first to collect feedback.")
        return

    samples = load_samples(DB_PATH)
    if len(samples) < min_samples:
        print(f"Only {len(samples)} samples in DB. Need at least {min_samples} to retrain.")
        return

    print(f"Loaded {len(samples)} feedback samples.")
    for s in samples:
        print(f"  [{s['feedback']:7s}] {os.path.basename(s['video_path'])} "
              f"→ {'FAKE' if s['label'] else 'REAL'} (conf={s['confidence']:.3f})")

    dataset = FeedbackDataset(samples)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Load base weights
    base_pth = os.path.join(BASE_DIR, "FaceForensics.pth")
    model = MyResNeXt().to(device)
    model.load_state_dict(torch.load(base_pth, map_location=device))

    # Freeze all except fc and layer4
    for name, param in model.named_parameters():
        if not (name.startswith("fc") or name.startswith("layer4")):
            param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nFine-tuning ResNeXt for {epochs} epoch(s) on {len(samples)} samples...")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x).squeeze(1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch}/{epochs}  loss={total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), CKPT_OUT)
    print(f"\nSaved fine-tuned weights → {CKPT_OUT}")
    print("To use: rename resnext_finetuned.pth to FaceForensics.pth and re-run api.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ResNeXt on feedback data")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--min_samples", type=int,   default=50,
                        help="Minimum feedback samples required before retraining")
    parser.add_argument("--stats", action="store_true",
                        help="Just show feedback DB stats, don't retrain")
    args = parser.parse_args()

    if args.stats:
        if not os.path.exists(DB_PATH):
            print("No feedback.db yet.")
        else:
            con = sqlite3.connect(DB_PATH)
            total   = con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            wrong   = con.execute("SELECT COUNT(*) FROM feedback WHERE feedback='wrong'").fetchone()[0]
            correct = con.execute("SELECT COUNT(*) FROM feedback WHERE feedback='correct'").fetchone()[0]
            con.close()
            print(f"Total samples : {total}")
            print(f"  Wrong       : {wrong}")
            print(f"  Correct     : {correct}")
            print(f"Ready to retrain: {'YES' if total >= args.min_samples else f'NO (need {args.min_samples - total} more)'}")
    else:
        retrain(args.epochs, args.lr, args.batch_size, args.min_samples)
