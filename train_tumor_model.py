#!/usr/bin/env python
"""
train_tumor_model.py  (GPU‑aware, AMP‑optional)
------------------------------------------------
* Accepts --data_path that may be root or any split folder.
* Prints device info.
* Default batch_size 32 (fits 8 GB @ 384×384).
* Optional --amp to enable mixed‑precision.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, classification_report


# ---------- Path helper ---------------------------------------------------- #
def resolve_dataset_root(path: Path) -> Path:
    return path.parent if path.name.lower() in {"train", "valid", "test"} else path


# ---------- Dataset -------------------------------------------------------- #
class TumorDataset(Dataset):
    def __init__(self, folder: Path, transform=None):
        self.folder = folder
        self.df = pd.read_csv(self.folder / "_classes.csv")
        self.image_paths = [self.folder / f for f in self.df.iloc[:, 0]]
        self.labels = self.df.iloc[:, 1:].fillna(0).values.astype(np.float32)
        self.transform = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


# ---------- Utility -------------------------------------------------------- #
def pos_weights(labels):
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    return torch.tensor(neg / (pos + 1e-6), dtype=torch.float32)


class EarlyStop:
    def __init__(self, patience=5):
        self.best, self.wait, self.patience = 1e9, 0, patience

    def step(self, loss):
        improved = loss < self.best
        self.best = min(self.best, loss)
        self.wait = 0 if improved else self.wait + 1
        return self.wait >= self.patience


# ---------- Training ------------------------------------------------------- #
def train(args):
    root = resolve_dataset_root(Path(args.data_path))
    train_dir, val_dir, test_dir = root / "train", root / "valid", root / "test"

    # -- transforms
    tf_train = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    tf_val = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    # -- datasets & loaders
    ds_tr, ds_va, ds_te = (TumorDataset(train_dir, tf_train),
                           TumorDataset(val_dir, tf_val),
                           TumorDataset(test_dir, tf_val))

    pin_mem = torch.cuda.is_available()
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True,
                       pin_memory=pin_mem, num_workers=2)
    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False,
                       pin_memory=pin_mem, num_workers=2)
    dl_te = DataLoader(ds_te, batch_size=args.bs, shuffle=False,
                       pin_memory=pin_mem, num_workers=2)

    # -- device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📟  Using device: {device}")
    if device.type == "cuda":
        print("🖥️   GPU:", torch.cuda.get_device_name(0))

    # -- model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, ds_tr.labels.shape[1])
    model.to(device)

    # -- loss, optim, sched
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights(ds_tr.labels).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=3)
    stopper = EarlyStop(args.pat)

    # -- AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_f1 = 0
    args.out.mkdir(parents=True, exist_ok=True)

    # -------- epoch loop -------- #
    for ep in range(1, args.epochs + 1):
        # ---- train ----
        model.train(); tr_loss = 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.amp):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(dl_tr.dataset)

        # ---- validate ----
        model.eval(); va_loss = 0; preds, labs = [], []
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    out = model(x)
                    loss = criterion(out, y)
                va_loss += loss.item() * x.size(0)
                preds.append(torch.sigmoid(out).cpu())
                labs.append(y.cpu())
        va_loss /= len(dl_va.dataset)
        preds_all, labs_all = torch.cat(preds), torch.cat(labs)
        f1 = f1_score(labs_all, (preds_all > 0.5), average="macro")

        print(f"Epoch {ep:02d} | Train {tr_loss:.4f} | Val {va_loss:.4f} | F1 {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.out / "best.pth")
        if stopper.step(va_loss):
            print("⏹️  Early stop.")
            break
        sched.step(va_loss)

    # ---- test ----
    model.load_state_dict(torch.load(args.out / "best.pth"))
    model.eval(); preds, labs = [], []
    with torch.no_grad():
        for x, y in dl_te:
            x = x.to(device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                preds.append(torch.sigmoid(model(x)).cpu())
            labs.append(y)
    preds_all, labs_all = torch.cat(preds), torch.cat(labs)
    print("\n🏁  Test F1:", f1_score(labs_all, (preds_all > 0.5), average="macro"))
    print("\n📄  Classification report:\n",
          classification_report(labs_all, (preds_all > 0.5), zero_division=0))


# ---------- CLI ------------------------------------------------------------ #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True,
                   help="Root OR any of train/valid/test sub‑folders")
    p.add_argument("--out", type=Path, default=Path("./outputs"))
    p.add_argument("--img", type=int, default=384)
    p.add_argument("--bs",  type=int, default=32, help="Batch size (32 fits 8 GB GPU)")
    p.add_argument("--lr",  type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--pat", type=int, default=5, help="Early‑stop patience")
    p.add_argument("--amp", action="store_true",
                   help="Enable mixed‑precision (recommended for GPU)")
    train(p.parse_args())
