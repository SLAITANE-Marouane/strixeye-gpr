"""Training script for the compact U-Net on synthetic GPR data.

Usage
-----
From the project root:

    python -m scripts.train

The script auto-detects CUDA, trains on the configured seed ranges, and
saves a versioned checkpoint to ``models/unet_vX.pt``.
"""

import os
import random
import time
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import (
    GPR_NT, GPR_NX, TRAIN_SEEDS, VAL_SEEDS, GLOBAL_SEED
)
from src.simulator import simulate_bscan
from src.preprocess import preprocess
from src.model import CompactUNet, bce_dice, get_device, next_checkpoint_path


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
def make_arrays(seeds):
    """Pre-generate (X, Y) arrays for a list of simulator seeds."""
    Xs, Ys = [], []
    for s in seeds:
        img, mask, _ = simulate_bscan(seed=s)
        Xs.append(preprocess(img)[None])
        Ys.append(mask[None].astype(np.float32))
    return np.stack(Xs).astype(np.float32), np.stack(Ys).astype(np.float32)


class GPRDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
def train(device, epochs=8, batch_size=8, lr=1e-3):
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    print("Building datasets ...")
    t0 = time.time()
    Xtr, Ytr = make_arrays(TRAIN_SEEDS)
    Xva, Yva = make_arrays(VAL_SEEDS)
    print(f"  train {Xtr.shape} | val {Xva.shape} | pos px {100.0 * Ytr.mean():.3f}% "
          f"({time.time() - t0:.1f} s)")

    model = CompactUNet(base=8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    tr_loader = DataLoader(GPRDataset(Xtr, Ytr), batch_size=batch_size,
                           shuffle=True, generator=torch.Generator().manual_seed(GLOBAL_SEED))
    va_loader = DataLoader(GPRDataset(Xva, Yva), batch_size=16, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state = float('inf'), None

    print(f"\nTraining on {device} for {epochs} epochs ...")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = bce_dice(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)

        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                vtot += bce_dice(model(xb), yb).item() * len(xb)

        tr_loss = tot / len(Xtr)
        va_loss = vtot / len(Xva)
        print(f"  epoch {ep}/{epochs}  train {tr_loss:.4f}  val {va_loss:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed:.1f} s | best val loss {best_val:.4f}")

    out_path = next_checkpoint_path()
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_path)
    print(f"Checkpoint saved: {out_path}")
    return out_path


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train GPR U-Net")
    parser.add_argument("--device", type=str, default=None,
                        help="torch device, e.g. cpu or cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")
    train(device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
