"""Compact U-Net for GPR hyperbola segmentation (notebook Sections 6–7).

Architecture
------------
- Encoder: three conv–conv blocks (8, 16, 32 channels) with BatchNorm/ReLU
  and 2x2 max-pooling.
- Bottleneck: 64 channels at 16x16 resolution.
- Decoder: transposed-convolution upsampling with skip concatenation.
- Head: 1x1 convolution producing per-pixel logits.

Loss: positively-weighted BCE + soft-Dice.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Architecture
# ------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm + ReLU."""
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CompactUNet(nn.Module):
    """Small U-Net for 128x128 B-scan segmentation (~121 k parameters)."""
    def __init__(self, base=8):
        super().__init__()
        self.enc1 = ConvBlock(1, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.bott = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ------------------------------------------------------------------
# Losses
# ------------------------------------------------------------------
def dice_loss(logits, targets, eps=1e-6):
    """Soft-Dice loss on sigmoid probabilities."""
    p = torch.sigmoid(logits).flatten(1)
    t = targets.flatten(1)
    inter = (p * t).sum(1)
    union = p.sum(1) + t.sum(1)
    return 1.0 - ((2 * inter + eps) / (union + eps)).mean()


def bce_dice(logits, targets, pos_weight=5.0):
    """Combined BCE (positive-weighted) + soft-Dice loss."""
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=torch.tensor(pos_weight)
    )
    return bce + dice_loss(logits, targets)


# ------------------------------------------------------------------
# Inference helpers
# ------------------------------------------------------------------
def predict_prob(model, img_pre, device=None):
    """Per-pixel tunnel-signature probability for one preprocessed B-scan.

    Parameters
    ----------
    model : nn.Module
    img_pre : ndarray, shape (H, W)
        Preprocessed B-scan.
    device : torch.device or None
        If ``None`` the model's current device is used.

    Returns
    -------
    prob : ndarray, shape (H, W)
        Sigmoid probabilities in [0, 1].
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    x = torch.from_numpy(img_pre[None, None].astype(np.float32)).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return prob


def batch_predict(model, imgs_pre, device):
    """Batch inference over many preprocessed B-scans.

    Parameters
    ----------
    model : nn.Module
    imgs_pre : list[ndarray] or ndarray
        If list, arrays must all be (H, W). If ndarray, shape (N, H, W).
    device : torch.device

    Returns
    -------
    probs : ndarray, shape (N, H, W)
    """
    model.eval()
    if isinstance(imgs_pre, list):
        arr = np.stack(imgs_pre).astype(np.float32)
    else:
        arr = imgs_pre.astype(np.float32)
    x = torch.from_numpy(arr[:, None, :, :]).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).cpu().numpy()[:, 0]
    return probs


# ------------------------------------------------------------------
# Device management
# ------------------------------------------------------------------
def list_compute_devices():
    """Return a list of (device, description) tuples available on this host."""
    devices = [('cpu', f'CPU ({os.cpu_count()} logical cores)')]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            devices.append((
                f'cuda:{i}',
                f'{name} ({vram_gb:.1f} GB VRAM)'
            ))
    return devices


def get_device(preferred=None):
    """Resolve a torch.device from a string or default to CPU."""
    if preferred is None:
        preferred = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return torch.device(preferred)


# ------------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------------
def find_latest_checkpoint(models_dir='models', prefix='unet_v'):
    """Return the latest versioned checkpoint path or None."""
    root = Path(models_dir)
    if not root.exists():
        return None
    candidates = sorted(root.glob(f'{prefix}*.pt'))
    return candidates[-1] if candidates else None


def next_checkpoint_path(models_dir='models', prefix='unet_v'):
    """Return the next versioned checkpoint path (e.g. models/unet_v1.pt)."""
    root = Path(models_dir)
    root.mkdir(parents=True, exist_ok=True)
    existing = [p for p in root.glob(f'{prefix}*.pt')]
    versions = []
    for p in existing:
        try:
            versions.append(int(p.stem.replace(prefix, '')))
        except ValueError:
            pass
    next_v = max(versions, default=0) + 1
    return root / f'{prefix}{next_v}.pt'
