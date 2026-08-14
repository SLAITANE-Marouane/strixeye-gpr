"""Survey-grid probability heat map and plan-view metrics (notebook Section 9).

A real survey acquires parallel B-scans at line positions y_1 .. y_M.
Each line yields a per-pixel probability map p_i(x, t) from the U-Net.
We compress each line to a per-trace confidence

    c_i(x) = max_{t >= t_min} p_i(x, t)

(discarding the direct-wave region t < t_min) and stack the c_i into a
2-D field C(x, y). Light Gaussian smoothing interpolates across the
discrete line spacing. Thresholding C at tau gives a hard plan-view
delineation; we report its IoU against the true tunnel footprint.
"""

import numpy as np
from scipy import ndimage

from .config import GPR_NT, GPR_NX, GPR_DT, GPR_DX


def line_confidence(prob, t_min_rows=12):
    """Per-trace max confidence after suppressing the direct wave."""
    return prob[t_min_rows:, :].max(axis=0)


def build_heatmap(line_probs, ys, y_spacing, sigma=(0.8, 1.5)):
    """Build a smooth 2-D detection-probability field C(x, y).

    Parameters
    ----------
    line_probs : list[ndarray]
        U-Net probability maps, one per survey line.
    ys : ndarray
        y-coordinate of each line [m].
    y_spacing : float
        Nominal spacing between lines [m].
    sigma : tuple
        Gaussian smoothing kernel in (line, trace) units.

    Returns
    -------
    heatmap : ndarray, shape (N_lines, GPR_NX)
        Smoothed probability field.
    extent : tuple
        ``(x_min, x_max, y_max + pad, y_min - pad)`` for matplotlib/plotly.
    xs : ndarray
        x-axis coordinates [m].
    """
    profiles = np.stack([line_confidence(p) for p in line_probs])
    heatmap = ndimage.gaussian_filter(profiles, sigma=sigma)
    xs = np.arange(GPR_NX) * GPR_DX
    pad = y_spacing / 2.0
    extent = (0.0, GPR_NX * GPR_DX, ys[-1] + pad, -pad)
    return heatmap, extent, xs


def compute_threshold_iou(heatmap, gt_mask, taus=None):
    """Compute plan-view IoU for a range of thresholds.

    Parameters
    ----------
    heatmap : ndarray, shape (N, GPR_NX)
    gt_mask : ndarray, same shape, bool
        True where the true tunnel footprint projects onto the plan.
    taus : ndarray or None
        Thresholds to scan. Default: 0.30 -> 0.95.

    Returns
    -------
    taus, ious : ndarray
    best_tau, best_iou : float
    """
    if taus is None:
        taus = np.linspace(0.30, 0.95, 14)
    ious = []
    for tau in taus:
        pm = heatmap >= tau
        ious.append((pm & gt_mask).sum() / max((pm | gt_mask).sum(), 1))
    ious = np.array(ious)
    best_idx = int(np.argmax(ious))
    return taus, ious, float(taus[best_idx]), float(ious[best_idx])


def make_gt_plan_mask(tunnels, ys, xs, survey_len_x):
    """Rasterise tunnel footprints into a plan-view boolean mask.

    Each tunnel is treated as a rectangle of width 2*radius centered on
    its axis segment, projected onto the (y, x) grid.
    """
    mask = np.zeros((len(ys), len(xs)), dtype=bool)
    for t in tunnels:
        x0, y0 = t['x0_m'], t['y0_m']
        x1, y1 = t['x1_m'], t['y1_m']
        r = t['radius_m']
        # Sample segment finely and mark within radius.
        n = max(50, int(np.hypot(x1 - x0, y1 - y0) / 0.05))
        xc = np.linspace(x0, x1, n)
        yc = np.linspace(y0, y1, n)
        for xi, yi in zip(xc, yc):
            mask |= (np.abs(ys[:, None] - yi) <= 0.5 * (ys[1] - ys[0])) & \
                    (np.abs(xs[None, :] - xi) <= r)
    return mask
