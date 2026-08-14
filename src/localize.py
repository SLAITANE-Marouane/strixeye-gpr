"""From predicted mask to physical tunnel parameters.

This module implements:

- classical diffraction-stack hyperbola focusing (baseline detector),
- least-squares hyperbola inversion (Section 2.4),
- mask post-processing and connected-component labelling,
- matching of detections to ground truth.

The hyperbola model is

    t^2 = a x^2 + b x + c

with closed-form inversion

    v = 2 / sqrt(a),   x0 = -b / (2a),   t0 = sqrt(c - a x0^2).
"""

import numpy as np
from scipy import ndimage
from skimage import measure

from .config import GPR_NT, GPR_NX, GPR_DT, GPR_DX, V_GRID
from .preprocess import preprocess


# ------------------------------------------------------------------
# Least-squares hyperbola inversion
# ------------------------------------------------------------------
def fit_hyperbola_lsq(xs, ts, v_range=(0.05, 0.20), t0_range=(10.0, 80.0)):
    """Linear LSQ fit of t^2 = a x^2 + b x + c -> (v, x0, t0) or None."""
    if len(xs) < 5:
        return None
    A = np.column_stack([xs ** 2, xs, np.ones_like(xs)])
    (a, b, c), *_ = np.linalg.lstsq(A, ts ** 2, rcond=None)
    if a <= 1e-9:
        return None
    x0 = -b / (2 * a)
    t0sq = c - a * x0 ** 2
    if t0sq <= 0:
        return None
    v, t0 = 2.0 / np.sqrt(a), np.sqrt(t0sq)
    if not (v_range[0] <= v <= v_range[1]):
        return None
    if not (0.5 <= x0 <= GPR_NX * GPR_DX - 0.5):
        return None
    if not (t0_range[0] <= t0 <= t0_range[1]):
        return None
    return float(v), float(x0), float(t0)


# ------------------------------------------------------------------
# Mask utilities
# ------------------------------------------------------------------
def mask_from_prob(prob, thr=0.5, min_size=10):
    """Threshold + morphological closing + small-component rejection."""
    m = ndimage.binary_closing(prob >= thr, structure=np.ones((3, 3)), iterations=1)
    lab = measure.label(m)
    out = np.zeros_like(m)
    for r in measure.regionprops(lab):
        if r.area >= min_size:
            out[lab == r.label] = True
    return out


def mask_from_fit(v, x0, t0, halfwidth=2):
    """Rasterise a fitted hyperbola into a binary mask."""
    m = np.zeros((GPR_NT, GPR_NX), dtype=bool)
    xs = np.arange(GPR_NX) * GPR_DX
    t = np.sqrt(np.maximum(t0 ** 2 + 4.0 * (xs - x0) ** 2 / v ** 2, 0.0)) / GPR_DT
    for j in range(GPR_NX):
        i = int(round(t[j]))
        if 0 <= i < GPR_NT:
            m[max(0, i - halfwidth):min(GPR_NT, i + halfwidth + 1), j] = True
    return m


# ------------------------------------------------------------------
# Mask -> components -> parameters
# ------------------------------------------------------------------
def localize_tunnels(pred_mask, max_tunnels=3):
    """Fit one hyperbola per connected component; return physical params."""
    lab = measure.label(pred_mask)
    comps = sorted(measure.regionprops(lab), key=lambda r: -r.area)
    out = []
    for r in comps[:max_tunnels]:
        rows, cols = r.coords[:, 0], r.coords[:, 1]
        fit = fit_hyperbola_lsq(cols * GPR_DX, rows * GPR_DT)
        if fit is None:
            continue
        v, x0, t0 = fit
        out.append(dict(
            v=float(v), x0_m=float(x0), t0_ns=float(t0),
            depth_m=float(v * t0 / 2.0), area=int(r.area),
            apex_row=int(rows.min())
        ))
    return out


# ------------------------------------------------------------------
# Classical baseline detector (diffraction-stack focusing)
# ------------------------------------------------------------------
def candidate_points(img_pre, row_min=12, pct=97.0):
    """Per-trace local maxima of |signal| above a percentile threshold."""
    norm = np.abs(img_pre).astype(np.float64)
    norm = norm / (norm.max() + 1e-12)
    thr = np.percentile(norm, pct)
    rows_all, cols_all = [], []
    for j in range(norm.shape[1]):
        col = norm[row_min:, j]
        lm = np.where((col[1:-1] >= col[:-2]) & (col[1:-1] >= col[2:])
                      & (col[1:-1] > thr))[0] + 1 + row_min
        rows_all.append(lm)
        cols_all.append(np.full_like(lm, j))
    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    return cols * GPR_DX, rows * GPR_DT


def focus_image(pre, half_width_m=1.2,
                t0_rows=np.arange(14, 112, 2),
                x0_cols=np.arange(6, GPR_NX - 6, 2)):
    """Diffraction-stack focusing volume F[v, t0, x0] over velocity grid."""
    W = int(half_width_m / GPR_DX)
    js = np.arange(-W, W + 1)
    dxs = js * GPR_DX
    t0s = t0_rows * GPR_DT
    F = np.zeros((len(V_GRID), len(t0s), len(x0_cols)))
    v_arr = np.array(V_GRID)
    for iv, v in enumerate(v_arr):
        for ix in range(len(x0_cols)):
            jcols = x0_cols[ix] + js
            cvalid = (jcols >= 0) & (jcols < GPR_NX)
            jc = np.clip(jcols, 0, GPR_NX - 1)
            T = np.sqrt(t0s[:, None] ** 2 + 4.0 * dxs[None, :] ** 2 / v ** 2)
            rows = (T / GPR_DT).astype(int)
            valid = (rows <= GPR_NT - 2) & cvalid[None, :]
            rows = np.clip(rows, 0, GPR_NT - 1)
            vals = np.where(valid, pre[rows, jc[None, :]], 0.0)
            F[iv, :, ix] = vals.sum(axis=1) / np.sqrt(np.maximum(valid.sum(axis=1), 1))
    return F, t0_rows, x0_cols


def _parabolic3(Fsm, iv, it, ix, t0_rows, x0_cols):
    """Sub-grid peak position via 1-D parabolic interpolation along each axis."""
    def p1(fm, f0, fp):
        return float(np.clip((fm - fp) / (2.0 * (fm - 2 * f0 + fp) + 1e-12), -1.0, 1.0))

    dv = p1(Fsm[iv - 1, it, ix], Fsm[iv, it, ix], Fsm[iv + 1, it, ix]) if 0 < iv < Fsm.shape[0] - 1 else 0.0
    dt = p1(Fsm[iv, it - 1, ix], Fsm[iv, it, ix], Fsm[iv, it + 1, ix]) if 0 < it < Fsm.shape[1] - 1 else 0.0
    dx = p1(Fsm[iv, it, ix - 1], Fsm[iv, it, ix], Fsm[iv, it, ix + 1]) if 0 < ix < Fsm.shape[2] - 1 else 0.0
    v = float(np.array(V_GRID)[iv] + dv * (np.array(V_GRID)[1] - np.array(V_GRID)[0]))
    t0 = float((t0_rows[it] + dt * (t0_rows[1] - t0_rows[0])) * GPR_DT)
    x0 = float((x0_cols[ix] + dx * (x0_cols[1] - x0_cols[0])) * GPR_DX)
    return v, t0, x0


def baseline_detect(pre, score_frac=0.45, max_dets=3, nms_r=3):
    """Classical detector: focus -> NMS peak -> sub-grid -> guarded LSQ refine."""
    F, t0_rows, x0_cols = focus_image(pre)
    Fsm = ndimage.gaussian_filter(F, sigma=(0, 1.0, 1.0))
    thr = score_frac * Fsm.max()
    dets = []
    Fw = Fsm.copy()
    for _ in range(max_dets):
        iv, it, ix = np.unravel_index(np.argmax(Fw), Fw.shape)
        if not np.isfinite(Fw[iv, it, ix]) or Fw[iv, it, ix] < thr:
            break
        v, t0, x0 = _parabolic3(Fsm, iv, it, ix, t0_rows, x0_cols)
        dets.append(dict(v=v, x0_m=x0, t0_ns=t0, depth_m=v * t0 / 2.0,
                         score=float(Fsm[iv, it, ix])))
        Fw[:, max(0, it - nms_r * 2):it + nms_r * 2 + 1,
           max(0, ix - nms_r):ix + nms_r + 1] = -np.inf

    # Guarded LSQ refinement on strong local maxima near each template.
    xs, ts = candidate_points(pre, pct=95.0)
    refined = []
    for det in dets:
        v, x0, t0 = det['v'], det['x0_m'], det['t0_ns']
        for _ in range(3):
            t_pred = np.sqrt(np.maximum(t0 ** 2 + 4.0 * (xs - x0) ** 2 / v ** 2, 0.0))
            near = np.abs(ts - t_pred) < 2.4
            if near.sum() < 6:
                break
            fit = fit_hyperbola_lsq(xs[near], ts[near])
            if fit is None or abs(fit[0] * fit[2] / 2.0 - det['depth_m']) >= 0.4:
                break
            v, x0, t0 = fit
        refined.append(dict(v=float(v), x0_m=float(x0), t0_ns=float(t0),
                            depth_m=float(v * t0 / 2.0), score=det['score']))
    return refined, F


# ------------------------------------------------------------------
# Matching detections to ground truth
# ------------------------------------------------------------------
def match_detections(dets, gt_params, x_tol=0.5, depth_tol=0.5):
    """Greedy 1-to-1 matching by lateral distance.

    Returns
    -------
    n_matched : int
    x_errs, depth_errs, v_errs : list[float]
    """
    matched, xe, de, ve = set(), [], [], []
    for g in gt_params:
        cands = [(abs(d['x0_m'] - g['x0_m']), i, d)
                 for i, d in enumerate(dets) if i not in matched]
        if not cands:
            continue
        xerr, i, db = min(cands)
        derr = abs(db['depth_m'] - g['depth_m'])
        if xerr < x_tol and derr < depth_tol:
            matched.add(i)
            xe.append(xerr)
            de.append(derr)
            ve.append(abs(db['v'] - g['v']))
    return len(matched), xe, de, ve


# ------------------------------------------------------------------
# Segmentation metrics
# ------------------------------------------------------------------
def seg_metrics_from_masks(pred_bool, gt_bool):
    """IoU, Dice, Precision, Recall, F1 from binary masks."""
    tp = int((pred_bool & gt_bool).sum())
    fp = int((pred_bool & ~gt_bool).sum())
    fn = int((~pred_bool & gt_bool).sum())
    iou = tp / max(tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return dict(
        IoU=iou,
        Dice=2 * tp / max(2 * tp + fp + fn, 1),
        Precision=prec,
        Recall=rec,
        F1=2 * prec * rec / max(prec + rec, 1e-12),
    )
