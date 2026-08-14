"""Physics-inspired synthetic GPR B-scan simulator.

The forward model follows the notebook (Section 3):

    B(x, t) = (R *t w)(x, t) + additive_noise + speckle

where R is a reflectivity image containing a direct wave, layered
background, clutter point scatterers, and tunnel diffraction hyperbolas.
The hyperbola equation is

    t^2(x) = t0^2 + 4 (x - x0)^2 / v^2,   t0 = 2 d / v.
"""

import numpy as np
from scipy import ndimage

from .config import GPR_NT, GPR_NX, GPR_DT, GPR_DX


def ricker_wavelet(f0_mhz, dt_ns, n=25):
    """Zero-phase Ricker (Mexican-hat) wavelet, peak-normalised."""
    f0 = f0_mhz * 1e-3                       # cycles per ns
    t = (np.arange(n) - (n - 1) / 2.0) * dt_ns
    a = (np.pi * f0) ** 2
    w = (1.0 - 2.0 * a * t ** 2) * np.exp(-a * t ** 2)
    return w / (np.abs(w).max() + 1e-12)


def simulate_bscan(seed=None, n_tunnels=None, tunnel_specs=None):
    """Generate one synthetic B-scan.

    Parameters
    ----------
    seed : int or None
        Seed for the scan-local RNG (reproducibility).
    n_tunnels : int or None
        Number of tunnels (random in {1,2,3} if None and no specs).
    tunnel_specs : list[dict] or None
        Optional deterministic tunnel list. Each dict must contain
        ``x0_m``, ``depth_m``, ``v`` and ``r_m``.

    Returns
    -------
    img : ndarray, shape (GPR_NT, GPR_NX)
        B-scan amplitude.
    mask : ndarray, shape (GPR_NT, GPR_NX), bool
        Ground-truth tunnel-signature mask (+/- 2 samples around crest).
    params : list[dict]
        Per-tunnel physical parameters.
    """
    rng = np.random.default_rng(seed)
    refl = np.zeros((GPR_NT, GPR_NX), dtype=np.float64)

    # (1) direct wave: strong horizontal band near t = 0
    refl[int(rng.integers(3, 6)), :] += rng.uniform(1.0, 1.4)

    # (2) layered background: undulating sub-horizontal reflectors
    depth_row = rng.uniform(14, 24)
    for _ in range(int(rng.integers(2, 4))):
        depth_row += rng.uniform(22, 42)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(1.0, 3.5)
        freq = rng.uniform(0.5, 1.5)
        b = depth_row + amp * np.sin(2 * np.pi * freq * np.arange(GPR_NX) / GPR_NX + phase)
        b = np.clip(b, 8, GPR_NT - 12)
        la = rng.uniform(0.15, 0.45) * rng.choice([-1.0, 1.0])
        for j in range(GPR_NX):
            refl[int(round(b[j])), j] += la

    # (3) clutter point scatterers (stones, roots)
    for _ in range(int(rng.integers(4, 12))):
        si = int(rng.integers(12, GPR_NT - 8))
        sj = int(rng.integers(0, GPR_NX))
        refl[si, sj] += rng.uniform(0.1, 0.3) * rng.choice([-1.0, 1.0])

    # (4) tunnel diffraction hyperbolas
    mask, params = np.zeros((GPR_NT, GPR_NX), dtype=bool), []
    if tunnel_specs is None:
        if n_tunnels is None:
            n_tunnels = int(rng.integers(1, 4))
        tunnel_specs = [dict(
            x0_m=float(rng.uniform(1.5, GPR_NX * GPR_DX - 1.5)),
            depth_m=float(rng.uniform(1.0, 2.8)),
            v=float(rng.uniform(0.08, 0.13)),
            r_m=float(rng.uniform(0.4, 0.8))
        ) for _ in range(n_tunnels)]

    xs = np.arange(GPR_NX) * GPR_DX
    for spec in tunnel_specs:
        v, x0, d = spec['v'], spec['x0_m'], spec['depth_m']
        t0 = 2.0 * d / v
        t_hyp = np.sqrt(t0 ** 2 + 4.0 * (xs - x0) ** 2 / v ** 2)
        A0 = float(rng.uniform(0.9, 1.4)) * (0.6 + 0.4 * spec['r_m'] / 0.8)
        w_lat = float(rng.uniform(0.35, 0.65))
        amp = A0 * np.exp(-0.5 * ((xs - x0) / w_lat) ** 2)
        ti = t_hyp / GPR_DT
        valid = (ti < GPR_NT - 3) & (ti > 10) & (amp > 0.10 * A0)
        for j in np.where(valid)[0]:
            ii = int(round(ti[j]))
            refl[ii, j] += amp[j]
            mask[max(0, ii - 2):min(GPR_NT, ii + 3), j] = True
        params.append(dict(
            x0_m=x0, depth_m=d, v=v, r_m=spec['r_m'], t0_ns=t0,
            apex_row=t0 / GPR_DT, apex_col=x0 / GPR_DX
        ))

    # (5) band limitation + noise
    wv = ricker_wavelet(float(rng.uniform(150, 250)), GPR_DT, 25)
    img = ndimage.convolve1d(refl, wv, axis=0, mode='nearest')
    img = img + rng.normal(0.0, 0.03, img.shape) * np.abs(img).max()
    img = img * (1.0 + rng.normal(0.0, 0.08, img.shape))
    return img, mask, params


def compute_tunnel_specs_for_line(y_line, tunnels, line_width_m=1.5):
    """Map plan-view tunnel segments to simulator specs for a flight line.

    For each tunnel, the closest point on the 2-D segment to the line at
    ``y = y_line`` is found. If the perpendicular distance is within
    ``line_width_m`` a hyperbola is generated at the projected x position
    with the tunnel's depth/radius/velocity.

    Parameters
    ----------
    y_line : float
        y-coordinate of the survey line [m].
    tunnels : list[dict]
        Each dict has ``x0_m, y0_m, x1_m, y1_m, depth_m, radius_m, v_m_ns``.
    line_width_m : float
        Lateral cutoff: lines farther than this from the tunnel axis see
        no appreciable hyperbola.

    Returns
    -------
    specs : list[dict]
        Arguments ready for :func:`simulate_bscan`.
    """
    specs = []
    for t in tunnels:
        x0, y0 = t['x0_m'], t['y0_m']
        x1, y1 = t['x1_m'], t['y1_m']
        dx_, dy_ = x1 - x0, y1 - y0
        seg_len2 = dx_ * dx_ + dy_ * dy_
        if seg_len2 < 1e-12:
            closest_x, closest_y = x0, y0
        else:
            t_proj = max(0.0, min(1.0, ((y_line - y0) * dy_) / seg_len2))
            closest_x = x0 + t_proj * dx_
            closest_y = y0 + t_proj * dy_
        dist = abs(closest_y - y_line)
        if dist > line_width_m:
            continue
        # Depth can gently vary across the line for sloping tunnels.
        depth = t['depth_m']
        # Simple linear interpolation of depth along segment.
        if seg_len2 > 1e-12:
            frac = ((closest_y - y0) * dy_) / seg_len2
            frac = max(0.0, min(1.0, frac))
            # Allow a +/-5% depth variation if tunnel is slanted.
            depth = t['depth_m'] * (1.0 + 0.05 * (frac - 0.5))
        # Amplitude attenuation with lateral distance from tunnel axis.
        attenuation = max(0.2, np.exp(-0.5 * (dist / max(0.3, t['radius_m'])) ** 2))
        specs.append(dict(
            x0_m=float(closest_x),
            depth_m=float(depth),
            v=float(t['v_m_ns']),
            r_m=float(t['radius_m']) * attenuation
        ))
    return specs
