"""3-D digital-twin reconstruction (notebook Section 10).

The detected tunnel is modelled as a cylindrical segment

    T = {(x, y, z) : (x - x_hat(y))^2 + (z - z_hat(y))^2 <= r^2}

where z is positive downward (depth). Detections are first linked across
survey lines into per-tunnel *tracks*; each track axis (x_hat, z_hat) is
then obtained by a Huber-robust line fit through its apex estimates, so
isolated false positives cannot drag the whole axis off. Because a
single-line diffraction hyperbola constrains the depth of the cylinder
top but not its radius, the radius is an explicit prior supplied by the
user or defaulted from the detected tunnels.

The original single-tunnel helpers (:func:`reconstruct_twin`,
:func:`evaluate_twin`) are kept for notebook parity; the app uses the
multi-tunnel variants (:func:`reconstruct_twins`, :func:`evaluate_twins`).
"""

import numpy as np
from scipy.optimize import least_squares


def cylinder_mesh(x_fn, z_fn, y0, y1, r, ny=30, nth=24):
    """Parametric cylinder surface around a curved axis (x_fn(y), z_fn(y))."""
    y = np.linspace(y0, y1, ny)
    th = np.linspace(0, 2 * np.pi, nth)
    Y, TH = np.meshgrid(y, th)
    return x_fn(Y) + r * np.cos(TH), Y, z_fn(Y) + r * np.sin(TH)


def fit_axis(lines_y, xs_axis, zs_axis):
    """Fit linear axes x(y) and z(y) from per-line apex estimates.

    Parameters
    ----------
    lines_y : ndarray
        y positions of survey lines.
    xs_axis, zs_axis : ndarray
        Estimated x0 and depth per line; NaN where no detection.

    Returns
    -------
    x_fn, z_fn : callable
        ``x_fn(y)`` and ``z_fn(y)`` evaluated by the fitted polynomials.
    ok : ndarray, bool
        Mask of lines with valid detections.
    """
    ok = ~np.isnan(xs_axis)
    if ok.sum() < 2:
        # Fall back to a constant axis at the mean valid estimate.
        x_mean = np.nanmean(xs_axis) if ok.any() else lines_y.mean()
        z_mean = np.nanmean(zs_axis) if ok.any() else 2.0
        x_fn = lambda y: np.full_like(np.asarray(y, dtype=float), x_mean)
        z_fn = lambda y: np.full_like(np.asarray(y, dtype=float), z_mean)
        return x_fn, z_fn, ok
    cx = np.polyfit(lines_y[ok], xs_axis[ok], 1)
    cz = np.polyfit(lines_y[ok], zs_axis[ok], 1)
    x_fn = lambda y: np.polyval(cx, np.asarray(y, dtype=float))
    z_fn = lambda y: np.polyval(cz, np.asarray(y, dtype=float))
    return x_fn, z_fn, ok


def reconstruct_twin(lines_y, line_dets, default_radius=0.55):
    """Build a digital twin from per-line detections.

    Parameters
    ----------
    lines_y : ndarray
    line_dets : list[list[dict]]
        Detections per line (first detection per line is used as axis).
    default_radius : float
        Radius prior [m].

    Returns
    -------
    dict with keys ``x_fn, z_fn, radius, xs, zs, ok, v_mean``.
    """
    xs_axis = np.array([d[0]['x0_m'] if d else np.nan for d in line_dets])
    zs_axis = np.array([d[0]['depth_m'] if d else np.nan for d in line_dets])
    vs_axis = np.array([d[0]['v'] if d else np.nan for d in line_dets])
    x_fn, z_fn, ok = fit_axis(lines_y, xs_axis, zs_axis)
    radius = default_radius
    return dict(
        x_fn=x_fn,
        z_fn=z_fn,
        radius=radius,
        xs=xs_axis,
        zs=zs_axis,
        ok=ok,
        v_mean=float(np.nanmean(vs_axis)) if ok.any() else None,
    )


def evaluate_twin(twin, tunnels_truth, lines_y):
    """Compute axis errors against ground truth for a single perpendicular tunnel.

    Parameters
    ----------
    twin : dict
        Output of :func:`reconstruct_twin`.
    tunnels_truth : list[dict]
        Ground-truth tunnel segments.
    lines_y : ndarray

    Returns
    -------
    dict with lateral and depth MAEs, velocity error, radius bias.
    """
    if not tunnels_truth:
        return dict(lateral_mae=None, depth_mae=None, v_err=None, radius_bias=None)
    # Use the longest / first tunnel as reference.
    t = max(tunnels_truth, key=lambda tt: np.hypot(tt['x1_m'] - tt['x0_m'], tt['y1_m'] - tt['y0_m']))
    x_true = np.linspace(t['x0_m'], t['x1_m'], len(lines_y))
    y_true = np.linspace(t['y0_m'], t['y1_m'], len(lines_y))
    z_true = np.full_like(y_true, t['depth_m'])

    ok = twin['ok']
    lateral_errs = np.abs(twin['xs'][ok] - np.interp(lines_y[ok], y_true, x_true))
    depth_errs = np.abs(twin['zs'][ok] - np.interp(lines_y[ok], y_true, z_true))
    return dict(
        lateral_mae=float(lateral_errs.mean()) if ok.any() else None,
        depth_mae=float(depth_errs.mean()) if ok.any() else None,
        v_err=abs(twin['v_mean'] - t['v_m_ns']) if twin['v_mean'] is not None else None,
        radius_bias=abs(twin['radius'] - t['radius_m']),
    )


# ------------------------------------------------------------------
# Multi-tunnel tracking, robust fits, per-tunnel evaluation
# ------------------------------------------------------------------
def track_detections(lines_y, line_dets, gate_m=1.0, min_len=3):
    """Link per-line detections into cross-line tunnel tracks.

    A greedy nearest-neighbour assignment: each detection joins the track
    whose last lateral position is closest, provided it is within
    ``gate_m`` metres and the track has not already absorbed a detection
    on the same line; otherwise a new track is started. Tracks with fewer
    than ``min_len`` detections are discarded as noise.

    Parameters
    ----------
    lines_y : ndarray
        y position of each survey line [m].
    line_dets : list[list[dict]]
        Detections per line (each with ``x0_m``, ``depth_m``, ``v``).
    gate_m : float
        Maximum lateral jump between consecutive detections of a track.
    min_len : int
        Minimum detections for a track to be considered a tunnel.

    Returns
    -------
    list[list[tuple[int, dict]]]
        Tracks as lists of ``(line_index, detection)`` pairs.
    """
    tracks, last_x = [], []
    for i, dets in enumerate(line_dets):
        fed = set()
        for d in dets:
            x = d['x0_m']
            best, best_dist = -1, gate_m
            for ti in range(len(tracks)):
                if ti in fed:
                    continue
                dist = abs(x - last_x[ti])
                if dist < best_dist:
                    best, best_dist = ti, dist
            if best >= 0:
                tracks[best].append((i, d))
                last_x[best] = x
                fed.add(best)
            else:
                tracks.append([(i, d)])
                last_x.append(x)
                fed.add(len(tracks) - 1)
    return [tr for tr in tracks if len(tr) >= min_len]


def _robust_line(y, x):
    """Huber-robust linear fit; returns ``fn(y)`` and per-point |residuals|."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    c0 = np.polyfit(y, x, 1)
    r0 = np.polyval(c0, y) - x
    mad = np.median(np.abs(r0 - np.median(r0)))
    f_scale = max(0.05, 1.4826 * mad)
    sol = least_squares(lambda c: c[0] * y + c[1] - x, c0,
                        loss='huber', f_scale=f_scale)
    c = sol.x
    fn = lambda yy: c[0] * np.asarray(yy, dtype=float) + c[1]
    return fn, np.abs(c[0] * y + c[1] - x)


def reconstruct_twins(lines_y, line_dets, default_radius=0.55,
                      gate_m=1.0, min_len=3):
    """Build one digital twin per tracked tunnel.

    Returns
    -------
    list[dict]
        Each twin has ``x_fn, z_fn, radius, ys, xs, zs, res_x, res_z,
        v_mean, n_det, y0, y1``. ``res_x``/``res_z`` are the per-line
        absolute residuals of the robust axis fits.
    """
    lines_y = np.asarray(lines_y, dtype=float)
    twins = []
    for tr in track_detections(lines_y, line_dets, gate_m, min_len):
        idx = np.array([i for i, _ in tr])
        ys = lines_y[idx]
        xs = np.array([d['x0_m'] for _, d in tr])
        zs = np.array([d['depth_m'] for _, d in tr])
        vs = np.array([d['v'] for _, d in tr])
        x_fn, res_x = _robust_line(ys, xs)
        z_fn, res_z = _robust_line(ys, zs)
        twins.append(dict(
            x_fn=x_fn, z_fn=z_fn, radius=default_radius,
            ys=ys, xs=xs, zs=zs, res_x=res_x, res_z=res_z,
            v_mean=float(np.mean(vs)),
            n_det=len(tr), y0=float(ys.min()), y1=float(ys.max()),
        ))
    return twins


def evaluate_twins(twins, tunnels_truth, lines_y):
    """Match each ground-truth tunnel to its best twin and score it.

    For every GT tunnel the twin minimising mean lateral + depth axis
    distance over the overlapping y range is selected. Metrics computed
    over that overlap: lateral/depth MAE of the fitted axes, velocity
    error, radius bias, and coverage (fraction of survey lines inside the
    overlap that carry a detection of the matched track).

    Returns
    -------
    list[dict]
        One entry per GT tunnel with keys ``tunnel`` (1-based),
        ``twin`` (1-based or None), ``lateral_mae``, ``depth_mae``,
        ``v_err``, ``radius_bias``, ``coverage``.
    """
    lines_y = np.asarray(lines_y, dtype=float)
    results = []
    for ti, t in enumerate(tunnels_truth):
        gy0, gy1 = sorted((t['y0_m'], t['y1_m']))
        entry = dict(tunnel=ti + 1, twin=None, lateral_mae=None,
                     depth_mae=None, v_err=None, radius_bias=None,
                     coverage=0.0)
        if gy1 - gy0 < 1e-9:  # tunnel parallel to flight lines: not handled
            results.append(entry)
            continue
        # GT axis as functions of y (ascending).
        frac = lambda yy: (yy - t['y0_m']) / (t['y1_m'] - t['y0_m'])
        x_of_y = lambda yy: t['x0_m'] + (t['x1_m'] - t['x0_m']) * frac(yy)

        best, best_score = None, np.inf
        for wi, tw in enumerate(twins):
            lo = max(gy0, tw['y0'])
            hi = min(gy1, tw['y1'])
            if hi - lo < 0.3:
                continue
            yy = np.linspace(lo, hi, 30)
            lat = float(np.abs(tw['x_fn'](yy) - x_of_y(yy)).mean())
            dep = float(np.abs(tw['z_fn'](yy) - t['depth_m']).mean())
            if lat + dep < best_score:
                best_score, best, best_lat, best_dep = lat + dep, wi, lat, dep
        if best is not None:
            tw = twins[best]
            lo = max(gy0, tw['y0'])
            hi = min(gy1, tw['y1'])
            n_lines = max(1, int(((lines_y >= lo) & (lines_y <= hi)).sum()))
            n_dets = int(((tw['ys'] >= lo) & (tw['ys'] <= hi)).sum())
            entry.update(
                twin=best + 1,
                lateral_mae=best_lat,
                depth_mae=best_dep,
                v_err=abs(tw['v_mean'] - t['v_m_ns']),
                radius_bias=abs(tw['radius'] - t['radius_m']),
                coverage=min(1.0, n_dets / n_lines),
            )
        results.append(entry)
    return results
