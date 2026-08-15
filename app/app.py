"""Streamlit front-end for the AI-GPR tunnel-detection research pipeline.

Run with:
    streamlit run app/app.py
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

# Make project root importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.components import ICON_PATH  # noqa: E402  (needed for page icon)

st.set_page_config(
    page_title="StrixEyE — AI-GPR Tunnel Detection",
    page_icon=str(ICON_PATH),
    layout="wide",
)

import torch  # noqa: E402
from streamlit.runtime.scriptrunner import get_script_run_ctx  # noqa: E402

from src.config import GPR_NT, GPR_NX, GPR_DT, GPR_DX, GLOBAL_SEED  # noqa: E402
from src.simulator import simulate_bscan, compute_tunnel_specs_for_line  # noqa: E402
from src.preprocess import preprocess, preprocess_stages  # noqa: E402
from src.model import (  # noqa: E402
    CompactUNet, get_device, batch_predict, find_latest_checkpoint
)
from src.localize import localize_tunnels, mask_from_prob  # noqa: E402
from src.fusion import build_heatmap, compute_threshold_iou, make_gt_plan_mask  # noqa: E402
from src.twin import reconstruct_twins, evaluate_twins  # noqa: E402

from app.state import (  # noqa: E402
    Keys, ensure, reset_pipeline, invalidate_if_config_changed
)
from app.i18n import _, language_selector  # noqa: E402
from app.components import (  # noqa: E402
    explanation, device_selector, device_badge, show_tutorial,
    generate_trajectory, plot_plan_view, plot_heatmap_interactive,
    plot_3d_twin, plot_residuals, plot_bscan, export_report, report_dataframe,
    inject_css, render_header, pipeline_status_bar, metric_cards,
    build_pdf_report,
    LOGO_PATH,
)

inject_css()


# ------------------------------------------------------------------
# Startup tutorial
# ------------------------------------------------------------------
# One-shot trigger: the dialog is invoked only on reruns where it was
# explicitly requested (first load, sidebar button, in-dialog navigation).
# Dismissing it with the close button therefore sticks, instead of the
# dialog reappearing on every interaction.
# NOTE: a separate sentinel marks the first load. Guarding on
# "tutorial_trigger" itself would resurrect the dialog, because showing
# it consumes (pops) the key and the guard would re-arm it on the next
# rerun.
if "tutorial_auto_shown" not in st.session_state:
    st.session_state["tutorial_auto_shown"] = True
    st.session_state["tutorial_trigger"] = True

if get_script_run_ctx() is not None and st.session_state.pop("tutorial_trigger", False):
    show_tutorial()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading U-Net checkpoint ... / Chargement du checkpoint U-Net ...")
def load_model(checkpoint_path, device_str):
    device = get_device(device_str)
    model = CompactUNet(base=8)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def random_tunnels(n, width, length):
    rng = np.random.default_rng()
    tunnels = []
    for _ in range(n):
        x0 = float(rng.uniform(1.5, width - 1.5))
        x1 = float(rng.uniform(1.5, width - 1.5))
        y0 = float(rng.uniform(1.0, length - 1.0))
        y1 = float(rng.uniform(1.0, length - 1.0))
        tunnels.append(dict(
            x0_m=x0, y0_m=y0, x1_m=x1, y1_m=y1,
            depth_m=float(rng.uniform(1.2, 2.8)),
            radius_m=float(rng.uniform(0.4, 0.8)),
            v_m_ns=float(rng.uniform(0.08, 0.13))
        ))
    return tunnels


def ensure_model():
    ckpt = find_latest_checkpoint()
    if ckpt is None:
        st.error(_("No trained checkpoint found in `models/`. Please run `python -m scripts.train` first."))
        st.stop()
    device = get_device(ensure(Keys.DEVICE, "cpu"))
    return load_model(str(ckpt), str(device))


def _time_step(name, fn):
    """Run a pipeline step, record its wall time, return fn's result."""
    t0 = time.time()
    out = fn()
    st.session_state.setdefault(Keys.RUN_TIMES, {})[name] = time.time() - t0
    return out


def _clamp(v, lo, hi):
    return max(lo, min(float(v), hi))


#: One B-scan covers exactly this many metres (model input geometry).
MAX_LINE_M = float(GPR_NX * GPR_DX)


# ------------------------------------------------------------------
# Pipeline steps (each writes its results into session state)
# ------------------------------------------------------------------
def step_bscans(trajectory, tunnels, seed):
    raw_scans, gt_masks, gt_params, ys = [], [], [], []
    for idx, line in enumerate(trajectory):
        if line['vertical']:
            continue
        ys.append(line['y'])
        specs = compute_tunnel_specs_for_line(line['y'], tunnels)
        # Empty spec list -> true negative scan (no phantom tunnels).
        img, mask, params = simulate_bscan(
            seed=int(seed) + 1000 + idx, tunnel_specs=specs)
        raw_scans.append(img)
        gt_masks.append(mask)
        gt_params.append(params)
    st.session_state[Keys.RAW_SCANS] = raw_scans
    st.session_state[Keys.GT_MASKS] = gt_masks
    st.session_state[Keys.GT_PARAMS] = gt_params
    st.session_state[Keys.YS] = np.array(ys)
    return _("Generated {n} B-scans.").format(n=len(raw_scans))


def step_preprocess():
    scans = st.session_state[Keys.RAW_SCANS]
    st.session_state[Keys.PREPROCESSED] = [preprocess(img) for img in scans]
    return _("Preprocessed {n} B-scans.").format(n=len(scans))


def step_detect(device):
    pre = st.session_state[Keys.PREPROCESSED]
    model = ensure_model()
    probs = batch_predict(model, pre, device)
    st.session_state[Keys.PROBS] = list(probs)
    return _("U-Net inference on {dev} — {n} lines.").format(
        dev=device_badge(device), n=len(probs))


def step_localize():
    probs = st.session_state[Keys.PROBS]
    line_dets = [localize_tunnels(mask_from_prob(p)) for p in probs]
    st.session_state[Keys.LINE_DETS] = line_dets
    n = sum(1 for d in line_dets if d)
    return _("Localised detections on {n}/{m} lines.").format(n=n, m=len(line_dets))


def step_heatmap(tunnels, line_spacing, survey_width):
    ys = st.session_state[Keys.YS]
    probs = st.session_state[Keys.PROBS]
    heatmap, extent, xs = build_heatmap(probs, ys, line_spacing)
    st.session_state[Keys.HEATMAP] = heatmap
    st.session_state[Keys.HEAT_EXTENT] = extent
    st.session_state[Keys.XS] = xs
    st.session_state[Keys.GT_PLAN_MASK] = make_gt_plan_mask(
        tunnels, ys, xs, survey_width)
    return _("Heat map built.")


def step_twin(tunnels):
    default_r = tunnels[0]['radius_m'] if tunnels else 0.55
    twins = reconstruct_twins(st.session_state[Keys.YS],
                              st.session_state[Keys.LINE_DETS],
                              default_radius=default_r)
    st.session_state[Keys.TWIN] = twins
    st.session_state[Keys.TWIN_ERRORS] = evaluate_twins(
        twins, tunnels, st.session_state[Keys.YS])
    return _("{n} twin(s) reconstructed.").format(n=len(twins))


def step_report(device):
    metrics = {}
    if Keys.HEATMAP in st.session_state:
        _taus, _ious, best_tau, best_iou = compute_threshold_iou(
            st.session_state[Keys.HEATMAP],
            st.session_state[Keys.GT_PLAN_MASK]
        )
        metrics['best_threshold'] = best_tau
        metrics['plan_view_iou'] = best_iou
    if Keys.LINE_DETS in st.session_state:
        metrics['lines_with_detections'] = sum(
            1 for d in st.session_state[Keys.LINE_DETS] if d)
        metrics['total_lines'] = len(st.session_state[Keys.LINE_DETS])
    if Keys.TWIN_ERRORS in st.session_state:
        results = st.session_state[Keys.TWIN_ERRORS]
        matched = [e for e in results if e.get('twin')]
        metrics['tunnels_matched'] = len(matched)
        metrics['tunnels_total'] = len(results)

        def _mean(key):
            vals = [e[key] for e in matched if e.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        metrics['lateral_mae'] = _mean('lateral_mae')
        metrics['depth_mae'] = _mean('depth_mae')
        metrics['v_err'] = _mean('v_err')
        metrics['radius_bias'] = _mean('radius_bias')
        metrics['mean_coverage'] = _mean('coverage')
    rt = st.session_state.get(Keys.RUN_TIMES, {})
    if 'detect' in rt:
        metrics['inference_time_s'] = rt['detect']
    metrics['device'] = str(device)
    st.session_state[Keys.REPORT_METRICS] = metrics
    return _("Report generated.")


def run_full_pipeline(trajectory, tunnels, seed, line_spacing,
                      survey_width, device):
    """Execute every step in order; results land in session state."""
    with st.status(_("Running full pipeline ..."), expanded=True) as status:
        st.write("🛰️ " + _time_step('bscans',
                 lambda: step_bscans(trajectory, tunnels, seed)))
        st.write("🧹 " + _time_step('preprocess', step_preprocess))
        st.write("🧠 " + _time_step('detect', lambda: step_detect(device)))
        st.write("📍 " + _time_step('localize', step_localize))
        st.write("🗺️ " + _time_step('heatmap',
                 lambda: step_heatmap(tunnels, line_spacing, survey_width)))
        st.write("🏗️ " + _time_step('twin', lambda: step_twin(tunnels)))
        st.write("📊 " + _time_step('report', lambda: step_report(device)))
        status.update(label=_("Pipeline complete ✔"), state="complete")


def _gt_visible():
    """Ground truth is shown unless presentation mode is on and unrevealed."""
    return (not st.session_state.get("pres_mode", False)) \
        or st.session_state.get("gt_revealed", False)


# ------------------------------------------------------------------
# Fragment viewers (slider + plot rerun in isolation, no full reload)
# ------------------------------------------------------------------
@st.fragment
def _bscan_viewer():
    scans = st.session_state[Keys.RAW_SCANS]
    idx = st.slider(_("Line index"), 0, len(scans) - 1, 0, key="bscan_slider")
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_bscan(ax, scans[idx],
               title=_("Raw B-scan – line {i}").format(i=idx),
               mask=st.session_state[Keys.GT_MASKS][idx] if _gt_visible() else None)
    st.pyplot(fig)
    if _gt_visible():
        st.caption(_("Ground-truth hyperbola mask overlaid in lime."))
    else:
        st.caption(_("🔒 Ground truth hidden (presentation mode)"))


@st.fragment
def _preprocess_viewer():
    n = len(st.session_state[Keys.PREPROCESSED])
    idx = st.slider(_("Line index"), 0, n - 1, 0, key="pre_slider")
    # Compute the stage cascade once, for the selected line only.
    stages = preprocess_stages(st.session_state[Keys.RAW_SCANS][idx])
    panels = [
        (stages['raw'], _("raw")),
        (stages['dewow'], _("1) dewow")),
        (stages['time_zero'], _("2) time-zero")),
        (stages['bg_removed'], _("3) bg-removed")),
        (stages['gained'], _("4) SEC+AGC")),
        (stages['final'], _("5) normalised")),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (arr, ttl) in zip(axes.flat, panels):
        plot_bscan(ax, arr, title=ttl)
    plt.tight_layout()
    st.pyplot(fig)


@st.fragment
def _ai_viewer():
    n = len(st.session_state[Keys.PROBS])
    idx = st.slider(_("Line index"), 0, n - 1, 0, key="ai_slider")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_bscan(axes[0], st.session_state[Keys.PREPROCESSED][idx],
               title=_("Preprocessed input"),
               gt_params=st.session_state[Keys.GT_PARAMS][idx] if _gt_visible() else None)
    prob = st.session_state[Keys.PROBS][idx]
    axes[1].imshow(prob, cmap='hot', aspect='auto', vmin=0, vmax=1,
                   extent=[0, GPR_NX * GPR_DX, GPR_NT * GPR_DT, 0])
    axes[1].contour(mask_from_prob(prob), colors='lime', linewidths=1.0,
                    extent=[0, GPR_NX * GPR_DX, GPR_NT * GPR_DT, 0])
    axes[1].set_title(_("U-Net probability (≥0.5 contour in lime)"))
    axes[1].set_xlabel(_("x [m]"))
    axes[1].set_ylabel(_("two-way time [ns]"))
    st.pyplot(fig)


@st.fragment
def _loc_viewer():
    n = len(st.session_state[Keys.LINE_DETS])
    idx = st.slider(_("Line index"), 0, n - 1, 0, key="loc_slider")
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_bscan(ax, st.session_state[Keys.PREPROCESSED][idx],
               title=_("Localization – line {i}").format(i=idx),
               dets=st.session_state[Keys.LINE_DETS][idx],
               gt_params=st.session_state[Keys.GT_PARAMS][idx] if _gt_visible() else None)
    st.pyplot(fig)

    rows = []
    for d in st.session_state[Keys.LINE_DETS][idx]:
        rows.append(dict(
            x0_est_m=d['x0_m'], depth_est_m=d['depth_m'],
            v_est_m_ns=d['v'], area_px=d['area']
        ))
    if _gt_visible():
        for g in st.session_state[Keys.GT_PARAMS][idx]:
            rows.append(dict(
                x0_true_m=g['x0_m'], depth_true_m=g['depth_m'],
                v_true_m_ns=g['v'], radius_m=g['r_m']
            ))
    if rows:
        st.dataframe(pd.DataFrame(rows).round(3))


@st.fragment
def _heatmap_viewer():
    tau = st.slider(_("Threshold"), 0.0, 1.0, 0.5, 0.05, key="heat_tau")
    st.plotly_chart(plot_heatmap_interactive(
        st.session_state[Keys.HEATMAP],
        st.session_state[Keys.HEAT_EXTENT],
        st.session_state[Keys.XS],
        st.session_state[Keys.YS],
        st.session_state.get(Keys.GT_PLAN_MASK) if _gt_visible() else None,
        tau
    ), width='stretch')

    if _gt_visible():
        _taus, _ious, best_tau, best_iou = compute_threshold_iou(
            st.session_state[Keys.HEATMAP],
            st.session_state[Keys.GT_PLAN_MASK]
        )
        st.write(_("Best threshold: **{tau:.2f}** → plan-view IoU **{iou:.3f}**").format(
            tau=best_tau, iou=best_iou))
    else:
        st.caption(_("🔒 Ground truth hidden (presentation mode)"))


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), width='stretch')
language_selector()
st.sidebar.markdown(f"#### {_('Control Panel')}")

if st.sidebar.button(_("📖 Show tutorial"), width='stretch'):
    st.session_state["tutorial_page"] = 0
    st.session_state["tutorial_trigger"] = True
    st.rerun()

st.sidebar.divider()

# 1. Device selection
device = device_selector()

# 2. Reproducibility
st.sidebar.header(_("Reproducibility"))
seed = st.sidebar.number_input(
    _("Global random seed"), min_value=0, max_value=99999,
    value=int(ensure(Keys.SEED, GLOBAL_SEED)),
    help=_("Fixes every random generator (tunnel layout, simulator noise, clutter) "
           "so the whole pipeline is reproducible. Click 'Set seed' to apply it; "
           "change the value to explore a different random realisation."),
)
ensure(Keys.SEED, seed)
if st.sidebar.button(_("Set seed")):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    st.sidebar.success(_("Seeds fixed."))

# 3. Survey configuration
st.sidebar.header(_("Survey configuration"))
# Clamp stored values to the current widget bounds (protects sessions
# created before the width cap was introduced).
if "survey_w" in st.session_state:
    st.session_state["survey_w"] = _clamp(st.session_state["survey_w"], 2.0, MAX_LINE_M)
survey_width = st.sidebar.number_input(
    _("Survey width [m]"), 2.0, MAX_LINE_M, min(10.0, MAX_LINE_M), 0.5,
    key="survey_w",
    help=_("Range 2.0–{m:.2f} m. Capped by the model input: one B-scan covers "
           "{m:.2f} m ({n} traces × {d} m).").format(m=MAX_LINE_M, n=GPR_NX, d=GPR_DX))
survey_length = st.sidebar.number_input(
    _("Survey length [m]"), 2.0, 50.0, 10.0, 0.5,
    key="survey_l",
    help=_("Range 2–50 m. Number of flight lines ≈ length ÷ line spacing."))
line_spacing = st.sidebar.selectbox(
    _("Line spacing [m]"), [0.25, 0.5, 1.0], index=1,
    help=_("Distance between adjacent flight lines. Smaller spacing = denser "
           "survey and more B-scans to process."))
st.sidebar.caption(
    _("Limits: width ≤ {w:.2f} m (model line length) · length ≤ 50 m · "
      "up to 5 tunnels.").format(w=MAX_LINE_M))
st.sidebar.number_input(
    _("Trace spacing [m]"), value=GPR_DX, disabled=True,
    help=_("Fixed by model input size ({n} traces -> {d:.2f} m line).").format(
        n=GPR_NX, d=GPR_NX * GPR_DX))
_patterns = ("lawnmower", "grid")
_pattern_labels = [_(p) for p in _patterns]
pattern = _patterns[_pattern_labels.index(
    st.sidebar.radio(_("Trajectory pattern"), _pattern_labels, index=0,
                     help=_("Lawnmower: the drone flies parallel lines back and forth along x, "
                            "stepping in y after each pass (one B-scan per line).\n\n"
                            "Grid: same lawnmower scan plus a second set of perpendicular lines "
                            "(shown in the plan view) for cross-line confirmation.")))]

# 4. Tunnel editor
st.sidebar.header(_("Ground-truth tunnels"))
n_tunnels = st.sidebar.number_input(_("Number of tunnels"), 0, 5, 2, key="n_tunnels",
                                    help=_("0–5 tunnels. Each tunnel adds its start/end "
                                           "points, depth, radius and soil velocity."))
st.sidebar.caption(
    _("Ranges — depth 0.5–4 m · radius 0.1–2 m · soil velocity 0.05–0.20 m/ns. "
      "Coordinates must lie inside the survey."))
tunnels = []


# Clamp any previously stored tunnel-widget values to the current survey
# bounds (e.g. after shrinking the survey or adding more tunnels).
for i in range(n_tunnels):
    for k, lo, hi in ((f"t{i}x0", 0.0, survey_width), (f"t{i}x1", 0.0, survey_width),
                      (f"t{i}y0", 0.0, survey_length), (f"t{i}y1", 0.0, survey_length)):
        if k in st.session_state:
            st.session_state[k] = _clamp(st.session_state[k], lo, hi)

for i in range(n_tunnels):
    with st.sidebar.expander(_("Tunnel {i}").format(i=i+1), expanded=i == 0):
        c1, c2 = st.columns(2)
        x0 = c1.number_input(_("Start x (m)"), 0.0, survey_width,
                             _clamp(3.0 + i * 2.0, 0.0, survey_width), 0.1, key=f"t{i}x0")
        y0 = c2.number_input(_("Start y (m)"), 0.0, survey_length,
                             _clamp(1.0 + i * 1.5, 0.0, survey_length), 0.1, key=f"t{i}y0")
        x1 = c1.number_input(_("End x (m)"), 0.0, survey_width,
                             _clamp(7.0 + i * 0.5, 0.0, survey_width), 0.1, key=f"t{i}x1")
        y1 = c2.number_input(_("End y (m)"), 0.0, survey_length,
                             _clamp(9.0 - i * 1.0, 0.0, survey_length), 0.1, key=f"t{i}y1")
        depth = st.number_input(_("Depth (m)"), 0.5, 4.0, 2.0 + i * 0.2, 0.1, key=f"t{i}d")
        radius = st.number_input(_("Radius (m)"), 0.1, 2.0, 0.55, 0.05, key=f"t{i}r")
        velocity = st.number_input(_("Soil velocity (m/ns)"), 0.05, 0.20, 0.10, 0.01, key=f"t{i}v")
        tunnels.append(dict(x0_m=x0, y0_m=y0, x1_m=x1, y1_m=y1,
                            depth_m=depth, radius_m=radius, v_m_ns=velocity))

def _apply_random_scenario():
    """on_click callback: runs BEFORE the rerun, so writing widget keys is
    allowed (writing them after the widgets are instantiated raises
    StreamlitAPIException)."""
    n = max(1, int(st.session_state.get("n_tunnels", 2)))
    w = float(st.session_state.get("survey_w", 10.0))
    length = float(st.session_state.get("survey_l", 10.0))
    for i, t in enumerate(random_tunnels(n, w, length)):
        st.session_state[f"t{i}x0"] = t['x0_m']
        st.session_state[f"t{i}y0"] = t['y0_m']
        st.session_state[f"t{i}x1"] = t['x1_m']
        st.session_state[f"t{i}y1"] = t['y1_m']
        st.session_state[f"t{i}d"] = t['depth_m']
        st.session_state[f"t{i}r"] = t['radius_m']
        st.session_state[f"t{i}v"] = t['v_m_ns']


st.sidebar.button(_("🎲 Random scenario"), width='stretch',
                  on_click=_apply_random_scenario)

# 5. Action buttons
st.sidebar.header(_("Run pipeline"))
if st.sidebar.button(_("▶ Run full pipeline"), type="primary", width='stretch'):
    reset_pipeline()
    st.session_state["pending_run_all"] = True

# 6. Presentation mode
st.sidebar.header(_("Presentation"))
pres = st.sidebar.toggle(_("🎬 Presentation mode"), key="pres_mode",
                         help=_("Hides all ground-truth overlays until you reveal "
                                "them — perfect for live demos."))
if pres:
    if st.session_state.get("gt_revealed", False):
        st.sidebar.success(_("Ground truth revealed 🎭"))
        if st.sidebar.button(_("🙈 Hide again"), width='stretch'):
            st.session_state["gt_revealed"] = False
            st.rerun()
    else:
        st.sidebar.button(_("🎭 Reveal ground truth"), width='stretch',
                          on_click=lambda: st.session_state.update(gt_revealed=True))
else:
    st.session_state.pop("gt_revealed", None)


# ------------------------------------------------------------------
# Configuration change -> invalidate stale results
# ------------------------------------------------------------------
config = dict(
    seed=int(seed), survey_width=survey_width, survey_length=survey_length,
    line_spacing=line_spacing, pattern=pattern, tunnels=tunnels,
)
if invalidate_if_config_changed(config):
    st.toast(_("Scene configuration changed — previous results were cleared."), icon="♻️")

trajectory = ensure(Keys.TRAJECTORY, generate_trajectory(
    survey_width, survey_length, line_spacing, pattern))


# ------------------------------------------------------------------
# Main page
# ------------------------------------------------------------------
render_header()
pipeline_status_bar()

# Execute pending full-pipeline run (deterministic one-shot flag).
if st.session_state.pop("pending_run_all", False):
    run_full_pipeline(trajectory, tunnels, seed, line_spacing,
                      survey_width, device)


def _done(key):
    return key in st.session_state


tab_labels = [
    _("1. Scene setup"),
    ("✅ " if _done(Keys.RAW_SCANS) else "") + _("2. B-scans"),
    ("✅ " if _done(Keys.PREPROCESSED) else "") + _("3. Preprocessing"),
    ("✅ " if _done(Keys.PROBS) else "") + _("4. AI detection"),
    ("✅ " if _done(Keys.LINE_DETS) else "") + _("5. Localization"),
    ("✅ " if _done(Keys.HEATMAP) else "") + _("6. Heat map"),
    ("✅ " if _done(Keys.TWIN) else "") + _("7. 3-D twin"),
    ("✅ " if _done(Keys.REPORT_METRICS) else "") + _("8. Report"),
]
tabs = st.tabs(tab_labels)

# ------------------------------------------------------------------
# Step 1
# ------------------------------------------------------------------
with tabs[0]:
    explanation("scene")
    st.plotly_chart(plot_plan_view(tunnels if _gt_visible() else [],
                                   trajectory, survey_width, survey_length, pattern),
                    width='stretch')
    if not _gt_visible():
        st.caption(_("🔒 Ground truth hidden (presentation mode)"))


# ------------------------------------------------------------------
# Step 2
# ------------------------------------------------------------------
with tabs[1]:
    explanation("bscan")
    if st.button(_("Generate B-scans"), key="gen_bscan", type="primary"):
        with st.spinner(_("Simulating B-scans ...")):
            msg = _time_step('bscans', lambda: step_bscans(trajectory, tunnels, seed))
        st.success(msg)

    if Keys.RAW_SCANS in st.session_state:
        _bscan_viewer()


# ------------------------------------------------------------------
# Step 3
# ------------------------------------------------------------------
with tabs[2]:
    explanation("preprocess")
    if st.button(_("Run preprocessing"), key="run_pre", type="primary"):
        if Keys.RAW_SCANS not in st.session_state:
            st.warning(_("Generate B-scans first (Step 2)."))
        else:
            with st.spinner(_("Preprocessing ...")):
                msg = _time_step('preprocess', step_preprocess)
            st.success(msg)

    if Keys.PREPROCESSED in st.session_state:
        _preprocess_viewer()


# ------------------------------------------------------------------
# Step 4
# ------------------------------------------------------------------
with tabs[3]:
    explanation("ai")
    if st.button(_("Run U-Net detection"), key="run_ai", type="primary"):
        if Keys.PREPROCESSED not in st.session_state:
            st.warning(_("Preprocess B-scans first (Step 3)."))
        else:
            with st.spinner(_("U-Net inference on {dev} ...").format(dev=device_badge(device))):
                msg = _time_step('detect', lambda: step_detect(device))
            dt = st.session_state[Keys.RUN_TIMES]['detect']
            n = len(st.session_state[Keys.PROBS])
            st.success(f"{msg} {dt:.2f} s ({dt / n * 1e3:.1f} {_('ms/line')}).")

    if Keys.PROBS in st.session_state:
        _ai_viewer()


# ------------------------------------------------------------------
# Step 5
# ------------------------------------------------------------------
with tabs[4]:
    explanation("localize")
    if st.button(_("Localize tunnels"), key="run_loc", type="primary"):
        if Keys.PROBS not in st.session_state:
            st.warning(_("Run AI detection first (Step 4)."))
        else:
            with st.spinner(_("Fitting hyperbolae ...")):
                msg = _time_step('localize', step_localize)
            st.success(msg)

    if Keys.LINE_DETS in st.session_state:
        _loc_viewer()


# ------------------------------------------------------------------
# Step 6
# ------------------------------------------------------------------
with tabs[5]:
    explanation("heatmap")
    if st.button(_("Build heat map"), key="run_heat", type="primary"):
        if Keys.PROBS not in st.session_state:
            st.warning(_("Run AI detection first (Step 4)."))
        else:
            with st.spinner(_("Fusing lines ...")):
                msg = _time_step('heatmap',
                                 lambda: step_heatmap(tunnels, line_spacing, survey_width))
            st.success(msg)

    if Keys.HEATMAP in st.session_state:
        _heatmap_viewer()


# ------------------------------------------------------------------
# Step 7
# ------------------------------------------------------------------
with tabs[6]:
    explanation("twin")
    if st.button(_("Reconstruct 3-D twin"), key="run_twin", type="primary"):
        if Keys.LINE_DETS not in st.session_state:
            st.warning(_("Localize tunnels first (Step 5)."))
        else:
            with st.spinner(_("Reconstructing twin ...")):
                msg = _time_step('twin', lambda: step_twin(tunnels))
            st.success(msg)

    if Keys.TWIN in st.session_state:
        twins = st.session_state[Keys.TWIN]
        results = st.session_state.get(Keys.TWIN_ERRORS, [])
        if not twins:
            st.warning(_("No reconstruction possible (a track needs ≥ 3 "
                         "detections on consecutive lines)."))
        else:
            col1, col2 = st.columns([3, 2])
            with col1:
                st.plotly_chart(plot_3d_twin(tunnels if _gt_visible() else [],
                                             twins, survey_width,
                                             survey_length, results),
                                width='stretch')
            with col2:
                if not _gt_visible():
                    st.info(_("🔒 Ground truth hidden (presentation mode)"))
                else:
                    st.subheader(_("Axis errors"))

                    def _r(v):
                        return round(v, 3) if v is not None else "—"

                    rows = [{
                        _("Tunnel"): e['tunnel'],
                        _("Twin"): e['twin'] if e['twin'] else "—",
                        _("Lateral MAE [m]"): _r(e['lateral_mae']),
                        _("Depth MAE [m]"): _r(e['depth_mae']),
                        _("Velocity error [m/ns]"): _r(e['v_err']),
                        _("Coverage"): (f"{e['coverage'] * 100:.0f}%"
                                        if e['coverage'] is not None else "—"),
                    } for e in results]
                    if rows:
                        st.dataframe(pd.DataFrame(rows), hide_index=True)

            st.subheader(_("Per-line residuals"))
            st.pyplot(plot_residuals(twins))
            st.caption(_("Residuals of the robust (Huber) axis fit per track — "
                         "spikes mark outlier detections the fit rejected."))


# ------------------------------------------------------------------
# Step 8
# ------------------------------------------------------------------
with tabs[7]:
    explanation("report")
    if st.button(_("Generate report"), key="run_report", type="primary"):
        msg = _time_step('report', lambda: step_report(device))
        st.success(msg)

    if Keys.REPORT_METRICS in st.session_state:
        metrics = st.session_state[Keys.REPORT_METRICS]

        cards = []
        if 'plan_view_iou' in metrics:
            cards.append((_("Plan-view IoU"), f"{metrics['plan_view_iou']:.3f}"))
        if 'lines_with_detections' in metrics:
            cards.append((_("Lines w/ detections"),
                          f"{metrics['lines_with_detections']}/{metrics['total_lines']}"))
        if metrics.get('lateral_mae') is not None:
            cards.append((_("Lateral MAE"), f"{metrics['lateral_mae']:.3f} m"))
        if metrics.get('depth_mae') is not None:
            cards.append((_("Depth MAE"), f"{metrics['depth_mae']:.3f} m"))
        if metrics.get('inference_time_s') is not None:
            cards.append((_("Inference time"), f"{metrics['inference_time_s']:.2f} s"))
        cards.append((_("Device"), metrics.get('device', '—')))
        if cards:
            metric_cards(cards, cols=min(4, len(cards)))
            st.write("")

        with st.expander(_("Full metrics (JSON)")):
            st.json(metrics)

        pdf_bytes = build_pdf_report(
            metrics,
            dict(width=survey_width, length=survey_length,
                 spacing=line_spacing, pattern=_(pattern),
                 n_tunnels=len(tunnels), seed=int(seed), device=str(device)),
            heatmap=st.session_state.get(Keys.HEATMAP),
            extent=st.session_state.get(Keys.HEAT_EXTENT),
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.download_button(_("⬇ Download JSON"),
                           json.dumps(metrics, indent=2, default=str),
                           "report.json", "application/json",
                           width='stretch')
        c2.download_button(_("⬇ Download CSV"),
                           report_dataframe(metrics).to_csv(index=False),
                           "report.csv", "text/csv",
                           width='stretch')
        c3.download_button(_("⬇ Download PDF"), pdf_bytes,
                           "strixeye_report.pdf", "application/pdf",
                           width='stretch')
        if c4.button(_("💾 Save to outputs/"), width='stretch'):
            export_report(metrics, "outputs/report.json")
            report_dataframe(metrics).to_csv("outputs/report.csv", index=False)
            st.success(_("Saved outputs/report.json and outputs/report.csv"))
