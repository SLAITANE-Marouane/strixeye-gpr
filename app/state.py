"""Streamlit session-state helpers for the GPR tunnel-detection app."""

import hashlib
import json

import streamlit as st


class Keys:
    # hardware / reproducibility
    DEVICE = "device"
    DEVICE_CONFIRMED = "device_confirmed"
    SEED = "seed"

    # survey / scene
    SURVEY_WIDTH = "survey_width"
    SURVEY_LENGTH = "survey_length"
    LINE_SPACING = "line_spacing"
    PATTERN = "pattern"
    TUNNELS = "tunnels"
    CONFIG_HASH = "config_hash"

    # pipeline results
    TRAJECTORY = "trajectory"
    RAW_SCANS = "raw_scans"
    GT_MASKS = "gt_masks"
    GT_PARAMS = "gt_params"
    PREPROCESSED = "preprocessed"
    PROBS = "probs"
    LINE_DETS = "line_dets"
    HEATMAP = "heatmap"
    HEAT_EXTENT = "heat_extent"
    XS = "xs"
    YS = "ys"
    GT_PLAN_MASK = "gt_plan_mask"
    TWIN = "twin"
    TWIN_ERRORS = "twin_errors"
    RUN_TIMES = "run_times"
    REPORT_METRICS = "report_metrics"


#: Result keys in pipeline order (used for invalidation and status chips).
PIPELINE_STEPS = [
    ("B-scans", [Keys.RAW_SCANS, Keys.GT_MASKS, Keys.GT_PARAMS, Keys.YS]),
    ("Preprocessing", [Keys.PREPROCESSED]),
    ("AI detection", [Keys.PROBS]),
    ("Localization", [Keys.LINE_DETS]),
    ("Heat map", [Keys.HEATMAP, Keys.HEAT_EXTENT, Keys.XS, Keys.GT_PLAN_MASK]),
    ("3-D twin", [Keys.TWIN, Keys.TWIN_ERRORS]),
    ("Report", [Keys.REPORT_METRICS]),
]


def ensure(key, default):
    """Return session-state value, initialising with default if absent."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def set_device(device):
    st.session_state[Keys.DEVICE] = device
    st.session_state[Keys.DEVICE_CONFIRMED] = True


def get_device():
    return ensure(Keys.DEVICE, "cpu")


def device_confirmed():
    return ensure(Keys.DEVICE_CONFIRMED, False)


def reset_pipeline(exclude=None):
    """Clear computed results (keeps configuration)."""
    exclude = set(exclude or [])
    keys = [Keys.TRAJECTORY, Keys.RUN_TIMES]
    for _, step_keys in PIPELINE_STEPS:
        keys.extend(step_keys)
    for k in keys:
        if k not in exclude:
            st.session_state.pop(k, None)


def config_hash(config):
    """Stable hash of the survey/scene configuration."""
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()


def invalidate_if_config_changed(config):
    """Clear all pipeline results when the configuration has changed.

    Returns True if results were invalidated on this run.
    """
    h = config_hash(config)
    if st.session_state.get(Keys.CONFIG_HASH) == h:
        return False
    had_results = any(k in st.session_state for k in (Keys.RAW_SCANS, Keys.PROBS))
    reset_pipeline()
    st.session_state[Keys.CONFIG_HASH] = h
    return had_results
