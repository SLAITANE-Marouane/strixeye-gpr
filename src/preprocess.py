"""Classical GPR B-scan preprocessing chain (notebook Section 4).

The standard processing stages are:

1. Dewow                – remove DC/low-frequency wow via running mean.
2. Time-zero correction – align first arrival to a reference sample.
3. Background removal   – subtract mean trace to suppress flat layers.
4. SEC gain + AGC       – compensate spreading/attenuation and balance
                           amplitudes with a local RMS gain.
5. Robust normalisation – clip to roughly [-1.5, 1.5].
"""

import numpy as np
from scipy import ndimage


def dewow(data, window=21):
    """Remove DC bias / low-frequency wow via running-mean subtraction."""
    trend = ndimage.uniform_filter1d(data, size=window, axis=0, mode='nearest')
    return data - trend


def time_zero_correct(data, search_rows=15, target_row=5):
    """Align first arrival (max |mean trace| near the top) to ``target_row``."""
    mean_trace = data.mean(axis=1)
    idx = int(np.argmax(np.abs(mean_trace[:search_rows])))
    shift = target_row - idx
    out = np.zeros_like(data)
    if shift > 0:
        out[shift:] = data[:-shift]
    elif shift < 0:
        out[:shift] = data[-shift:]
    else:
        out = data.copy()
    return out


def background_removal(data):
    """Suppress horizontally coherent events (direct wave, flat layers)."""
    return data - data.mean(axis=1, keepdims=True)


def sec_gain(data, alpha=1.0):
    """Spreading-compensation gain g(t) ~ t^alpha."""
    t = np.arange(data.shape[0], dtype=np.float64) + 1.0
    return data * ((t / t[0]) ** alpha)[:, None]


def agc(data, window=31):
    """Automatic gain control: divide by local RMS in a sliding window."""
    sq = ndimage.uniform_filter1d(data ** 2, size=window, axis=0, mode='nearest')
    return data / (np.sqrt(sq) + 1e-6 * np.abs(data).max())


def preprocess_stages(data):
    """Full conditioning chain, returning every intermediate stage.

    Returns
    -------
    dict with keys ``raw``, ``dewow``, ``time_zero``, ``bg_removed``,
    ``gained`` and ``final`` (the float32 normalised output, identical
    to :func:`preprocess`).
    """
    d = dewow(data)
    t = time_zero_correct(d)
    b = background_removal(t)
    g = agc(sec_gain(b, alpha=1.0))
    scale = np.percentile(np.abs(g), 99.5) + 1e-9
    final = np.clip(g / scale, -1.5, 1.5).astype(np.float32)
    return dict(raw=data, dewow=d, time_zero=t, bg_removed=b,
                gained=g, final=final)


def preprocess(data):
    """Full conditioning chain -> float32 array in roughly [-1.5, 1.5]."""
    return preprocess_stages(data)['final']
