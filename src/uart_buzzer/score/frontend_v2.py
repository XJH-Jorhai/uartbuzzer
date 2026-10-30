"""Versioned boundary correction: use centered 20ms energy, retain v1 salience.

v1 and its first probe remain unchanged. Parameters were selected after its
early-boundary failure; additional transition fixture is evaluated afterwards.
"""
import numpy as np
from .frontend import extract_candidates as extract_v1, decode_melody


def extract_candidates(pcm, sample_rate=16000, **kwargs):
    c = extract_v1(pcm, sample_rate, **kwargs)
    x = np.asarray(pcm, dtype=np.float64)
    old = np.asarray(c['activity'], dtype=float)
    old_gate = np.clip((20 * np.log10(np.maximum(c['loudness'], 1e-12)) + 45) / 20, 0, 1)
    tonal = np.divide(old, old_gate, out=np.zeros_like(old), where=old_gate > 1e-9)
    harmonic = np.divide(c['salience'], old[:, None], out=np.zeros_like(c['salience']), where=old[:, None] > 1e-9)
    width = max(2, round(sample_rate * .020))
    starts = np.minimum(np.rint(c['frame_times_s'] * sample_rate).astype(int), len(x)-1)
    padded = np.pad(x, (width // 2, width))
    energy = np.r_[0., np.cumsum(padded * padded)]
    rms = np.sqrt(np.maximum((energy[starts + width] - energy[starts]) / width, 0))
    peak = max(float(np.max(rms)), 1e-12)
    activity = np.clip((20 * np.log10(np.maximum(rms / peak, 1e-12)) + 45) / 20, 0, 1) * tonal
    if peak < 1e-7:
        activity[:] = 0
    c.update(frontend_id='harmonic-salience-v2-short-energy', activity=activity.astype(np.float32),
             salience=np.clip(harmonic * activity[:, None], 0, 1).astype(np.float32),
             loudness=np.clip(rms / peak, 0, 1).astype(np.float32))
    c['settings']['energy_window_s'] = .020
    return c
