"""Original NumPy harmonic-salience research front end; no UART/device coupling.

Scores are uncalibrated relative evidence, not posterior probabilities. Frame time
is the start of an output time cell after compensating the centered FFT window.
This is an explicit melody baseline, not a claim of full polyphonic transcription.
"""
import hashlib
import numpy as np

VERSION = 'harmonic-salience-v1'


def extract_candidates(pcm, sample_rate=16000, midi_min=48, midi_max=96,
                       hop_s=.01, frame_length=2048, n_fft=4096, cancel_check=None, progress=None):
    x = np.asarray(pcm, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError('finite nonempty mono PCM required')
    if (not isinstance(sample_rate,int) or not 8000 <= sample_rate <= 96000 or
        not np.isfinite(hop_s) or not .005 <= hop_s <= .1 or
        not isinstance(midi_min,int) or not isinstance(midi_max,int) or not 0 <= midi_min <= midi_max <= 127 or
        not isinstance(frame_length,int) or not isinstance(n_fft,int) or frame_length>8192 or n_fft>16384 or len(x)>sample_rate*600):
        raise ValueError('invalid sample rate or pitch range')
    hop = max(1, round(sample_rate * hop_s))
    pitches = np.arange(midi_min, midi_max + 1)
    hz = 440. * 2. ** ((pitches - 69.) / 12.)
    if hz[-1] >= sample_rate / 2 or frame_length < 32 or n_fft < frame_length:
        raise ValueError('FFT/range invalid')
    n_frames = max(1, int(np.ceil(len(x) / hop)))
    padded = np.pad(x, (frame_length // 2, frame_length))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop][:n_frames]
    window = np.hanning(frame_length)
    harm = np.arange(1, 9)
    positions = hz[:, None] * harm[None, :] * n_fft / sample_rate
    lo = np.floor(positions).astype(int)
    frac = positions - lo
    valid = lo + 1 <= n_fft // 2
    lo = np.clip(lo, 0, n_fft // 2 - 1)
    weights = 1. / harm
    raw = np.empty((n_frames, len(pitches)))
    rms = np.empty(n_frames)
    flatness = np.empty(n_frames)
    flux = np.empty(n_frames)
    previous = np.zeros(n_fft // 2 + 1)
    band = (np.fft.rfftfreq(n_fft, 1 / sample_rate) >= hz[0]) & (np.fft.rfftfreq(n_fft, 1 / sample_rate) <= min(6000, sample_rate / 2))
    for begin in range(0, n_frames, 256):
        if cancel_check and cancel_check():raise InterruptedError('曲谱分析已取消')
        if progress:progress(begin/n_frames)
        f = frames[begin:begin + 256]
        mag = np.abs(np.fft.rfft(f * window, n=n_fft, axis=1))
        values = mag[:, lo] * (1 - frac) + mag[:, lo + 1] * frac
        raw[begin:begin + len(f)] = np.sum(values * valid * weights, axis=2)
        rms[begin:begin + len(f)] = np.sqrt(np.mean(f * f, axis=1))
        m = mag[:, band] + 1e-12
        flatness[begin:begin + len(f)] = np.exp(np.mean(np.log(m), axis=1)) / np.mean(m, axis=1)
        flux[begin:begin + len(f)] = np.sum(np.maximum(mag - np.vstack([previous, mag[:-1]]), 0), axis=1)
        previous = mag[-1].copy()
    tonal = np.clip((.70 - flatness) / .60, 0, 1)
    peak = max(float(np.max(rms)), 1e-12)
    db = 20 * np.log10(np.maximum(rms / peak, 1e-12))
    activity = np.clip((db + 45) / 20, 0, 1) * tonal
    if peak < 1e-7:
        activity[:] = 0
    floor = np.quantile(raw, .35, axis=1, keepdims=True)
    raw = np.maximum(raw - floor, 0)
    salience = raw / np.maximum(raw.max(axis=1, keepdims=True), 1e-12)
    salience *= activity[:, None]
    onset = np.clip(flux / max(float(np.quantile(flux, .95)), 1e-12), 0, 1) * activity
    return dict(schema='discrete-note-candidates-v1', frontend_id=VERSION,
                duration_s=len(x) / sample_rate, frame_times_s=np.arange(n_frames) * hop / sample_rate,
                frame_time_semantics='left edge of output cell; centered analysis window compensated',
                hop_s=hop / sample_rate, midi_pitches=pitches, pitch_hz=hz,
                salience=salience.astype(np.float32), onset_strength=onset.astype(np.float32),
                activity=activity.astype(np.float32), loudness=np.clip(rms / peak, 0, 1).astype(np.float32),
                confidence_semantics='uncalibrated harmonic evidence; not a probability',
                source_PCM_sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
                settings=dict(midi_min=midi_min, midi_max=midi_max, frame_length=frame_length, n_fft=n_fft,
                              harmonic_weights=weights.tolist(), batch_frames=256))


def decode_melody(candidates, minimum_note_s=.06, cancel_check=None, progress=None):
    """Device-agnostic one-voice baseline; retain lattice for response optimizer."""
    s = np.asarray(candidates['salience'], dtype=float)
    a = np.asarray(candidates['activity'], dtype=float)
    onset = np.asarray(candidates['onset_strength'], dtype=float)
    midi = np.asarray(candidates['midi_pitches'])
    hop = candidates['hop_s']; count, pitches = s.shape
    observations = np.column_stack([.05 + 2 * (1 - a[:, None]) + 2 * (1 - s) * a[:, None], 2 * a])
    trans = .4 + .008 * np.minimum(np.abs(midi[:, None] - midi[None, :]), 24)
    np.fill_diagonal(trans, 0)
    full = np.full((pitches + 1, pitches + 1), .3)
    full[:pitches, :pitches] = trans; full[-1, -1] = 0
    cost = observations[0].copy(); back = np.zeros((count, pitches + 1), np.int16)
    for t in range(1, count):
        if t%256==0:
            if cancel_check and cancel_check():raise InterruptedError("主旋律解码已取消")
            if progress:progress(t/count)
        edge = full * (1 - .65 * onset[t])
        options = cost[:, None] + edge
        back[t] = np.argmin(options, axis=0)
        cost = observations[t] + np.min(options, axis=0)
    path = np.empty(count, int); path[-1] = np.argmin(cost)
    for t in range(count - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    events = []; begin = 0
    for end in range(1, count + 1):
        if end < count and path[end] == path[begin]:
            continue
        k = path[begin]
        if k < pitches and (end - begin) * hop >= minimum_note_s:
            events.append(dict(onset_s=begin * hop, duration_s=min(end * hop, candidates['duration_s']) - begin * hop,
                               midi=int(midi[k]), frequency_hz=float(candidates['pitch_hz'][k]),
                               confidence=float(np.mean(s[begin:end, k])),
                               velocity_norm=float(np.mean(candidates['loudness'][begin:end])), voice=0))
        begin = end
    return events, path
