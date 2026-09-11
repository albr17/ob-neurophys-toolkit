import numpy as np

from ob_neurophys_toolkit.lfp import detect_troughs

def test_detect_troughs_runs():
    fs = 1000
    t = np.arange(0, 1, 1 / fs)
    x = np.sin(2 * np.pi * 50 * t)[None, :]
    peaks, props = detect_troughs(
        x,
        channel_index=0,
        sfreq=fs,
        threshold_sd=0.5,
        min_distance_ms=1.0,
    )
    assert len(peaks) > 0
