from __future__ import annotations

from typing import Mapping
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, spectrogram

DEFAULT_BANDS = {
    "low_gamma": (30.0, 70.0),
    "high_gamma": (70.0, 100.0),
    "HFO": (130.0, 180.0),
}

def resample_raw_mne(raw, new_sfreq: float = 1000.0, n_jobs: int = -1):
    """Return a resampled copy of an MNE Raw object."""
    out = raw.copy()
    out.resample(new_sfreq, n_jobs=n_jobs)
    return out

def compute_welch_psd(raw, picks=None, n_fft: int = 2**15, **kwargs):
    """Compute Welch PSD using MNE's Raw.compute_psd."""
    return raw.compute_psd(method="welch", picks=picks, n_fft=n_fft, **kwargs)

def compute_representative_spectrogram(
    data: np.ndarray,
    fs: float,
    nperseg_s: float = 1.0,
):
    """Compute representative SciPy spectrogram used for visualization."""
    nperseg = int(round(nperseg_s * fs))
    return spectrogram(data, fs=fs, nperseg=nperseg)

def extract_band_power(
    data: np.ndarray,
    fs: float,
    window_sec: float = 30.0,
    fft_size: int = 32768,
    bands: Mapping[str, tuple[float, float]] | None = None,
    reducer: str = "median",
):
    """Extract band power from a Hann-windowed spectrogram."""
    bands = DEFAULT_BANDS if bands is None else bands
    f, t, sxx = spectrogram(
        data,
        fs=fs,
        nperseg=int(round(window_sec * fs)),
        nfft=fft_size,
        window="hann",
        scaling="spectrum",
    )
    out = {}
    for name, (low, high) in bands.items():
        mask = (f >= low) & (f <= high)
        if not np.any(mask):
            out[name] = np.nan
            continue
        vals = sxx[mask]
        if reducer == "median":
            out[name] = float(np.median(vals))
        elif reducer == "mean":
            out[name] = float(np.mean(vals))
        else:
            raise ValueError("reducer must be 'median' or 'mean'")
    return out

def bandpass_filter(
    data: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 1,
    axis: int = -1,
):
    """Zero-phase Butterworth band-pass filter."""
    sos = butter(
        order,
        [low_hz / (fs / 2.0), high_hz / (fs / 2.0)],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, data, axis=axis)

def detect_troughs(
    data: np.ndarray,
    channel_index: int,
    sfreq: float,
    threshold_sd: float = 2.0,
    min_distance_ms: float = 1.0,
):
    """Detect troughs as local maxima in the inverted signal."""
    channel_data = np.asarray(data)[channel_index]
    neg_data = -channel_data
    threshold = threshold_sd * np.std(neg_data)
    distance = max(1, int(round(min_distance_ms * sfreq / 1000.0)))
    return find_peaks(neg_data, distance=distance, height=threshold)

def compute_pac_comodulogram(
    signal: np.ndarray,
    fs: float = 1000.0,
    phase_range: tuple[float, float] = (1.0, 10.0),
    amplitude_range: tuple[float, float] = (10.0, 200.0),
    n_phase: int = 50,
    n_amplitude: int = 100,
    low_fq_width: float = 2.0,
    method: str = "tort",
):
    """Compute PAC using pactools.Comodulogram."""
    try:
        from pactools import Comodulogram
    except ImportError as exc:
        raise ImportError(
            "Install optional PAC dependencies with: pip install -e '.[pactools]'"
        ) from exc

    est = Comodulogram(
        fs=fs,
        low_fq_range=np.linspace(*phase_range, n_phase),
        high_fq_range=np.linspace(*amplitude_range, n_amplitude),
        low_fq_width=low_fq_width,
        method=method,
        progress_bar=False,
    )
    est.fit(np.asarray(signal))
    return est

def make_kcsd2d(ele_pos: np.ndarray, pots: np.ndarray, **kwargs):
    """Construct a KCSD2D estimator with explicit experiment-specific settings."""
    try:
        from kcsd import KCSD2D
    except ImportError as exc:
        raise ImportError(
            "Install optional kCSD dependencies with: pip install -e '.[kcsd]'"
        ) from exc
    return KCSD2D(ele_pos, pots, **kwargs)
