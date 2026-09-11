from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture

def compute_isi(
    spike_times,
    fs: float,
    max_isi_ms: float = 100.0,
    bin_ms: float = 1.0,
):
    """Compute consecutive-spike ISI histogram."""
    spike_times = np.sort(np.asarray(spike_times, dtype=np.int64))
    if spike_times.size < 2:
        return np.array([]), np.array([])
    isi_ms = np.diff(spike_times) / fs * 1000.0
    edges = np.arange(0, max_isi_ms + bin_ms, bin_ms)
    counts, edges = np.histogram(isi_ms, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts

def compute_autocorrelogram(
    spike_times,
    fs: float,
    window_ms: float = 100.0,
    bin_ms: float = 1.0,
    normalize: bool = False,
):
    """Compute symmetric autocorrelogram and zero the zero-lag bin."""
    spike_times = np.sort(np.asarray(spike_times, dtype=np.int64))
    if spike_times.size < 2:
        return np.array([]), np.array([])

    half_window_samples = int(round(window_ms * fs / 1000.0))
    bin_samples = max(1, int(round(bin_ms * fs / 1000.0)))
    edges = np.arange(
        -half_window_samples,
        half_window_samples + bin_samples,
        bin_samples,
    )
    counts = np.zeros(len(edges) - 1, dtype=np.int64)

    for i, t0 in enumerate(spike_times):
        j = i + 1
        while j < spike_times.size:
            dt = spike_times[j] - t0
            if dt > half_window_samples:
                break
            for lag in (dt, -dt):
                idx = np.searchsorted(edges, lag, side="right") - 1
                if 0 <= idx < counts.size:
                    counts[idx] += 1
            j += 1

    centers_samples = 0.5 * (edges[:-1] + edges[1:])
    zero_bin = np.argmin(np.abs(centers_samples))
    counts[zero_bin] = 0

    if normalize and counts.max() > 0:
        counts = counts.astype(float) / counts.max()

    return centers_samples / fs * 1000.0, counts

def _robust_upper_thresh(vals: np.ndarray, mult: float = 5.0) -> float:
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    if mad < 1e-12:
        mad = np.std(vals) * 0.6745 + 1e-12
    return med + mult * 1.4826 * mad

def extract_spike_waveforms(
    x,
    spike_times,
    fs: float,
    window_s: float = 0.004,
    baseline_subtract: bool = True,
    reject_artifacts: bool = True,
    ptp_thresh_mult: float = 5.0,
    abs_amp_thresh_mult: float = 5.0,
    deviation_thresh_mult: float = 5.0,
    pre_ptp_thresh_mult: float = 4.0,
    pre_abs_thresh_mult: float = 4.0,
):
    """Extract ±window waveforms and optionally reject artifacts robustly."""
    x = np.asarray(x, dtype=float)
    spike_times = np.asarray(spike_times, dtype=int)
    window_samples = int(round(window_s * fs))
    epochs = []

    for st in spike_times:
        if st - window_samples < 0 or st + window_samples >= x.size:
            continue
        ep = x[st - window_samples: st + window_samples + 1].copy()
        if baseline_subtract:
            pre1 = max(1, int(0.75 * window_samples))
            ep -= np.median(ep[:pre1])
        epochs.append(ep)

    if not epochs:
        return np.array([]), np.empty((0, 2 * window_samples + 1)), np.array([], bool)

    epochs = np.stack(epochs)
    keep = np.ones(len(epochs), dtype=bool)

    if reject_artifacts and len(epochs) >= 10:
        ptp_vals = np.ptp(epochs, axis=1)
        absmax_vals = np.max(np.abs(epochs), axis=1)
        keep &= ptp_vals <= _robust_upper_thresh(ptp_vals, ptp_thresh_mult)
        keep &= absmax_vals <= _robust_upper_thresh(absmax_vals, abs_amp_thresh_mult)

        pre_region = epochs[:, :window_samples]
        pre_ptp = np.ptp(pre_region, axis=1)
        pre_abs = np.max(np.abs(pre_region), axis=1)
        keep &= pre_ptp <= _robust_upper_thresh(pre_ptp, pre_ptp_thresh_mult)
        keep &= pre_abs <= _robust_upper_thresh(pre_abs, pre_abs_thresh_mult)

        if np.sum(keep) >= 10:
            ref = np.median(epochs[keep], axis=0)
            dev = np.sqrt(np.mean((epochs - ref) ** 2, axis=1))
            keep &= dev <= _robust_upper_thresh(dev, deviation_thresh_mult)

    kept = epochs[keep]
    if len(kept) == 0:
        return np.array([]), kept, keep

    return np.mean(kept, axis=0), kept, keep

def classify_firing_response(
    post_bin_values,
    baseline_mean: float,
    baseline_sd: float,
    min_bins: int = 16,
):
    """Classify firing response using the manuscript rule."""
    vals = np.asarray(post_bin_values, dtype=float)
    upper = baseline_mean + baseline_sd
    lower = baseline_mean - baseline_sd
    n_up = int(np.sum(vals > upper))
    n_down = int(np.sum(vals < lower))
    if n_up >= min_bins:
        return "increased"
    if n_down >= min_bins:
        return "decreased"
    return "unchanged"

def build_event_aligned_intervals(
    spike_times_s,
    event_times_s,
    window_s: float = 0.025,
):
    """Collect spike-event intervals within ±window."""
    spikes = np.asarray(spike_times_s, dtype=float)
    events = np.asarray(event_times_s, dtype=float)
    out = []
    for ev in events:
        dt = spikes - ev
        out.extend(dt[(dt >= -window_s) & (dt <= window_s)])
    return np.asarray(out)

def compute_psth(
    intervals_s,
    window_s: float = 0.025,
    bin_width_s: float = 0.001,
    density: bool = True,
):
    """Compute a PSTH over ±window."""
    intervals = np.asarray(intervals_s, dtype=float)
    edges = np.arange(-window_s, window_s + bin_width_s, bin_width_s)
    hist, edges = np.histogram(intervals, bins=edges, density=density)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist, edges

@dataclass
class GaussianPeakFit:
    detected_peak_s: float
    mean_s: float
    sd_s: float

def fit_local_gaussians_to_psth(
    intervals_s,
    bin_edges,
    peak_height: float = 0.01,
    fit_half_width_s: float = 0.003,
    random_state: int = 100,
):
    """Detect PSTH peaks and fit one Gaussian locally around each peak."""
    intervals = np.asarray(intervals_s, dtype=float)
    hist, edges = np.histogram(intervals, bins=bin_edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak_idx, _ = find_peaks(hist, height=peak_height)
    peak_positions = centers[peak_idx]

    fits = []
    for pos in peak_positions:
        mask = (intervals >= pos - fit_half_width_s) & (intervals <= pos + fit_half_width_s)
        if np.sum(mask) < 2:
            continue
        gmm = GaussianMixture(
            n_components=1,
            means_init=np.array([[pos]]),
            random_state=random_state,
        )
        gmm.fit(intervals[mask].reshape(-1, 1))
        fits.append(
            GaussianPeakFit(
                detected_peak_s=float(pos),
                mean_s=float(gmm.means_.ravel()[0]),
                sd_s=float(np.sqrt(gmm.covariances_).ravel()[0]),
            )
        )
    return fits, peak_positions

def slope_sd(peak_positions_s) -> float:
    """SD of first differences between consecutive PSTH peak positions."""
    p = np.sort(np.asarray(peak_positions_s, dtype=float))
    if p.size < 3:
        return np.nan
    return float(np.std(np.diff(p)))

def classify_gamma_organization(
    slope_sd_value: float,
    threshold: float = 0.00133,
):
    """Manuscript-specific empirical gamma-organization classifier."""
    if np.isnan(slope_sd_value):
        return "unclassified"
    return "high-gamma-organized" if slope_sd_value > threshold else "non-gamma-organized"
