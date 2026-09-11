import numpy as np

from ob_neurophys_toolkit.lfp import bandpass_filter, detect_troughs
from ob_neurophys_toolkit.spikes import (
    build_event_aligned_intervals,
    compute_psth,
    fit_local_gaussians_to_psth,
    slope_sd,
    classify_gamma_organization,
)

fs = 1000
rng = np.random.default_rng(42)

t = np.arange(0, 10, 1 / fs)
lfp = np.sin(2 * np.pi * 85 * t) + 0.2 * rng.normal(size=t.size)
lfp_2d = lfp[None, :]

filtered = bandpass_filter(lfp_2d, fs, 30, 200, order=1)
trough_idx, _ = detect_troughs(
    filtered,
    channel_index=0,
    sfreq=fs,
    threshold_sd=2.0,
    min_distance_ms=1.0,
)

spike_times_s = np.sort(rng.uniform(0, 10, 1000))
event_times_s = trough_idx / fs
intervals = build_event_aligned_intervals(spike_times_s, event_times_s, window_s=0.025)
centers, hist, edges = compute_psth(intervals, window_s=0.025, bin_width_s=0.001)
fits, peaks = fit_local_gaussians_to_psth(intervals, edges)
metric = slope_sd(peaks)
label = classify_gamma_organization(metric)

print("Detected troughs:", len(trough_idx))
print("PSTH peaks:", len(peaks))
print("Slope SD:", metric)
print("Classification:", label)
