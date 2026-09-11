import numpy as np

from ob_neurophys_toolkit.spikes import (
    compute_isi,
    classify_firing_response,
    slope_sd,
    classify_gamma_organization,
)

def test_compute_isi():
    centers, counts = compute_isi([0, 1000, 2000], fs=1000, max_isi_ms=2000, bin_ms=1000)
    assert counts.sum() == 2

def test_classify_firing_response_increased():
    vals = np.ones(30) * 2.0
    assert classify_firing_response(vals, baseline_mean=0.0, baseline_sd=1.0, min_bins=16) == "increased"

def test_slope_sd_regular_peaks():
    peaks = np.array([0.0, 0.01, 0.02, 0.03])
    assert np.isclose(slope_sd(peaks), 0.0)

def test_gamma_classifier():
    assert classify_gamma_organization(0.002) == "high-gamma-organized"
    assert classify_gamma_organization(0.001) == "non-gamma-organized"
