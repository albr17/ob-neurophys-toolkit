from __future__ import annotations

import numpy as np
import statsmodels.api as sm

def fit_poisson_glm(spike_counts, hfo_power):
    """Fit Poisson GLM: spike count ~ HFO power."""
    y = np.asarray(spike_counts, dtype=float)
    x = np.asarray(hfo_power, dtype=float)
    if y.shape != x.shape:
        raise ValueError("spike_counts and hfo_power must have identical shape")
    X = sm.add_constant(x)
    return sm.GLM(y, X, family=sm.families.Poisson()).fit()

def surrogate_beta_test(
    observed_spike_counts,
    hfo_power,
    n_surrogates: int = 1000,
    random_seed: int | None = 123,
):
    """Empirical two-sided surrogate test for the HFO-power GLM beta.

    Surrogates preserve total spike count and recording duration while randomly
    redistributing spikes across the existing time bins.
    """
    counts = np.asarray(observed_spike_counts, dtype=int)
    power = np.asarray(hfo_power, dtype=float)

    if counts.shape != power.shape:
        raise ValueError("observed_spike_counts and hfo_power must have identical shape")
    if np.any(counts < 0):
        raise ValueError("spike counts must be non-negative")

    obs_fit = fit_poisson_glm(counts, power)
    observed_beta = float(obs_fit.params[1])

    rng = np.random.default_rng(random_seed)
    total_spikes = int(counts.sum())
    n_bins = len(counts)

    surrogate_betas = np.empty(n_surrogates, dtype=float)
    if total_spikes == 0:
        surrogate_betas[:] = np.nan
        return {
            "observed_beta": observed_beta,
            "surrogate_betas": surrogate_betas,
            "empirical_p": np.nan,
        }

    for i in range(n_surrogates):
        draws = rng.integers(0, n_bins, size=total_spikes)
        surr_counts = np.bincount(draws, minlength=n_bins)
        surrogate_betas[i] = float(fit_poisson_glm(surr_counts, power).params[1])

    extreme = np.sum(np.abs(surrogate_betas) >= abs(observed_beta))
    empirical_p = (extreme + 1) / (n_surrogates + 1)

    return {
        "observed_beta": observed_beta,
        "surrogate_betas": surrogate_betas,
        "empirical_p": float(empirical_p),
    }

def spike_field_coherence_elephant(
    lfp_signal,
    spike_train,
    sampling_rate,
    **kwargs,
):
    """Generic Elephant spike-field coherence wrapper.

    IMPORTANT: the exact original manuscript SFC script was not available
    when this toolkit was assembled. Verify this wrapper against the original
    analysis before using it for exact reproduction.
    """
    try:
        import quantities as pq
        from neo import AnalogSignal
        from elephant.spectral import spike_field_coherence
    except ImportError as exc:
        raise ImportError(
            "Install optional Elephant dependencies with: pip install -e '.[elephant]'"
        ) from exc

    signal = np.asarray(lfp_signal, dtype=float)
    analog = AnalogSignal(signal, units=pq.V, sampling_rate=sampling_rate * pq.Hz)
    return spike_field_coherence(analog, spike_train, **kwargs)
