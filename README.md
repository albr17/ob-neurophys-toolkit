# ob-neurophys-toolkit

olfactory-bulb electrophysiology analyses.

## Modules

- `lfp.py`: MNE resampling, Welch PSD, spectrogram/band power, PAC, trough detection, kCSD helper
- `spikes.py`: ISI, autocorrelograms, waveform extraction, firing-rate classification, PSTHs, Gaussian peak fitting, gamma-organization classification
- `coupling.py`: spike-HFO Poisson GLM, surrogate testing, Elephant spike-field coherence wrapper

License: This repository is publicly available for transparency and academic reference.

## Notes

Defaults reproduce the manuscript analysis scripts where the exact settings were available. Uncertain or unavailable details are explicitly flagged in `METHODS_MAPPING.md`.


