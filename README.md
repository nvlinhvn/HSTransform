# Hyperbolic S-Transform

[![PyPI version](https://img.shields.io/pypi/v/HSTransform?color=blue)](https://pypi.org/project/HSTransform/)
[![PyPI Downloads](https://static.pepy.tech/badge/hstransform)](https://pepy.tech/project/hstransform)
[![PyPI Downloads/month](https://static.pepy.tech/badge/hstransform/month)](https://pepy.tech/project/hstransform)
[![CI](https://github.com/nvlinhvn/HSTransform/actions/workflows/ci.yml/badge.svg)](https://github.com/nvlinhvn/HSTransform/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)](https://github.com/nvlinhvn/HSTransform/actions)
[![Tests](https://img.shields.io/badge/tests-62%20passed-brightgreen)](https://github.com/nvlinhvn/HSTransform/actions)
[![Python](https://img.shields.io/badge/python-%3E%3D3.12-blue)](https://pypi.org/project/HSTransform/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](https://mypy-lang.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-orange)](https://github.com/astral-sh/ruff)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FPES.2018.8423487-blue)](https://ieeexplore.ieee.org/document/8423487)

## 1. Introduction

HSTransform is a Python package for the Hyperbolic [S-transform](https://en.wikipedia.org/wiki/S_transform) [[1]](#1). The S-transform is a time-frequency representation that combines the best properties of the Short-time Fourier Transform and the Wavelet Transform — simultaneous frequency content and temporal localisation.

The hyperbolic variant introduces an asymmetric Gaussian window via a hyperbolic coordinate transformation. Unlike the standard S-transform (which has equal causal and anti-causal decay), the hyperbolic window decays at different rates before and after each time point, controlled by `forwardtaper` and `backwardtaper`. This makes it better suited for detecting one-sided transients such as voltage sags, fault inception, and impulsive disturbances in power systems.

This package implements the algorithm from the author's [published paper](https://ieeexplore.ieee.org/document/8423487).

---

## 2. What changed in v0.2.0

### 2.1 Algorithmic corrections

Three bugs were found and fixed in the core algorithm, two of which were MATLAB→Python porting errors.

**Bug 1 — Window was a scalar, not a vector (MATLAB port)**

The hyperbolic Gaussian window must be a 1D vector applied pointwise to the shifted FFT spectrum. The original code called `np.sum(g)` without an axis argument, which collapses the entire matrix to a scalar. In MATLAB, `sum(A)` on a 2D matrix sums along columns; in NumPy, `np.sum(A)` without `axis=` sums all elements. The fix: `np.sum(g, axis=0)`.

With a scalar window, all frequency rows were effectively weighted by the same constant — the time-frequency localisation of the transform was completely broken.

**Bug 2 — DC row alternating pattern (MATLAB port)**

The DC row of the S-transform (frequency = 0) should be the signal mean repeated at every time step. The original code used `1 & np.arange(1, n+1)`, which in MATLAB is logical AND (all ones) but in Python/NumPy is bitwise AND, producing `[1, 0, 1, 0, ...]`. Every other DC-row sample was zeroed. The fix: `np.ones(n)`.

**Bug 3 — Window was real-valued, losing asymmetry (algorithmic)**

Even after fixing Bug 1, the frequency-domain window was computed as a real-valued sum over all time samples without a complex phase term. The correct frequency-domain window is the Discrete Fourier Transform of the time-domain hyperbolic Gaussian — and because the window is asymmetric in time (`forwardtaper ≠ backwardtaper`), its DFT is genuinely complex. Without the phase term, all causal/anti-causal information was averaged out, making the output identical to a symmetric window regardless of the taper values.

The fix replaces the old sum with a proper DFT:
```python
g = (k / sqrt(2π)) * exp(-x(t)² * k² / 2)   # time-domain hyperbolic Gaussian
G = fft(g) * dt                                # frequency-domain window (complex)
```

**Bug 4 — FFT slice off-by-one (MATLAB port)**

The shifted FFT spectrum was sliced as `h[k+1:k+n+1]` instead of `h[k:k+n]`. In MATLAB, `h(k+1:k+n)` with 1-based indexing translates to Python's `h[k:k+n]`, not `h[k+1:k+n+1]`. The off-by-one shifted every frequency row by one bin.

### 2.2 Modern packaging

| Before (v0.1) | After (v0.2) |
|---|---|
| `setup.py` + `requirements.txt` | `pyproject.toml` (PEP 517, hatchling) |
| `pandas` as hard runtime dependency | Removed — only `numpy` and `scipy` required |
| `pytest` listed as runtime dependency | Moved to `[dev]` optional group |
| No type annotations | Full `mypy --strict` type hints |
| Python 3.10 | Python ≥ 3.12 |

### 2.3 Algorithmic corrections in v0.2.1

Two additional bugs were found and fixed in the core window construction:

**Bug 5 — Frequency scaling used bin index instead of physical frequency**

The Gaussian exponent used `k` (DFT bin index) where physical frequency `f_k = k / (N·Δt)` is required. The window width was incorrect whenever the record duration `N·Δt ≠ 1 s`, and the transform was not invariant to changes of time unit. The fix: replace `k_f = float(k)` with `f_k = float(k) / (n * dt)`.

**Bug 6 — Hyperbola centred by vertical subtraction instead of horizontal translation**

The code subtracted `x(0) = (λf−λb)√c / (2λfλb)` from the coordinate (vertical shift) instead of translating the hyperbola's argument by `ζ = (λf−λb)√c / (2√(λfλb))` (horizontal shift). Both produce `X(0) = 0`, but they are geometrically different curves with different slopes and asymptotic behaviour. This caused incorrect window shapes for all asymmetric (`λf ≠ λb`) configurations. The reference construction is from Pinnegar & Mansinha (2003).

### 2.4 Robustness improvements in v0.2.2

- **Constructor validation**: `forwardtaper`, `backwardtaper`, and `curvature` must be finite and positive; invalid values raise `ValueError` immediately on construction.
- **Signal validation**: signal must be 1-D, ≥ 2 samples, finite (no NaN or ±inf).
- **Time array validation**: `time_values` must match signal length, be strictly ascending, and be uniformly spaced.
- **Frequency parameter validation**: `minf`, `fsamplingrate`, and `maxf` are checked for valid integer types and ranges before any computation.
- **Configurable `maxf`**: `fit_transform` now accepts a `maxf` keyword to limit the highest frequency bin returned.
- **`frequency_axis` helper**: new method returns the physical frequency in Hz for each row of the transform output, matching the same `minf`/`fsamplingrate`/`maxf` arguments.

### 2.5 Test suite

The original package had 2 tests. v0.2.2 ships with 90 tests covering:

- Input validation (wrong type, NaN, inf, non-1D, length mismatch, non-uniform time)
- Constructor validation (zero/negative/nan tapers and curvature)
- Mathematical invariants: linearity, voice synthesis, window normalization (DC = 1), pure-tone constant ridge
- Hyperbolic window properties: symmetric tapers → real window; asymmetric tapers → complex window; impulse asymmetric decay
- Parametrized across taper values, curvature, signal frequencies, and sampling rates
- Regression guards for all six fixed bugs, including time-unit invariance and minf > 0 bin selection
- Robustness guards for all new parameter validation paths

Coverage: 97%.

---

## 3. Installation

```bash
pip install HSTransform
```

For development:

```bash
git clone https://github.com/nvlinhvn/HSTransform
cd HSTransform
pip install -e ".[dev]"
```

---

## 4. Usage

```python
import numpy as np
from hstransform import HSTransform

# Create a signal: 1000 samples at 1 kHz (bin k == k Hz exactly)
fs = 1000
n  = 1000
t  = np.arange(n) / fs          # endpoint-exclusive: dt = 0.001 s, n*dt = 1 s

V = 220 * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)
V_sag = np.where((t >= 0.2) & (t <= 0.25), 0.5 * V, V)

# Run the transform
hs = HSTransform(
    forwardtaper=0.2,   # decay rate toward future samples (t > τ)
    backwardtaper=0.1,  # decay rate toward past samples (t < τ)
    curvature=0.25,     # controls onset of hyperbolic asymmetry; use ≲ (record duration)²
)
S = hs.fit_transform(t, V_sag)   # shape: (n_freq, n_time), complex

# S[k, :] = time series of frequency bin k (= k Hz when n*dt = 1 s)
# np.abs(S[50, :]) = instantaneous amplitude envelope at 50 Hz
```

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `forwardtaper` | `0.2` | Controls window decay toward future samples (`t > τ`, negative lag). Larger = slower decay = wider tail toward the future. |
| `backwardtaper` | `0.1` | Controls window decay toward past samples (`t < τ`, positive lag). Smaller = narrower tail into the past = sharper event-onset detection. |
| `curvature` | `312.5` | Controls when the hyperbolic asymmetry activates. The asymmetric term in the coordinate transform is `(λf−λb)√(t²+curvature)`, which only differs meaningfully from a constant when `t² ≳ curvature`. **Rule of thumb: set `curvature ≲ (record duration)²`.** For `t = np.arange(n)` (sample-indexed), default 312.5 is appropriate. For `t ∈ [0,1]` (normalised), use `curvature ≈ 0.01–0.25`; otherwise the asymmetry is invisible. |

When `forwardtaper == backwardtaper`, the window is a symmetric Gaussian.

### `fit_transform` signature

```python
S = hs.fit_transform(
    time_values,        # 1D array of time samples
    input_signal,       # 1D array of signal values (same length)
    minf=0,             # minimum frequency row index
    fsamplingrate=1,    # frequency step (1 = all rows, 2 = every other row, …)
)
# Returns: complex ndarray of shape ((maxf - minf) // fsamplingrate + 1, n)
# maxf = min(900, n // 2)
```

---

## 5. Run tests

```bash
pytest tests/ --cov=hstransform
```

---

## 6. Examples

### 6.1 Power quality disturbances — spectrogram comparison

Four voltage disturbance types (Sag, Swell, Spike, Notch) on a 50 Hz, 220 V signal, compared between HST and Morlet Wavelet:

![Voltage Disturbance](img/power_quality_disturbance.png)

HST evaluates all frequency rows simultaneously, so transient events like spikes and notches appear as clear vertical features across the full spectrum. Morlet concentrates energy around one scale (the dominant frequency), which limits its ability to characterize broadband or impulsive disturbances. The HST y-axis is in exact Hz; Morlet's is a wavelet scale with no direct Hz mapping.

### 6.2 Power quality disturbances — real/imaginary trajectories

The HST output is complex-valued. Plotting `Re(S[k, t])` vs `Im(S[k, t])` at a given frequency traces a trajectory in the complex plane that is characteristic of the disturbance type:

![Real-Imag Trajectory](img/power_quality_disturbance_trajectory.png)

Each disturbance type produces a geometrically distinct shape — circles, ellipses, figure-8s, or patterns with abrupt discontinuities — depending on how the disturbance modifies the amplitude and phase at that frequency. Morlet output is real-valued and cannot produce this class of features.

### 6.3 Power system faults — spectrogram comparison

Four 3-phase fault scenarios (Healthy, SLGF, DLGF, 3-Phase) with additive harmonics and noise. Asymmetric HST (`λf=0.3`, `λb=0.02`) vs symmetric HST vs Morlet:

![Fault Spectrogram](img/fault_current.png)

HST captures the full frequency evolution through the fault — the collapse of the fundamental and the emergence of high-frequency transient content are both visible. Morlet tracks a single scale and goes dark when the dominant component disappears, leaving the post-fault transient undetected. The asymmetric window sharpens the fault onset by reducing pre-event smearing relative to a symmetric window.

### 6.4 Power system faults — real/imaginary trajectories

Asymmetric HST complex trajectories at 50, 100, 150, and 200 Hz for each fault type:

![Fault Trajectory](img/fault_trajectory.png)

Each fault type traces a distinct shape in the complex plane at each frequency — from clean circles in the healthy case to interrupted or starburst patterns under fault conditions. The shapes differ across fault types and across frequency rows, making them usable as discriminative features for fault classification. Morlet provides no equivalent because its output is real-valued and carries no phase information.

**When to use HST instead of Morlet:**

| Scenario | Best choice | Reason |
|---|---|---|
| Fault inception / one-sided transients | Asymmetric HST | Sharp onset detection; no pre-ringing |
| Fault or disturbance type classification | HST | Complex trajectory is a discriminative feature; Morlet is real-only |
| Broadband impulsive events (spike, lightning) | HST | Captures all frequency bins simultaneously |
| Frequency axis must be in Hz | HST | Morlet scale has no direct Hz mapping |
| Stationary periodic signals only | Either | Morlet is simpler; HST adds no advantage |

---

## 7. References

<a id="1">[1]</a>
Stockwell, R.G., Mansinha, L. & Lowe, R.P. (1996).
Localization of the complex spectrum: the S transform.
*IEEE Trans. Signal Process.*, 44(4), 998–1001. doi:10.1109/78.492555

---

## 8. Citation

If you use HSTransform in your research, please cite:

> Linh V Nguyen (2024). HSTransform (Version 0.2.1) [Computer software]. GitHub: github.com/nvlinhvn/HSTransform

---

## 9. Contact

Questions, issues, or suggestions: open a GitHub issue or email linhvietnguyen.ee@gmail.com
