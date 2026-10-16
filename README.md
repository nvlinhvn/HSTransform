# Hyperbolic S-Transform

[![PyPI version](https://img.shields.io/pypi/v/HSTransform?color=blue)](https://pypi.org/project/HSTransform/)
[![PyPI Downloads](https://static.pepy.tech/badge/hstransform)](https://pepy.tech/project/hstransform)
[![PyPI Downloads/month](https://static.pepy.tech/badge/hstransform/month)](https://pepy.tech/project/hstransform)
[![CI](https://github.com/nvlinhvn/HSTransform/actions/workflows/ci.yml/badge.svg)](https://github.com/nvlinhvn/HSTransform/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)](https://github.com/nvlinhvn/HSTransform/actions)
[![Tests](https://img.shields.io/badge/tests-116%20passed-brightgreen)](https://github.com/nvlinhvn/HSTransform/actions)
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

The original package had 2 tests. The current suite ships with 116 tests covering:

- Input validation (wrong type, NaN, inf, non-1D, length mismatch, non-uniform time, complex timestamps, float32 and large-origin time grids)
- Constructor validation (zero/negative/nan tapers and curvature)
- Mathematical invariants: linearity, voice synthesis, window normalization (DC = 1), pure-tone constant ridge
- Hyperbolic window properties: symmetric tapers → real window; asymmetric tapers → complex window; impulse asymmetric decay
- Parametrized across taper values, curvature, signal frequencies, and sampling rates
- Regression guards for all six fixed bugs, including time-unit invariance and minf > 0 bin selection
- Independent direct-sum oracle tests that verify production `fit_transform` against reference weights built without calling any internal method
- Robustness guards for all new parameter validation paths
- Timestamp validation: epoch grids (Unix timestamp origin ~1.7×10⁹), float32 grids, sinusoidal timing jitter, linear drift

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
# np.abs(S[50, :]) = localized complex spectral magnitude at 50 Hz
#   For a real cosine of amplitude A, the mean magnitude across time is A/2
#   (not A) because the DFT splits energy between +k and -k bins.
#   Individual sample magnitudes vary around A/2 due to negative-frequency
#   image leakage; the A/2 approximation holds when image leakage is negligible.
```

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `forwardtaper` | `0.2` | Controls window decay toward **future** samples (`t > τ`, negative lag). Smaller = narrower future-looking tail = less pre-event energy from an upcoming transient. For sharp onset detection with minimal pre-event bleed, **decrease `forwardtaper`**. |
| `backwardtaper` | `0.1` | Controls window decay toward **past** samples (`t < τ`, positive lag). Smaller = narrower past-looking tail = faster post-event decay. |
| `curvature` | `312.5` | Controls when the hyperbolic asymmetry activates. The asymmetric term in the coordinate transform is `(λf−λb)√(t²+curvature)`, which only differs meaningfully from a constant when `t² ≳ curvature`. **Rule of thumb: set `curvature ≲ (record duration)²`.** For `t = np.arange(n)` (sample-indexed), default 312.5 is appropriate. For `t ∈ [0,1]` (normalised), use `curvature ≈ 0.01–0.25`; otherwise the asymmetry is invisible. |

When `forwardtaper == backwardtaper`, the window is a symmetric Gaussian.

### `fit_transform` and `frequency_axis` signatures

```python
S = hs.fit_transform(
    time_values,        # 1D uniformly-spaced time array, same length as signal
    input_signal,       # 1D finite real or complex signal
    minf=0,             # minimum frequency bin index (default 0 = include DC)
    fsamplingrate=1,    # bin step (1 = every bin, 2 = every other, …)
    maxf=None,          # maximum bin (default: min(900, n // 2))
)
# Returns: complex ndarray of shape (len(bins), n)
# where bins = np.arange(minf, effective_maxf+1, fsamplingrate)

# Get physical Hz label for each row.
# Use mean spacing — matches what the transform uses internally.
dt = float(np.mean(np.diff(time_values)))
freqs = hs.frequency_axis(n, dt, minf=0, fsamplingrate=1, maxf=None)
# freqs[j] = bins[j] / (n * dt)  in Hz
```

### Time-axis convention and frequency labeling

Use `t = np.arange(n) / fs` (endpoint-exclusive) rather than `np.linspace(0, T, n)`.  
When the record duration `n * dt = 1 s` exactly, row `k` maps to exactly `k` Hz.  
For other durations, use `frequency_axis()` to get the correct Hz labels.

### Finite circular-window contract

The implementation normalizes a single-period window cropped to `n` samples and treats the time axis as circular (via `fftfreq`). The continuous Gaussian window has infinite support; this discrete version loses tail energy for broad windows (small frequency bin or large taper values). This is expected behaviour, not a bug: the window integrates to 1 over the cropped period so DC is preserved, but magnitude values for broad windows are not directly comparable to the continuous formula.

The finite approximation is accurate when two independent conditions hold simultaneously:
1. **Negligible tails**: the window must fit within the cropped period — satisfied at high frequency bins or small tapers.
2. **Adequate sampling**: the window must span enough samples to be well resolved — satisfied at low frequency bins or large tapers.

These conditions pull in opposite directions. In particular, very high frequency bins can have _worse_ continuous-transform accuracy due to undersampling (a narrow window resolved by only a fraction of a sample). Use `frequency_axis()` together with the rule-of-thumb `window_SD_samples ≈ γ · N / k` (where `γ` is the taper value) to assess whether both conditions are met for a given configuration.

### Phase convention

The carrier phase is relative to `t[0] = 0` (sample zero), not the physical start time.  
If your time axis starts at `t₀ ≠ 0` and you need phase relative to that physical origin,
multiply each row `k` by `exp(-2πi · f_k · t₀)` after the transform.

---

## 5. Run tests

```bash
pytest tests/ --cov=hstransform
```

---

## 6. Examples

### 6.1 Power quality disturbances

Four voltage disturbance types on a 50 Hz, 220 V signal:

| Type | Definition |
|---|---|
| **Sag** | 50% of nominal for 0.5 s |
| **Swell** | 150% of nominal for 0.8 s |
| **Spike** | 500% of nominal for 5 ms (sub-cycle, ≈ ¼ cycle at 50 Hz) |
| **Notch** | 0 V for 80 ms |

Shown across three sampling and duration configurations (HST vs Morlet Wavelet):

**Config A — standard (1 kHz / 2 s)**

![PQ Config A](notebooks/pq_comparison_config_a_notebook.png)

**Config B — long record (1 kHz / 10 s)**

![PQ Config B](notebooks/pq_comparison_config_b_user.png)

**Config C — sub-cycle events (10 kHz / 0.2 s, 200 Hz carrier)**

![PQ Config C](notebooks/pq_comparison_config_c_subcycle.png)

Config C shows the advantage of HST's exact linear Hz axis most clearly. At 10 kHz sampling, the 0.3 ms spike appears as a sharp broadband burst spanning the full frequency axis — every row is labeled in exact Hz. The Morlet CWT uses log-spaced pseudo-frequency scales, so the y-axis is not directly comparable and the sub-cycle event is less sharply resolved.

### 6.2 Power quality disturbances — real/imaginary trajectories

The HST output is complex-valued. Plotting `Re(S[k, t])` vs `Im(S[k, t])` at a given frequency traces a trajectory in the complex plane that is characteristic of the disturbance type:

![Real-Imag Trajectory Animated](notebooks/pq_trajectory_animated.gif)

*Static version: [power_quality_disturbance_trajectory.png](notebooks/power_quality_disturbance_trajectory.png)*

Each disturbance type produces a geometrically distinct shape at 50 Hz — nested circles (sag), larger circle (swell), brief deflection (spike: sub-cycle impulse kicks the orbit off-track momentarily), and a pinched orbit (notch: brief zero-crossing collapses the radius). The real `'morl'` Morlet wavelet returns real coefficients, so `Im(CWT) = 0` identically — no trajectory is possible. The complex Morlet (`'cmor'`) does support trajectories but was not evaluated here.

### 6.3 Power system faults — real/imaginary trajectories

Four 3-phase fault scenarios with synthetic phase-current disturbances. Each fault type injects a distinct HF transient at inception, with burst frequencies chosen to be non-harmonic (not multiples of 50 Hz):

| Fault type | HF burst | Signature |
|---|---|---|
| **Healthy** | None | All three phases balanced; zero-sequence ≈ 0 |
| **SLGF** (Phase A) | 120 Hz on Ia only | Ia >> Ib ≈ Ic; zero-sequence ≠ 0 |
| **DLGF** (Phase A-B) | 175 Hz on Ia and Ib | Ia ≈ Ib > Ic; zero-sequence ≠ 0 |
| **3-Phase** | 220 Hz on all phases (120°-displaced) | Ia ≈ Ib ≈ Ic >> pre-fault; zero-sequence ≈ 0 |

Asymmetric HST (`λf=0.3`, `λb=0.02`) complex trajectories at 50, 100, 150, and 200 Hz for each fault type:

![Fault Trajectory Animated](notebooks/fault_trajectory_animated.gif)

*Static version: [fault_trajectory.png](notebooks/fault_trajectory.png)*

Each fault type traces a distinct shape in the complex plane at each frequency — clean circles (healthy), collapsed or interrupted orbits (SLGF), paired distortions (DLGF), and starburst patterns (3-Phase). The shapes differ across both fault types and frequency rows, making them usable as discriminative features for classification. The asymmetric window (`λf=0.3`, `λb=0.02`) sharpens the post-fault onset edge; for minimal pre-event smearing, use a smaller `forwardtaper` (e.g. `λf=0.02, λb=0.3`).

### 6.4 Asymmetric window demo

The hyperbolic Gaussian window decays at different rates before and after each time point. With `forwardtaper ≠ backwardtaper`, the magnitude envelope is visibly asymmetric around the impulse:

![Asymmetric Window Demo](notebooks/asymmetric_window_demo.png)

Left: unit impulse input. Centre: magnitude `|S(τ, k)|` at a fixed frequency row — both windows peak at the impulse location (zero lag); the asymmetric window (orange) has unequal tails relative to the symmetric one (teal). Right: log-scale decay profile separating the backward (past-input, positive lag, after impulse) and forward (future-input, negative lag, before impulse) tails. Larger `forwardtaper` = slower decay toward future samples = wider forward tail.

---

## 7. References

<a id="1">[1]</a>
Stockwell, R.G., Mansinha, L. & Lowe, R.P. (1996).
Localization of the complex spectrum: the S transform.
*IEEE Trans. Signal Process.*, 44(4), 998–1001. doi:10.1109/78.492555

---

## 8. Citation

If you use HSTransform in your research, please cite:

> Linh V Nguyen (2024). HSTransform (Version 0.2.4) [Computer software]. GitHub: github.com/nvlinhvn/HSTransform

---

## 9. Contact

Questions, issues, or suggestions: open a GitHub issue or email linhvietnguyen.ee@gmail.com
