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

### 2.3 Test suite

The original package had 2 tests. v0.2 ships with 62 tests covering:

- Input validation (wrong type, NaN, string arrays)
- Mathematical invariants: linearity, voice synthesis, window normalization (DC = 1), pure-tone constant ridge
- Hyperbolic window properties: symmetric tapers → real window; asymmetric tapers → complex window; impulse asymmetric decay
- Parametrized across taper values, curvature, signal frequencies, and sampling rates
- Regression guards for all four fixed bugs

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

# Create a signal (e.g. voltage with a 0.5 s sag)
t = np.linspace(0, 10, 1000)
V = 220 * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)
V_sag = np.where((t >= 2) & (t <= 2.5), 0.5 * V, V)

# Run the transform
hs = HSTransform(
    forwardtaper=0.2,   # decay rate for t < τ (anti-causal)
    backwardtaper=0.1,  # decay rate for t > τ (causal)
    curvature=312.5,    # controls onset of hyperbolic asymmetry
)
S = hs.fit_transform(t, V_sag)   # shape: (n_freq, n_time), complex

# S[k, :] = time series of the k-th frequency component
# np.abs(S[50, :]) = instantaneous amplitude at 50 Hz
```

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `forwardtaper` | `0.2` | Window decay rate looking backward from τ (anti-causal side). Larger = slower decay = wider window. |
| `backwardtaper` | `0.1` | Window decay rate looking forward from τ (causal side). |
| `curvature` | `312.5` | Activates the hyperbolic asymmetry for `\|t\| ≳ √curvature`. For sample-indexed time (`t = np.arange(n)`), the default is appropriate. For normalised time (`t ∈ [0,1]`), use `curvature ≈ 0.25`. |

When `forwardtaper == backwardtaper`, the window is a standard symmetric Gaussian and the result is real-valued.

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

Four voltage disturbance types (Sag, Swell, Spike, Notch) on a 50 Hz, 220 V signal:

![Voltage Disturbance](https://raw.githubusercontent.com/nvlinhvn/HSTransform/main/img/power_quality_disturbance.png)

**Why HST wins here:**

- **Spike (column 3)**: HST shows a sharp bright vertical stripe spanning _all_ frequency bins simultaneously. A voltage spike is broadband by physics — energy appears at every frequency at the instant of the impulse. HST captures this correctly because it evaluates every frequency row in parallel. Morlet shows only a faint blob at the 50 Hz pseudo-scale and misses the broadband signature entirely.

- **Notch (column 4)**: HST shows a clean dark gap in all rows during the zero-voltage window — the "silence" is visible across the full spectrum. Morlet shows a slight dimming of the single 50 Hz blob; the silence is not interpretable without knowing it represents a complete dropout.

- **Sag / Swell (columns 1–2)**: HST shows a uniform amplitude shift across all bins during the disturbance (darker/brighter stripes). Morlet shows a comparable blob size change but with no frequency breakdown.

- **Frequency axis**: HST row k = exact k Hz. Morlet's y-axis is a wavelet scale — there is no direct mapping to Hz without additional calibration.

### 6.2 Power quality disturbances — real/imaginary trajectories

The HST output is complex. Plotting `Re(S[k, t])` vs `Im(S[k, t])` traces a Lissajous-like trajectory in the complex plane that is unique per disturbance type:

![Real-Imag Trajectory](https://raw.githubusercontent.com/nvlinhvn/HSTransform/main/img/power_quality_disturbance_trajectory.png)

**What to read in this plot:**

- **5 Hz row**: A pure 50 Hz sine traces a clean circle at every sub-harmonic frequency. Sag compresses the circle radius (amplitude drops). Swell expands it. Spike collapses it to a flat ellipse (the transient energy is concentrated at the spike instant, not spread around the 5 Hz cycle). Notch shows a circle with an abrupt angular discontinuity at the zero-voltage event.

- **10 Hz row**: Figure-8 Lissajous patterns differentiate Sag/Swell (scale change only) from Spike/Notch (shape change) — all four are geometrically distinct.

- **Morlet has no equivalent**: Morlet CWT output is real-valued. It cannot produce a complex trajectory, so this entire class of features — phasor geometry, shape fingerprinting per frequency — is unavailable with Morlet.

### 6.3 Power system faults — spectrogram comparison

Four 3-phase fault scenarios (Healthy, SLGF, DLGF, 3-Phase) with additive odd harmonics and white noise. Asymmetric HST (`λf=0.3`, `λb=0.02`) vs symmetric HST vs Morlet, log-power scale, 0–300 Hz:

![Fault Spectrogram](https://raw.githubusercontent.com/nvlinhvn/HSTransform/main/img/fault_current.png)

**What this plot shows — and why HST wins:**

- **Pre-fault (t < 0.4 s)**: All three methods show the 50 Hz fundamental. The signal is clean and both transforms agree.

- **At fault onset (cyan dashed line)**: The fault current collapses. The HST rows show a dramatic, instantaneous change across the entire frequency spectrum — the bright 50 Hz band drops and a high-frequency transient burst appears (at 150 Hz for SLGF, 200 Hz for DLGF, 250 Hz for 3-Phase). The asymmetric window (`λb=0.02`) sharpens the onset: energy accumulates right at the fault line with no pre-ringing.

- **Post-fault**: The Morlet row goes **entirely black**. Morlet is tracking a single scale (the one corresponding to the 50 Hz fundamental). When the fault current collapses, that band disappears — and Morlet has no other scales to fall back on. It is blind to the high-frequency transient burst that characterises the fault type. The HST rows remain informative: they show the lower post-fault amplitude with the characteristic HF band still visible above the 50 Hz baseline.

- **Fault type discrimination**: Asymmetric HST distinguishes SLGF, DLGF, and 3-Phase by the frequency of the transient band (150 / 200 / 250 Hz respectively). Morlet cannot discriminate — all three fault columns are equally black post-fault.

**The core difference**: Morlet operates at one scale at a time. HST evaluates _all_ frequency rows simultaneously in a single pass. When the dominant frequency disappears (fault collapse), HST still captures the rest of the spectrum; Morlet goes dark.

### 6.4 Power system faults — real/imaginary trajectories

Asymmetric HST complex trajectories at 50, 100, 150, and 200 Hz for each fault type:

![Fault Trajectory](https://raw.githubusercontent.com/nvlinhvn/HSTransform/main/img/fault_trajectory.png)

**Why this plot is the strongest argument for HST over Morlet:**

- **50 Hz row**: Healthy signal traces a clean circle (pure phasor rotation at the fundamental). SLGF (single line-to-ground) shows the same circle with a sharp spike-out at t=0.4 s when Phase A collapses — the circle is interrupted. DLGF and 3-Phase produce starburst patterns because multiple collapsed phases create competing phasors that interfere at the fundamental frequency. All four fault types are geometrically distinct with no overlap.

- **150 Hz row**: SLGF shows a tight cluster with a small hexagonal trace (the 150 Hz burst unique to this fault type). DLGF and 3-Phase show broader scatter because their transient bursts are at 200–250 Hz, not 150 Hz — the 150 Hz row stays near zero for them. This frequency specificity means HST can identify not just _that_ a fault occurred but _which type_ and _at which frequency_.

- **200 Hz row**: DLGF shows a distinct compact rectangular Lissajous (200 Hz transient present). 3-Phase shows a larger and differently-oriented rectangle (250 Hz transient leaks into the 200 Hz row). Healthy shows a large open square — purely the 4× harmonic aliasing of the 50 Hz fundamental, with no transient.

- **Morlet cannot produce any of this**: Morlet output is real-valued, has no exact frequency axis, and provides no complex phase information. The phasor geometry and per-frequency fingerprinting shown above are structurally impossible with a wavelet transform.

**When to use HST instead of Morlet:**

| Scenario | Best choice | Reason |
|---|---|---|
| Fault inception detection (one-sided step) | Asymmetric HST | No pre-ringing; sharp boundary at fault onset |
| Fault type classification | HST (any) | Complex trajectory shape is a discriminative feature; Morlet is real-only |
| Broadband impulse (spike, lightning) | HST | Absolute frequency axis; captures all bins simultaneously |
| Stationary periodic signal, no transients | Either | Morlet is simpler; HST adds no advantage |
| Frequency must be in Hz, not scale | HST | Morlet scale → Hz requires per-wavelet calibration |

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
