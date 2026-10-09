# Hyperbolic S-Transform

[![PyPI Downloads](https://static.pepy.tech/badge/hstransform)](https://pepy.tech/project/hstransform)
[![PyPI Downloads/month](https://static.pepy.tech/badge/hstransform/month)](https://pepy.tech/project/hstransform)
[![Python](https://img.shields.io/badge/python-%3E%3D3.12-blue)](https://pypi.org/project/HSTransform/)
[![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)](https://github.com/nvlinhvn/HSTransform/actions)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-orange)](https://github.com/astral-sh/ruff)

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

### Power quality disturbance

Compare HS-transform vs. Morlet Wavelet Transform:

![Voltage Disturbance](https://raw.githubusercontent.com/nvlinhvn/HSTransform/main/img/power_quality_disturbance.png)

The HS-transform detects transient disturbances (notch, spike) that are not clearly visible in the Morlet Wavelet transform output.

![Real-Imag Trajectory](https://raw.githubusercontent.com/nvlinhvn/HSTransform/main/img/power_quality_disturbance_trajectory.png)

Different voltage disturbance types produce distinct real-imaginary trajectories at different frequencies.

### Power system faults

![Different Faults](https://raw.githubusercontent.com/nvlinhvn/HSTransform/main/img/fault_current.png)

Both Wavelet and S-transform detect fault inception. The HS-transform shows lower noise sensitivity under high distortion.

![Real-Imag Trajectory](https://raw.githubusercontent.com/nvlinhvn/HSTransform/main/img/fault_trajectory.png)

Different fault types produce distinct real-imaginary trajectories at varying frequencies.

---

## 7. References

<a id="1">[1]</a>
Stockwell, R.G., Mansinha, L. & Lowe, R.P. (1996).
Localization of the complex spectrum: the S transform.
*IEEE Trans. Signal Process.*, 44(4), 998–1001. doi:10.1109/78.492555

---

## 8. Citation

If you use HSTransform in your research, please cite:

> Linh V Nguyen (2024). HSTransform (Version 0.2) [Computer software]. GitHub: github.com/nvlinhvn/HSTransform

---

## 9. Contact

Questions, issues, or suggestions: open a GitHub issue or email linhvietnguyen.ee@gmail.com
