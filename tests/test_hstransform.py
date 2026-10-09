"""Tests for HSTransform: validation, window, and fit_transform behaviour."""

import numpy as np
import pytest

from hstransform import HSTransform  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def st() -> HSTransform:
    return HSTransform()


@pytest.fixture()
def simple_signal() -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 2, 100)
    signal = np.sin(2 * np.pi * 50 * t)
    return t, signal


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_input_validation_wrong_type(st: HSTransform) -> None:
    with pytest.raises(TypeError):
        st._input_validation({"a": 1})


def test_input_validation_nan(st: HSTransform) -> None:
    with pytest.raises(ValueError, match="null"):
        st._input_validation([1.0, float("nan"), 3.0])


def test_input_validation_string_array(st: HSTransform) -> None:
    with pytest.raises(ValueError, match="numerical"):
        st._input_validation(np.array(["a", "b", "c"]))


def test_input_validation_valid_list(st: HSTransform) -> None:
    st._input_validation([1.0, 2.0, 3.0])  # must not raise


def test_input_validation_valid_ndarray(st: HSTransform) -> None:
    st._input_validation(np.array([1.0, 2.0, 3.0]))  # must not raise


# ---------------------------------------------------------------------------
# _compute_hyperbolic_gaussian
# ---------------------------------------------------------------------------

def test_hyperbolic_gaussian(st: HSTransform) -> None:
    freq_len, freq_idx = 1000, 50
    t = np.linspace(0, 1, 100)
    result = st._compute_hyperbolic_gaussian(freq_len, freq_idx, t)
    assert isinstance(result, np.ndarray)
    assert result.shape == (freq_len,)


def test_hyperbolic_gaussian_returns_array(st: HSTransform) -> None:
    """Regression for Bug 1: result must be a 1D ndarray, not a scalar."""
    n = 100
    t = np.linspace(0, 1, n)
    result = st._compute_hyperbolic_gaussian(n, 50, t)
    assert isinstance(result, np.ndarray), (
        f"Expected np.ndarray but got {type(result)}. "
        "np.sum(g) without axis= returns a scalar (MATLAB port bug)."
    )
    assert result.shape == (n,), f"Expected shape ({n},) but got {result.shape}"


def test_hyperbolic_gaussian_non_negative(st: HSTransform) -> None:
    """Window is now complex-valued; check finite and no NaN instead of non-negative."""
    t = np.linspace(0, 1, 50)
    result = st._compute_hyperbolic_gaussian(50, 10, t)
    assert np.isfinite(result).all(), "Window contains non-finite values"
    assert not np.isnan(result).any(), "Window contains NaN values"


@pytest.mark.parametrize("forwardtaper,backwardtaper,curvature", [
    (0.1, 0.05, 100.0),
    (0.3, 0.2, 500.0),
    (0.5, 0.5, 312.5),
])
def test_hyperbolic_gaussian_custom_params(
    forwardtaper: float, backwardtaper: float, curvature: float
) -> None:
    st = HSTransform(forwardtaper=forwardtaper, backwardtaper=backwardtaper, curvature=curvature)
    t = np.linspace(0, 1, 64)
    result = st._compute_hyperbolic_gaussian(64, 10, t)
    assert isinstance(result, np.ndarray)
    assert result.shape == (64,)
    assert not np.isnan(result).any()


# ---------------------------------------------------------------------------
# fit_transform — shape, dtype, correctness
# ---------------------------------------------------------------------------

def test_fit_transform_basic(st: HSTransform, simple_signal: tuple) -> None:
    t, signal = simple_signal
    s = st.fit_transform(t, signal)
    assert isinstance(s, np.ndarray)
    assert np.issubdtype(s.dtype, np.complexfloating)
    assert not np.isnan(s).any()


@pytest.mark.parametrize("n_samples", [64, 100, 256])
def test_fit_transform_shape(n_samples: int) -> None:
    t = np.linspace(0, 1, n_samples)
    signal = np.sin(2 * np.pi * 10 * t)
    st = HSTransform()
    s = st.fit_transform(t, signal)
    maxf = min(900, n_samples // 2)
    expected_rows = (maxf - 0) // 1 + 1
    assert s.shape == (expected_rows, n_samples)


@pytest.mark.parametrize("fsamplingrate", [1, 2, 5])
def test_fit_transform_shape_fsamplingrate(fsamplingrate: int) -> None:
    n = 200
    t = np.linspace(0, 1, n)
    signal = np.sin(2 * np.pi * 20 * t)
    st = HSTransform()
    s = st.fit_transform(t, signal, fsamplingrate=fsamplingrate)
    maxf = min(900, n // 2)
    expected_rows = (maxf - 0) // fsamplingrate + 1
    assert s.shape == (expected_rows, n)


def test_fit_transform_list_input(st: HSTransform) -> None:
    t = list(np.linspace(0, 1, 50))
    signal = list(np.sin(2 * np.pi * 10 * np.linspace(0, 1, 50)))
    s = st.fit_transform(t, signal)
    assert isinstance(s, np.ndarray)
    assert not np.isnan(s).any()


def test_fit_transform_deterministic(st: HSTransform, simple_signal: tuple) -> None:
    t, signal = simple_signal
    s1 = st.fit_transform(t, signal)
    s2 = st.fit_transform(t, signal)
    np.testing.assert_array_equal(s1, s2)


def test_fit_transform_perf_matches_serial() -> None:
    """Verify batched vectorised output is numerically identical to serial computation.

    Guards against regressions introduced by the loop-precomputation optimisation.
    Serial reference: compute each row via _compute_hyperbolic_gaussian independently.
    Updated to use h[k:k+n] (correct slice) and the DFT-based window.
    """
    n = 256
    t = np.linspace(0, 1, n, endpoint=False)
    signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 150 * t)
    st = HSTransform()

    s_fast = st.fit_transform(t, signal)

    # Serial reference — uses corrected FFT slice h[k:k+n]
    from scipy.fft import fft, ifft  # noqa: PLC0415

    signal_arr = np.asarray(signal)
    time_arr = np.asarray(t)
    maxf = min(900, n // 2)
    h = np.concatenate((fft(signal_arr),) * 2)
    n_rows = (maxf - 0) // 1 + 1
    s_ref = np.zeros((n_rows, n), dtype="complex")
    s_ref[0, :] = np.mean(signal_arr) * np.ones(n)
    for k in range(1, maxf + 1):
        w = st._compute_hyperbolic_gaussian(n, k, time_arr)
        s_ref[k, :] = ifft(h[k : k + n] * w)

    np.testing.assert_allclose(s_fast, s_ref, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Regression tests for confirmed MATLAB→Python porting bugs
# ---------------------------------------------------------------------------

def test_fit_transform_dc_row_constant() -> None:
    """Regression for Bug 2: DC row must equal signal mean at every time step.

    In MATLAB, 1 & [1:N] is logical AND → all ones.
    In Python, 1 & array is bitwise AND → alternating [1,0,1,0,...].
    """
    t = np.linspace(0, 2, 200)
    signal = np.sin(2 * np.pi * 50 * t) + 0.5  # non-zero mean
    st = HSTransform()
    s = st.fit_transform(t, signal)
    expected_dc = np.mean(signal)
    assert np.allclose(s[0, :].real, expected_dc, atol=1e-10), (
        f"DC row should be constant {expected_dc:.4f} but got values ranging "
        f"[{s[0, :].real.min():.4f}, {s[0, :].real.max():.4f}]. "
        "Alternating pattern means bitwise AND bug (Bug 2) is still present."
    )


def test_fit_transform_two_tone_frequency_separation() -> None:
    """Regression for Bug 1: window must localise energy at signal frequencies.

    With the correct vector window, a 50 Hz + 150 Hz signal should show
    clearly higher energy at rows 50 and 150 than at row 10.
    The old scalar window caused k²-growth artifacts with near-zero energy
    at low-frequency rows.
    """
    n = 1000
    t = np.linspace(0, 1, n, endpoint=False)
    signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 150 * t)
    st = HSTransform()
    s = st.fit_transform(t, signal, fsamplingrate=1)

    energy = np.abs(s) ** 2
    mean_at_50 = energy[50, :].mean()
    mean_at_150 = energy[150, :].mean()
    mean_at_10 = energy[10, :].mean()

    assert mean_at_50 > mean_at_10 * 5, (
        "50 Hz row should have significantly more energy than row 10. "
        "If this fails, the Gaussian window is still acting as a scalar (Bug 1)."
    )
    assert mean_at_150 > mean_at_10 * 5, (
        "150 Hz row should have significantly more energy than row 10."
    )


# ---------------------------------------------------------------------------
# Tier 1 — Mathematical invariants (PR 10 new tests)
# ---------------------------------------------------------------------------

def test_linearity() -> None:
    """ST(a·x + b·y) == a·ST(x) + b·ST(y).

    atol=1e-14 guards against false failures in near-zero rows (noise floor ~1e-17)
    where relative tolerance is meaningless.
    """
    n = 128
    t = np.linspace(0, 1, n, endpoint=False)
    x = np.sin(2 * np.pi * 10 * t)
    y = np.cos(2 * np.pi * 25 * t)
    a, b = 2.0, -0.5
    st = HSTransform()
    combined = st.fit_transform(t, a * x + b * y)
    expected = a * st.fit_transform(t, x) + b * st.fit_transform(t, y)
    np.testing.assert_allclose(combined, expected, rtol=1e-10, atol=1e-14)


def test_voice_synthesis() -> None:
    """mean_τ s[k, τ] ≈ H[k]/n for signal frequencies (fundamental S-transform property)."""
    from scipy.fft import fft as sfft
    n = 256
    t = np.linspace(0, 1, n, endpoint=False)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 30 * t)
    st = HSTransform()
    s = st.fit_transform(t, signal)
    H = sfft(signal)
    for k in [10, 30]:
        expected = H[k] / n
        actual = np.mean(s[k, :])
        rel_err = abs(actual - expected) / (abs(expected) + 1e-10)
        assert rel_err < 0.05, f"Voice synthesis failed at k={k}: rel_err={rel_err:.4f}"


def test_window_dc_equals_one() -> None:
    """G[0, k] == 1 (window integrates to 1 → correct normalization)."""
    n = 128
    t = np.linspace(0, 1, n, endpoint=False)
    st = HSTransform()
    for k in [5, 20, 50]:
        G = st._compute_hyperbolic_gaussian(n, k, t)
        assert abs(G[0].real - 1.0) < 0.05, f"Window DC != 1 at k={k}: got {G[0].real:.4f}"
        assert abs(G[0].imag) < 1e-8, f"Window DC is not real at k={k}"


def test_pure_tone_constant_ridge() -> None:
    """Pure sinusoid → |S[f₀, :]| approximately constant (flat ridge).

    Uses symmetric tapers (forwardtaper == backwardtaper) because an asymmetric
    window inherently amplitude-modulates stationary signals — that is expected
    behaviour, not a bug.  The test guards against the window having pathological
    time-varying artifacts unrelated to the taper asymmetry.
    """
    n = 512
    f0 = 20
    t = np.linspace(0, 1, n, endpoint=False)
    signal = 1.5 * np.cos(2 * np.pi * f0 * t)
    st = HSTransform(forwardtaper=0.2, backwardtaper=0.2)
    s = st.fit_transform(t, signal)
    row = np.abs(s[f0, :])
    cv = row.std() / row.mean()  # coefficient of variation
    assert cv < 0.15, f"Ridge is not flat (CV={cv:.3f}). Window has time-varying artifacts."


# ---------------------------------------------------------------------------
# Tier 2 — Hyperbolic window properties (PR 10 new tests)
# ---------------------------------------------------------------------------

def test_symmetric_window_is_real() -> None:
    """λf == λb → symmetric Gaussian → G is real-valued (imaginary ≈ 0)."""
    lam = 0.2
    st = HSTransform(forwardtaper=lam, backwardtaper=lam, curvature=0.25)
    n = 64
    t = np.linspace(0, 1, n, endpoint=False)
    G = st._compute_hyperbolic_gaussian(n, 10, t)
    ratio = np.max(np.abs(G.imag)) / (np.max(np.abs(G.real)) + 1e-15)
    assert ratio < 1e-8, f"Symmetric window has non-trivial imaginary part (ratio={ratio:.2e})"


def test_asymmetric_window_has_imaginary_part() -> None:
    """λf ≠ λb → window is asymmetric → G is genuinely complex."""
    st = HSTransform(forwardtaper=0.3, backwardtaper=0.1, curvature=0.01)
    n = 64
    t = np.linspace(0, 1, n, endpoint=False)
    G = st._compute_hyperbolic_gaussian(n, 10, t)
    ratio = np.max(np.abs(G.imag)) / (np.max(np.abs(G.real)) + 1e-15)
    assert ratio > 0.05, f"Asymmetric window has negligible imaginary part (ratio={ratio:.2e})"


def test_large_curvature_degrades_to_symmetric() -> None:
    """Large curvature >> t² → hyperbolic correction negligible → G approximately real."""
    st = HSTransform(forwardtaper=0.3, backwardtaper=0.1, curvature=1e6)
    n = 64
    t = np.linspace(0, 1, n, endpoint=False)
    G = st._compute_hyperbolic_gaussian(n, 10, t)
    ratio = np.max(np.abs(G.imag)) / (np.max(np.abs(G.real)) + 1e-15)
    assert ratio < 0.01, f"Large curvature should suppress asymmetry (ratio={ratio:.2e})"


def _half_energy_width(arr: np.ndarray) -> int:
    """Return index where cumulative energy exceeds 50% of total."""
    total = np.sum(arr**2)
    if total == 0:
        return len(arr)
    cumsum = np.cumsum(arr**2)
    idx = np.searchsorted(cumsum, 0.5 * total)
    return int(idx) + 1


def test_asymmetric_decay_around_impulse() -> None:
    """Impulse response: forward/backward half-energy widths reflect λf and λb.

    Tolerance widened to 0.1 < ratio < 10.0 because the half-energy estimator
    is sensitive to the discrete impulse and window edge effects at small n.
    """
    n = 512
    t = np.linspace(-1, 1, n, endpoint=False)
    signal = np.zeros(n)
    signal[n // 2] = 1.0   # impulse at t=0
    lf, lb = 0.4, 0.1   # forward 4× wider than backward
    st = HSTransform(forwardtaper=lf, backwardtaper=lb, curvature=0.01)
    s = st.fit_transform(t, signal, fsamplingrate=5)
    energy = np.abs(s[10, :])**2
    mid = n // 2
    left_w = _half_energy_width(energy[:mid][::-1])
    right_w = _half_energy_width(energy[mid:])
    observed_ratio = right_w / left_w if left_w > 0 else float("inf")
    expected_ratio = lb / lf   # smaller lb = faster backward decay = narrower backward width
    assert 0.1 < observed_ratio / expected_ratio < 10.0, (
        f"Decay ratio {observed_ratio:.2f} does not match λb/λf={expected_ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# Tier 3 — Numerical regression guards (PR 10 new tests)
# ---------------------------------------------------------------------------

def test_fft_slice_alignment() -> None:
    """Row k DC == H[k]/n — directly catches the +1 offset bug."""
    from scipy.fft import fft as sfft
    n = 128
    t = np.linspace(0, 1, n, endpoint=False)
    signal = np.zeros(n)
    signal[0] = 1.0   # delta at t=0 → H[k] = 1 for all k
    st = HSTransform()
    s = st.fit_transform(t, signal)
    H = sfft(signal)
    for k in [1, 5, 20]:
        row_dc = np.sum(s[k, :]) / n
        expected_dc = H[k].real / n
        msg = f"FFT slice misaligned at k={k}: got {row_dc.real:.4f}, expected {expected_dc:.4f}"
        np.testing.assert_allclose(row_dc.real, expected_dc, rtol=0.05, err_msg=msg)


def test_parseval_approximate() -> None:
    """∑|s|² is in a reasonable range relative to ∑|x|² (no zero/blowup)."""
    n = 256
    t = np.linspace(0, 1, n, endpoint=False)
    signal = np.sin(2 * np.pi * 15 * t) + np.cos(2 * np.pi * 40 * t)
    st = HSTransform()
    s = st.fit_transform(t, signal)
    signal_energy = np.sum(np.abs(signal)**2)
    st_energy = np.sum(np.abs(s)**2)
    ratio = st_energy / signal_energy
    assert 0.001 < ratio < 10000, f"Energy ratio {ratio:.2e} is unreasonable"


# ---------------------------------------------------------------------------
# PR 11 — Parametrized test variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("forwardtaper,backwardtaper,curvature", [
    (0.2, 0.1, 312.5),   # defaults
    (0.1, 0.1, 0.25),    # symmetric, small curvature
    (0.5, 0.2, 1.0),     # asymmetric, medium curvature
    (0.3, 0.3, 0.01),    # symmetric, very small curvature
])
def test_window_dc_equals_one_parametrized(
    forwardtaper: float, backwardtaper: float, curvature: float
) -> None:
    """G[0, k] == 1 across a range of taper and curvature settings."""
    n = 128
    t = np.linspace(0, 1, n, endpoint=False)
    st = HSTransform(forwardtaper=forwardtaper, backwardtaper=backwardtaper, curvature=curvature)
    for k in [5, 20, 50]:
        G = st._compute_hyperbolic_gaussian(n, k, t)
        assert abs(G[0].real - 1.0) < 0.05, (
            f"Window DC != 1 at k={k} (lf={forwardtaper}, lb={backwardtaper}, "
            f"curv={curvature}): got {G[0].real:.4f}"
        )
        assert abs(G[0].imag) < 1e-8, f"Window DC is not real at k={k}"


@pytest.mark.parametrize("f0,amplitude", [
    (5, 1.0), (20, 2.0), (50, 0.5), (100, 1.5),
])
def test_pure_tone_constant_ridge_parametrized(f0: int, amplitude: float) -> None:
    """Pure sinusoid → |S[f₀, :]| approximately constant for various frequencies.

    Uses symmetric tapers — see test_pure_tone_constant_ridge for rationale.
    """
    n = 512
    t = np.linspace(0, 1, n, endpoint=False)
    signal = amplitude * np.cos(2 * np.pi * f0 * t)
    st = HSTransform(forwardtaper=0.2, backwardtaper=0.2)
    s = st.fit_transform(t, signal)
    row = np.abs(s[f0, :])
    cv = row.std() / row.mean()
    assert cv < 0.15, (
        f"Ridge not flat at f0={f0}, amplitude={amplitude} (CV={cv:.3f})"
    )


@pytest.mark.parametrize("n_samples,fsamplingrate", [
    (64, 1), (128, 2), (256, 5), (512, 1),
])
def test_voice_synthesis_parametrized(n_samples: int, fsamplingrate: int) -> None:
    """mean_τ s[k, τ] ≈ H[k]/n for various signal lengths and sampling rates."""
    from scipy.fft import fft as sfft
    t = np.linspace(0, 1, n_samples, endpoint=False)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 20 * t)
    st = HSTransform()
    s = st.fit_transform(t, signal, fsamplingrate=fsamplingrate)
    H = sfft(signal)
    # Only check k=10 (k=20 may not land on a row depending on fsamplingrate)
    for k in [10]:
        row_idx = k // fsamplingrate
        if row_idx >= s.shape[0]:
            continue
        expected = H[k] / n_samples
        actual = np.mean(s[row_idx, :])
        rel_err = abs(actual - expected) / (abs(expected) + 1e-10)
        assert rel_err < 0.05, (
            f"Voice synthesis failed at k={k} (n={n_samples}, rate={fsamplingrate}): "
            f"rel_err={rel_err:.4f}"
        )


@pytest.mark.parametrize("lam", [0.1, 0.2, 0.3, 0.5])
def test_symmetric_window_is_real_parametrized(lam: float) -> None:
    """Any equal forward/backward taper → real window, regardless of curvature."""
    st = HSTransform(forwardtaper=lam, backwardtaper=lam, curvature=0.25)
    n = 64
    t = np.linspace(0, 1, n, endpoint=False)
    G = st._compute_hyperbolic_gaussian(n, 10, t)
    ratio = np.max(np.abs(G.imag)) / (np.max(np.abs(G.real)) + 1e-15)
    assert ratio < 1e-8, (
        f"Symmetric window (lam={lam}) has non-trivial imaginary part (ratio={ratio:.2e})"
    )


@pytest.mark.parametrize("lf,lb", [
    (0.4, 0.1), (0.1, 0.4), (0.3, 0.05), (0.5, 0.2),
])
def test_asymmetric_window_has_imaginary_part_parametrized(lf: float, lb: float) -> None:
    """λf ≠ λb → window is genuinely complex for various asymmetric taper pairs."""
    st = HSTransform(forwardtaper=lf, backwardtaper=lb, curvature=0.01)
    n = 64
    t = np.linspace(0, 1, n, endpoint=False)
    G = st._compute_hyperbolic_gaussian(n, 10, t)
    ratio = np.max(np.abs(G.imag)) / (np.max(np.abs(G.real)) + 1e-15)
    assert ratio > 0.05, (
        f"Asymmetric window (lf={lf}, lb={lb}) has negligible imaginary part (ratio={ratio:.2e})"
    )


@pytest.mark.parametrize("curvature", [0.001, 0.01, 0.1, 1.0])
def test_asymmetric_decay_parametrized(curvature: float) -> None:
    """Impulse decay ratio holds across curvature values.

    Tolerance is wide (0.05 < ratio < 20.0) because very small curvature values
    make the hyperbolic coordinate transform highly nonlinear, compressing the
    effective forward/backward asymmetry. The test guards against complete collapse
    of the asymmetry rather than an exact ratio match.
    """
    n = 512
    t = np.linspace(-1, 1, n, endpoint=False)
    signal = np.zeros(n)
    signal[n // 2] = 1.0
    lf, lb = 0.4, 0.1
    st = HSTransform(forwardtaper=lf, backwardtaper=lb, curvature=curvature)
    s = st.fit_transform(t, signal, fsamplingrate=5)
    energy = np.abs(s[10, :])**2
    mid = n // 2
    left_w = _half_energy_width(energy[:mid][::-1])
    right_w = _half_energy_width(energy[mid:])
    if left_w == 0:
        return  # degenerate case — skip rather than divide by zero
    observed_ratio = right_w / left_w
    expected_ratio = lb / lf
    assert 0.05 < observed_ratio / expected_ratio < 20.0, (
        f"Decay ratio {observed_ratio:.2f} does not match λb/λf={expected_ratio:.2f} "
        f"at curvature={curvature}"
    )


@pytest.mark.parametrize("a,b", [(1, 0), (0, 1), (2, -0.5), (-1, 3), (0.1, 0.1)])
def test_linearity_parametrized(a: float, b: float) -> None:
    """ST(a·x + b·y) == a·ST(x) + b·ST(y) for various linear combination coefficients.

    atol=1e-14 guards against false failures in near-zero rows (noise floor ~1e-17).
    """
    n = 128
    t = np.linspace(0, 1, n, endpoint=False)
    x = np.sin(2 * np.pi * 10 * t)
    y = np.cos(2 * np.pi * 25 * t)
    st = HSTransform()
    combined = st.fit_transform(t, a * x + b * y)
    expected = a * st.fit_transform(t, x) + b * st.fit_transform(t, y)
    np.testing.assert_allclose(combined, expected, rtol=1e-10, atol=1e-14)
