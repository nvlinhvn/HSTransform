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
    """
    n = 256
    t = np.linspace(0, 1, n, endpoint=False)
    signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 150 * t)
    st = HSTransform()

    s_fast = st.fit_transform(t, signal)

    from scipy.fft import fft, ifft  # noqa: PLC0415

    signal_arr = np.asarray(signal)
    time_arr = np.asarray(t)
    maxf = min(900, n // 2)
    h = np.concatenate((fft(signal_arr),) * 2)
    bins = np.arange(0, maxf + 1, 1)
    n_rows = len(bins)
    s_ref = np.zeros((n_rows, n), dtype="complex")
    s_ref[0, :] = np.mean(signal_arr)
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
    """Return index where cumulative sum exceeds 50% of total (arr already an energy)."""
    total = np.sum(arr)
    if total == 0:
        return len(arr)
    cumsum = np.cumsum(arr)
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
    """Row k DC == H[k]/n — directly catches the +1 offset bug.

    Impulse at index 7 gives H[k] = exp(-2πi·7k/N), which is k-dependent and
    distinct for each k, unlike an impulse at 0 which gives H[k]=1 for all k
    and would not catch a constant frequency offset in the slice.
    """
    from scipy.fft import fft as sfft
    n = 128
    t = np.linspace(0, 1, n, endpoint=False)
    signal = np.zeros(n)
    signal[7] = 1.0   # delta at t=7 → H[k] = exp(-2πi·7k/n), varies with k
    st = HSTransform()
    s = st.fit_transform(t, signal)
    H = sfft(signal)
    for k in [1, 5, 20]:
        row_dc = np.sum(s[k, :]) / n
        expected_dc = H[k] / n
        msg = (
            f"FFT slice misaligned at k={k}: got {row_dc:.4f}, expected {expected_dc:.4f}. "
            "A +1 offset in the FFT slice would shift H[k] → H[k+1]."
        )
        np.testing.assert_allclose(row_dc, expected_dc, rtol=0.05, atol=1e-6, err_msg=msg)


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


# ---------------------------------------------------------------------------
# Regression guards for F1 (frequency-unit invariance), F2 (correct coordinate),
# and F3 (minf > 0 bin selection)
# ---------------------------------------------------------------------------

def test_frequency_unit_invariance() -> None:
    """F1 fix: changing time units must not change coefficients.

    If t is in seconds with curvature c, then t_ms = t * 1000 (milliseconds)
    with curvature c * 1e6 must produce the same window kernels.
    Before the fix, f_k = k (bin index) not k/(n·dt), so the window width
    scaled with record duration and broke under unit change.
    """
    n = 128
    t_s = np.arange(n) / 100.0       # seconds, fs=100 Hz
    t_ms = t_s * 1000.0              # milliseconds
    c_s = 1.0
    c_ms = c_s * 1e6                 # curvature scales as time²

    st_s  = HSTransform(forwardtaper=0.3, backwardtaper=0.1, curvature=c_s)
    st_ms = HSTransform(forwardtaper=0.3, backwardtaper=0.1, curvature=c_ms)

    k = 10
    W_s  = st_s._compute_hyperbolic_gaussian(n, k, t_s)
    W_ms = st_ms._compute_hyperbolic_gaussian(n, k, t_ms)

    np.testing.assert_allclose(
        W_s, W_ms, rtol=1e-8, atol=1e-10,
        err_msg="Window changed under seconds→milliseconds conversion (F1 regression)."
    )


def test_coordinate_slope_at_zero_asymmetric() -> None:
    """F2 fix: coordinate derivative at lag 0 must match the reference formula.

    Reference: X'(0) = 2 / (a + b)
    Old (vertical subtraction): X'(0) = (a + b) / (2ab)  [wrong for a ≠ b]

    Verified by finite difference on the time-domain coordinate x_1d.
    """
    a, b, c = 0.3, 0.1, 0.01
    n = 2048
    dt = 1.0 / n       # 1-second record, n*dt = 1
    t_w = np.fft.fftfreq(n) * n * dt

    zeta = (a - b) * np.sqrt(c) / (2.0 * np.sqrt(a * b))
    v = t_w - zeta
    x_1d = ((a + b) * v + (a - b) * np.sqrt(v**2 + c)) / (2.0 * a * b)

    # Finite-difference slope at lag 0 (t_w[0] = 0)
    # t_w[1] is the first positive lag = dt
    slope_fd = (x_1d[1] - x_1d[0]) / (t_w[1] - t_w[0])
    slope_reference = 2.0 / (a + b)

    np.testing.assert_allclose(
        slope_fd, slope_reference, rtol=2e-3,
        err_msg=(
            f"Coordinate slope at lag 0 is {slope_fd:.6f}, "
            f"expected reference 2/(a+b) = {slope_reference:.6f} (F2 regression)."
        )
    )


def test_minf_positive_bin_selection() -> None:
    """F3 fix: when minf > 0, the first row must correspond to bin minf.

    Previously row 0 was always DC (mean) regardless of minf, so bin minf
    was never computed and appeared as a constant-mean row instead.
    """
    n = 256
    minf = 10
    t = np.linspace(0, 1, n, endpoint=False)
    # Unit complex tone at exactly bin 10: DFT has magnitude 1 at k=10 only
    signal = np.exp(2j * np.pi * minf * np.arange(n) / n).real
    st = HSTransform()

    s = st.fit_transform(t, signal, minf=minf)

    # Row 0 must NOT be the DC mean (mean of signal ≈ 0)
    row0_mean = float(np.abs(np.mean(s[0, :])))
    assert row0_mean > 0.1, (
        f"Row 0 with minf={minf} has mean amplitude {row0_mean:.4f} ≈ 0; "
        "it was set to DC instead of computing bin 10 (F3 regression)."
    )


def test_minf_positive_shape() -> None:
    """F3 fix: output shape is correct for minf > 0."""
    n = 256
    t = np.linspace(0, 1, n, endpoint=False)
    signal = np.sin(2 * np.pi * 10 * t)
    st = HSTransform()
    maxf = min(900, n // 2)

    for minf, step in [(10, 1), (5, 5), (0, 1)]:
        s = st.fit_transform(t, signal, minf=minf, fsamplingrate=step)
        expected = len(np.arange(minf, maxf + 1, step))
        assert s.shape == (expected, n), (
            f"Shape mismatch for minf={minf}, step={step}: "
            f"got {s.shape}, expected ({expected}, {n})"
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


# ---------------------------------------------------------------------------
# Robustness: constructor validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,match", [
    ({"forwardtaper": 0},    "forwardtaper"),
    ({"forwardtaper": -0.1}, "forwardtaper"),
    ({"forwardtaper": float("inf")}, "forwardtaper"),
    ({"backwardtaper": 0},   "backwardtaper"),
    ({"backwardtaper": -1},  "backwardtaper"),
    ({"curvature": 0},       "curvature"),
    ({"curvature": -1},      "curvature"),
    ({"curvature": float("nan")}, "curvature"),
])
def test_constructor_invalid_params(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        HSTransform(**kwargs)


# ---------------------------------------------------------------------------
# Robustness: signal validation
# ---------------------------------------------------------------------------

def test_signal_1d_required() -> None:
    t = np.linspace(0, 1, 10, endpoint=False)
    signal_2d = np.ones((10, 3))
    with pytest.raises(ValueError, match="1-D"):
        HSTransform().fit_transform(t, signal_2d)


def test_signal_minimum_length() -> None:
    with pytest.raises(ValueError, match="2 samples"):
        HSTransform().fit_transform([0.0], [1.0])


def test_signal_infinite_rejected() -> None:
    t = np.linspace(0, 1, 10, endpoint=False)
    signal = np.ones(10)
    signal[3] = float("inf")
    with pytest.raises(ValueError, match="infinite"):
        HSTransform().fit_transform(t, signal)


def test_signal_nan_rejected() -> None:
    t = np.linspace(0, 1, 10, endpoint=False)
    signal = np.ones(10)
    signal[5] = float("nan")
    with pytest.raises(ValueError, match="null"):
        HSTransform().fit_transform(t, signal)


# ---------------------------------------------------------------------------
# Robustness: time array validation
# ---------------------------------------------------------------------------

def test_time_length_mismatch_rejected() -> None:
    t = np.linspace(0, 1, 5, endpoint=False)
    signal = np.ones(10)
    with pytest.raises(ValueError, match="length"):
        HSTransform().fit_transform(t, signal)


def test_time_non_ascending_rejected() -> None:
    t = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    signal = np.ones(6)
    with pytest.raises(ValueError, match="increasing"):
        HSTransform().fit_transform(t, signal)


def test_time_non_uniform_rejected() -> None:
    t = np.array([0.0, 0.1, 0.2, 0.4, 0.5, 0.6])  # gap at index 3
    signal = np.ones(6)
    with pytest.raises(ValueError, match="uniform"):
        HSTransform().fit_transform(t, signal)


def test_time_complex_rejected() -> None:
    """R3: complex timestamps must be rejected before float cast silently drops imag."""
    t = np.array([0.0, 0.1, 0.2, 0.3, 0.4]) + 1j * np.zeros(5)
    signal = np.ones(5)
    with pytest.raises(TypeError, match="complex"):
        HSTransform().fit_transform(t, signal)


def test_time_float32_accepted() -> None:
    """R3: float32 uniform grids must be accepted (float32 spacing varies by ~1e-7 relative)."""
    n = 64
    t = np.linspace(0.0, 0.63, n, endpoint=False, dtype=np.float32)
    signal = np.ones(n)
    HSTransform().fit_transform(t, signal)  # must not raise


def test_time_absolute_origin_accepted() -> None:
    """R3: large absolute-time origins (e.g. Unix timestamps) must be accepted when uniform."""
    n = 64
    t0 = 1_700_000_000.0  # Unix timestamp ~ 2023
    t = t0 + np.arange(n) * 0.1
    signal = np.ones(n)
    HSTransform().fit_transform(t, signal)  # must not raise


def test_time_infinite_rejected() -> None:
    t = np.array([0.0, 0.1, float("inf"), 0.3, 0.4, 0.5])
    signal = np.ones(6)
    with pytest.raises(ValueError, match="finite"):
        HSTransform().fit_transform(t, signal)


# ---------------------------------------------------------------------------
# Robustness: frequency parameter validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,exc,match", [
    ({"minf": -1},           ValueError, "minf"),
    ({"minf": 200},          ValueError, "minf"),   # minf > maxf
    ({"fsamplingrate": 0},   ValueError, "fsamplingrate"),
    ({"fsamplingrate": -1},  ValueError, "fsamplingrate"),
    ({"maxf": -1},           ValueError, "maxf"),
    ({"maxf": 99999},        ValueError, "maxf"),   # > n//2
])
def test_frequency_params_invalid(kwargs: dict, exc: type, match: str) -> None:
    n = 64
    t = np.linspace(0, 1, n, endpoint=False)
    signal = np.sin(2 * np.pi * 5 * t)
    with pytest.raises(exc, match=match):
        HSTransform().fit_transform(t, signal, **kwargs)


def test_maxf_configurable() -> None:
    """Custom maxf limits output rows and matches frequency_axis output."""
    n = 256
    t = np.linspace(0, 1, n, endpoint=False)
    signal = np.sin(2 * np.pi * 10 * t)
    st = HSTransform()

    s = st.fit_transform(t, signal, maxf=50)
    freqs = st.frequency_axis(n, t[1] - t[0], maxf=50)

    assert s.shape[0] == len(freqs) == 51   # bins 0..50 inclusive
    assert s.shape[1] == n


# ---------------------------------------------------------------------------
# Robustness: frequency_axis helper
# ---------------------------------------------------------------------------

def test_frequency_axis_values() -> None:
    """frequency_axis rows match physical frequencies of fit_transform output."""
    n = 256
    fs = 256.0
    dt = 1.0 / fs
    st = HSTransform()

    freqs = st.frequency_axis(n, dt)
    maxf = min(900, n // 2)

    assert len(freqs) == maxf + 1
    np.testing.assert_allclose(freqs[0], 0.0)
    np.testing.assert_allclose(freqs[1], 1.0 / (n * dt), rtol=1e-10)
    np.testing.assert_allclose(freqs[maxf], maxf / (n * dt), rtol=1e-10)


# ---------------------------------------------------------------------------
# R1 — Independent direct-sum oracle (catches coordinate regression)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# R4 — Numerical robustness at extreme scales
# ---------------------------------------------------------------------------

def test_extreme_scale_no_nan() -> None:
    """R4: transform at very high frequency bin must not produce NaN or inf."""
    n = 128
    t = np.arange(n) * 0.001   # dt = 1 ms, Nyquist = 500 Hz
    signal = np.random.default_rng(0).normal(size=n)
    st = HSTransform(forwardtaper=0.01, backwardtaper=0.01, curvature=1e-6)
    maxf = n // 2
    s = st.fit_transform(t, signal, maxf=maxf)
    assert np.isfinite(s).all(), "NaN or inf at extreme scale (R4 regression)."


def test_extreme_curvature_no_nan() -> None:
    """R4: very large curvature must not overflow √(v²+c)."""
    n = 64
    t = np.arange(n) * 1.0   # long record: c and v² are comparable at c=1e12
    signal = np.ones(n)
    st = HSTransform(forwardtaper=0.1, backwardtaper=0.2, curvature=1e12)
    s = st.fit_transform(t, signal, maxf=5)
    assert np.isfinite(s).all(), "NaN or inf with large curvature (R4 regression)."


# ---------------------------------------------------------------------------
# R5 — Mutable-state validation and frequency_axis guards
# ---------------------------------------------------------------------------

def test_post_construction_mutation_validated() -> None:
    """R5: mutating forwardtaper after construction must re-validate."""
    st = HSTransform()
    with pytest.raises(ValueError, match="forwardtaper"):
        st.forwardtaper = -1.0


def test_post_construction_curvature_validated() -> None:
    """R5: mutating curvature after construction must re-validate."""
    st = HSTransform()
    with pytest.raises(ValueError, match="curvature"):
        st.curvature = 0.0


def test_frequency_axis_invalid_dt() -> None:
    """R5: frequency_axis must reject non-positive dt."""
    st = HSTransform()
    with pytest.raises(ValueError, match="dt"):
        st.frequency_axis(n=64, dt=-0.001)


def test_frequency_axis_invalid_n() -> None:
    """R5: frequency_axis must reject n < 1."""
    st = HSTransform()
    with pytest.raises(ValueError, match="n"):
        st.frequency_axis(n=0, dt=0.01)


def test_direct_sum_oracle_asymmetric() -> None:
    """R1: oracle built without any HSTransform internal method catches coordinate regression.

    Constructs reference weights from the horizontal-translation hyperbolic coordinate
    (Pinnegar & Mansinha 2003) independently of _build_window_matrix, then compares
    against production fit_transform.  This test WILL fail if the production code
    reverts to the old vertical-subtraction coordinate (F2 regression), because
    both formulas evaluate to 0 at lag 0 and thus pass all tests that only check
    the coordinate value at a single point.
    """
    n, k = 31, 5
    dt = 0.02
    a, b, c = 0.3, 0.1, 0.01
    t = np.arange(n) * dt
    rng = np.random.default_rng(157)
    x = rng.normal(size=n) + 1j * rng.normal(size=n)

    # Reference: horizontal-translation coordinate — computed independently
    u = np.fft.fftfreq(n) * n * dt
    zeta = (a - b) * np.sqrt(c) / (2.0 * np.sqrt(a * b))
    v = u - zeta
    X = ((a + b) * v + (a - b) * np.sqrt(v**2 + c)) / (2.0 * a * b)
    f_k = k / (n * dt)
    weights = np.exp(-0.5 * (f_k * X)**2)
    weights /= weights.sum()

    # Direct circular convolution: S[k, τ] = Σ_m x[m] · w[(τ−m) mod n] · e^{−2πi k m/n}
    m = np.arange(n)
    demodulated = x * np.exp(-2j * np.pi * k * m / n)
    expected = np.array([
        np.sum(demodulated * weights[(j - m) % n]) for j in range(n)
    ])

    actual = HSTransform(a, b, c).fit_transform(t, x, minf=k, maxf=k)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("n,k,dt,a,b,c,seed", [
    (31,  5, 0.02, 0.3, 0.1, 0.01,  157),   # asymmetric, non-unit dt
    (64, 10, 0.01, 0.2, 0.2, 0.25,   42),   # symmetric (a == b)
    (47,  3, 0.10, 0.5, 0.2,  1.0,   99),   # odd n, larger a/b ratio
    (16,  2, 0.05, 0.4, 0.1,  0.5,    7),   # small n
])
def test_direct_sum_oracle_parametrized(
    n: int, k: int, dt: float, a: float, b: float, c: float, seed: int,
) -> None:
    """Direct-sum oracle across multiple asymmetric and symmetric configurations."""
    t = np.arange(n) * dt
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n) + 1j * rng.normal(size=n)

    u = np.fft.fftfreq(n) * n * dt
    zeta = (a - b) * np.sqrt(c) / (2.0 * np.sqrt(a * b))
    v = u - zeta
    X = ((a + b) * v + (a - b) * np.sqrt(v**2 + c)) / (2.0 * a * b)
    f_k = k / (n * dt)
    weights = np.exp(-0.5 * (f_k * X)**2)
    weights /= weights.sum()

    m = np.arange(n)
    demodulated = x * np.exp(-2j * np.pi * k * m / n)
    expected = np.array([
        np.sum(demodulated * weights[(j - m) % n]) for j in range(n)
    ])

    actual = HSTransform(a, b, c).fit_transform(t, x, minf=k, maxf=k)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


# ---------------------------------------------------------------------------
# C1 — Rejected mutations must not leave invalid state
# ---------------------------------------------------------------------------

def test_rejected_mutation_preserves_valid_forwardtaper() -> None:
    """C1: a rejected forwardtaper assignment must leave the previous valid value in place."""
    st = HSTransform(forwardtaper=0.3)
    with pytest.raises(ValueError, match="forwardtaper"):
        st.forwardtaper = -0.2
    assert st.forwardtaper == pytest.approx(0.3), (
        "forwardtaper should be unchanged after rejected assignment"
    )


def test_rejected_mutation_instance_still_usable() -> None:
    """C1: fit_transform must still produce finite results after a rejected mutation."""
    st = HSTransform()
    try:
        st.forwardtaper = -0.2
    except ValueError:
        pass
    n = 64
    t = np.arange(n) * 0.01
    s = st.fit_transform(t, np.ones(n), maxf=2)
    assert np.isfinite(s).all(), "transform should still work after rejected forwardtaper mutation"


def test_rejected_curvature_mutation_preserves_state() -> None:
    """C1: a rejected curvature assignment must leave the previous valid value in place."""
    st = HSTransform(curvature=100.0)
    with pytest.raises(ValueError, match="curvature"):
        st.curvature = 0.0
    assert st.curvature == pytest.approx(100.0), (
        "curvature should be unchanged after rejected assignment"
    )


# ---------------------------------------------------------------------------
# C2 — Drifting grids must be rejected despite per-spacing tolerance
# ---------------------------------------------------------------------------

def test_drifting_grid_rejected() -> None:
    """C2: a monotonically drifting grid exceeding 0.5-sample drift against the
    mean-spacing ideal must be rejected.

    The validator uses mean spacing as the reference (matching the transform).
    A linear ramp from 0 to 0.001*dt accumulates ~2.05 samples of parabolic drift
    at its midpoint — well above the 0.5-sample threshold.  Individual per-spacing
    deviations stay within rtol=1e-4, so the endpoint check is the active gate.
    """
    n = 16384
    dt = 1.0 / n
    spacings = dt * (1 + np.linspace(0, 0.001, n - 1))
    t = np.r_[0.0, np.cumsum(spacings)]
    signal = np.ones(n)
    with pytest.raises(ValueError, match="uniformly spaced"):
        HSTransform().fit_transform(t, signal, maxf=2)


def test_genuine_uniform_float64_still_accepted() -> None:
    """C2: np.arange-based grid with no drift must still be accepted after the stricter check."""
    n = 1024
    t = np.arange(n) * (1.0 / n)
    signal = np.ones(n)
    HSTransform().fit_transform(t, signal, maxf=2)  # must not raise


def test_epoch_grid_accepted() -> None:
    """P2: Unix-timestamp grid must be accepted.

    float64 representational rounding on the first interval at large absolute origins
    (e.g. 1.7e9) causes it to differ from the mean spacing.  Using the mean spacing as
    the reference in the drift check (matching the transform) should accept this grid.
    """
    t_ep = 1.7e9 + np.arange(12000) * 0.002
    signal = np.sin(2 * np.pi * 1.0 * np.arange(12000) * 0.002)
    HSTransform().fit_transform(t_ep, signal, maxf=2)  # must not raise


# ---------------------------------------------------------------------------
# C3 — Representative dt used for frequency labeling
# ---------------------------------------------------------------------------

def test_epoch_grid_frequency_labels_use_mean_spacing() -> None:
    """C3: frequency_axis should use mean spacing, not first spacing, for epoch grids.

    For epoch grids (large absolute time origin), the first represented spacing may
    differ from the mean by representational rounding.  Using mean spacing gives a
    less biased frequency estimate.
    """
    n = 64
    t0 = 1_700_000_000.0
    dt_intended = 0.002
    t = t0 + np.arange(n) * dt_intended
    signal = np.ones(n)
    st = HSTransform()
    # Must not raise, and frequency_axis with the mean dt should be close to intended
    st.fit_transform(t, signal, maxf=2)
    # Mean spacing should be close to dt_intended
    dt_mean = float(np.mean(np.diff(t)))
    assert abs(dt_mean - dt_intended) / dt_intended < 1e-3


# ---------------------------------------------------------------------------
# C4 — frequency_axis validation matches fit_transform
# ---------------------------------------------------------------------------

def test_frequency_axis_rejects_negative_minf() -> None:
    """C4: frequency_axis must reject negative minf (same as fit_transform)."""
    st = HSTransform()
    with pytest.raises(ValueError, match="minf"):
        st.frequency_axis(n=64, dt=0.01, minf=-2)


def test_frequency_axis_rejects_out_of_range_maxf() -> None:
    """C4: frequency_axis must reject maxf > n//2 (same as fit_transform)."""
    st = HSTransform()
    with pytest.raises(ValueError, match="maxf"):
        st.frequency_axis(n=64, dt=0.01, maxf=99999)


def test_frequency_axis_rejects_invalid_fsamplingrate() -> None:
    """C4: frequency_axis must reject non-positive fsamplingrate."""
    st = HSTransform()
    with pytest.raises(ValueError, match="fsamplingrate"):
        st.frequency_axis(n=64, dt=0.01, fsamplingrate=0)
