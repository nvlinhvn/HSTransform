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
    t = np.linspace(0, 1, 50)
    result = st._compute_hyperbolic_gaussian(50, 10, t)
    assert (result >= 0).all()


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

    # Serial reference
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
        s_ref[k, :] = ifft(h[k + 1 : k + n + 1] * w)

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
