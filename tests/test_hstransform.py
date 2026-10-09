"""
Unit Test for HS functions
- Test Gaussian Windowed
- Test HS-transform output
"""
import numpy as np
import pytest
from hstransform import HSTransform


def test_hyperbolic_gaussian():
    l, n = 1000, 50
    t = np.linspace(0, 1, 100)
    st = HSTransform()
    result = st._compute_hyperbolic_gaussian(l, n, t)
    # Returns a 1D array of length l (one weight per frequency sample), not a scalar
    assert isinstance(result, np.ndarray)
    assert result.shape == (l,)


# --- Regression test for Bug 1: _compute_hyperbolic_gaussian must return a 1D array, not a scalar ---

def test_hyperbolic_gaussian_returns_array():
    """w_hy must be a 1D ndarray of length n so it shapes the spectrum per time step."""
    n = 100
    t = np.linspace(0, 1, n)
    st = HSTransform()
    result = st._compute_hyperbolic_gaussian(n, 50, t)
    assert isinstance(result, np.ndarray), (
        f"Expected np.ndarray but got {type(result)}. "
        "np.sum(g) without axis= returns a scalar (MATLAB port bug)."
    )
    assert result.shape == (n,), f"Expected shape ({n},) but got {result.shape}"


def test_fit_transform_dc_row_constant():
    """
    Regression for Bug 2: the DC row (frequency=0) must equal the signal mean
    at every time step. In MATLAB 1 & [1:N] is logical AND → all ones.
    In Python 1 & array is bitwise AND → alternating [1,0,1,0,...].
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


def test_fit_transform():
    t = np.linspace(0, 2, 100)
    signal = np.sin(2 * np.pi * 50 * t)
    st = HSTransform()
    s_transformed = st.fit_transform(t, signal)
    assert s_transformed.shape == (min(900, len(signal) // 2) + 1, len(signal))
    assert isinstance(s_transformed, np.ndarray)
    assert np.issubdtype(s_transformed.dtype, np.complexfloating)
    assert not np.isnan(s_transformed).any()


def test_fit_transform_two_tone_frequency_separation():
    """
    Regression for Bug 1: with the window correctly applied as a vector,
    a two-tone signal (50 Hz + 150 Hz) must show localised energy at those
    frequency rows, not a k²-growth artifact across all rows.
    """
    fs = 1000
    n = 1000
    t = np.linspace(0, 1, n, endpoint=False)
    signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 150 * t)
    st = HSTransform()
    s = st.fit_transform(t, signal, fsamplingrate=1)

    energy = np.abs(s) ** 2
    mean_energy_at_50 = energy[50, :].mean()
    mean_energy_at_150 = energy[150, :].mean()
    mean_energy_at_10 = energy[10, :].mean()

    assert mean_energy_at_50 > mean_energy_at_10 * 5, (
        "50 Hz row should have significantly more energy than row 10. "
        "If this fails, the Gaussian window is still acting as a scalar (Bug 1)."
    )
    assert mean_energy_at_150 > mean_energy_at_10 * 5, (
        "150 Hz row should have significantly more energy than row 10."
    )


if __name__ == "__main__":
    pytest.main()
