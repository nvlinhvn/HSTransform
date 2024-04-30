import numpy as np
import pytest
from HSTransform.hstransform import HSTransform

def test_hyperbolic_gaussian():
    # Create a sinusoidal signal
    t = np.linspace(0, 1, 100)
    signal = np.sin(2 * np.pi * 50 * t)

    # Compute the S-transform
    st = HSTransform()

    # Test the _compute_hyperbolic_gaussian method
    result = st._compute_hyperbolic_gaussian(1000, 50, t)
    assert isinstance(result, np.float64)

def test_fit_transform():
    # Create a sinusoidal signal
    t = np.linspace(0, 2, 100)
    signal = np.sin(2 * np.pi * 50 * t)

    # Compute the S-transform
    st = HSTransform()
    S_transformed = st.fit_transform(t, signal)

    # Add some assertions here to test the S-transform output
    # Dimension
    assert S_transformed.shape == (min(900, len(signal) // 2)+1, len(signal))
    # Type of output
    assert isinstance(S_transformed, np.ndarray)
    assert np.issubdtype(S_transformed.dtype, np.complexfloating)
    # any null
    assert not np.isnan(S_transformed).any()

if __name__ == "__main__":
    pytest.main()
