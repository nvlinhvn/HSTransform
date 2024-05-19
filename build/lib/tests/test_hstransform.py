"""
Unit Test for HS functions
- Test Gaussian Windowed
- Test HS-transform output
"""
import numpy as np
import pytest
from hstransform import HSTransform

def test_hyperbolic_gaussian():
    """
    Tests aims to validate the output 
    of Gaussian Window of a time array
    """
    # Create a sinusoidal signal
    t = np.linspace(0, 1, 100)

    # Compute the S-transform
    st = HSTransform()

    # Test the _compute_hyperbolic_gaussian method
    result = st._compute_hyperbolic_gaussian(1000, 50, t)
    assert isinstance(result, np.float64)

def test_fit_transform():
    """
    Tests aims to validate the S-transform output 
    - shape
    - type 
    - check any null
    """
    # Create a sinusoidal signal
    t = np.linspace(0, 2, 100)
    signal = np.sin(2 * np.pi * 50 * t)

    # Compute the S-transform
    st = HSTransform()
    s_transformed = st.fit_transform(t, signal)

    # Add some assertions here to test the S-transform output
    # Dimension
    assert s_transformed.shape == (min(900, len(signal) // 2)+1, len(signal))
    # Type of output
    assert isinstance(s_transformed, np.ndarray)
    assert np.issubdtype(s_transformed.dtype, np.complexfloating)
    # any null
    assert not np.isnan(s_transformed).any()

if __name__ == "__main__":
    pytest.main()
