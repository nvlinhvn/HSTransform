# Hyperbolic S-Transform

Package for Hyperbolic S-transform

## 1. Introduction

HS Transform is a Python package for performing Hyperbolic [S-transform](https://en.wikipedia.org/wiki/S_transform) [[1]](#1). The S-transform is a time-frequency representation that combines the best properties of the Short-time Fourier Transform and the Wavelet Transform. It provides simultaneous information on both the frequency content and temporal localization of a signal.

This is a part of my [published paper](https://ieeexplore.ieee.org/document/8423487)

### References

<a id="1">[1]</a>
Stockwell, R.G., Mansinha, L. & Lowe, R.P., (1996).
Localization of the complex spectrum: the S transform.
IEEE Trans. Signal Process., 44(4), 998–1001, doi:10.1109/78.492555

## 2. Dependencies

HS Transform requires the following Python packages:

- numpy
- scipy
- pandas
- matplotlib
- pytest

## 3. How to Install

You can install HS Transform using pip:

```
pip install HSTransform
```

## 4. Run tests

After installation, you can test the package using the included test scripts:
`pytest tests/`

## 5. Example

Here’s an example of how to use HS Transform to analyze a signal with voltage disturbance and power system fault:

<pre>
from hstransform import HyperbolicSTransform as HSTransform

# Create input signal (for example: Voltage signal)
t = np.linspace(0, 10, 100) # timeseries
V_m = 220*np.sqrt(2)  # peak voltage
f_V = 50  # frequency
phi_V = 0  # phase

V_clean = V_m * np.sin(2 * np.pi * f_V * t + phi_V)
# Create voltage sag/dip (80% of the nominal voltage for 0.15 second)
V_sag = np.where((t >= 2) & (t <= 3.5), 0.5 * V_clean, V_clean)

# Create an instance of HSTransform

hs = HSTransform()

# Perform the transform
signal = V_sag
S_transformed = hs.fit_transform(t, signal)
</pre>

### 5.1 Power Quality Disturbance

![](https://raw.githubusercontent.com/nvlinhvn/HSTransform/main/img/power_quality_disturbance.png)

The figure showed HS-transform is able to detect the transient disturbances like notch, spike. Meanwhile, those signals from Morlet Wavelet transform are not obviously recognized.

![](https://raw.githubusercontent.com/nvlinhvn/HSTransform/main/img/power_quality_disturbance_trajectory.png)
We can see also different types of voltage disturbance can generate different real-imaginary trajectory in S-transform at different frequencies.

### 5.2 Power System Faults

![](https://raw.githubusercontent.com/nvlinhvn/HSTransform/main/img/fault_current.png)

As can be seen, both Wavelet and S-transform are able to detect when the fault occur (huge change in current magnitude). Wavelet transform seems more sensitive with noise with high distortion compared with HS-transform.

![](https://raw.githubusercontent.com/nvlinhvn/HSTransform/main/img/fault_trajectory.png)

We can also observeo different types of faults can generate different real-imaginary trajectory in S-transform at varying level of frequencies.

## 6. Communication

If you have any questions, issues, or suggestions for HS Transform, please open an issue on the GitHub repository.
or contact email: linhvietnguyen.ee@gmail.com

## 7. Citation

If you use HS Transform in your research, please cite it as follows:
Linh V Nguyen (2024). HS Transform (Version 0.1) [Computer software]. GitHub: github.com/nvlinhvn/hstransform
