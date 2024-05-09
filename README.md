# Hyperbolic S-Transform

Package for Hyperbolic S-transform

## 1. Introduction

HS Transform is a Python package for performing Hyperbolic S-transforms. The S-transform is a time-frequency representation that combines the best properties of the Short-time Fourier Transform and the Wavelet Transform. It provides simultaneous information on both the frequency content and temporal localization of a signal.

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

```
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
S_transformed = st.fit_transform(t, signal)
```

![alt text](./img/power_quality_disturbance.png)
![alt text](./img/power_quality_disturbance_trajectory.png)
![alt text](./img/fault_current.png)
![alt text](./img/fault_trajectory.png)

## 6. Communication

If you have any questions, issues, or suggestions for HS Transform, please open an issue on the GitHub repository.
or contact email: linhvietnguyen.ee@gmail.com

## 7. Citation

If you use HS Transform in your research, please cite it as follows:
Linh V Nguyen (2024). HS Transform (Version 0.1) [Computer software]. GitHub: github.com/nvlinhvn/hstransform
