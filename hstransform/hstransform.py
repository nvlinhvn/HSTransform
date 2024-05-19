import dataclasses
from typing import Union

import numpy as np
from scipy.fft import fft, ifft
import pandas as pd


@dataclasses.dataclass
class HSTransform:
    """
    A class used to represent the S-transform with a hyperbolic Gaussian window.

    ...

    Attributes
    ----------
    forwardtaper : float
        forward taper value
    backwardtaper : float
        backward taper value
    curvature : float
        curvature value
    """

    forwardtaper: float = 0.2
    backwardtaper: float = 0.1
    curvature: float = 312.5

    def _input_validation(self, input_signal):
        """
        Validates the input signal.

        Parameters
        ----------
            input_signal : np.ndarray, pd.Series, or list

        Raises
        ------
            TypeError: If input_signal is not a numpy array, pandas Series, or list.
            ValueError: If input_signal contains non-numerical values.
        """
        valid_types = (np.ndarray, pd.Series, list)
        if not isinstance(input_signal, valid_types):
            raise TypeError(f"input_signal must be one of the following types: {valid_types}, not {type(input_signal)}.")

        input_array = np.array(input_signal)
        if np.isnan(input_array).any():
            raise ValueError("input_signal contains null values.")

        if not np.issubdtype(input_array.dtype, np.number):
            raise ValueError("input_signal should only contain numerical values.")

    def _compute_hyperbolic_gaussian(self, l: int, n: int, time: np.ndarray) -> np.ndarray:
        """
        Computes the hyperbolic Gaussian window.

        Parameters
        ----------
            l : int
                length of the signal
            n : int
                frequency point
            time : np.ndarray
                time values of the signal

        Returns
        -------
            g : np.ndarray
                hyperbolic Gaussian window
        """
        vectorf = np.arange(0, l)
        vectorf1 = vectorf**2
        lambdaf = self.forwardtaper
        lambdab = self.backwardtaper
        lambda_val = self.curvature
        x = (lambdaf + lambdab) * time / (2 * lambdaf * lambdab) + (lambdaf - lambdab) * np.sqrt(time**2 + lambda_val) / (2 * lambdaf * lambdab)
        x = np.tile(x, (1, 2)).T
        vectorf2 = -vectorf1 * x**2 / (2 * n**2)
        g = 2 * np.abs(vectorf) * np.exp(vectorf2) / ((lambdaf + lambdab) * np.sqrt(2 * np.pi))
        return np.sum(g)

    def fit_transform(self,
                      time_values: Union[pd.Series, np.ndarray, list],
                      input_signal: Union[pd.Series, np.ndarray, list],
                      minf: int = 0,
                      fsamplingrate: int = 1) -> np.ndarray:
        """
        Computes the S-transform of the input signal.

        Parameters
        ----------
            time_values : np.ndarray
                time values of the signal
            input_signal : np.ndarray
                input signal
            minf : int
                minimum frequency point
            fsamplingrate : int
                frequency sampling rate

        Returns
        -------
            S : np.ndarray
                S-transform of the input signal
        """
        # Validate the input
        self._input_validation(input_signal)

        # Convert to numpy arrays if they are not array types
        if not isinstance(time_values, np.ndarray):
            time_values = np.array(time_values)
        if not isinstance(input_signal, np.ndarray):
            input_signal = np.array(input_signal)

        n = len(input_signal)
        # Make sure the max frequency to be optimized (Cover the 6th, 12th, or 18th harmonic respectively)
        maxf = min(900, n // 2)

        # Compute the fft of input
        h = fft(input_signal)
        h = np.concatenate((h, h))

        # S output
        s = np.zeros(((maxf - minf + 1) // fsamplingrate, n), dtype='complex')
        s[0, :] = np.mean(input_signal) * (1 & np.arange(1, n + 1))

        # Increment the frequency point
        for k in range(fsamplingrate, maxf - minf + 1, fsamplingrate):
            w_hy = self._compute_hyperbolic_gaussian(n, minf + k, time_values)
            s[k // fsamplingrate, :] = ifft(h[minf + k + 1:minf + k + n+1] * w_hy)

        return s
