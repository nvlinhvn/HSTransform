import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.fft import fft, ifft

_ArrayLike = npt.NDArray[Any] | Sequence[float]


@dataclasses.dataclass
class HSTransform:
    """S-transform with a hyperbolic Gaussian window.

    Combines properties of the Short-time Fourier Transform and Wavelet Transform.
    """

    forwardtaper: float = 0.2
    backwardtaper: float = 0.1
    curvature: float = 312.5

    def _input_validation(self, input_signal: _ArrayLike) -> None:
        """Validate that input_signal is array-like and contains only finite numbers.

        Raises:
            TypeError: if input_signal is not array-like (ndarray, list, or similar sequence).
            ValueError: if input_signal contains NaN or non-numeric values.
        """
        try:
            input_array = np.asarray(input_signal)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"input_signal must be array-like (ndarray, list, pd.Series, …), "
                f"got {type(input_signal)}."
            ) from exc

        if not isinstance(input_signal, (np.ndarray, list, Sequence)):
            raise TypeError(
                f"input_signal must be array-like (ndarray, list, pd.Series, …), "
                f"got {type(input_signal)}."
            )

        if not np.issubdtype(input_array.dtype, np.number):
            raise ValueError("input_signal should only contain numerical values.")

        if np.isnan(input_array).any():
            raise ValueError("input_signal contains null values.")

    def _compute_hyperbolic_gaussian(
        self, freq_len: int, n: int, time: npt.NDArray[np.floating[Any]]
    ) -> npt.NDArray[np.floating[Any]]:
        """Compute the hyperbolic Gaussian window.

        Args:
            freq_len: length of the frequency vector (determines output length).
            n: frequency index used to scale the Gaussian.
            time: 1D array of time values.

        Returns:
            1D array of length freq_len: the summed window weights per time step.
        """
        vectorf = np.arange(0, freq_len)
        vectorf1 = vectorf**2
        lambdaf = self.forwardtaper
        lambdab = self.backwardtaper
        lambda_val = self.curvature
        x = (
            (lambdaf + lambdab) * time / (2 * lambdaf * lambdab)
            + (lambdaf - lambdab) * np.sqrt(time**2 + lambda_val) / (2 * lambdaf * lambdab)
        )
        x = np.tile(x, (1, 2)).T
        vectorf2 = -vectorf1 * x**2 / (2 * n**2)
        g = 2 * np.abs(vectorf) * np.exp(vectorf2) / ((lambdaf + lambdab) * np.sqrt(2 * np.pi))
        # axis=0 matches MATLAB sum(g) behaviour: sum along columns → 1D array of length freq_len
        result: npt.NDArray[np.floating[Any]] = np.sum(g, axis=0)
        return result

    def fit_transform(
        self,
        time_values: _ArrayLike,
        input_signal: _ArrayLike,
        minf: int = 0,
        fsamplingrate: int = 1,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]:
        """Compute the S-transform of the input signal.

        Args:
            time_values: time values corresponding to each sample.
            input_signal: the signal to transform.
            minf: minimum frequency index.
            fsamplingrate: frequency sampling step.

        Returns:
            Complex ndarray of shape ((maxf - minf) // fsamplingrate + 1, n).
        """
        self._input_validation(input_signal)

        time_arr: npt.NDArray[np.floating[Any]] = np.asarray(time_values)
        signal_arr: npt.NDArray[np.floating[Any]] = np.asarray(input_signal)

        n = len(signal_arr)
        # Cap at 900 to cover up to the 18th harmonic of 50 Hz at 1 kHz sampling
        maxf = min(900, n // 2)

        h: npt.NDArray[np.complexfloating[Any, Any]] = np.concatenate((fft(signal_arr),) * 2)

        # Number of rows = DC row (0) + one row per frequency step up to maxf
        n_rows = (maxf - minf) // fsamplingrate + 1
        s: npt.NDArray[np.complexfloating[Any, Any]] = np.zeros((n_rows, n), dtype="complex")
        # MATLAB 1&[1:N] is logical AND (all ones); Python 1&array is bitwise AND ([1,0,1,0,...]).
        s[0, :] = np.mean(signal_arr) * np.ones(n)

        for k in range(fsamplingrate, maxf - minf + 1, fsamplingrate):
            w_hy = self._compute_hyperbolic_gaussian(n, minf + k, time_arr)
            s[k // fsamplingrate, :] = ifft(h[minf + k + 1 : minf + k + n + 1] * w_hy)

        return s
