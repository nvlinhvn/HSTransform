import dataclasses
from collections.abc import Sequence

import numpy as np
from scipy.fft import fft, ifft


@dataclasses.dataclass
class HSTransform:
    """S-transform with a hyperbolic Gaussian window.

    Combines properties of the Short-time Fourier Transform and Wavelet Transform.
    """

    forwardtaper: float = 0.2
    backwardtaper: float = 0.1
    curvature: float = 312.5

    def _input_validation(self, input_signal: np.ndarray | Sequence[float]) -> None:
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

    def _compute_hyperbolic_gaussian(self, l: int, n: int, time: np.ndarray) -> np.ndarray:
        """Compute the hyperbolic Gaussian window.

        Args:
            l: length of the frequency vector (determines output length).
            n: frequency index used to scale the Gaussian.
            time: 1D array of time values.

        Returns:
            1D array of length l: the summed window weights per time step.
        """
        vectorf = np.arange(0, l)
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
        # axis=0 matches MATLAB sum(g) behaviour: sum along columns → 1D array of length l
        return np.sum(g, axis=0)

    def fit_transform(
        self,
        time_values: np.ndarray | Sequence[float],
        input_signal: np.ndarray | Sequence[float],
        minf: int = 0,
        fsamplingrate: int = 1,
    ) -> np.ndarray:
        """Compute the S-transform of the input signal.

        Args:
            time_values: time values corresponding to each sample.
            input_signal: the signal to transform.
            minf: minimum frequency index.
            fsamplingrate: frequency sampling step.

        Returns:
            Complex ndarray of shape ((maxf - minf + 1) // fsamplingrate, n).
        """
        self._input_validation(input_signal)

        time_values = np.asarray(time_values)
        input_signal = np.asarray(input_signal)

        n = len(input_signal)
        # Cap at 900 to cover up to the 18th harmonic of 50 Hz at 1 kHz sampling
        maxf = min(900, n // 2)

        h = fft(input_signal)
        h = np.concatenate((h, h))

        # Number of rows = DC row (0) + one row per frequency step up to maxf
        n_rows = (maxf - minf) // fsamplingrate + 1
        s = np.zeros((n_rows, n), dtype="complex")
        # MATLAB: 1 & [1:N] is logical AND → all ones. Python: 1 & array is bitwise AND → [1,0,1,0,...].
        s[0, :] = np.mean(input_signal) * np.ones(n)

        for k in range(fsamplingrate, maxf - minf + 1, fsamplingrate):
            w_hy = self._compute_hyperbolic_gaussian(n, minf + k, time_values)
            s[k // fsamplingrate, :] = ifft(h[minf + k + 1 : minf + k + n + 1] * w_hy)

        return s
