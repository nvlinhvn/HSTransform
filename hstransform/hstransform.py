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
        """Compute the hyperbolic Gaussian window for a single frequency index.

        Thin wrapper around _build_window_matrix for a single frequency step.

        Args:
            freq_len: length of the frequency vector (determines output length).
            n: frequency index used to scale the Gaussian.
            time: 1D array of time values.

        Returns:
            1D array of length freq_len: window weights.
        """
        freq_steps = np.array([n], dtype=np.intp)
        W = self._build_window_matrix(freq_len, time, freq_steps)
        result: npt.NDArray[np.floating[Any]] = W[0]
        return result

    def _build_window_matrix(
        self,
        n: int,
        time: npt.NDArray[np.floating[Any]],
        freq_steps: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.floating[Any]]:
        """Compute the hyperbolic Gaussian window for all frequency steps at once.

        Precomputes the x-transform and the base quadratic form A = vectorf² × x² once,
        then evaluates exp(-A / (2k²)) for each k without redundant allocations.

        Args:
            n: signal length (also the frequency vector length).
            time: 1D array of time values, shape (n,).
            freq_steps: 1D array of frequency indices, shape (K,).

        Returns:
            Window matrix of shape (K, n): row j is w_hy for freq_steps[j].
        """
        lambdaf = self.forwardtaper
        lambdab = self.backwardtaper
        lambda_val = self.curvature
        norm = 2.0 / ((lambdaf + lambdab) * np.sqrt(2.0 * np.pi))

        # x depends only on time — compute once for all frequency steps
        x_1d: npt.NDArray[np.floating[Any]] = (
            (lambdaf + lambdab) * time / (2.0 * lambdaf * lambdab)
            + (lambdaf - lambdab)
            * np.sqrt(time**2 + lambda_val)
            / (2.0 * lambdaf * lambdab)
        )
        # Note: the original code used np.tile(x_1d, (1,2)).T → shape (2n,1), which produces
        # two identical copies of x_1d. The resulting (2n,n) matrix is therefore two stacked
        # copies of an (n,n) matrix, so sum(g, axis=0) = 2 × sum(g[:n], axis=0).
        # We exploit this to halve the matrix size and computation.
        x_col: npt.NDArray[np.floating[Any]] = x_1d.reshape(-1, 1)  # shape (n, 1)

        # vectorf and its square — same for every k
        vectorf: npt.NDArray[np.floating[Any]] = np.arange(0, n, dtype=float)

        # Precompute A = vectorf² ⊗ x²: shape (n, n) — the expensive part, done once.
        # Factor of 2 accounts for the removed tile duplication.
        A: npt.NDArray[np.floating[Any]] = (vectorf**2)[None, :] * x_col**2
        prefactor = norm * 2.0 * vectorf  # absorbs the duplicate-tile factor of 2

        # Build the full window matrix in a single loop that only does cheap ops per k
        K = len(freq_steps)
        W: npt.NDArray[np.floating[Any]] = np.empty((K, n))
        for i, k in enumerate(freq_steps):
            g = prefactor * np.exp(-A / (2.0 * float(k) ** 2))
            W[i] = np.sum(g, axis=0)

        return W

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

        # Frequency indices for which we compute the transform
        freq_steps = np.arange(fsamplingrate, maxf - minf + 1, fsamplingrate, dtype=np.intp)
        k_abs = (minf + freq_steps).astype(np.intp)  # absolute frequency indices

        # Build all window vectors in one pass (x and A precomputed once)
        W = self._build_window_matrix(n, time_arr, k_abs)

        # Build the full FFT slice matrix: row j = h[k+1 : k+n+1], shape (K, n)
        H: npt.NDArray[np.complexfloating[Any, Any]] = np.stack(
            [h[k + 1 : k + n + 1] for k in k_abs]
        )

        # Batch IFFT across all frequency rows in one call
        row_indices = (freq_steps // fsamplingrate).astype(np.intp)
        s[row_indices, :] = ifft(H * W, axis=1)

        return s
