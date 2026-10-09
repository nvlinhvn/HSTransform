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
    ) -> npt.NDArray[np.complexfloating[Any, Any]]:
        """Compute the hyperbolic Gaussian window for a single frequency index.

        The window is computed as the DFT of the time-domain hyperbolic Gaussian,
        yielding a complex-valued frequency-domain window that correctly encodes
        both amplitude and phase for asymmetric (λf ≠ λb) tapers.

        Thin wrapper around _build_window_matrix for a single frequency step.

        Args:
            freq_len: length of the frequency vector (determines output length).
            n: frequency index used to scale the Gaussian.
            time: 1D array of time values.

        Returns:
            1D complex array of length freq_len: frequency-domain window weights.
        """
        freq_steps = np.array([n], dtype=np.intp)
        W = self._build_window_matrix(freq_len, time, freq_steps)
        result: npt.NDArray[np.complexfloating[Any, Any]] = W[0]
        return result

    def _build_window_matrix(
        self,
        n: int,
        time: npt.NDArray[np.floating[Any]],
        freq_steps: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.complexfloating[Any, Any]]:
        """Compute the hyperbolic Gaussian window for all frequency steps at once.

        The correct algorithm computes the frequency-domain window as the DFT of the
        time-domain hyperbolic Gaussian g(t) = (|f|/√2π) exp(-x(t)²·f²/2), where
        x(t) is the hyperbolic coordinate transform. This yields a complex-valued
        window that correctly represents asymmetric (λf ≠ λb) tapers via phase.

        The DFT approach also naturally produces W[0] = 1 (correct DC normalisation),
        fixing the original bug where the DC was forced to zero by a prefactor of 2f.

        Args:
            n: signal length (also the frequency vector length).
            time: 1D array of time values, shape (n,).
            freq_steps: 1D array of frequency indices, shape (K,).

        Returns:
            Complex window matrix of shape (K, n): row j is W for freq_steps[j].
        """
        lambdaf = self.forwardtaper
        lambdab = self.backwardtaper
        lambda_val = self.curvature

        # dt from signal time axis (used to scale DFT to approximate continuous FT)
        dt = float(time[1] - time[0]) if len(time) > 1 else 1.0

        # Centered time axis for window evaluation — independent of signal time offset.
        # np.fft.fftfreq(n) * n * dt gives indices [0, 1, ..., n//2, -n//2+1, ..., -1] * dt,
        # which is the natural centered, zero-origin axis for a DFT window.
        t_w: npt.NDArray[np.floating[Any]] = np.fft.fftfreq(n) * n * dt

        # Hyperbolic coordinate transform on centered axis
        x_1d: npt.NDArray[np.floating[Any]] = (
            (lambdaf + lambdab) * t_w / (2.0 * lambdaf * lambdab)
            + (lambdaf - lambdab) * np.sqrt(t_w**2 + lambda_val) / (2.0 * lambdaf * lambdab)
        )
        # Centre x so the window peaks at t=0 regardless of forward/backward asymmetry
        x_1d = x_1d - x_1d[0]

        K = len(freq_steps)
        W: npt.NDArray[np.complexfloating[Any, Any]] = np.empty((K, n), dtype=complex)
        for i, k in enumerate(freq_steps):
            k_f = float(k)
            # Time-domain hyperbolic Gaussian: g(t) = (|f|/√2π) exp(-x²·f²/2)
            g = (k_f / np.sqrt(2.0 * np.pi)) * np.exp(-x_1d**2 * k_f**2 / 2.0)
            # Normalize so ∫g(t)dt = 1 (required for voice synthesis: mean_τ S[k,τ] = H[k]/n).
            # The hyperbolic coordinate transform stretches time, so the raw Gaussian does not
            # integrate to 1 in general; dividing by the discrete integral corrects this.
            g_integral = float(np.sum(g)) * dt
            if g_integral > 1e-15:
                g = g / g_integral
            # Frequency-domain window = DFT of g, scaled by dt to approximate ∫g(t)e^{-iωt}dt.
            # W[0] = sum(g)*dt = 1 after normalization (unit DC).
            W[i] = fft(g) * dt
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

        # Build all window vectors in one pass
        W = self._build_window_matrix(n, time_arr, k_abs)

        # Build the FFT slice matrix: row j = h[k : k+n], shape (K, n)
        # (corrected from the original MATLAB 1-based indexing h[k+1 : k+n+1])
        H: npt.NDArray[np.complexfloating[Any, Any]] = np.stack(
            [h[k : k + n] for k in k_abs]
        )

        # Batch IFFT across all frequency rows in one call
        row_indices = (freq_steps // fsamplingrate).astype(np.intp)
        s[row_indices, :] = ifft(H * W, axis=1)

        return s
