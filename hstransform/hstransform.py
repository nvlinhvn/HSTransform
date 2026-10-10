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
            freq_len: signal length (determines DFT size and lag axis).
            n: frequency index used to scale the Gaussian.
            time: 1D array of time values (used for dt).

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

        The correct algorithm:
        1. Translates the hyperbola horizontally by ζ = (a−b)√c / (2√ab) so that
           X(0) = 0 while preserving the correct asymmetric slope (fixing the
           previous vertical-subtraction error).
        2. Uses physical frequency f_k = k / (N·Δt) rather than bin index k in the
           Gaussian exponent, so the window width is invariant to record duration.

        Args:
            n: signal length (also the DFT size).
            time: 1D array of time values, shape (n,).
            freq_steps: 1D array of frequency indices, shape (K,).

        Returns:
            Complex window matrix of shape (K, n): row j is W for freq_steps[j].
        """
        a = self.forwardtaper    # controls negative-lag (future, t > τ) decay
        b = self.backwardtaper   # controls positive-lag (past, t < τ) decay
        c = self.curvature

        dt = float(time[1] - time[0]) if len(time) > 1 else 1.0

        # Centered lag axis: t_w[0]=0, wraps to [-T/2, T/2)
        t_w: npt.NDArray[np.floating[Any]] = np.fft.fftfreq(n) * n * dt

        # Horizontal translation: ζ = (a−b)√c / (2√ab) centres the hyperbola at lag 0.
        # This is the correct reference construction (Pinnegar & Mansinha 2003 eq. 7-8);
        # the old code applied a vertical subtraction which changes the window shape.
        zeta = (a - b) * np.sqrt(c) / (2.0 * np.sqrt(a * b))
        v = t_w - zeta
        x_1d: npt.NDArray[np.floating[Any]] = (
            (a + b) * v + (a - b) * np.sqrt(v**2 + c)
        ) / (2.0 * a * b)

        K = len(freq_steps)
        W: npt.NDArray[np.complexfloating[Any, Any]] = np.empty((K, n), dtype=complex)
        for i, k in enumerate(freq_steps):
            # Physical frequency in Hz: f_k = k / (N·Δt).
            # Using bin index k directly was only correct when N·Δt = 1 s (record duration).
            f_k = float(k) / (n * dt)
            # Time-domain hyperbolic Gaussian: g(t) = (|f|/√2π) exp(-x²·f²/2)
            g = (f_k / np.sqrt(2.0 * np.pi)) * np.exp(-x_1d**2 * f_k**2 / 2.0)
            # Normalize so ∫g(t)dt = 1, guaranteeing W[0] = 1 (voice synthesis property).
            g_integral = float(np.sum(g)) * dt
            if g_integral > 1e-15:
                g = g / g_integral
            # Frequency-domain window = DFT of g, scaled by dt to approximate ∫g(t)e^{−iωt}dt.
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
            minf: minimum frequency bin index (default 0 = include DC row).
            fsamplingrate: frequency bin step (1 = all bins, 2 = every other, …).

        Returns:
            Complex ndarray of shape (len(bins), n) where
            bins = np.arange(minf, maxf+1, fsamplingrate) and
            maxf = min(900, n // 2).
        """
        self._input_validation(input_signal)

        time_arr: npt.NDArray[np.floating[Any]] = np.asarray(time_values)
        signal_arr: npt.NDArray[np.floating[Any]] = np.asarray(input_signal)

        n = len(signal_arr)
        maxf = min(900, n // 2)

        h: npt.NDArray[np.complexfloating[Any, Any]] = np.concatenate((fft(signal_arr),) * 2)

        # All requested bin indices, inclusive of minf and maxf
        bins: npt.NDArray[np.intp] = np.arange(minf, maxf + 1, fsamplingrate, dtype=np.intp)
        n_rows = len(bins)
        s: npt.NDArray[np.complexfloating[Any, Any]] = np.zeros((n_rows, n), dtype="complex")

        # DC bin (k=0): fill with signal mean
        dc_mask = bins == 0
        if dc_mask.any():
            s[dc_mask, :] = np.mean(signal_arr)

        # Positive-frequency bins: compute via windowed IFFT
        pos_mask = bins > 0
        k_abs = bins[pos_mask]
        if len(k_abs) > 0:
            W = self._build_window_matrix(n, time_arr, k_abs)
            # Build shifted FFT slice matrix: row j = h[k : k+n]
            # (corrected from original MATLAB 1-based indexing h[k+1 : k+n+1])
            H: npt.NDArray[np.complexfloating[Any, Any]] = np.stack(
                [h[k : k + n] for k in k_abs]
            )
            s[pos_mask, :] = ifft(H * W, axis=1)

        return s
