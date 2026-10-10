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

    def __post_init__(self) -> None:
        self._validate_params(self.forwardtaper, self.backwardtaper, self.curvature)

    def __setattr__(self, name: str, value: object) -> None:
        """Re-validate taper/curvature when a field is mutated after construction."""
        super().__setattr__(name, value)
        if name in ("forwardtaper", "backwardtaper", "curvature"):
            self._validate_params(self.forwardtaper, self.backwardtaper, self.curvature)

    @staticmethod
    def _validate_params(forwardtaper: float, backwardtaper: float, curvature: float) -> None:
        if not np.isfinite(forwardtaper) or forwardtaper <= 0:
            raise ValueError("forwardtaper must be a finite positive number.")
        if not np.isfinite(backwardtaper) or backwardtaper <= 0:
            raise ValueError("backwardtaper must be a finite positive number.")
        if not np.isfinite(curvature) or curvature <= 0:
            raise ValueError("curvature must be a finite positive number.")

    def _input_validation(
        self,
        input_signal: _ArrayLike,
        time_values: _ArrayLike | None = None,
    ) -> None:
        """Validate signal and optional time array.

        Raises:
            TypeError: if input is not array-like or time is not real-valued.
            ValueError: if input contains NaN/inf, shapes mismatch, time is
                non-uniform/non-ascending, or signal is not 1-D.
        """
        try:
            signal_arr = np.asarray(input_signal)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"input_signal must be array-like (ndarray, list, …), "
                f"got {type(input_signal)}."
            ) from exc

        if not isinstance(input_signal, (np.ndarray, list, Sequence)):
            raise TypeError(
                f"input_signal must be array-like (ndarray, list, …), "
                f"got {type(input_signal)}."
            )

        if signal_arr.ndim != 1:
            raise ValueError(
                f"input_signal must be 1-D, got shape {signal_arr.shape}."
            )

        if len(signal_arr) < 2:
            raise ValueError("input_signal must contain at least 2 samples.")

        if not np.issubdtype(signal_arr.dtype, np.number):
            raise ValueError("input_signal should only contain numerical values.")

        if not np.isfinite(signal_arr).all():
            if np.isnan(signal_arr).any():
                raise ValueError("input_signal contains null values.")
            raise ValueError("input_signal contains infinite values.")

        if time_values is not None:
            # Reject complex timestamps before casting (np.asarray would silently
            # discard the imaginary part, hiding a likely caller error).
            _tv_raw = np.asarray(time_values)
            if np.iscomplexobj(_tv_raw):
                raise TypeError(
                    "time_values must be a 1-D array of real numbers; "
                    "got a complex-valued array."
                )
            try:
                time_arr = _tv_raw.astype(float)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "time_values must be a 1-D array of real numbers."
                ) from exc

            if time_arr.ndim != 1:
                raise ValueError(
                    f"time_values must be 1-D, got shape {time_arr.shape}."
                )

            if len(time_arr) != len(signal_arr):
                raise ValueError(
                    f"time_values length ({len(time_arr)}) must match "
                    f"input_signal length ({len(signal_arr)})."
                )

            if not np.isfinite(time_arr).all():
                raise ValueError("time_values must be finite.")

            spacings = np.diff(time_arr)
            dt_ref = float(spacings[0])
            if dt_ref <= 0:
                raise ValueError("time_values must be strictly increasing.")
            # Use a tolerance scaled to dt_ref so float32 grids and grids with
            # large absolute origins (e.g. Unix timestamps) are accepted when
            # spacing is genuinely uniform.  atol handles near-zero dt_ref.
            atol = max(dt_ref * 1e-4, 1e-9 * (float(time_arr[-1]) - float(time_arr[0])))
            if not np.allclose(spacings, dt_ref, rtol=1e-4, atol=atol):
                raise ValueError("time_values must be uniformly spaced.")

    def _compute_hyperbolic_gaussian(
        self, freq_len: int, n: int, time: npt.NDArray[np.floating[Any]]
    ) -> npt.NDArray[np.complexfloating[Any, Any]]:
        """Compute the hyperbolic Gaussian window for a single frequency index.

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

        Args:
            n: signal length (also the DFT size).
            time: 1D array of time values, shape (n,).
            freq_steps: 1D array of frequency indices, shape (K,).

        Returns:
            Complex window matrix of shape (K, n): row j is W for freq_steps[j].
        """
        a = self.forwardtaper    # controls negative-lag (t > τ) decay
        b = self.backwardtaper   # controls positive-lag (t < τ) decay
        c = self.curvature

        dt = float(time[1] - time[0]) if len(time) > 1 else 1.0

        # Centered lag axis: t_w[0]=0, wraps to [-T/2, T/2)
        t_w: npt.NDArray[np.floating[Any]] = np.fft.fftfreq(n) * n * dt

        # Horizontal translation: ζ = (a−b)√c / (2√ab) centres the hyperbola at lag 0.
        sqrt_c = np.sqrt(c)
        zeta = (a - b) * sqrt_c / (2.0 * np.sqrt(a * b))
        v = t_w - zeta
        # Use np.hypot for √(v²+c): avoids potential overflow when |v|≫√c at extreme scales.
        x_1d: npt.NDArray[np.floating[Any]] = (
            (a + b) * v + (a - b) * np.hypot(v, sqrt_c)
        ) / (2.0 * a * b)

        K = len(freq_steps)
        W: npt.NDArray[np.complexfloating[Any, Any]] = np.empty((K, n), dtype=complex)
        for i, k in enumerate(freq_steps):
            # Physical frequency: f_k = k / (N·Δt)
            f_k = float(k) / (n * dt)
            # Clamp exponent to avoid underflow warnings at extreme scales; the
            # Gaussian is effectively 0 well before this threshold.
            exponent = np.clip(-0.5 * (f_k * x_1d) ** 2, -700.0, 0.0)
            g = (f_k / np.sqrt(2.0 * np.pi)) * np.exp(exponent)
            g_integral = float(np.sum(g)) * dt
            if g_integral > 1e-15:
                g = g / g_integral
            W[i] = fft(g) * dt
        return W

    def frequency_axis(
        self,
        n: int,
        dt: float,
        minf: int = 0,
        fsamplingrate: int = 1,
        maxf: int | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """Return the physical frequency in Hz for each row of fit_transform output.

        Pass the same minf, fsamplingrate, and maxf values used in fit_transform.

        Args:
            n: signal length (same as len(input_signal)).
            dt: sample spacing in seconds (time_values[1] - time_values[0]).
            minf: same value passed to fit_transform (default 0).
            fsamplingrate: same value passed to fit_transform (default 1).
            maxf: same value passed to fit_transform (default: min(900, n//2)).

        Returns:
            1D float array of length matching fit_transform output rows,
            where freqs[j] = physical frequency in Hz of row j.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"dt must be a finite positive number, got {dt}.")
        if n < 1:
            raise ValueError(f"n must be ≥ 1, got {n}.")
        effective_maxf = (min(900, n // 2)) if maxf is None else int(maxf)
        bins = np.arange(minf, effective_maxf + 1, fsamplingrate, dtype=np.intp)
        result: npt.NDArray[np.floating[Any]] = bins / (n * dt)
        return result

    def fit_transform(
        self,
        time_values: _ArrayLike,
        input_signal: _ArrayLike,
        minf: int = 0,
        fsamplingrate: int = 1,
        maxf: int | None = None,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]:
        """Compute the S-transform of the input signal.

        Args:
            time_values: uniformly spaced time values, same length as input_signal.
            input_signal: 1-D finite real or complex signal to transform.
            minf: minimum frequency bin index (default 0 = include DC row).
            fsamplingrate: frequency bin step (1 = every bin, 2 = every other, …).
            maxf: maximum frequency bin index (default: min(900, n // 2)).
                  Must satisfy 0 ≤ minf ≤ maxf ≤ n // 2.

        Returns:
            Complex ndarray of shape (len(bins), n) where
            bins = np.arange(minf, effective_maxf+1, fsamplingrate).
        """
        self._input_validation(input_signal, time_values)

        time_arr: npt.NDArray[np.floating[Any]] = np.asarray(time_values, dtype=float)
        signal_arr: npt.NDArray[np.floating[Any]] = np.asarray(input_signal)

        n = len(signal_arr)
        default_maxf = min(900, n // 2)
        if maxf is None:
            effective_maxf = default_maxf
        else:
            if not isinstance(maxf, (int, np.integer)) or isinstance(maxf, bool):
                raise TypeError("maxf must be an integer.")
            if maxf < 0 or maxf > n // 2:
                raise ValueError(f"maxf must be in [0, n//2={n//2}], got {maxf}.")
            effective_maxf = int(maxf)

        if not isinstance(minf, (int, np.integer)) or isinstance(minf, bool):
            raise TypeError("minf must be a non-negative integer.")
        if minf < 0:
            raise ValueError(f"minf must be ≥ 0, got {minf}.")
        if minf > effective_maxf:
            raise ValueError(f"minf ({minf}) must be ≤ maxf ({effective_maxf}).")

        if not isinstance(fsamplingrate, (int, np.integer)) or isinstance(fsamplingrate, bool):
            raise TypeError("fsamplingrate must be a positive integer.")
        if fsamplingrate <= 0:
            raise ValueError(f"fsamplingrate must be > 0, got {fsamplingrate}.")

        h: npt.NDArray[np.complexfloating[Any, Any]] = np.concatenate((fft(signal_arr),) * 2)

        bins: npt.NDArray[np.intp] = np.arange(
            minf, effective_maxf + 1, fsamplingrate, dtype=np.intp
        )
        n_rows = len(bins)
        s: npt.NDArray[np.complexfloating[Any, Any]] = np.zeros((n_rows, n), dtype="complex")

        dc_mask = bins == 0
        if dc_mask.any():
            s[dc_mask, :] = np.mean(signal_arr)

        pos_mask = bins > 0
        k_abs = bins[pos_mask]
        if len(k_abs) > 0:
            W = self._build_window_matrix(n, time_arr, k_abs)
            H: npt.NDArray[np.complexfloating[Any, Any]] = np.stack(
                [h[k : k + n] for k in k_abs]
            )
            s[pos_mask, :] = ifft(H * W, axis=1)

        return s
