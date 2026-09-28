"""Phase and baseline correction of a measured spectrum.

These are the two corrections that have to be right before a lineshape fit
means anything, and the two that are quickest to judge by eye -- so the GUI
applies them live while a slider moves, and the fit then runs on the result.

Phase
-----
The correction multiplies the complex spectrum by
``exp(i * (ph0 + ph1 * (nu - pivot) / (nu_high - nu_low)))`` and keeps the real
part: ``ph0`` is the zero-order phase, ``ph1`` the first-order phase accumulated
across the full width of the spectrum (both in degrees), and ``pivot`` the
frequency at which ``ph1`` has no effect.  That is TopSpin's convention up to
where it puts the pivot.

A spectrum loaded without its imaginary part (a text export, a co-add of
``1r`` files, a synthetic test) gets one from the Hilbert transform, as
TopSpin's ``ht`` does, so it can still be rephased.

Baseline
--------
A polynomial fitted to the points outside the signal and subtracted.  The
signal-free points are the outer ``edge`` fraction of the spectrum at each
end, which is how wideline quadrupolar data is usually recorded: the pattern in
the middle, a stretch of noise on either side.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

__all__ = ["analytic_signal", "phase", "auto_phase0", "baseline", "process"]


def analytic_signal(real: np.ndarray, imag: np.ndarray | None = None) -> np.ndarray:
    """``real + i imag``, reconstructing ``imag`` by Hilbert transform if absent."""
    real = np.asarray(real, dtype=float)
    if imag is not None:
        imag = np.asarray(imag, dtype=float)
        if imag.shape != real.shape:
            raise ValueError("real and imaginary parts must have the same shape")
        return real + 1j * imag
    return hilbert(real)


def _ramp(freq_mhz: np.ndarray, pivot_mhz: float | None) -> np.ndarray:
    """``(nu - pivot) / width``: the first-order phase weight of each point."""
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    width = float(freq_mhz.max() - freq_mhz.min()) if freq_mhz.size > 1 else 0.0
    if width <= 0:
        return np.zeros_like(freq_mhz)
    if pivot_mhz is None:
        pivot_mhz = 0.5 * (freq_mhz.max() + freq_mhz.min())
    return (freq_mhz - pivot_mhz) / width


def phase(
    freq_mhz: np.ndarray,
    spectrum: np.ndarray,
    ph0: float = 0.0,
    ph1: float = 0.0,
    pivot_mhz: float | None = None,
) -> np.ndarray:
    """Rephase a complex spectrum; returns the complex result.

    ``ph0`` and ``ph1`` are in degrees; ``pivot_mhz`` defaults to the centre.
    """
    angle = np.deg2rad(ph0 + ph1 * _ramp(freq_mhz, pivot_mhz))
    return np.asarray(spectrum) * np.exp(1j * angle)


def auto_phase0(spectrum: np.ndarray) -> float:
    """Zero-order phase (degrees) that maximises the integral of the real part.

    Closed form: the integral of ``Re(S e^{i phi})`` is ``|sum S| cos(phi +
    arg sum S)``, largest at ``phi = -arg sum S``.  Exact for a spectrum that is
    all absorption once phased, a good start otherwise.
    """
    total = np.sum(np.asarray(spectrum))
    if total == 0:
        return 0.0
    angle = -np.rad2deg(np.angle(total))
    return float((angle + 180.0) % 360.0 - 180.0)


def baseline(
    freq_mhz: np.ndarray,
    intensity: np.ndarray,
    order: int = 1,
    edge: float = 0.1,
) -> np.ndarray:
    """Polynomial baseline fitted to the outer ``edge`` fraction at each end.

    Returns the baseline itself (subtract it to correct).  ``order < 0`` gives
    zero.  The abscissa is scaled to ``[-1, 1]`` so high orders stay well
    conditioned on an absolute MHz axis.
    """
    x = np.asarray(freq_mhz, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if order < 0 or x.size == 0:
        return np.zeros_like(y)
    if not 0.0 < edge <= 0.5:
        raise ValueError("edge must be a fraction in (0, 0.5]")
    count = max(int(round(edge * x.size)), order + 1)
    mask = np.zeros(x.size, dtype=bool)
    mask[:count] = True
    mask[-count:] = True
    if mask.sum() <= order:
        raise ValueError("too few baseline points for this polynomial order")
    span = x.max() - x.min()
    t = (x - x.min()) / span * 2.0 - 1.0 if span > 0 else np.zeros_like(x)
    coefficients = np.polynomial.polynomial.polyfit(t[mask], y[mask], order)
    return np.polynomial.polynomial.polyval(t, coefficients)


def process(
    freq_mhz: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray | None = None,
    ph0: float = 0.0,
    ph1: float = 0.0,
    pivot_mhz: float | None = None,
    baseline_order: int = -1,
    baseline_edge: float = 0.1,
) -> np.ndarray:
    """Phase then baseline-correct; returns the corrected real spectrum.

    With no phase change and no ``imag`` the input comes back untouched, so a
    real-only spectrum is never pushed through a Hilbert round trip for
    nothing.
    """
    real = np.asarray(real, dtype=float)
    if ph0 or ph1:
        corrected = np.real(
            phase(freq_mhz, analytic_signal(real, imag), ph0, ph1, pivot_mhz)
        )
    else:
        corrected = real.copy()
    if baseline_order >= 0:
        corrected = corrected - baseline(
            freq_mhz, corrected, baseline_order, baseline_edge
        )
    return corrected
