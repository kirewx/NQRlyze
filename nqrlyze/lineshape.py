"""Turning transition frequencies into a broadened powder lineshape.

The Alderman-Solum-Grant interpolation spreads each triangle's intensity over
the frequency band its three corners span, as a triangular density peaking at
the middle corner frequency -- which is exactly the distribution of a linearly
varying frequency over a uniformly weighted triangle.

Rather than evaluating that density bin by bin, the density is accumulated
through its derivatives: slope changes go into one array, jumps into a second,
and two cumulative sums rebuild the lineshape in ``O(n_bins + n_triangles)``.

Broadening is applied in the Fourier domain with the exact analytic transforms
of the Lorentzian and Gaussian, so there is no kernel-truncation error:

    Lorentzian, FWHM wL:  exp(-pi * wL * |t|)
    Gaussian,   FWHM wG:  exp(-pi**2 * wG**2 * t**2 / (4 ln 2))
"""

from __future__ import annotations

import numpy as np
from scipy.fft import next_fast_len, irfft, rfft

__all__ = ["accumulate_asg", "accumulate_sticks", "broaden", "voigt_kernel"]

#: Frequency separations below this fraction of a bin are treated as coincident.
_DEGENERATE = 0.05

#: Bin offsets that make each integration level integrate exactly (see _deposit).
_JUMP_SHIFT = 0.5
_SLOPE_SHIFT = 1.0


def _deposit(
    target: np.ndarray,
    position: np.ndarray,
    value: np.ndarray,
    x0: float,
    dx: float,
    shift: float = 0.0,
    spill: np.ndarray | None = None,
    drop_below: bool = False,
):
    """Add ``value`` at ``position`` with linear interpolation between bins.

    ``shift`` moves the deposit by that many bins before interpolating, which is
    how each integration level is made to integrate exactly.  Bin ``k`` holds
    the *average* density over ``[x_k - dx/2, x_k + dx/2]``, so:

    * a delta needs no shift -- splitting it between two bins conserves area;
    * a step of height ``J`` at ``p`` fills only the part of its own bin above
      ``p``, which is reproduced by shifting it half a bin right;
    * a slope change ``m`` at ``p`` must build the ramp ``m (x - p)``, and the
      double cumulative sum starts that ramp one bin early unless it is shifted
      a full bin right.

    Without these shifts a lineshape comes out displaced by a bin and every
    vertical edge carries a half-bin area error.

    Deposits past the right edge are discarded: the lineshape is rebuilt by a
    left-to-right cumulative sum, so they cannot reach the visible axis.
    Deposits *before* the left edge still matter and are folded in exactly.  A
    slope change at ``p < x0`` contributes slope ``m`` from bin 0 plus a
    constant ``m (x0 - p)``, which is written into ``spill``, the array
    integrated one time fewer.  A step before the left edge is already a
    constant, and a delta before it is off screen and dropped.
    """
    if position.size == 0:
        return
    exact = (position - x0) / dx
    u = exact + shift
    lo = np.floor(u).astype(np.int64)
    frac = u - lo
    keep = lo < target.size - 1
    if drop_below:
        keep &= lo >= 0
    lo, frac, val, exact = lo[keep], frac[keep], value[keep], exact[keep]

    below = lo < 0
    if spill is not None and np.any(below):
        np.add.at(spill, 0, float(np.sum(val[below] * (-exact[below]) * dx)))
    lo = np.where(below, 0, lo)
    frac = np.where(below, 0.0, frac)
    np.add.at(target, lo, val * (1.0 - frac))
    np.add.at(target, lo + 1, val * frac)


def accumulate_asg(
    freqs: np.ndarray,
    amps: np.ndarray,
    triangles: np.ndarray,
    weights: np.ndarray,
    x0: float,
    dx: float,
    n_bins: int,
) -> np.ndarray:
    """Bin ASG triangles into a histogram of intensity per bin.

    Parameters
    ----------
    freqs, amps
        ``(n_orientations, n_transitions)`` frequencies and transition
        probabilities.
    triangles
        ``(n_triangles, 3)`` indices into the orientation axis.
    weights
        ``(n_triangles,)`` solid-angle fractions.
    x0, dx, n_bins
        Uniform frequency axis: bin ``k`` is centred at ``x0 + k * dx``.
    """
    freqs = np.asarray(freqs, dtype=float)
    amps = np.asarray(amps, dtype=float)

    # (n_triangles, 3, n_transitions) -> flatten triangle and transition axes.
    corner_f = freqs[triangles]
    corner_a = amps[triangles]
    n_tri, _, n_trans = corner_f.shape
    f = np.sort(corner_f.transpose(0, 2, 1).reshape(-1, 3), axis=1)
    total = (
        corner_a.mean(axis=1).reshape(-1)
        * np.repeat(weights, n_trans)
    )

    f1, f2, f3 = f[:, 0], f[:, 1], f[:, 2]
    span = f3 - f1
    tol = _DEGENERATE * dx

    slope = np.zeros(n_bins + 2)   # integrated twice
    jump = np.zeros(n_bins + 2)    # integrated once
    direct = np.zeros(n_bins + 2)  # added as-is


    is_point = span <= tol
    if np.any(is_point):
        _deposit(
            direct,
            0.5 * (f1[is_point] + f3[is_point]),
            total[is_point],
            x0,
            dx,
            drop_below=True,
        )

    rest = ~is_point
    if np.any(rest):
        a, b, c = f1[rest], f2[rest], f3[rest]
        w = total[rest]
        width = c - a
        height = 2.0 / width  # peak of a unit-area triangular density
        left_flat = (b - a) <= tol
        right_flat = (c - b) <= tol
        general = ~(left_flat | right_flat)

        # Right triangle with its vertical edge at ``a``.
        if np.any(left_flat):
            edge = height[left_flat] * w[left_flat]
            m = height[left_flat] / width[left_flat] * w[left_flat]
            _deposit(jump, a[left_flat], edge, x0, dx, shift=_JUMP_SHIFT)
            _deposit(slope, a[left_flat], -m, x0, dx, shift=_SLOPE_SHIFT, spill=jump)
            _deposit(slope, c[left_flat], m, x0, dx, shift=_SLOPE_SHIFT, spill=jump)

        # Right triangle with its vertical edge at ``c``.
        if np.any(right_flat):
            edge = height[right_flat] * w[right_flat]
            m = height[right_flat] / width[right_flat] * w[right_flat]
            _deposit(slope, a[right_flat], m, x0, dx, shift=_SLOPE_SHIFT, spill=jump)
            _deposit(slope, c[right_flat], -m, x0, dx, shift=_SLOPE_SHIFT, spill=jump)
            _deposit(jump, c[right_flat], -edge, x0, dx, shift=_JUMP_SHIFT)

        if np.any(general):
            ag, bg, cg = a[general], b[general], c[general]
            hg, wg = height[general], w[general]
            m1 = hg / (bg - ag) * wg
            m2 = hg / (cg - bg) * wg
            _deposit(slope, ag, m1, x0, dx, shift=_SLOPE_SHIFT, spill=jump)
            _deposit(slope, bg, -(m1 + m2), x0, dx, shift=_SLOPE_SHIFT, spill=jump)
            _deposit(slope, cg, m2, x0, dx, shift=_SLOPE_SHIFT, spill=jump)

    density = np.cumsum(np.cumsum(slope) * dx + jump)
    return (density * dx + direct)[:n_bins]


def accumulate_sticks(
    freqs: np.ndarray,
    amps: np.ndarray,
    weights: np.ndarray,
    x0: float,
    dx: float,
    n_bins: int,
) -> np.ndarray:
    """Bin one stick per orientation and transition, no interpolation."""
    freqs = np.asarray(freqs, dtype=float)
    amps = np.asarray(amps, dtype=float)
    values = (amps * np.asarray(weights)[:, None]).reshape(-1)
    target = np.zeros(n_bins + 2)
    _deposit(target, freqs.reshape(-1), values, x0, dx, drop_below=True)
    return target[:n_bins]


def broaden(
    spectrum: np.ndarray,
    dx: float,
    lorentz_fwhm: float = 0.0,
    gauss_fwhm: float = 0.0,
) -> np.ndarray:
    """Convolve with a Lorentzian and/or Gaussian of the given FWHM.

    Widths are in the same units as ``dx``.  Zero widths return the input
    untouched.  The transform is zero padded, so no intensity wraps around.
    """
    if lorentz_fwhm < 0 or gauss_fwhm < 0:
        raise ValueError("broadening widths must be non-negative")
    if lorentz_fwhm == 0 and gauss_fwhm == 0:
        return np.asarray(spectrum, dtype=float)

    n = spectrum.size
    padded = next_fast_len(2 * n)
    ft = rfft(spectrum, n=padded)
    t = np.fft.rfftfreq(padded, d=dx)
    transfer = np.ones_like(t)
    if lorentz_fwhm > 0:
        transfer = transfer * np.exp(-np.pi * lorentz_fwhm * np.abs(t))
    if gauss_fwhm > 0:
        transfer = transfer * np.exp(
            -(np.pi**2) * gauss_fwhm**2 * t**2 / (4.0 * np.log(2.0))
        )
    return irfft(ft * transfer, n=padded)[:n]


def voigt_kernel(
    x: np.ndarray, lorentz_fwhm: float = 0.0, gauss_fwhm: float = 0.0
) -> np.ndarray:
    """Unit-area Voigt profile centred at zero, for plotting and tests."""
    x = np.asarray(x, dtype=float)
    dx = float(x[1] - x[0])
    delta = np.zeros_like(x)
    delta[np.argmin(np.abs(x))] = 1.0
    return broaden(delta, dx, lorentz_fwhm, gauss_fwhm) / dx
