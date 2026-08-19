"""Combining sub-spectra into one pattern before fitting.

Wideline quadrupolar patterns are usually too broad to excite in one go, so they
are recorded piecewise -- stepped transmitter offsets (VOCS/WURST-QCPMG pieces),
or simply several windows -- and the pieces are joined afterwards.  Because
:mod:`nqrlyze` keeps every spectrum on an absolute MHz axis, joining them is
exact: each piece is interpolated onto a common grid and combined.

``sum``
    Plain co-addition.  The right choice when the pieces were acquired
    identically and each carries the same intensity scale.
``mean``
    Co-addition divided by how many pieces cover each point, so that regions
    measured twice are not twice as tall.  Use when the offset steps overlap
    unevenly.
``skyline``
    Point-by-point maximum -- the traditional way to present stepped-frequency
    data, but it distorts intensities and should not be fitted quantitatively.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .spectrum import Spectrum

__all__ = ["coadd", "common_axis"]

_MODES = ("sum", "mean", "skyline")


def common_axis(spectra: Sequence[Spectrum], dx_mhz: float | None = None) -> np.ndarray:
    """Uniform axis spanning every spectrum, at the finest spacing present."""
    if not spectra:
        raise ValueError("no spectra given")
    low = min(float(s.freq_mhz[0]) for s in spectra)
    high = max(float(s.freq_mhz[-1]) for s in spectra)
    if dx_mhz is None:
        dx_mhz = min(
            float(np.abs(np.median(np.diff(s.freq_mhz))))
            for s in spectra
            if s.freq_mhz.size > 1
        )
    if dx_mhz <= 0:
        raise ValueError("step size must be positive")
    n = int(np.floor((high - low) / dx_mhz)) + 1
    return low + dx_mhz * np.arange(n)


def coadd(
    spectra: Sequence[Spectrum],
    mode: str = "sum",
    dx_mhz: float | None = None,
    weights: Sequence[float] | None = None,
    normalize_each: bool = False,
) -> Spectrum:
    """Join sub-spectra onto one absolute frequency axis.

    Parameters
    ----------
    spectra
        The pieces, in any order.
    mode
        ``"sum"``, ``"mean"`` or ``"skyline"`` -- see the module docstring.
    dx_mhz
        Step of the output axis; defaults to the finest step among the inputs.
    weights
        Per-piece scale factors applied before combining.
    normalize_each
        Scale every piece to unit peak height first.  Convenient for display,
        but it destroys the relative intensities a quantitative fit relies on.
    """
    spectra = list(spectra)
    if not spectra:
        raise ValueError("no spectra given")
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if weights is None:
        weights = [1.0] * len(spectra)
    if len(weights) != len(spectra):
        raise ValueError("weights and spectra must have the same length")

    axis = common_axis(spectra, dx_mhz)
    stack = np.zeros((len(spectra), axis.size))
    coverage = np.zeros(axis.size)

    for k, (spec, weight) in enumerate(zip(spectra, weights)):
        piece = spec.normalized() if normalize_each else spec
        stack[k] = np.interp(
            axis, piece.freq_mhz, piece.intensity, left=0.0, right=0.0
        ) * weight
        inside = (axis >= piece.freq_mhz[0]) & (axis <= piece.freq_mhz[-1])
        coverage += inside

    if mode == "skyline":
        combined = stack.max(axis=0)
    else:
        combined = stack.sum(axis=0)
        if mode == "mean":
            combined = np.divide(
                combined, coverage, out=np.zeros_like(combined), where=coverage > 0
            )

    references = {s.reference for s in spectra if s.reference}
    reference = references.pop() if len(references) == 1 else 0.0
    meta = {
        "coadd_mode": mode,
        "n_pieces": len(spectra),
        "coverage": coverage,
        "sources": [s.meta.get("source", "") for s in spectra],
    }
    return Spectrum(axis, combined, reference, meta)
