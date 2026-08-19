"""Powder lineshape simulation for one or more quadrupolar sites.

Frequency axes are **absolute, in MHz**.  That is the only convention that
serves NMR and NQR equally well: a Bruker axis converts to it exactly, and in
pure NQR there is no reference frequency to offset from.  Helpers convert to and
from the ppm and kHz-offset scales people actually plot.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

import numpy as np

from .constants import get_nucleus
from .hamiltonian import eigen_transitions, shift_tensor
from .lineshape import accumulate_asg, accumulate_sticks, broaden
from .powder import PowderGrid, asg_grid

__all__ = [
    "Site",
    "Experiment",
    "simulate",
    "simulate_sites",
    "suggest_window",
    "ppm_axis",
    "khz_axis",
]


@dataclass
class Site:
    """One crystallographic site.

    Frequencies are in MHz, shifts and shift-tensor parameters in ppm, angles in
    degrees.  ``lorentz`` and ``gauss`` are FWHM values in MHz.
    """

    cq: float = 0.0
    eta: float = 0.0
    iso: float = 0.0
    span: float = 0.0
    skew: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    lorentz: float = 0.0
    gauss: float = 0.0
    weight: float = 1.0
    label: str = ""

    def shift_matrix(self) -> np.ndarray | None:
        """Chemical shift tensor in the EFG frame, or ``None`` if there is none."""
        if self.iso == 0.0 and self.span == 0.0:
            return None
        return shift_tensor(
            self.iso, self.span, self.skew, self.alpha, self.beta, self.gamma
        )


@dataclass
class Experiment:
    """What was measured: nucleus, field, and which transitions are observed."""

    spin: float
    larmor: float = 0.0
    """Larmor frequency in MHz.  ``0`` means pure NQR."""
    reference: float | None = None
    """Frequency of 0 ppm, in MHz.  Defaults to ``larmor``."""
    transitions: str = "all"
    rf_average: str = "auto"
    nucleus: str = ""

    @classmethod
    def from_nucleus(
        cls,
        symbol: str,
        field: float | None = None,
        larmor: float | None = None,
        **kwargs,
    ) -> "Experiment":
        """Build from an isotope symbol plus either a field (T) or a Larmor (MHz)."""
        nuc = get_nucleus(symbol)
        if field is not None and larmor is not None:
            raise ValueError("give either field or larmor, not both")
        if field is not None:
            larmor = nuc.larmor(field)
        elif larmor is None:
            larmor = 0.0
        return cls(spin=nuc.spin, larmor=larmor, nucleus=nuc.symbol, **kwargs)

    @property
    def reference_frequency(self) -> float:
        return self.larmor if self.reference is None else self.reference


def ppm_axis(freq_mhz: np.ndarray, reference: float) -> np.ndarray:
    """Absolute MHz -> ppm relative to ``reference`` (MHz)."""
    if reference == 0:
        raise ValueError("a ppm scale needs a non-zero reference frequency")
    return (np.asarray(freq_mhz) - reference) / reference * 1e6


def khz_axis(freq_mhz: np.ndarray, reference: float) -> np.ndarray:
    """Absolute MHz -> kHz offset from ``reference`` (MHz)."""
    return (np.asarray(freq_mhz) - reference) * 1e3


def _needs_four_octants(sites: Sequence[Site]) -> bool:
    for site in sites:
        mat = site.shift_matrix()
        if mat is None:
            continue
        if np.max(np.abs(mat - np.diag(np.diag(mat)))) > 1e-12 * max(
            1.0, np.max(np.abs(mat))
        ):
            return True
    return False


def _internal_axis(freq_mhz: np.ndarray, sites: Sequence[Site]):
    """Uniform internal grid covering the requested window plus a broadening margin."""
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    if freq_mhz.size < 2:
        raise ValueError("the frequency axis needs at least two points")
    diffs = np.diff(freq_mhz)
    if np.any(diffs <= 0) and np.any(diffs >= 0):
        raise ValueError("the frequency axis must be monotonic")
    dx = float(np.abs(np.median(diffs)))
    if dx <= 0:
        raise ValueError("the frequency axis has zero spacing")

    # Broadening pulls intensity from outside the window in.  The histogram
    # itself handles the window edge exactly, so the margin only has to cover
    # the convolution: a Gaussian is negligible past a few widths, but a
    # Lorentzian falls off as 1/x^2 and needs tens of them to reach 1e-4.
    lorentz = max((s.lorentz for s in sites), default=0.0)
    gauss = max((s.gauss for s in sites), default=0.0)
    margin = 50.0 * lorentz + 6.0 * gauss + 10.0 * dx
    n_max = 4_000_000
    lo_guess = float(np.min(freq_mhz)) - margin
    hi_guess = float(np.max(freq_mhz)) + margin
    if (hi_guess - lo_guess) / dx > n_max:
        margin = max(0.5 * (n_max * dx - np.ptp(freq_mhz)), 10.0 * dx)
    lo = float(np.min(freq_mhz)) - margin
    hi = float(np.max(freq_mhz)) + margin
    n_bins = int(np.ceil((hi - lo) / dx)) + 1
    return lo, dx, n_bins


def simulate_sites(
    freq_mhz: np.ndarray,
    sites: Sequence[Site],
    experiment: Experiment,
    grid: PowderGrid | None = None,
    divisions: int = 30,
    normalize: bool = False,
) -> np.ndarray:
    """Simulate each site separately.

    Returns ``(n_sites, len(freq_mhz))`` of intensity density (per MHz), with
    each site's ``weight`` already applied.  Keeping the sites apart is what
    lets the fitter solve their amplitudes by linear least squares.
    """
    sites = list(sites)
    if not sites:
        raise ValueError("at least one site is required")
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    flipped = freq_mhz[0] > freq_mhz[-1]
    query = freq_mhz[::-1] if flipped else freq_mhz

    x0, dx, n_bins = _internal_axis(query, sites)
    internal = x0 + dx * np.arange(n_bins)

    if grid is None:
        octants = 4 if _needs_four_octants(sites) else 1
        grid = asg_grid(divisions, octants=octants)

    out = np.empty((len(sites), query.size))
    for k, site in enumerate(sites):
        freqs, amps = eigen_transitions(
            grid.directions,
            experiment.spin,
            site.cq,
            site.eta,
            experiment.larmor,
            shift_ppm=site.shift_matrix(),
            reference=experiment.reference_frequency,
            transitions=experiment.transitions,
            rf_average=experiment.rf_average,
        )
        if grid.interpolated:
            hist = accumulate_asg(
                freqs, amps, grid.triangles, grid.weights, x0, dx, n_bins
            )
        else:
            hist = accumulate_sticks(freqs, amps, grid.weights, x0, dx, n_bins)

        density = hist / dx
        if site.lorentz > 0 or site.gauss > 0:
            density = broaden(density, dx, site.lorentz, site.gauss)
        if normalize:
            area = np.trapezoid(density, internal)
            if area > 0:
                density = density / area
        out[k] = np.interp(query, internal, density, left=0.0, right=0.0) * site.weight

    return out[:, ::-1] if flipped else out


def simulate(
    freq_mhz: np.ndarray,
    sites: Sequence[Site] | Site,
    experiment: Experiment,
    grid: PowderGrid | None = None,
    divisions: int = 30,
    normalize: bool = False,
) -> np.ndarray:
    """Total powder lineshape, summed over sites."""
    if isinstance(sites, Site):
        sites = [sites]
    return simulate_sites(
        freq_mhz, sites, experiment, grid, divisions, normalize
    ).sum(axis=0)


def suggest_window(
    sites: Sequence[Site] | Site,
    experiment: Experiment,
    divisions: int = 10,
    padding: float = 5.0,
) -> tuple[float, float]:
    """A frequency window (MHz) that contains the whole pattern.

    Finds the extreme transition frequencies over a coarse powder grid and pads
    them for broadening.  Saves guessing a centre and a spectral width by hand,
    which is otherwise the first thing that goes wrong when the coupling is
    large or the spin is high.
    """
    if isinstance(sites, Site):
        sites = [sites]
    sites = list(sites)
    if not sites:
        raise ValueError("at least one site is required")

    grid = asg_grid(divisions)
    low, high = np.inf, -np.inf
    for site in sites:
        freqs, amps = eigen_transitions(
            grid.directions,
            experiment.spin,
            site.cq,
            site.eta,
            experiment.larmor,
            shift_ppm=site.shift_matrix(),
            reference=experiment.reference_frequency,
            transitions=experiment.transitions,
            rf_average=experiment.rf_average,
        )
        peak = float(np.max(amps)) if amps.size else 0.0
        visible = freqs[amps > 1e-6 * peak] if peak > 0 else freqs
        if visible.size:
            low = min(low, float(visible.min()))
            high = max(high, float(visible.max()))

    if not np.isfinite(low) or not np.isfinite(high):
        centre = experiment.larmor or 1.0
        return centre * 0.99, centre * 1.01

    widest = max((s.lorentz + s.gauss for s in sites), default=0.0)
    margin = padding * widest + 0.08 * (high - low)
    if margin <= 0:
        margin = max(0.01 * max(abs(low), abs(high)), 1e-3)
    return low - margin, high + margin
