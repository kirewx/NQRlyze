"""Closed-form results used to prove the simulator right.

These are independent of the simulation path: they come from the literature (or
from diagonalising the bare quadrupolar Hamiltonian) rather than from the powder
machinery, so agreement with them pins down the Hamiltonian, the units and the
frequency conventions without needing QUEST -- or anything else -- to compare
against.

The second-order coefficients below were checked against exact diagonalisation
term by term; note that the sign of the ``eta cos 2phi`` term in ``C`` is the
one that reproduces the exact result, and is printed with the other sign in some
of the literature.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .hamiltonian import quadrupolar_hamiltonian

__all__ = [
    "cq_range_for_line",
    "nu_q",
    "nqr_frequencies",
    "nqr_frequency_spin_three_halves",
    "second_order_ct_shift",
    "second_order_ct_isotropic_shift",
    "second_order_ct_extremes",
]


def nu_q(spin: float, cq: float) -> float:
    """Quadrupolar frequency ``nu_Q = 3 Cq / (2I(2I-1))``."""
    return 3.0 * cq / (2.0 * spin * (2.0 * spin - 1.0))


def nqr_frequencies(spin: float, cq: float, eta: float) -> np.ndarray:
    """Exact zero-field transition frequencies, ascending, in MHz.

    Obtained by diagonalising the quadrupolar Hamiltonian alone.  Levels are
    Kramers doublets, so the distinct frequencies are the differences between
    doublet energies.
    """
    energies = np.sort(np.linalg.eigvalsh(quadrupolar_hamiltonian(spin, cq, eta)))
    # Merge Kramers doublets with a tolerance relative to Cq -- rounding to a
    # fixed number of decimals fails once Cq reaches hundreds of MHz.
    tolerance = 1e-9 * max(abs(cq), 1.0)
    levels = [energies[0]]
    for value in energies[1:]:
        if value - levels[-1] > tolerance:
            levels.append(value)
    levels = np.array(levels)
    diffs = [
        levels[j] - levels[i]
        for i in range(len(levels))
        for j in range(i + 1, len(levels))
    ]
    return np.array(sorted(diffs))


def nqr_frequency_spin_three_halves(cq: float, eta: float) -> float:
    """``nu = (Cq/2) sqrt(1 + eta^2/3)`` -- the textbook I = 3/2 NQR line.

    A single line cannot separate ``Cq`` from ``eta``: only this combination is
    observable.  Fitting both to one I = 3/2 NQR resonance is therefore
    ill-posed, and :mod:`nqrlyze` will happily report a perfectly correlated
    pair unless one of them is fixed or a second field is measured.
    """
    return 0.5 * cq * np.sqrt(1.0 + eta**2 / 3.0)


def _abc(eta: float, phi):
    cos2 = np.cos(2.0 * np.asarray(phi))
    a = -27.0 / 8.0 + 9.0 / 4.0 * eta * cos2 - 3.0 / 8.0 * eta**2 * cos2**2
    b = (
        30.0 / 8.0
        - 0.5 * eta**2
        - 2.0 * eta * cos2
        + 0.75 * eta**2 * cos2**2
    )
    c = (
        -3.0 / 8.0
        + eta**2 / 3.0
        - 0.25 * eta * cos2
        - 3.0 / 8.0 * eta**2 * cos2**2
    )
    return a, b, c


def second_order_ct_shift(
    spin: float, cq: float, eta: float, larmor: float, theta, phi
) -> np.ndarray:
    """Second-order shift of the central transition, in MHz.

    ``theta`` and ``phi`` (radians) orient the magnetic field in the EFG frame:
    ``theta`` from ``Vzz``, ``phi`` from ``Vxx``.  Valid for half-integer spin
    in the high-field limit.
    """
    a, b, c = _abc(eta, phi)
    cos_theta = np.cos(np.asarray(theta))
    prefactor = -(nu_q(spin, cq) ** 2 / (6.0 * larmor)) * (spin * (spin + 1) - 0.75)
    return prefactor * (a * cos_theta**4 + b * cos_theta**2 + c)


def second_order_ct_isotropic_shift(
    spin: float, cq: float, eta: float, larmor: float
) -> float:
    """Isotropic part of the second-order shift, in MHz.

    ``-(nu_Q^2 / 30 nu_L)(I(I+1) - 3/4)(1 + eta^2/3)`` -- the shift that
    survives magic-angle spinning, and the sphere average of
    :func:`second_order_ct_shift`.
    """
    return (
        -(nu_q(spin, cq) ** 2 / (30.0 * larmor))
        * (spin * (spin + 1) - 0.75)
        * (1.0 + eta**2 / 3.0)
    )


def second_order_ct_extremes(
    spin: float, cq: float, eta: float, larmor: float, samples: int = 2001
) -> tuple[float, float]:
    """Lowest and highest second-order shift over all orientations, in MHz.

    These bracket the central-transition powder pattern, so they are a sharp
    check on the *width* a simulation produces, independent of intensities.
    """
    cos_theta = np.linspace(0.0, 1.0, samples)
    phi = np.linspace(0.0, np.pi / 2.0, samples)
    a, b, c = _abc(eta, phi)
    values = (
        a[:, None] * cos_theta[None, :] ** 4
        + b[:, None] * cos_theta[None, :] ** 2
        + c[:, None]
    )
    prefactor = -(nu_q(spin, cq) ** 2 / (6.0 * larmor)) * (spin * (spin + 1) - 0.75)
    scaled = prefactor * values
    return float(scaled.min()), float(scaled.max())


def cq_range_for_line(
    spin: float,
    frequency: float,
    eta_range: tuple[float, float] = (0.0, 1.0),
    transitions: int | Sequence[int] | None = None,
    samples: int = 201,
) -> tuple[float, float]:
    """Bracket ``Cq`` from one observed zero-field line.

    Zero-field transition frequencies are strictly proportional to ``Cq``, so a
    line at ``frequency`` fixes ``Cq`` up to the unknown ``eta`` and the unknown
    identity of the transition.  This returns the smallest and largest ``Cq``
    consistent with the line over the given ``eta`` range.

    Use it to set the bounds of an NQR fit.  An NQR resonance is typically
    thousands of times narrower than the range of ``Cq`` a search would
    otherwise have to cover, and a global optimiser cannot find a needle that
    sharp -- give it this bracket and it converges immediately.

    Parameters
    ----------
    transitions
        Which transition the line is, indexed in order of ascending frequency
        at fixed ``eta``.  ``None`` considers all of them, which is the
        conservative choice when the assignment is unknown.
    """
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    low, high = sorted(eta_range)
    if not (0.0 <= low <= high <= 1.0):
        raise ValueError("eta_range must lie within [0, 1]")

    if isinstance(transitions, int):
        wanted = [transitions]
    elif transitions is None:
        wanted = None
    else:
        wanted = list(transitions)

    candidates: list[float] = []
    for eta in np.linspace(low, high, samples):
        # Frequencies scale linearly with Cq, so a unit Cq gives the shape.
        unit = nqr_frequencies(spin, 1.0, float(eta))
        unit = unit[unit > 1e-12]
        chosen = unit if wanted is None else [unit[i] for i in wanted]
        candidates.extend(frequency / value for value in chosen)
    if not candidates:
        raise ValueError("no zero-field transitions for this spin")
    return float(min(candidates)), float(max(candidates))
