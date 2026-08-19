"""Exact Zeeman + quadrupolar (+ chemical shift) Hamiltonian.

Everything is expressed in the **principal axis system of the electric field
gradient** (EFG PAS).  Instead of rotating the quadrupolar tensor into the lab
frame we rotate the magnetic field direction into the EFG PAS, which needs only
a unit vector ``n`` per crystallite and keeps the quadrupolar term diagonal in
its own natural form.  Powder averaging is therefore an average over
orientations of ``n`` on the unit sphere.

Conventions (see ``docs/conventions.md``)
-----------------------------------------
Energies and frequencies are in **MHz**.  With ``nu_L`` the Larmor frequency,
``Cq = e*Q*Vzz/h`` and ``eta = (Vxx - Vyy)/Vzz`` with
``|Vzz| >= |Vyy| >= |Vxx|``::

    H/h = -nu_L * (n . I)
          + Cq / (4I(2I-1)) * [ 3Iz^2 - I(I+1) + eta*(Ix^2 - Iy^2) ]
          - nu_ref * 1e-6 * (n . delta . n) * (n . I)

The last (chemical shift) term uses the IUPAC shift convention: a *larger*
``delta`` moves a resonance to *higher* frequency.  It vanishes identically when
``nu_ref = 0``, i.e. in pure NQR.

A transition frequency is ``E_j - E_i`` for ``j > i`` with the eigenvalues
sorted ascending, i.e. the energy absorbed -- always a positive number.  In the
high-field limit that is ``nu_L`` plus the quadrupolar/shift corrections, and
``simulate`` reports the offset from ``nu_L`` unless asked for absolute
frequencies.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "spin_operators",
    "quadrupolar_hamiltonian",
    "shift_tensor",
    "euler_zyz",
    "transition_indices",
    "eigen_transitions",
]


def spin_operators(spin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(Ix, Iy, Iz)`` for spin ``I`` in the ``|I, m>`` basis.

    The basis is ordered by *descending* ``m``: index 0 is ``m = +I``.
    """
    two_i = round(2 * spin)
    if two_i < 1 or abs(2 * spin - two_i) > 1e-9:
        raise ValueError(f"spin must be a positive multiple of 1/2, got {spin}")
    dim = two_i + 1
    m = np.arange(spin, -spin - 0.5, -1.0)
    iz = np.diag(m).astype(complex)
    ip = np.zeros((dim, dim), dtype=complex)
    for k in range(1, dim):
        ip[k - 1, k] = np.sqrt(spin * (spin + 1) - m[k] * (m[k] + 1))
    im = ip.conj().T
    ix = (ip + im) / 2.0
    iy = (ip - im) / 2.0j
    return ix, iy, iz


def quadrupolar_hamiltonian(spin: float, cq: float, eta: float) -> np.ndarray:
    """First-principles quadrupolar Hamiltonian in the EFG PAS, in MHz.

    ``cq`` is in MHz.  Spin-1/2 has no quadrupole moment and yields zeros.
    """
    ix, iy, iz = spin_operators(spin)
    dim = ix.shape[0]
    if spin < 1.0:
        return np.zeros((dim, dim), dtype=complex)
    scale = cq / (4.0 * spin * (2.0 * spin - 1.0))
    return scale * (
        3.0 * (iz @ iz)
        - spin * (spin + 1.0) * np.eye(dim, dtype=complex)
        + eta * (ix @ ix - iy @ iy)
    )


def euler_zyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Rotation matrix ``Rz(alpha) Ry(beta) Rz(gamma)``, angles in degrees.

    Used to carry a tensor from its own PAS into the EFG PAS via
    ``T_efg = R @ T_pas @ R.T``.
    """
    a, b, c = np.radians([alpha, beta, gamma])

    def rz(t):
        return np.array(
            [[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]]
        )

    def ry(t):
        return np.array(
            [[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]]
        )

    return rz(a) @ ry(b) @ rz(c)


def shift_tensor(
    iso: float,
    span: float = 0.0,
    skew: float = 0.0,
    alpha: float = 0.0,
    beta: float = 0.0,
    gamma: float = 0.0,
) -> np.ndarray:
    """Chemical shift tensor in the EFG PAS, in ppm.

    Uses the Herzfeld-Berger convention: ``span = d11 - d33 >= 0`` and
    ``skew = 3(d22 - iso)/span`` in ``[-1, 1]``, with ``d11 >= d22 >= d33``.
    The Euler angles rotate the shift PAS into the EFG PAS (ZYZ, degrees).
    """
    if span < 0:
        raise ValueError("span must be non-negative")
    skew = float(np.clip(skew, -1.0, 1.0))
    d22 = iso + span * skew / 3.0
    d11 = iso + span * (3.0 - skew) / 6.0
    d33 = iso - span * (3.0 + skew) / 6.0
    pas = np.diag([d11, d22, d33])
    rot = euler_zyz(alpha, beta, gamma)
    return rot @ pas @ rot.T


def transition_indices(spin: float, which: str | list[tuple[int, int]]):
    """Level-index pairs ``(i, j)``, ``i < j``, for the requested transitions.

    Levels are numbered by ascending energy, which in the high-field limit runs
    ``m = +I, +I-1, ... -I``.  ``which`` may be ``"all"``, ``"ct"`` (central
    transition, half-integer spin only), ``"satellites"``, ``"single"``
    (all ``|dm| = 1``), or an explicit list of pairs.
    """
    dim = round(2 * spin) + 1
    if isinstance(which, (list, tuple)) and not isinstance(which, str):
        pairs = [(int(a), int(b)) for a, b in which]
        for a, b in pairs:
            if not (0 <= a < b < dim):
                raise ValueError(f"transition {(a, b)} outside 0..{dim - 1}")
        return pairs
    which = which.lower()
    half_integer = abs(spin - round(spin)) > 1e-9
    single = [(k, k + 1) for k in range(dim - 1)]
    if which == "all":
        return [(a, b) for a in range(dim) for b in range(a + 1, dim)]
    if which == "single":
        return single
    if which == "ct":
        if not half_integer:
            raise ValueError("central transition requires half-integer spin")
        return [(dim // 2 - 1, dim // 2)]
    if which == "satellites":
        if not half_integer:
            return single
        return [p for p in single if p != (dim // 2 - 1, dim // 2)]
    raise ValueError(f"unknown transition selection {which!r}")


def eigen_transitions(
    directions: np.ndarray,
    spin: float,
    cq: float,
    eta: float,
    larmor: float,
    shift_ppm: np.ndarray | None = None,
    reference: float | None = None,
    transitions: str | list[tuple[int, int]] = "all",
    rf_average: str = "auto",
    drop_degenerate: bool | None = None,
    chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact transition frequencies and intensities for a set of orientations.

    Parameters
    ----------
    directions
        ``(N, 3)`` unit vectors: the magnetic field direction expressed in the
        EFG PAS.
    spin, cq, eta
        Nuclear spin, quadrupolar coupling constant in MHz, asymmetry.
    larmor
        Larmor frequency in MHz.  ``0`` gives pure NQR.
    shift_ppm
        Optional ``(3, 3)`` chemical shift tensor in the EFG PAS, in ppm.
    reference
        Reference frequency in MHz for the ppm scale; defaults to ``larmor``.
    transitions
        See :func:`transition_indices`.
    rf_average
        ``"perpendicular"`` averages the rf field over the plane normal to
        ``B0`` (the powder NMR case), ``"isotropic"`` over the full sphere (the
        NQR case).  ``"auto"`` picks by whether ``larmor`` is zero.
    drop_degenerate
        Zero the intensity of transitions *within* a degenerate level.  At zero
        field the levels are Kramers doublets, so pairing every level with every
        other produces same-energy pairs at zero frequency which carry large
        matrix elements but are not resonances.  Defaults to on when
        ``larmor`` is zero and off otherwise, where a genuine zero-frequency
        transition would be meaningful.

    Returns
    -------
    frequencies, intensities
        Both ``(N, n_transitions)``.  Frequencies are absolute (``E_j - E_i``)
        in MHz and always positive.  Intensities are the powder-averaged
        ``|<i| I.e |j>|^2`` transition probabilities.
    """
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape (N, 3)")
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("direction vectors must be non-zero")
    directions = directions / norms

    ix, iy, iz = spin_operators(spin)
    ops = np.stack([ix, iy, iz])  # (3, d, d)
    hq = quadrupolar_hamiltonian(spin, cq, eta)
    pairs = transition_indices(spin, transitions)
    ia = np.array([p[0] for p in pairs])
    ib = np.array([p[1] for p in pairs])

    if rf_average == "auto":
        rf_average = "isotropic" if larmor == 0.0 else "perpendicular"
    if rf_average not in ("isotropic", "perpendicular"):
        raise ValueError(f"unknown rf_average {rf_average!r}")

    ref = larmor if reference is None else reference
    if drop_degenerate is None:
        drop_degenerate = larmor == 0.0
    degeneracy_tol = 1e-9 * (abs(cq) + abs(larmor) + abs(ref) * 1e-6 + 1e-30)

    n_dir = directions.shape[0]
    freqs = np.empty((n_dir, len(pairs)))
    amps = np.empty((n_dir, len(pairs)))

    for start in range(0, n_dir, chunk):
        n = directions[start : start + chunk]  # (c, 3)
        # Zeeman (+ isotropic and anisotropic chemical shift) along n.
        weight = np.full(n.shape[0], larmor)
        if shift_ppm is not None and ref != 0.0:
            delta_lab = np.einsum("ci,ij,cj->c", n, shift_ppm, n)
            weight = weight + ref * 1e-6 * delta_lab
        # H = -weight * (n . I) + Hq
        n_dot_i = np.einsum("ck,kpq->cpq", n, ops)
        ham = -weight[:, None, None] * n_dot_i + hq[None]

        energies, vectors = np.linalg.eigh(ham)  # ascending
        # Transform the spin operators into the eigenbasis.  ``vectors[c, :, p]``
        # is eigenvector p, so u[k, c, p, q] = <p| I_k |q>.
        u = np.einsum("cip,kij,cjq->kcpq", vectors.conj(), ops, vectors)
        sel = u[:, :, ia, ib]  # (3, c, T)
        mod2 = np.abs(sel) ** 2
        total = mod2.sum(axis=0)  # |u|^2
        if rf_average == "isotropic":
            inten = total / 3.0
        else:
            along = np.abs(np.einsum("ck,kcT->cT", n, sel)) ** 2
            inten = 0.5 * (total - along)

        transition_freqs = energies[:, ib] - energies[:, ia]
        if drop_degenerate:
            inten = np.where(np.abs(transition_freqs) <= degeneracy_tol, 0.0, inten)
        freqs[start : start + chunk] = transition_freqs
        amps[start : start + chunk] = inten

    return freqs, amps
