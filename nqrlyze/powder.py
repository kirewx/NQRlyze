"""Powder averaging grids.

The default is the interpolation scheme of Alderman, Solum and Grant
(*J. Chem. Phys.* **1986**, 84, 3717), which is what makes QUEST fast: one
octant of the unit sphere is tessellated by projecting a triangular grid from
the face of an octahedron, and each triangle contributes a *continuous* band of
intensity rather than a single stick.  A few hundred orientations then give
lineshapes whose singularities are as sharp as the frequency grid allows.

A brute-force spiral ("golden section") grid is provided as an independent
reference for tests -- it needs orders of magnitude more orientations for the
same lineshape quality, which is exactly why the ASG scheme is worth having.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PowderGrid", "asg_grid", "spiral_grid"]

#: Sign patterns that tile the unit sphere.  A Zeeman + quadrupolar Hamiltonian
#: is invariant under each of the three pi-rotations about the EFG principal
#: axes and, by time reversal, under ``n -> -n``; one octant is therefore
#: enough.  A chemical shift tensor that is not diagonal in the EFG frame
#: breaks the pi-rotation symmetry but not the ``n -> -n`` symmetry, so four
#: octants are needed and eight are never required.
_OCTANTS = np.array(
    [
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
)


@dataclass(frozen=True)
class PowderGrid:
    """Orientations plus the triangles that interpolate between them.

    Attributes
    ----------
    directions
        ``(N, 3)`` unit vectors in the EFG principal axis system.
    triangles
        ``(T, 3)`` integer indices into ``directions``.  Empty for grids that
        do not support interpolation.
    weights
        ``(T,)`` solid-angle fraction of each triangle, summing to 1.  For a
        non-interpolating grid this is ``(N,)`` per-orientation weights.
    interpolated
        Whether ``triangles`` should be used.
    """

    directions: np.ndarray
    triangles: np.ndarray
    weights: np.ndarray
    interpolated: bool

    @property
    def n_orientations(self) -> int:
        return self.directions.shape[0]


def _octant_tessellation(divisions: int):
    """Vertices on the plane ``x + y + z = 1`` and the triangles joining them."""
    if divisions < 1:
        raise ValueError("divisions must be >= 1")
    n = divisions
    index = -np.ones((n + 1, n + 1), dtype=int)
    points = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            index[a, b] = len(points)
            points.append((a, b, n - a - b))
    verts = np.array(points, dtype=float) / n

    tris = []
    for a in range(n):
        for b in range(n - a):
            tris.append((index[a, b], index[a + 1, b], index[a, b + 1]))
    for a in range(n - 1):
        for b in range(n - 1 - a):
            tris.append((index[a + 1, b], index[a, b + 1], index[a + 1, b + 1]))
    return verts, np.array(tris, dtype=int)


def asg_grid(divisions: int = 22, octants: int = 1) -> PowderGrid:
    """Alderman-Solum-Grant interpolation grid.

    Parameters
    ----------
    divisions
        Number of subdivisions along each edge of the octahedral face.  The
        octant carries ``(divisions + 1)(divisions + 2) / 2`` orientations and
        ``divisions**2`` triangles.
    octants
        1 for a Hamiltonian diagonal in the EFG frame, 4 when a chemical shift
        tensor is tilted away from it.

    Notes
    -----
    Radial projection from the plane ``x + y + z = 1`` onto the sphere carries a
    Jacobian ``d(Omega) = d / |p|**3 dA`` with ``d = 1/sqrt(3)`` the distance
    from the origin to the plane; since every triangle has the same area, the
    solid angle of a triangle is proportional to ``<1 / |p|**3>`` over it.
    """
    if octants not in (1, 4):
        raise ValueError("octants must be 1 or 4")
    verts, tris = _octant_tessellation(divisions)

    radii = np.linalg.norm(verts, axis=1)
    unit = verts / radii[:, None]
    # Solid-angle weight per triangle, averaged over its three corners.
    weights = np.mean(1.0 / radii[tris] ** 3, axis=1)

    if octants == 1:
        directions = unit
        triangles = tris
        tri_weights = weights
    else:
        n_vert = unit.shape[0]
        directions = np.concatenate([unit * s for s in _OCTANTS[:octants]])
        triangles = np.concatenate(
            [tris + k * n_vert for k in range(octants)]
        )
        tri_weights = np.tile(weights, octants)

    tri_weights = tri_weights / tri_weights.sum()
    return PowderGrid(directions, triangles, tri_weights, interpolated=True)


def spiral_grid(n_orientations: int = 20000, hemisphere: bool = True) -> PowderGrid:
    """Golden-section spiral over the sphere; equal weights, no interpolation.

    Slow to converge, but free of the assumptions ASG makes -- used in the test
    suite as an independent check of the interpolated lineshapes.
    """
    if n_orientations < 1:
        raise ValueError("n_orientations must be >= 1")
    k = np.arange(n_orientations) + 0.5
    z_span = 1.0 if hemisphere else 2.0
    z = 1.0 - z_span * k / n_orientations
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    phi = np.pi * (1.0 + np.sqrt(5.0)) * k
    directions = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    weights = np.full(n_orientations, 1.0 / n_orientations)
    return PowderGrid(
        directions, np.empty((0, 3), dtype=int), weights, interpolated=False
    )
