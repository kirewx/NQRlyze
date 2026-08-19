"""Powder grids, interpolation and broadening."""

import numpy as np
import pytest

from nqrlyze.analytic import second_order_ct_extremes
from nqrlyze.lineshape import accumulate_asg, broaden, voigt_kernel
from nqrlyze.powder import asg_grid, spiral_grid
from nqrlyze.simulate import Experiment, Site, simulate


@pytest.mark.parametrize("divisions", [4, 10, 22, 40])
def test_asg_weights_sum_to_one(divisions):
    grid = asg_grid(divisions)
    assert grid.weights.sum() == pytest.approx(1.0)
    assert np.all(grid.weights > 0)
    assert np.allclose(np.linalg.norm(grid.directions, axis=1), 1.0)
    assert grid.triangles.shape == (divisions**2, 3)


@pytest.mark.parametrize("divisions", [10, 22, 40, 80])
def test_asg_weights_reproduce_spherical_averages(divisions):
    """<nz^2> = 1/3 exactly; higher moments converge with the grid."""
    grid = asg_grid(divisions)
    nz = grid.directions[:, 2]
    average = lambda f: float(np.sum(grid.weights * np.mean(f[grid.triangles], axis=1)))
    assert average(nz**2) == pytest.approx(1 / 3, abs=1e-12)
    assert average(nz**4) == pytest.approx(1 / 5, abs=20.0 / divisions**2)


def test_four_octant_grid_covers_the_sphere():
    grid = asg_grid(12, octants=4)
    single = asg_grid(12, octants=1)
    assert grid.n_orientations == 4 * single.n_orientations
    assert grid.weights.sum() == pytest.approx(1.0)
    # Four octants and their antipodes span every sign combination.
    signs = {tuple(np.sign(d).astype(int)) for d in grid.directions if np.all(d != 0)}
    assert len(signs) == 4


def test_asg_triangle_conserves_intensity():
    """One triangle deposits exactly its weight, wherever it lands."""
    x0, dx, n_bins = 0.0, 0.01, 4000
    for corners in ([3.0, 7.0, 12.0], [5.0, 5.0, 9.0], [4.0, 8.5, 8.5], [6.0, 6.0, 6.0]):
        freqs = np.array(corners).reshape(3, 1)
        amps = np.ones((3, 1))
        histogram = accumulate_asg(
            freqs, amps, np.array([[0, 1, 2]]), np.array([1.0]), x0, dx, n_bins
        )
        assert histogram.sum() == pytest.approx(1.0, rel=1e-6)
        assert np.all(histogram >= -1e-12)


def test_asg_triangle_has_the_expected_shape():
    """A linear frequency over a uniform triangle gives a triangular density."""
    x0, dx, n_bins = 0.0, 0.002, 6000
    a, b, c = 2.0, 5.0, 9.0
    histogram = accumulate_asg(
        np.array([[a], [b], [c]]),
        np.ones((3, 1)),
        np.array([[0, 1, 2]]),
        np.array([1.0]),
        x0,
        dx,
        n_bins,
    )
    axis = x0 + dx * np.arange(n_bins)
    density = histogram / dx
    peak = 2.0 / (c - a)
    assert density.max() == pytest.approx(peak, rel=1e-3)
    assert axis[np.argmax(density)] == pytest.approx(b, abs=2 * dx)
    assert density[axis < a].max() < 1e-9
    assert density[axis > c].max() < 1e-9
    midpoint = np.argmin(np.abs(axis - 0.5 * (a + b)))
    assert density[midpoint] == pytest.approx(peak * 0.5, rel=1e-3)


def test_narrow_window_matches_a_cropped_wide_one():
    """Intensity outside the window must be accounted for, not clamped.

    Simulating a narrow window directly and cropping a wide simulation have to
    agree, which they only do because a triangle straddling the window edge is
    deposited exactly.  The residual difference is the Lorentzian tail beyond
    the convolution margin, a few parts in 100000 of the peak.
    """
    experiment = Experiment(spin=2.5, larmor=130.3, transitions="ct")
    site = Site(cq=6.0, eta=0.3, lorentz=0.001)
    grid = asg_grid(24)
    wide = np.linspace(130.24, 130.33, 9001)
    narrow = wide[(wide > 130.28) & (wide < 130.30)]
    full = simulate(wide, site, experiment, grid=grid)
    cropped = full[(wide > 130.28) & (wide < 130.30)]
    direct = simulate(narrow, site, experiment, grid=grid)
    assert np.max(np.abs(direct - cropped)) < 2e-4 * full.max()


def test_broadening_conserves_area_and_width():
    x = np.linspace(-80, 80, 32001)
    for lorentz, gauss in [(2.0, 0.0), (0.0, 3.0), (1.5, 2.5)]:
        kernel = voigt_kernel(x, lorentz, gauss)
        assert np.trapezoid(kernel, x) == pytest.approx(1.0, abs=0.02)
        half = kernel.max() / 2
        inside = np.where(kernel >= half)[0]
        measured = x[inside[-1]] - x[inside[0]]
        expected = 0.5346 * lorentz + np.sqrt(0.2166 * lorentz**2 + gauss**2)
        assert measured == pytest.approx(expected, rel=0.01)


def test_broadening_is_a_no_op_at_zero_width():
    y = np.random.default_rng(0).normal(size=256)
    assert np.array_equal(broaden(y, 0.1, 0.0, 0.0), y)
    with pytest.raises(ValueError):
        broaden(y, 0.1, -1.0, 0.0)


def test_asg_matches_a_brute_force_powder_average():
    """The whole point of ASG: the same lineshape from far fewer orientations."""
    experiment = Experiment(spin=2.5, larmor=130.318, transitions="ct")
    site = Site(cq=5.0, eta=0.5, lorentz=0.0008)
    x = np.linspace(130.28, 130.34, 3001)
    reference = simulate(x, site, experiment, grid=spiral_grid(400_000))
    reference /= np.trapezoid(reference, x)
    interpolated = simulate(x, site, experiment, divisions=40)
    interpolated /= np.trapezoid(interpolated, x)
    assert np.max(np.abs(interpolated - reference)) < 0.02 * reference.max()


def test_pattern_width_matches_the_analytic_second_order_edges():
    """The powder pattern must span exactly the range the closed form predicts."""
    spin, cq, eta, larmor = 2.5, 4.0, 0.6, 104.2
    experiment = Experiment(spin=spin, larmor=larmor, transitions="ct")
    low, high = second_order_ct_extremes(spin, cq, eta, larmor)
    x = np.linspace(larmor + low - 0.01, larmor + high + 0.01, 20001)
    y = simulate(x, Site(cq=cq, eta=eta), experiment, divisions=60)
    significant = x[y > 1e-4 * y.max()]
    assert significant[0] - larmor == pytest.approx(low, abs=2e-4)
    assert significant[-1] - larmor == pytest.approx(high, abs=2e-4)
