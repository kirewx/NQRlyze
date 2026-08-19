"""The Hamiltonian, checked against results that do not come from this code."""

import numpy as np
import pytest

from nqrlyze.analytic import (
    nqr_frequencies,
    nqr_frequency_spin_three_halves,
    second_order_ct_isotropic_shift,
    second_order_ct_shift,
)
from nqrlyze.hamiltonian import (
    eigen_transitions,
    quadrupolar_hamiltonian,
    spin_operators,
    transition_indices,
)
from nqrlyze.powder import spiral_grid

RANDOM_DIRECTIONS = np.array(
    [
        [0.3, 0.5, 0.81],
        [-0.7, 0.2, 0.68],
        [0.1, -0.9, 0.42],
        [0.577, 0.577, 0.577],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
)


@pytest.mark.parametrize("spin", [1.0, 1.5, 2.5, 3.0, 3.5, 4.5])
def test_spin_operators_obey_su2(spin):
    ix, iy, iz = spin_operators(spin)
    dim = round(2 * spin) + 1
    assert ix.shape == (dim, dim)
    # [Ix, Iy] = i Iz and cyclic permutations.
    assert np.allclose(ix @ iy - iy @ ix, 1j * iz)
    assert np.allclose(iy @ iz - iz @ iy, 1j * ix)
    assert np.allclose(iz @ ix - ix @ iz, 1j * iy)
    # I^2 = I(I+1)
    total = ix @ ix + iy @ iy + iz @ iz
    assert np.allclose(total, spin * (spin + 1) * np.eye(dim))
    for operator in (ix, iy, iz):
        assert np.allclose(operator, operator.conj().T)


def test_quadrupolar_hamiltonian_is_traceless():
    for spin in (1.0, 1.5, 2.5, 3.5):
        matrix = quadrupolar_hamiltonian(spin, 7.3, 0.4)
        assert abs(np.trace(matrix)) < 1e-12


@pytest.mark.parametrize("eta", [0.0, 0.15, 0.5, 0.87, 1.0])
def test_spin_three_halves_nqr_matches_closed_form(eta):
    """nu = (Cq/2) sqrt(1 + eta^2/3), and independent of orientation."""
    cq = 60.0
    expected = nqr_frequency_spin_three_halves(cq, eta)
    freqs, amps = eigen_transitions(RANDOM_DIRECTIONS, 1.5, cq, eta, larmor=0.0)
    observed = freqs[amps > 1e-9]
    assert observed.size > 0
    assert np.allclose(observed, expected, rtol=0, atol=1e-9)
    assert np.allclose(nqr_frequencies(1.5, cq, eta), expected)


def test_spin_five_halves_nqr_frequencies_and_intensity_ratio():
    """For eta = 0 the two NQR lines are at 3Cq/20 and 6Cq/20, in a 1.6:1 ratio."""
    cq = 10.0
    freqs, amps = eigen_transitions(RANDOM_DIRECTIONS, 2.5, cq, 0.0, larmor=0.0)
    keep = amps[0] > 1e-9
    lines = np.unique(np.round(freqs[0][keep], 9))
    assert not np.any(lines == 0.0), "zero-frequency pairs must be suppressed"

    assert np.allclose(lines, [3 * cq / 20, 6 * cq / 20])

    intensity = {}
    for value, amp in zip(freqs[0][keep], amps[0][keep]):
        intensity[round(value, 9)] = intensity.get(round(value, 9), 0.0) + amp
    low, high = intensity[round(3 * cq / 20, 9)], intensity[round(6 * cq / 20, 9)]
    assert low / high == pytest.approx(1.6, rel=1e-9)


def test_double_quantum_transitions_are_forbidden_at_zero_asymmetry():
    freqs, amps = eigen_transitions(RANDOM_DIRECTIONS, 2.5, 10.0, 0.0, larmor=0.0)
    pairs = transition_indices(2.5, "all")
    # Levels are Kramers doublets: 0,1 = +-1/2, 2,3 = +-3/2, 4,5 = +-5/2.
    for column, (a, b) in enumerate(pairs):
        if {a // 2, b // 2} == {0, 2}:  # +-1/2 <-> +-5/2, dm = 2
            assert np.all(amps[:, column] < 1e-12)


@pytest.mark.parametrize(
    "spin, cq, eta", [(1.5, 2.0, 0.0), (2.5, 3.0, 0.6), (3.5, 5.0, 0.95), (2.5, 1.0, 1.0)]
)
@pytest.mark.parametrize("larmor", [300.0, 3000.0])
def test_central_transition_matches_second_order_perturbation_theory(
    spin, cq, eta, larmor
):
    """In the high-field limit the exact result must reduce to the textbook one."""
    rng = np.random.default_rng(7)
    theta = rng.uniform(0, np.pi, 24)
    phi = rng.uniform(0, 2 * np.pi, 24)
    directions = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        axis=1,
    )
    exact = eigen_transitions(
        directions, spin, cq, eta, larmor, transitions="ct"
    )[0][:, 0]
    approximate = larmor + second_order_ct_shift(spin, cq, eta, larmor, theta, phi)
    spread = np.ptp(approximate - larmor)
    assert np.max(np.abs(exact - approximate)) < 2e-3 * spread


def test_isotropic_second_order_shift_matches_the_powder_average():
    grid = spiral_grid(200_000)
    for spin, cq, eta, larmor in [(2.5, 5.0, 0.4, 130.3), (3.5, 8.0, 0.8, 78.2)]:
        freqs = eigen_transitions(
            grid.directions, spin, cq, eta, larmor, transitions="ct"
        )[0][:, 0]
        expected = second_order_ct_isotropic_shift(spin, cq, eta, larmor)
        assert (freqs - larmor).mean() == pytest.approx(expected, rel=2e-3)


def test_powder_pattern_is_invariant_under_larmor_sign():
    """A negative gyromagnetic ratio must not change a powder spectrum."""
    grid = spiral_grid(4000)
    positive = eigen_transitions(grid.directions, 2.5, 6.0, 0.35, 90.0)
    negative = eigen_transitions(grid.directions, 2.5, 6.0, 0.35, -90.0)
    assert np.allclose(np.sort(positive[0].ravel()), np.sort(negative[0].ravel()))
    assert np.allclose(np.sort(positive[1].ravel()), np.sort(negative[1].ravel()))


def test_chemical_shift_moves_a_line_the_right_way():
    """A larger delta must move a resonance to higher frequency."""
    larmor = 100.0
    directions = np.array([[0.0, 0.0, 1.0]])
    zero = eigen_transitions(
        directions, 2.5, 0.0, 0.0, larmor, transitions="ct"
    )[0][0, 0]
    shifted = eigen_transitions(
        directions,
        2.5,
        0.0,
        0.0,
        larmor,
        shift_ppm=np.eye(3) * 100.0,
        transitions="ct",
    )[0][0, 0]
    assert shifted - zero == pytest.approx(larmor * 100e-6, rel=1e-9)


def test_chemical_shift_is_inert_without_a_field():
    """In NQR there is no reference frequency, so a shift cannot do anything."""
    plain = eigen_transitions(RANDOM_DIRECTIONS, 1.5, 40.0, 0.3, 0.0)
    with_shift = eigen_transitions(
        RANDOM_DIRECTIONS, 1.5, 40.0, 0.3, 0.0, shift_ppm=np.diag([300.0, 10.0, -50.0])
    )
    assert np.allclose(plain[0], with_shift[0])


def test_transition_selection():
    assert transition_indices(2.5, "ct") == [(2, 3)]
    assert transition_indices(1.5, "ct") == [(1, 2)]
    assert transition_indices(1.5, "single") == [(0, 1), (1, 2), (2, 3)]
    assert transition_indices(1.5, "satellites") == [(0, 1), (2, 3)]
    assert len(transition_indices(2.5, "all")) == 15
    with pytest.raises(ValueError):
        transition_indices(1.0, "ct")
    with pytest.raises(ValueError):
        transition_indices(1.5, "nonsense")
