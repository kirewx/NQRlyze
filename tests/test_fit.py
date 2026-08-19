"""The fitter: can it get the parameters back?"""

import numpy as np
import pytest

from nqrlyze.analytic import (
    cq_range_for_line,
    nqr_frequencies,
    nqr_frequency_spin_three_halves,
)
from nqrlyze.fit import FitParameter, default_parameters, fit
from nqrlyze.simulate import Experiment, Site, simulate
from nqrlyze.spectrum import Spectrum


def _noisy(x, sites, experiment, noise=0.01, offset=0.0, seed=0, divisions=90):
    y = simulate(x, sites, experiment, divisions=divisions)
    y = y / y.max()
    y = y + np.random.default_rng(seed).normal(0, noise, y.size) + offset
    return Spectrum(x, y, reference=experiment.reference_frequency)


def test_central_transition_round_trip():
    """Cq, eta, shift and both broadenings, recovered from a poor starting guess."""
    experiment = Experiment.from_nucleus("27Al", field=11.7449, transitions="ct")
    truth = Site(cq=5.2, eta=0.42, iso=62.0, lorentz=0.0012, gauss=0.0020)
    x = np.linspace(experiment.larmor - 0.045, experiment.larmor + 0.020, 2400)
    data = _noisy(x, [truth], experiment, noise=0.01, offset=0.03)

    free = [
        FitParameter(0, "cq", 0.5, 12.0),
        FitParameter(0, "eta", 0.0, 1.0),
        FitParameter(0, "iso", -200.0, 300.0),
        FitParameter(0, "lorentz", 0.0, 0.02),
        FitParameter(0, "gauss", 0.0, 0.02),
    ]
    start = [Site(cq=2.0, eta=0.9, iso=0.0, lorentz=0.005, gauss=0.005)]
    result = fit(data, start, experiment, free, baseline_order=0)

    assert result.success
    assert result.r_squared > 0.99
    best = result.sites[0]
    assert best.cq == pytest.approx(truth.cq, abs=0.02)
    assert best.eta == pytest.approx(truth.eta, abs=0.01)
    assert best.iso == pytest.approx(truth.iso, abs=1.0)
    assert best.lorentz == pytest.approx(truth.lorentz, abs=0.0004)
    assert best.gauss == pytest.approx(truth.gauss, abs=0.0004)
    for name in ("cq", "eta", "lorentz", "gauss"):
        assert np.isfinite(result.uncertainty(0, name))
    assert "cq" in result.report()


def test_nqr_round_trip_for_spin_five_halves():
    """Both NQR lines together determine Cq and eta.

    The search is bounded with :func:`cq_range_for_line` from the position of
    the lower line -- an NQR resonance is far too sharp for an unbounded global
    search to stumble onto, and bracketing it is the normal workflow.
    """
    experiment = Experiment.from_nucleus("127I", larmor=0.0)
    truth = Site(cq=1200.0, eta=0.18, lorentz=0.6)
    lines = nqr_frequencies(2.5, truth.cq, truth.eta)
    x = np.linspace(0.85 * lines[0], 1.08 * lines[1], 4000)
    data = _noisy(x, [truth], experiment, noise=0.005, divisions=40)

    low, high = cq_range_for_line(2.5, lines[0], transitions=0)
    assert low <= truth.cq <= high
    free = [
        FitParameter(0, "cq", low, high),
        FitParameter(0, "eta", 0.0, 1.0),
        FitParameter(0, "lorentz", 0.05, 3.0),
    ]
    result = fit(
        data, [Site(cq=0.5 * (low + high), eta=0.5, lorentz=1.0)], experiment, free,
        baseline_order=0, divisions=20, coarse_divisions=8,
    )
    assert result.sites[0].cq == pytest.approx(truth.cq, rel=5e-3)
    assert result.sites[0].eta == pytest.approx(truth.eta, abs=0.02)
    assert result.sites[0].lorentz == pytest.approx(truth.lorentz, rel=0.15)


def test_two_sites_are_separated():
    """Two overlapping sites, with amplitudes solved rather than searched."""
    experiment = Experiment.from_nucleus("23Na", field=9.4, transitions="ct")
    truth = [
        Site(cq=1.6, eta=0.15, iso=5.0, lorentz=0.0006, weight=1.0, label="Na1"),
        Site(cq=2.8, eta=0.80, iso=-12.0, lorentz=0.0006, weight=0.5, label="Na2"),
    ]
    x = np.linspace(experiment.larmor - 0.012, experiment.larmor + 0.006, 2200)
    data = _noisy(x, truth, experiment, noise=0.004)

    free = [
        FitParameter(0, "cq", 0.5, 4.0), FitParameter(0, "eta", 0.0, 1.0),
        FitParameter(0, "iso", -60.0, 60.0),
        FitParameter(1, "cq", 0.5, 4.0), FitParameter(1, "eta", 0.0, 1.0),
        FitParameter(1, "iso", -60.0, 60.0),
        FitParameter(0, "lorentz", 0.0002, 0.003),
    ]

    def tie(sites):
        sites[1].lorentz = sites[0].lorentz
        return sites

    start = [
        Site(cq=1.0, eta=0.5, iso=0.0, lorentz=0.001, label="Na1"),
        Site(cq=3.5, eta=0.5, iso=0.0, lorentz=0.001, label="Na2"),
    ]
    result = fit(data, start, experiment, free, tie=tie, baseline_order=0)

    found = sorted(result.sites, key=lambda s: s.cq)
    assert found[0].cq == pytest.approx(1.6, abs=0.05)
    assert found[1].cq == pytest.approx(2.8, abs=0.05)
    assert found[0].eta == pytest.approx(0.15, abs=0.08)
    assert found[1].eta == pytest.approx(0.80, abs=0.08)
    assert np.all(result.amplitudes >= 0)
    ratio = result.amplitudes[1] / result.amplitudes[0]
    assert ratio == pytest.approx(0.5, rel=0.15)


def test_amplitudes_and_baseline_are_solved_exactly():
    """A sloping baseline and an arbitrary scale must not need to be guessed."""
    experiment = Experiment.from_nucleus("27Al", field=9.4, transitions="ct")
    truth = Site(cq=3.0, eta=0.3, lorentz=0.001)
    x = np.linspace(experiment.larmor - 0.02, experiment.larmor + 0.01, 1500)
    clean = simulate(x, truth, experiment, divisions=90)
    ramp = 0.4 + 2.5 * (x - x[0]) / (x[-1] - x[0])
    data = Spectrum(x, 137.0 * clean / clean.max() + ramp, experiment.larmor)

    result = fit(
        data, [Site(cq=3.0, eta=0.3, lorentz=0.001)], experiment,
        parameters=[], baseline_order=1,
    )
    assert result.rmsd < 0.02 * np.ptp(data.intensity)
    assert np.max(np.abs(result.baseline - ramp)) < 0.05 * np.ptp(ramp)


def test_fit_rejects_inconsistent_input():
    experiment = Experiment(spin=1.5, larmor=50.0)
    data = Spectrum(np.linspace(24.0, 26.0, 100), np.zeros(100))
    with pytest.raises(ValueError):
        fit(data, [Site(cq=50.0)], experiment, [FitParameter(3, "cq")])
    with pytest.raises(ValueError):
        fit(data, [Site(cq=50.0)], experiment,
            [FitParameter(0, "cq"), FitParameter(0, "cq")])
    with pytest.raises(ValueError):
        FitParameter(0, "not_a_parameter")
    with pytest.raises(ValueError):
        FitParameter(0, "cq", 5.0, 1.0)
    with pytest.raises(ValueError):
        fit(data, [Site(cq=50.0)], experiment, [], weights=np.ones(7))


def test_default_parameters_cover_the_requested_options():
    sites = [Site(cq=2.0), Site(cq=3.0)]
    names = {(p.site, p.name) for p in default_parameters(sites)}
    assert (0, "cq") in names and (1, "eta") in names
    assert (1, "lorentz") not in names  # shared broadening: first site only
    assert (0, "span") not in names
    with_csa = {(p.site, p.name) for p in default_parameters(sites, fit_csa=True)}
    assert (0, "span") in with_csa and (1, "beta") in with_csa
    per_site = {(p.site, p.name) for p in default_parameters(sites, shared_broadening=False)}
    assert (1, "gauss") in per_site


def test_one_nqr_line_cannot_separate_cq_from_eta():
    """A single I = 3/2 resonance fixes only Cq * sqrt(1 + eta^2/3).

    This is a property of the experiment, not of the fitter.  The fit reproduces
    the data perfectly while landing away from the true Cq, and the two
    parameters come back essentially perfectly correlated -- which is how the
    result announces that the problem is under-determined.
    """
    experiment = Experiment.from_nucleus("35Cl", larmor=0.0)
    truth = Site(cq=60.0, eta=0.40, lorentz=0.02)
    combination = nqr_frequency_spin_three_halves(truth.cq, truth.eta)
    x = np.linspace(combination - 0.6, combination + 0.6, 1200)
    data = _noisy(x, [truth], experiment, noise=0.003, divisions=20)

    low, high = cq_range_for_line(1.5, combination)
    assert low <= truth.cq <= high
    free = [
        FitParameter(0, "cq", low, high),
        FitParameter(0, "eta", 0.0, 1.0),
        FitParameter(0, "lorentz", 0.002, 0.1),
    ]
    result = fit(
        data, [Site(cq=high, eta=0.05, lorentz=0.03)], experiment, free,
        baseline_order=0, divisions=12, coarse_divisions=6,
    )
    best = result.sites[0]
    assert result.r_squared > 0.99
    # The observable combination is recovered even though Cq alone is not.
    assert nqr_frequency_spin_three_halves(best.cq, best.eta) == pytest.approx(
        combination, rel=1e-3
    )
    names = result.parameter_names
    correlation = result.correlation[names.index("site1.cq"), names.index("site1.eta")]
    assert abs(correlation) > 0.99
