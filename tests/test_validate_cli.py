"""The QUEST comparison harness and the command line."""

import json

import numpy as np
import pytest

from _bruker_fixture import write_bruker_1d
from nqrlyze.cli import main
from nqrlyze.config import build_axis, load_job, load_spectrum, parse_experiment
from nqrlyze.io import write_ascii
from nqrlyze.simulate import Experiment, Site, simulate
from nqrlyze.spectrum import Spectrum
from nqrlyze.validate import compare, find_singularities, run_manifest


def _reference(tmp_path, name, sites, experiment, span=0.06, points=4000, scale=1.0):
    """Stand in for a spectrum exported from QUEST."""
    centre = experiment.larmor if experiment.larmor else 30.0
    x = np.linspace(centre - span, centre + span / 2, points)
    y = simulate(x, sites, experiment, divisions=70) * scale
    path = tmp_path / name
    write_ascii(path, Spectrum(x, y, experiment.reference_frequency), unit="MHz")
    return path


def test_compare_accepts_a_matching_simulation(tmp_path):
    experiment = Experiment.from_nucleus("27Al", larmor=130.318, transitions="ct")
    sites = [Site(cq=5.2, eta=0.42, lorentz=0.001)]
    from nqrlyze.io import read_ascii

    reference = read_ascii(_reference(tmp_path, "ref.txt", sites, experiment))
    result = compare(reference, sites, experiment, name="self")
    assert result.passed
    assert result.rmsd < 1e-3
    assert result.correlation > 0.9999
    assert abs(result.peak_shift_khz) < 0.1
    assert "PASS" in result.report()


def test_compare_is_insensitive_to_scale_and_offset(tmp_path):
    """QUEST's vertical scale is arbitrary, so it must not affect the verdict."""
    experiment = Experiment.from_nucleus("27Al", larmor=130.318, transitions="ct")
    sites = [Site(cq=4.0, eta=0.2, lorentz=0.001)]
    from nqrlyze.io import read_ascii

    reference = read_ascii(_reference(tmp_path, "r2.txt", sites, experiment, scale=735.0))
    shifted = Spectrum(
        reference.freq_mhz, reference.intensity + 12.5, reference.reference
    )
    result = compare(shifted, sites, experiment)
    assert result.passed
    assert result.scale > 0


def test_compare_flags_a_wrong_coupling_constant(tmp_path):
    experiment = Experiment.from_nucleus("27Al", larmor=130.318, transitions="ct")
    from nqrlyze.io import read_ascii

    reference = read_ascii(
        _reference(tmp_path, "r3.txt", [Site(cq=5.2, eta=0.42, lorentz=0.001)], experiment)
    )
    result = compare(reference, [Site(cq=4.4, eta=0.42, lorentz=0.001)], experiment)
    assert not result.passed
    assert result.rmsd > 0.02


def test_find_singularities_locates_the_pattern_edges():
    """A second-order central-transition pattern has two prominent horns."""
    experiment = Experiment(spin=2.5, larmor=130.3, transitions="ct")
    x = np.linspace(130.26, 130.32, 8000)
    y = simulate(x, Site(cq=5.0, eta=0.3, lorentz=0.0004), experiment, divisions=70)
    peaks = find_singularities(Spectrum(x, y, 130.3), prominence=0.03)
    assert 2 <= peaks.size <= 4
    assert np.all(np.diff(peaks) > 0)


def test_manifest_round_trip(tmp_path):
    experiment = Experiment.from_nucleus("27Al", larmor=130.318, transitions="ct")
    sites = [Site(cq=5.2, eta=0.42, lorentz=0.001)]
    _reference(tmp_path, "case1.txt", sites, experiment)
    manifest = {
        "tolerance": {"rmsd": 0.01, "peak_shift_khz": 0.5, "singularity_khz": 1.0},
        "cases": [
            {
                "name": "27Al CT",
                "file": "case1.txt",
                "unit": "MHz",
                "experiment": {
                    "nucleus": "27Al", "larmor": 130.318, "transitions": "ct",
                },
                "sites": [{"cq": 5.2, "eta": 0.42, "lorentz": 0.001}],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    results = run_manifest(path, divisions=70)
    assert len(results) == 1 and results[0].passed
    assert main(["validate", str(path), "--divisions", "70"]) == 0


def test_cli_template_and_info(capsys):
    assert main(["template", "job"]) == 0
    job = json.loads(capsys.readouterr().out)
    assert "experiment" in job and "fit" in job
    assert main(["template", "manifest"]) == 0
    assert "cases" in json.loads(capsys.readouterr().out)

    assert main(["info", "27Al", "--field", "11.7449", "--cq", "5.2"]) == 0
    out = capsys.readouterr().out
    assert "Larmor" in out and "Vzz" in out
    assert main(["info", "not-an-isotope"]) == 2


def test_cli_coadd_and_fit_end_to_end(tmp_path, capsys):
    """Bruker in, stepped-frequency pieces joined, fitted, plotted."""
    experiment = Experiment.from_nucleus("23Na", field=9.4, transitions="ct")
    truth = Site(cq=1.9, eta=0.35, iso=4.0, lorentz=0.0006)
    sf = experiment.larmor

    # Two overlapping sub-spectra, as a stepped-frequency experiment produces.
    for index, centre_khz in enumerate((-6.0, 2.0)):
        size = 2048
        sw_p = 12_000.0
        offset_ppm = (centre_khz * 1e-3 + sw_p / 2e6) / sf * 1e6
        index_axis = np.arange(size, dtype=float)
        freq = sf + (offset_ppm * sf - index_axis * sw_p / size) * 1e-6
        y = simulate(np.sort(freq), truth, experiment, divisions=70)
        y = 3.0e5 * y / y.max()
        write_bruker_1d(
            tmp_path / f"expt{index}", y[::-1], sf=sf, sw_p=sw_p,
            offset=offset_ppm, ns=32,
        )

    combined = tmp_path / "combined.txt"
    assert main([
        "coadd", str(tmp_path / "expt0"), str(tmp_path / "expt1"),
        "--mode", "mean", "--scale-by-scans", "-o", str(combined),
    ]) == 0
    assert combined.is_file()

    job = {
        "experiment": {"nucleus": "23Na", "field": 9.4, "transitions": "ct"},
        "data": {
            "format": "bruker",
            "paths": [str(tmp_path / "expt0"), str(tmp_path / "expt1")],
            "coadd": {"mode": "mean", "scale_by_scans": True},
            "normalize": True,
        },
        "sites": [{"label": "Na1", "cq": 1.0, "eta": 0.6, "iso": 0.0,
                   "lorentz": 0.001, "gauss": 0.0}],
        "fit": {
            "baseline_order": 0,
            "divisions": 40,
            "free": [
                {"site": 0, "name": "cq", "lower": 0.5, "upper": 4.0},
                {"site": 0, "name": "eta", "lower": 0.0, "upper": 1.0},
                {"site": 0, "name": "iso", "lower": -40.0, "upper": 40.0},
                {"site": 0, "name": "lorentz", "lower": 0.0002, "upper": 0.004},
            ],
        },
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job))

    plot = tmp_path / "fit.png"
    out = tmp_path / "result.json"
    assert main(["fit", str(job_path), "--plot", str(plot), "-o", str(out),
                 "--correlations"]) == 0
    text = capsys.readouterr().out
    assert "fit converged" in text
    assert plot.is_file() and plot.stat().st_size > 5000
    fitted = json.loads(out.read_text())
    assert fitted["sites"][0]["cq"] == pytest.approx(truth.cq, abs=0.05)
    assert fitted["sites"][0]["eta"] == pytest.approx(truth.eta, abs=0.06)


def test_cli_simulate_from_an_axis_block(tmp_path, capsys):
    job = {
        "experiment": {"nucleus": "35Cl", "larmor": 0.0},
        "axis": {"unit": "MHz", "low": 29.0, "high": 32.0, "points": 2000},
        "sites": [{"cq": 60.0, "eta": 0.25, "lorentz": 0.03}],
    }
    path = tmp_path / "sim.json"
    path.write_text(json.dumps(job))
    out = tmp_path / "sim.txt"
    assert main(["simulate", str(path), "-o", str(out)]) == 0
    assert out.is_file()
    from nqrlyze.io import read_ascii

    spectrum = read_ascii(out)
    from nqrlyze.analytic import nqr_frequency_spin_three_halves

    peak = spectrum.freq_mhz[np.argmax(spectrum.intensity)]
    assert peak == pytest.approx(nqr_frequency_spin_three_halves(60.0, 0.25), abs=0.01)


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        parse_experiment({"nucleus": "27Al", "field": 9.4, "bogus": 1})
    with pytest.raises(ValueError):
        parse_experiment({"nucleus": "27Al", "field": 9.4, "larmor": 100.0})
    with pytest.raises(ValueError):
        build_axis({"unit": "ppm", "low": 0, "high": 1}, reference=0.0)


def test_load_spectrum_windows_in_ppm(tmp_path):
    x = np.linspace(129.9, 130.7, 2000)
    write_ascii(tmp_path / "s.txt", Spectrum(x, np.ones_like(x), 130.3), unit="MHz")
    spectrum = load_spectrum(
        {"format": "ascii", "path": "s.txt", "reference": 130.3,
         "window_ppm": [-500, 500]},
        tmp_path,
    )
    assert spectrum.ppm.min() >= -500.001 and spectrum.ppm.max() <= 500.001
