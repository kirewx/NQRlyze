"""The local GUI: its HTTP API, and the page it serves."""

import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from nqrlyze import Experiment, Site, Spectrum, simulate, suggest_window
from nqrlyze.fit import FitParameter, fit
from nqrlyze.webapp import (
    STATE,
    STATIC,
    _decimate,
    api_bracket,
    api_clear,
    api_demo,
    api_fit,
    api_job,
    api_load,
    api_nuclei,
    api_simulate,
    api_window,
    create_server,
)

EXPERIMENT = {"nucleus": "27Al", "field": 11.7449, "transitions": "ct"}
SITES = [{"cq": 5.2, "eta": 0.42, "iso": 62.0, "lorentz": 0.0012, "gauss": 0.002}]
MODEL = {"experiment": EXPERIMENT, "sites": SITES}


@pytest.fixture(autouse=True)
def _clean_state():
    api_clear({})
    yield
    api_clear({})
    STATE.jobs.clear()


def test_page_is_self_contained():
    """No external requests: a local tool has to work offline."""
    html = (STATIC / "index.html").read_text()
    assert "<title>NQRlyze</title>" in html
    # The w3.org SVG string is an XML namespace, never fetched; drop it before
    # looking for anything that would actually leave the machine.
    offline = html.replace("http://www.w3.org/2000/svg", "")
    for forbidden in ("http://", "https://", "<script src", "<link href", "@import"):
        assert forbidden not in offline, f"page references {forbidden}"
    # Theme is defined for both modes, not only one.
    assert '[data-theme="dark"]' in html


def test_nuclei_list_is_quadrupolar_only():
    nuclei = api_nuclei({})["nuclei"]
    assert len(nuclei) > 40
    assert all(n["spin"] >= 1.0 for n in nuclei)
    assert any(n["symbol"] == "27Al" and n["spin"] == 2.5 for n in nuclei)


def test_window_matches_the_library_helper():
    got = api_window(MODEL)
    expected = suggest_window(
        [Site(**SITES[0])], Experiment.from_nucleus("27Al", field=11.7449, transitions="ct")
    )
    assert got["low"] == pytest.approx(expected[0])
    assert got["high"] == pytest.approx(expected[1])


def test_simulate_returns_a_normalised_trace():
    r = api_simulate(MODEL)
    assert len(r["total"]["x"]) == len(r["total"]["y"]) > 100
    assert max(r["total"]["y"]) == pytest.approx(1.0, abs=1e-9)
    assert len(r["components"]) == 1
    assert r["low"] < r["high"]


def test_simulate_honours_an_explicit_window():
    body = dict(MODEL, window={"low": 130.40, "high": 130.42, "points": 300})
    r = api_simulate(body)
    assert r["low"] == pytest.approx(130.40) and r["high"] == pytest.approx(130.42)
    assert len(r["total"]["x"]) == 300
    with pytest.raises(ValueError):
        api_simulate(dict(MODEL, window={"low": 130.42, "high": 130.40}))


def test_zeeman_perturbed_nqr_splits_the_line():
    """A small Larmor term on top of the quadrupole interaction is a valid
    experiment: the degenerate +-m levels split and the pattern widens."""
    site = [{"cq": 72.0, "eta": 0.1, "lorentz": 0.001}]
    pure = api_simulate({"experiment": {"spin": 1.5, "larmor": 0.0, "reference": 0,
                                        "transitions": "all"}, "sites": site})
    zeeman = api_simulate({"experiment": {"spin": 1.5, "larmor": 0.5, "reference": 0,
                                          "transitions": "all"}, "sites": site})
    assert max(pure["total"]["y"]) == pytest.approx(1.0, abs=1e-9)
    assert max(zeeman["total"]["y"]) == pytest.approx(1.0, abs=1e-9)
    assert zeeman["reference"] == 0.0
    assert (zeeman["high"] - zeeman["low"]) > (pure["high"] - pure["low"])


def test_pure_nqr_central_transition_has_no_intensity():
    """At zero field the central transition carries nothing.  The GUI defaults
    used to combine (NQR, transitions=ct) silently into this empty spectrum;
    it now switches to "all" and explains a flat result."""
    model = {"experiment": {"spin": 2.5, "larmor": 0.0, "transitions": "ct"},
             "sites": [{"cq": 1.0, "eta": 0.0, "lorentz": 0.001}]}
    assert max(api_simulate(model)["total"]["y"]) == 0.0
    model["experiment"]["transitions"] = "all"
    assert max(api_simulate(model)["total"]["y"]) == pytest.approx(1.0, abs=1e-9)


def test_decimate_keeps_the_envelope():
    """Striding can step over a singularity; keeping block extremes cannot."""
    x = np.linspace(0.0, 1.0, 50_000)
    y = np.zeros_like(x)
    y[31_111] = 7.0  # a one-sample spike, exactly what a horn looks like
    dx, dy = _decimate(x, y, limit=2000)
    assert dy.size <= 2100
    assert dy.max() == pytest.approx(7.0)
    assert dx[np.argmax(dy)] == pytest.approx(x[31_111])
    short = np.linspace(0, 1, 50)
    assert _decimate(short, short, limit=2000)[0].size == 50


def test_demo_stores_data_and_reports_the_truth():
    r = api_demo(dict(MODEL, noise=0.01, points=800))
    assert r["points"] == 800
    assert STATE.data is not None and len(STATE.data) == 800
    assert r["truth"][0]["cq"] == pytest.approx(5.2)
    assert max(r["data"]["y"]) > 0.5


def test_load_reads_two_column_text(tmp_path):
    x = np.linspace(30.0, 31.0, 400)
    y = np.exp(-(((x - 30.5) / 0.05) ** 2))
    path = tmp_path / "spec.txt"
    path.write_text("\n".join(f"{a} {b}" for a, b in zip(x, y)))

    r = api_load({"path": str(path), "format": "ascii", "unit": "MHz"})
    assert r["points"] == 400
    assert max(r["data"]["y"]) == pytest.approx(1.0)
    assert STATE.data is not None

    pasted = api_load({"text": path.read_text(), "name": "dropped.txt", "unit": "MHz"})
    assert pasted["label"] == "dropped.txt"

    api_clear({})
    assert STATE.data is None
    with pytest.raises(FileNotFoundError):
        api_load({"path": str(tmp_path / "absent.txt"), "format": "ascii"})
    with pytest.raises(ValueError):
        api_load({"path": "  "})


def test_bracket_wraps_the_analytic_helper():
    r = api_bracket({"experiment": {"nucleus": "35Cl", "larmor": 0.0}, "sites": SITES,
                     "frequency": 30.79})
    assert r["low"] < 60.0 < r["high"]


def _await_job(job_id, timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = api_job({"job": job_id})
        if state["state"] != "running":
            return state
        time.sleep(0.3)
    raise AssertionError("fit did not finish")


def test_fit_runs_as_a_background_job():
    api_demo(dict(MODEL, noise=0.01, points=900))
    body = dict(
        MODEL,
        sites=[dict(SITES[0], cq=4.6, eta=0.6)],
        free=[{"site": 0, "name": "cq"}, {"site": 0, "name": "eta"}],
        strategy="local",
        divisions=24,
    )
    started = api_fit(body)
    assert api_job({"job": started["job"]})["state"] in ("running", "done")
    state = _await_job(started["job"])
    assert state["state"] == "done", state.get("error")
    result = state["result"]
    assert result["r_squared"] > 0.95
    assert result["sites"][0]["cq"] == pytest.approx(5.2, abs=0.15)
    assert len(result["model"]["x"]) == len(result["residual"]["x"])
    assert "0.cq" in result["uncertainties"]


def test_fit_reports_bad_requests_rather_than_hanging():
    api_demo(dict(MODEL, points=400))
    state = _await_job(api_fit(dict(MODEL, free=[]))["job"])
    assert state["state"] == "error" and "tick at least one" in state["error"].lower()

    api_clear({})
    state = _await_job(api_fit(dict(MODEL, free=[{"site": 0, "name": "cq"}]))["job"])
    assert state["state"] == "error" and "load or synthesise" in state["error"].lower()

    with pytest.raises(KeyError):
        api_job({"job": "nope"})


FREE_SITE_PARAMETERS = [
    FitParameter(0, "cq", 0.0, 13.2),
    FitParameter(0, "eta", 0.0, 1.0),
    FitParameter(0, "iso", -270.0, 330.0),
    FitParameter(0, "lorentz", 0.0, 0.03),
    FitParameter(0, "gauss", 0.0, 0.03),
]
TRUTH = Site(cq=5.2, eta=0.42, iso=62.0, lorentz=0.0012, gauss=0.0020)


def _noisy_central_transition():
    experiment = Experiment.from_nucleus("27Al", field=11.7449, transitions="ct")
    low, high = suggest_window([TRUTH], experiment)
    x = np.linspace(low, high, 2000)
    y = simulate(x, TRUTH, experiment, divisions=60)
    y = y / y.max() + np.random.default_rng(0).normal(0, 0.015, x.size)
    return experiment, Spectrum(x, y, experiment.reference_frequency)


@pytest.mark.parametrize(
    "cq, eta, iso",
    [(4.4, 0.7, 30.0), (1.5, 0.1, -100.0), (9.0, 0.9, 150.0)],
)
def test_multistart_recovers_the_truth_from_a_poor_start(cq, eta, iso):
    """The contract: a handful of restarts finds the answer from a bad guess.

    Deliberately asserts only what multi-start *achieves*, not that a single
    refinement fails. Whether a lone local fit falls into the usual trap -- a
    wrong Cq hiding behind inflated broadening -- turns out to depend on the
    SciPy version, so requiring it to fail encoded one optimiser's behaviour as
    if it were physics, and broke on SciPy 1.18.
    """
    experiment, data = _noisy_central_transition()
    start = [Site(cq=cq, eta=eta, iso=iso, lorentz=0.003, gauss=0.003)]
    result = fit(
        data, start, experiment, FREE_SITE_PARAMETERS,
        baseline_order=0, divisions=34, global_search=False, restarts=8,
    )
    assert result.sites[0].cq == pytest.approx(TRUTH.cq, abs=0.08)
    assert result.sites[0].eta == pytest.approx(TRUTH.eta, abs=0.08)
    assert result.r_squared > 0.99


def test_multistart_is_never_worse_than_a_single_refinement():
    """Restarts include the caller's own starting point, so they cannot lose."""
    experiment, data = _noisy_central_transition()
    start = [Site(cq=4.4, eta=0.7, iso=30.0, lorentz=0.003, gauss=0.003)]
    common = dict(
        baseline_order=0, divisions=34, global_search=False,
    )
    plain = fit(data, start, experiment, FREE_SITE_PARAMETERS, **common)
    many = fit(data, start, experiment, FREE_SITE_PARAMETERS, restarts=8, **common)
    assert many.r_squared >= plain.r_squared - 1e-6


def test_http_surface():
    server = create_server("127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(base + "/", timeout=10) as response:
            assert response.status == 200
            assert b"<title>NQRlyze</title>" in response.read()

        request = urllib.request.Request(
            base + "/api/nuclei", data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            assert len(json.loads(response.read())["nuclei"]) > 40

        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(base + "/nope", timeout=10)
        assert missing.value.code == 404

        bad = urllib.request.Request(
            base + "/api/simulate", data=b"not json",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as broken:
            urllib.request.urlopen(bad, timeout=10)
        assert broken.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_page_has_quest_style_window_controls():
    """The window is entered as centre frequency + spectral width, not From/To."""
    html = (STATIC / "index.html").read_text()
    assert 'id="win-centre"' in html and 'id="win-width"' in html
    assert 'id="win-low"' not in html and 'id="win-high"' not in html


def test_page_keeps_field_editable_in_nqr_mode():
    """The NQR box is a preset, not a lock: it must not disable the field and
    Larmor inputs (that is what forbade Zeeman-perturbed NQR)."""
    html = (STATIC / "index.html").read_text()
    assert '"larmor").disabled' not in html
    assert '"field").disabled' not in html
    assert "no observable intensity" in html


def test_process_phases_and_does_not_accumulate():
    """Corrections are always applied to the spectrum as loaded: sending the
    same settings twice gives the same answer, and zero gets the original."""
    from nqrlyze.webapp import api_autophase, api_process

    original = api_demo(dict(MODEL, noise=0.0))
    assert original["has_imag"] is False
    first = api_process({"ph0": 40.0, "ph1": -30.0, "baseline_order": 1})
    again = api_process({"ph0": 40.0, "ph1": -30.0, "baseline_order": 1})
    assert first["data"]["y"] == again["data"]["y"]
    assert first["data"]["y"] != original["data"]["y"]
    assert first["processing"]["ph0"] == 40.0
    back = api_process({})
    assert np.allclose(back["data"]["y"], original["data"]["y"])

    # The fitted copy follows the processing.
    assert STATE.data is not None and STATE.raw is not None
    assert not np.shares_memory(STATE.data.intensity, STATE.raw.intensity)

    # Auto phase undoes a zero-order error on top of the Hilbert-rebuilt part.
    api_process({"ph0": 70.0})
    fixed = api_autophase({})
    assert abs(fixed["processing"]["ph0"]) < 5.0


def test_process_needs_data_and_sane_settings():
    from nqrlyze.webapp import api_process

    with pytest.raises(ValueError):
        api_process({"ph0": 10.0})
    api_demo(MODEL)
    with pytest.raises(ValueError):
        api_process({"baseline_order": 12})


def test_new_data_resets_processing():
    from nqrlyze.webapp import api_process

    api_demo(MODEL)
    api_process({"ph0": 30.0})
    assert api_demo(MODEL)["processing"]["ph0"] == 0.0


def test_page_has_processing_and_export_controls():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for control in ('id="ph0"', 'id="ph1"', 'id="bl-order"', 'id="autophase"',
                    'id="export-svg"', "/api/process", "/api/autophase"):
        assert control in html
