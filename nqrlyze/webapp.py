"""A small local web interface: ``nqrlyze gui``.

Runs a plain :mod:`http.server` on the loopback interface and serves one
self-contained page.  No framework, no build step, no network access -- the
whole thing is the standard library plus the package itself, so it starts as
fast as a script and works offline.

The browser is only a front end.  Every number it shows comes from the same
functions the command line and the test suite use, so what you tune by hand and
what you fit are the same simulation.

The loaded spectrum lives here, on the server, rather than being shipped back
and forth: the page receives a decimated copy to draw, while fits run against
the full data.

Security: the server binds to 127.0.0.1 and reads files the user names, which
is appropriate for a local tool driven by the person sitting at the machine.
Do not expose it on a public interface.
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from .analytic import cq_range_for_line
from .coadd import coadd
from .config import parse_experiment, parse_sites
from .constants import NUCLEI
from .fit import FitParameter, fit
from .io import read_ascii, read_bruker_series
from .simulate import Experiment, Site, simulate_sites, suggest_window
from .spectrum import Spectrum

STATIC = Path(__file__).parent / "static"

#: Most points ever sent to the browser; fits always use the full data.
DISPLAY_LIMIT = 4000


class _State:
    """Everything one session of the GUI holds on to."""

    def __init__(self):
        self.data: Spectrum | None = None
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()


STATE = _State()


def _decimate(x: np.ndarray, y: np.ndarray, limit: int = DISPLAY_LIMIT):
    """Thin a trace for display, keeping the extremes of each discarded run.

    Plain striding can step straight over a sharp singularity.  Taking the
    minimum and maximum of every block instead keeps the envelope, so a
    decimated powder pattern still shows its horns.
    """
    n = x.size
    if n <= limit:
        return x, y
    block = int(np.ceil(n / (limit / 2)))
    usable = (n // block) * block
    xb = x[:usable].reshape(-1, block)
    yb = y[:usable].reshape(-1, block)
    lows = yb.argmin(axis=1)
    highs = yb.argmax(axis=1)
    first = np.minimum(lows, highs)
    second = np.maximum(lows, highs)
    rows = np.arange(xb.shape[0])
    out_x = np.empty(xb.shape[0] * 2)
    out_y = np.empty(xb.shape[0] * 2)
    out_x[0::2], out_y[0::2] = xb[rows, first], yb[rows, first]
    out_x[1::2], out_y[1::2] = xb[rows, second], yb[rows, second]
    if usable < n:
        out_x = np.append(out_x, x[usable:])
        out_y = np.append(out_y, y[usable:])
    return out_x, out_y


def _trace(x: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
    dx, dy = _decimate(np.asarray(x, float), np.asarray(y, float))
    return {"x": [round(float(v), 9) for v in dx], "y": [float(v) for v in dy]}


def _bounds(name: str, value: float) -> tuple[float, float]:
    """Automatic search bounds around the value the user has dialled in.

    The GUI workflow is to get close by eye and then refine, so the bounds are
    generous around the current value rather than covering all of parameter
    space.  For a wide search, bound ``Cq`` from a line position with
    :func:`nqrlyze.cq_range_for_line` and use a job file.
    """
    if name == "cq":
        return (0.0, max(3.0 * abs(value), 1.0)) if value else (0.0, 1.0)
    if name == "eta":
        return 0.0, 1.0
    if name == "iso":
        return value - 300.0, value + 300.0
    if name in ("lorentz", "gauss"):
        return 0.0, max(10.0 * value, 1e-3)
    if name == "span":
        return 0.0, max(3.0 * value, 200.0)
    if name == "skew":
        return -1.0, 1.0
    if name in ("alpha", "gamma"):
        return 0.0, 360.0
    if name == "beta":
        return 0.0, 180.0
    raise ValueError(f"no automatic bounds for {name!r}")


def _payload_model(body: dict) -> tuple[Experiment, list[Site]]:
    return parse_experiment(body["experiment"]), parse_sites(body["sites"])


def api_nuclei(_body: dict) -> dict:
    return {
        "nuclei": [
            {
                "symbol": nuc.symbol,
                "spin": nuc.spin,
                "gamma": nuc.gamma,
                "quadrupole_moment": nuc.quadrupole_moment,
            }
            for nuc in sorted(NUCLEI.values(), key=lambda n: (n.spin, n.symbol))
            if nuc.spin >= 1.0
        ]
    }


def api_window(body: dict) -> dict:
    experiment, sites = _payload_model(body)
    low, high = suggest_window(sites, experiment)
    return {"low": low, "high": high, "reference": experiment.reference_frequency}


def api_simulate(body: dict) -> dict:
    experiment, sites = _payload_model(body)
    window = body.get("window") or {}
    if window.get("low") is None or window.get("high") is None:
        low, high = suggest_window(sites, experiment)
    else:
        low, high = float(window["low"]), float(window["high"])
    if high <= low:
        raise ValueError("the upper edge of the window must exceed the lower one")
    points = int(window.get("points") or 2400)
    points = max(64, min(points, 20000))
    x = np.linspace(low, high, points)

    components = simulate_sites(
        x, sites, experiment, divisions=int(body.get("divisions") or 24)
    )
    total = components.sum(axis=0)
    peak = float(np.max(total)) if total.size else 0.0
    scale = 1.0 / peak if peak > 0 else 1.0
    return {
        "total": _trace(x, total * scale),
        "components": [_trace(x, component * scale) for component in components],
        "reference": experiment.reference_frequency,
        "low": low,
        "high": high,
    }


def _spectrum_payload(spectrum: Spectrum, label: str) -> dict:
    return {
        "data": _trace(spectrum.freq_mhz, spectrum.intensity),
        "reference": spectrum.reference,
        "low": float(spectrum.freq_mhz[0]),
        "high": float(spectrum.freq_mhz[-1]),
        "points": len(spectrum),
        "label": label,
    }


def api_load(body: dict) -> dict:
    """Load Bruker directories, text files, or a pasted/dropped two-column file."""
    normalize = bool(body.get("normalize", True))
    if body.get("text"):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write(body["text"])
            temporary = handle.name
        spectrum = read_ascii(
            temporary,
            unit=body.get("unit", "MHz"),
            reference=float(body.get("reference") or 0.0),
        )
        label = body.get("name") or "pasted data"
        Path(temporary).unlink(missing_ok=True)
    else:
        raw = str(body.get("path", "")).strip()
        if not raw:
            raise ValueError("give a file or directory path, or drop a file on the page")
        paths = [line.strip() for line in raw.splitlines() if line.strip()]
        for path in paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"{path} does not exist")
        if body.get("format", "bruker") == "bruker":
            pieces = read_bruker_series(
                paths, scale_by_scans=bool(body.get("scale_by_scans", False))
            )
        else:
            pieces = [
                read_ascii(
                    path,
                    unit=body.get("unit", "MHz"),
                    reference=float(body.get("reference") or 0.0),
                )
                for path in paths
            ]
        spectrum = (
            pieces[0]
            if len(pieces) == 1
            else coadd(pieces, mode=body.get("coadd_mode", "mean"))
        )
        label = (
            Path(paths[0]).name
            if len(paths) == 1
            else f"{len(paths)} spectra ({body.get('coadd_mode', 'mean')})"
        )

    if normalize:
        spectrum = spectrum.normalized()
    with STATE.lock:
        STATE.data = spectrum
    return _spectrum_payload(spectrum, label)


def api_demo(body: dict) -> dict:
    """Synthesise a spectrum from the current parameters so a fit can be tried."""
    experiment, sites = _payload_model(body)
    low, high = suggest_window(sites, experiment)
    x = np.linspace(low, high, int(body.get("points") or 3000))
    total = simulate_sites(x, sites, experiment, divisions=60).sum(axis=0)
    peak = float(np.max(total)) or 1.0
    rng = np.random.default_rng(int(body.get("seed") or 0))
    noise = float(body.get("noise", 0.015))
    y = total / peak + rng.normal(0.0, noise, x.size)
    spectrum = Spectrum(x, y, experiment.reference_frequency)
    with STATE.lock:
        STATE.data = spectrum
    payload = _spectrum_payload(spectrum, f"synthetic ({noise * 100:.1f} % noise)")
    payload["truth"] = [asdict(site) for site in sites]
    return payload


def api_clear(_body: dict) -> dict:
    with STATE.lock:
        STATE.data = None
    return {"cleared": True}


def api_bracket(body: dict) -> dict:
    """Bracket Cq from a line position -- the practical way to start an NQR fit."""
    experiment, _ = _payload_model(body)
    low, high = cq_range_for_line(
        experiment.spin,
        float(body["frequency"]),
        transitions=body.get("transition"),
    )
    return {"low": low, "high": high}


def _run_fit(job_id: str, body: dict) -> None:
    try:
        experiment, sites = _payload_model(body)
        with STATE.lock:
            spectrum = STATE.data
        if spectrum is None:
            raise ValueError("load or synthesise a spectrum before fitting")

        free = []
        for entry in body.get("free", []):
            index, name = int(entry["site"]), str(entry["name"])
            low = entry.get("lower")
            high = entry.get("upper")
            if low is None or high is None:
                low, high = _bounds(name, float(getattr(sites[index], name)))
            free.append(FitParameter(index, name, float(low), float(high)))
        if not free:
            raise ValueError("tick at least one parameter to fit")

        share = bool(body.get("share_broadening", True)) and len(sites) > 1

        def tie(trial):
            if share:
                for site in trial[1:]:
                    site.lorentz = trial[0].lorentz
                    site.gauss = trial[0].gauss
            return trial

        strategy = str(body.get("strategy", "multistart"))
        result = fit(
            spectrum,
            sites,
            experiment,
            free,
            baseline_order=int(body.get("baseline_order", 0)),
            global_search=strategy == "global",
            restarts=8 if strategy == "multistart" else 0,
            divisions=int(body.get("divisions") or 40),
        )

        payload = {
            "sites": [asdict(site) for site in result.sites],
            "uncertainties": {
                f"{site}.{name}": (None if not np.isfinite(value) else float(value))
                for (site, name), value in result.uncertainties.items()
            },
            "amplitudes": [float(v) for v in result.amplitudes],
            "model": _trace(result.freq_mhz, result.model),
            "components": [
                _trace(result.freq_mhz, component + result.baseline)
                for component in result.components
            ],
            "baseline": _trace(result.freq_mhz, result.baseline),
            "residual": _trace(result.freq_mhz, result.residual),
            "report": result.report(),
            "rmsd": float(result.rmsd),
            "r_squared": float(result.r_squared),
            "success": bool(result.success),
            "message": result.message,
            "evaluations": int(result.n_evaluations),
            "reference": experiment.reference_frequency,
        }
        with STATE.lock:
            STATE.jobs[job_id] = {"state": "done", "result": payload}
    except Exception as exc:  # surfaced to the browser, not swallowed
        with STATE.lock:
            STATE.jobs[job_id] = {
                "state": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }


def api_fit(body: dict) -> dict:
    """Start a fit in the background and return a job handle to poll."""
    job_id = uuid.uuid4().hex[:12]
    with STATE.lock:
        STATE.jobs[job_id] = {"state": "running"}
    threading.Thread(target=_run_fit, args=(job_id, body), daemon=True).start()
    return {"job": job_id}


def api_job(body: dict) -> dict:
    with STATE.lock:
        job = STATE.jobs.get(str(body.get("job")))
    if job is None:
        raise KeyError("unknown job")
    return job


ROUTES = {
    "/api/nuclei": api_nuclei,
    "/api/window": api_window,
    "/api/simulate": api_simulate,
    "/api/load": api_load,
    "/api/demo": api_demo,
    "/api/clear": api_clear,
    "/api/bracket": api_bracket,
    "/api/fit": api_fit,
    "/api/job": api_job,
}


class Handler(BaseHTTPRequestHandler):
    server_version = "nqrlyze"

    def log_message(self, *_args):  # keep the console for our own messages
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(
            status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8"
        )

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            page = STATIC / "index.html"
            if not page.is_file():
                self._json(500, {"error": f"missing {page}"})
                return
            self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0]
        handler = ROUTES.get(path)
        if handler is None:
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError) as exc:
            self._json(400, {"error": f"malformed request: {exc}"})
            return
        try:
            self._json(200, handler(body))
        except Exception as exc:
            self._json(400, {"error": f"{type(exc).__name__}: {exc}"})


def create_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> int:
    server = create_server(host, port)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"nqrlyze gui on {url}  (ctrl-c to stop)")
    if open_browser:
        import webbrowser

        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0
