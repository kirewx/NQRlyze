"""JSON job files: describe an experiment, its sites and what to fit.

Kept to the standard library on purpose -- a job file is data, and a fit that
can be re-run from a file is a fit you can put in a paper's supporting
information.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .constants import get_nucleus
from .fit import FitParameter
from .simulate import Experiment, Site
from .spectrum import Spectrum

__all__ = [
    "load_job",
    "parse_experiment",
    "parse_sites",
    "parse_parameters",
    "dump_job",
    "load_spectrum",
    "build_axis",
]

_SITE_FIELDS = {
    "cq", "eta", "iso", "span", "skew", "alpha", "beta", "gamma",
    "lorentz", "gauss", "weight", "label",
}


def parse_experiment(spec: dict[str, Any]) -> Experiment:
    """Build an :class:`Experiment` from a job file's ``experiment`` block."""
    spec = dict(spec)
    nucleus = spec.pop("nucleus", None)
    field = spec.pop("field", None)
    larmor = spec.pop("larmor", None)
    spin = spec.pop("spin", None)
    extra = {
        key: spec.pop(key)
        for key in ("reference", "transitions", "rf_average")
        if key in spec
    }
    if spec:
        raise ValueError(f"unknown keys in experiment block: {sorted(spec)}")

    if nucleus:
        nuc = get_nucleus(nucleus)
        if field is not None and larmor is not None:
            raise ValueError("give either field or larmor, not both")
        if field is not None:
            larmor = nuc.larmor(field)
        return Experiment(
            spin=nuc.spin, larmor=larmor or 0.0, nucleus=nuc.symbol, **extra
        )
    if spin is None:
        raise ValueError("experiment needs either a nucleus or a spin")
    return Experiment(spin=float(spin), larmor=float(larmor or 0.0), **extra)


def parse_sites(spec: list[dict[str, Any]]) -> list[Site]:
    """Build the site list from a job file's ``sites`` block."""
    if not spec:
        raise ValueError("at least one site is required")
    sites = []
    for entry in spec:
        unknown = set(entry) - _SITE_FIELDS
        if unknown:
            raise ValueError(f"unknown site keys: {sorted(unknown)}")
        sites.append(Site(**entry))
    return sites


def parse_parameters(spec, sites) -> list[FitParameter]:
    """Build the free-parameter list from a job file's ``fit.free`` block."""
    params = []
    for entry in spec:
        entry = dict(entry)
        site = int(entry.pop("site", 0))
        name = entry.pop("name")
        lower = entry.pop("lower", None)
        upper = entry.pop("upper", None)
        if entry:
            raise ValueError(f"unknown keys in free parameter: {sorted(entry)}")
        params.append(FitParameter(site, name, lower, upper))
    return params


def load_job(path: str | Path) -> dict[str, Any]:
    """Read a job file and expand its ``experiment`` and ``sites`` blocks."""
    path = Path(path)
    job = json.loads(path.read_text())
    if "experiment" not in job:
        raise ValueError(f"{path}: no 'experiment' block")
    job["experiment"] = parse_experiment(job["experiment"])
    job["sites"] = parse_sites(job.get("sites", []))
    fit_spec = job.get("fit", {})
    if "free" in fit_spec:
        fit_spec["free"] = parse_parameters(fit_spec["free"], job["sites"])
    job["fit"] = fit_spec
    job["_path"] = path
    return job


def dump_job(path: str | Path, experiment: Experiment, sites, fit_spec=None) -> None:
    """Write a job file, e.g. to record the result of a fit."""
    payload: dict[str, Any] = {
        "experiment": {
            key: value
            for key, value in {
                "nucleus": experiment.nucleus,
                "spin": None if experiment.nucleus else experiment.spin,
                "larmor": experiment.larmor,
                "reference": experiment.reference,
                "transitions": experiment.transitions,
            }.items()
            if value not in (None, "")
        },
        "sites": [
            {k: v for k, v in asdict(site).items() if v not in (0.0, "")}
            or {"cq": 0.0}
            for site in sites
        ],
    }
    if fit_spec:
        payload["fit"] = fit_spec
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def load_spectrum(spec: dict[str, Any], base_dir: str | Path = ".") -> "Spectrum":
    """Load the ``data`` block of a job file into a single :class:`Spectrum`.

    Accepts one dataset (``path``) or several (``paths``), in which case the
    pieces are joined with :func:`nqrlyze.coadd.coadd` before anything else
    happens -- which is how stepped-frequency data has to be handled.
    Windowing is applied last, in whichever unit is convenient.
    """
    from .coadd import coadd
    from .io import read_ascii, read_bruker, read_bruker_series

    base = Path(base_dir)
    spec = dict(spec)
    fmt = spec.pop("format", "bruker").lower()
    paths = spec.pop("paths", None)
    single = spec.pop("path", None)
    if paths is None and single is None:
        raise ValueError("data block needs 'path' or 'paths'")
    if paths is not None and single is not None:
        raise ValueError("give either 'path' or 'paths', not both")
    entries = [single] if paths is None else list(paths)
    resolved = [str((base / entry).resolve()) for entry in entries]

    coadd_spec = dict(spec.pop("coadd", {}) or {})
    procno = spec.pop("procno", 1)

    if fmt == "bruker":
        pieces = read_bruker_series(
            resolved, procno, scale_by_scans=coadd_spec.pop("scale_by_scans", False)
        )
    elif fmt == "ascii":
        unit = spec.pop("unit", "MHz")
        reference = spec.pop("reference", 0.0)
        column = spec.pop("column", 1)
        pieces = [read_ascii(p, unit, reference, column) for p in resolved]
    else:
        raise ValueError(f"unknown data format {fmt!r}; use 'bruker' or 'ascii'")

    if len(pieces) == 1 and not coadd_spec:
        spectrum = pieces[0]
    else:
        spectrum = coadd(pieces, **{"mode": "mean", **coadd_spec})

    if "window_mhz" in spec:
        low, high = spec.pop("window_mhz")
        spectrum = spectrum.crop(low, high)
    if "window_ppm" in spec:
        low, high = sorted(spec.pop("window_ppm"))
        ref = spectrum.reference
        if ref == 0:
            raise ValueError("window_ppm needs a reference frequency")
        spectrum = spectrum.crop(ref * (1 + low * 1e-6), ref * (1 + high * 1e-6))
    if "window_khz" in spec:
        low, high = sorted(spec.pop("window_khz"))
        ref = spectrum.reference
        spectrum = spectrum.crop(ref + low * 1e-3, ref + high * 1e-3)
    if spec.pop("normalize", False):
        spectrum = spectrum.normalized()
    if spec:
        raise ValueError(f"unknown keys in data block: {sorted(spec)}")
    return spectrum


def build_axis(spec: dict[str, Any], reference: float) -> "np.ndarray":
    """Build a simulation axis in absolute MHz from an ``axis`` block."""
    import numpy as np

    spec = dict(spec)
    unit = spec.pop("unit", "MHz").lower()
    low = float(spec.pop("low"))
    high = float(spec.pop("high"))
    points = int(spec.pop("points", 4096))
    if spec:
        raise ValueError(f"unknown keys in axis block: {sorted(spec)}")
    if low >= high:
        low, high = high, low
    if unit == "mhz":
        return np.linspace(low, high, points)
    if unit == "khz":
        return reference + np.linspace(low, high, points) * 1e-3
    if unit == "ppm":
        if reference == 0:
            raise ValueError("a ppm axis needs a reference frequency")
        return reference * (1.0 + np.linspace(low, high, points) * 1e-6)
    raise ValueError(f"unknown axis unit {unit!r}")
