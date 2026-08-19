"""Proving that this simulator and QUEST agree.

Nothing here compares to QUEST automatically -- QUEST is a MATLAB program and
this is a Python package, so the only honest bridge is a set of spectra QUEST
itself produced.  The workflow is:

1. In QUEST, simulate a handful of cases that span what you actually measure --
   different spins, ``Cq``, ``eta``, and both the high-field and the NQR limit.
   Export each as a two-column text file.
2. Write a manifest naming each file and the parameters that produced it.
3. Run ``nqrlyze validate manifest.json``.

The manifest is the contract.  If a case disagrees, the metrics say *how*:
a shifted peak means a frequency-convention mismatch, a stretched pattern means
a ``Cq``, ``eta`` or ``nu_L`` mismatch, and matching singularity positions with
mismatched intensities point at the transition selection or the rf averaging.

The intensity scale and any constant offset are fitted out before comparing,
because those carry no physics -- QUEST's vertical scale is arbitrary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import parse_experiment, parse_sites
from .io.ascii import read_ascii
from .simulate import Experiment, Site, simulate
from .spectrum import Spectrum

__all__ = [
    "Comparison",
    "compare",
    "find_singularities",
    "run_manifest",
    "MANIFEST_TEMPLATE",
]

MANIFEST_TEMPLATE: dict[str, Any] = {
    "description": "QUEST reference spectra for nqrlyze validation",
    "tolerance": {"rmsd": 0.02, "peak_shift_khz": 1.0, "singularity_khz": 2.0},
    "cases": [
        {
            "name": "27Al CT, Cq = 5.2 MHz, eta = 0.42",
            "file": "quest_27Al_ct.txt",
            "unit": "kHz",
            "experiment": {
                "nucleus": "27Al",
                "larmor": 130.318,
                "transitions": "ct",
            },
            "sites": [{"cq": 5.2, "eta": 0.42, "lorentz": 0.001}],
        },
        {
            "name": "35Cl NQR",
            "file": "quest_35Cl_nqr.txt",
            "unit": "MHz",
            "experiment": {"nucleus": "35Cl", "larmor": 0.0},
            "sites": [{"cq": 60.0, "eta": 0.2, "lorentz": 0.01}],
        },
    ],
}


@dataclass
class Comparison:
    """How closely a simulation reproduces a reference spectrum."""

    name: str
    n_points: int
    rmsd: float
    """Root-mean-square difference as a fraction of the reference peak height."""
    max_deviation: float
    correlation: float
    peak_shift_khz: float
    singularity_shifts_khz: np.ndarray
    reference_singularities_mhz: np.ndarray
    simulated_singularities_mhz: np.ndarray
    scale: float
    offset: float
    passed: bool
    notes: list[str] = field(default_factory=list)

    def report(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        lines = [
            f"[{mark}] {self.name}",
            f"       points {self.n_points}   RMSD {100 * self.rmsd:.3f} % of peak"
            f"   max dev {100 * self.max_deviation:.3f} %"
            f"   correlation {self.correlation:.6f}",
            f"       peak shift {self.peak_shift_khz:+.3f} kHz",
        ]
        if self.singularity_shifts_khz.size:
            shifts = ", ".join(f"{v:+.2f}" for v in self.singularity_shifts_khz)
            lines.append(f"       singularity shifts / kHz: {shifts}")
        else:
            lines.append("       singularity shifts: none matched")
        for note in self.notes:
            lines.append(f"       note: {note}")
        return "\n".join(lines)


def find_singularities(
    spectrum: Spectrum, prominence: float = 0.02, min_separation_khz: float = 0.0
) -> np.ndarray:
    """Frequencies (MHz) of the local maxima that define a powder pattern.

    Powder singularities -- the steps and horns whose positions carry ``Cq`` and
    ``eta`` -- are far more diagnostic than the overall lineshape, because they
    depend on frequencies alone and not on intensities or broadening.
    """
    from scipy.signal import find_peaks

    y = np.asarray(spectrum.intensity, float)
    span = np.max(y) - np.min(y)
    if span <= 0:
        return np.empty(0)
    normalized = (y - np.min(y)) / span
    dx_khz = float(np.abs(np.median(np.diff(spectrum.freq_mhz)))) * 1e3
    distance = (
        max(1, int(round(min_separation_khz / dx_khz))) if dx_khz > 0 else 1
    )
    peaks, _ = find_peaks(normalized, prominence=prominence, distance=distance)
    return spectrum.freq_mhz[peaks]


def _match(reference: np.ndarray, simulated: np.ndarray, window_mhz: float):
    """Pair up singularities that fall within ``window_mhz`` of each other."""
    shifts = []
    for value in reference:
        if simulated.size == 0:
            continue
        nearest = simulated[np.argmin(np.abs(simulated - value))]
        if abs(nearest - value) <= window_mhz:
            shifts.append((nearest - value) * 1e3)
    return np.array(shifts)


def compare(
    reference: Spectrum,
    sites: Sequence[Site],
    experiment: Experiment,
    name: str = "case",
    divisions: int = 60,
    tolerance: dict[str, float] | None = None,
    match_window_khz: float = 20.0,
) -> Comparison:
    """Simulate on the reference's own axis and quantify the agreement."""
    tolerance = {
        "rmsd": 0.02,
        "peak_shift_khz": 1.0,
        "singularity_khz": 2.0,
        **(tolerance or {}),
    }

    x = reference.freq_mhz
    y = np.asarray(reference.intensity, float)
    model = simulate(x, list(sites), experiment, divisions=divisions)

    # QUEST's vertical scale and zero are arbitrary: fit them out.
    design = np.stack([model, np.ones_like(model)], axis=1)
    (scale, offset), *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ np.array([scale, offset])

    height = float(np.max(y) - np.min(y))
    residual = fitted - y
    rmsd = float(np.sqrt(np.mean(residual**2))) / height if height else np.inf
    max_dev = float(np.max(np.abs(residual))) / height if height else np.inf
    if np.std(y) > 0 and np.std(fitted) > 0:
        correlation = float(np.corrcoef(y, fitted)[0, 1])
    else:
        correlation = np.nan

    peak_shift = (x[int(np.argmax(fitted))] - x[int(np.argmax(y))]) * 1e3

    simulated = Spectrum(x, fitted, reference.reference)
    reference_peaks = find_singularities(reference)
    simulated_peaks = find_singularities(simulated)
    shifts = _match(reference_peaks, simulated_peaks, match_window_khz * 1e-3)

    notes: list[str] = []
    if scale <= 0:
        notes.append("best-fit scale is not positive -- check the sign of the data")
    if reference_peaks.size and not shifts.size:
        notes.append("no singularity matched inside the window")
    if len(reference_peaks) != len(simulated_peaks):
        notes.append(
            f"{len(reference_peaks)} singularities in the reference vs "
            f"{len(simulated_peaks)} simulated"
        )

    passed = (
        rmsd <= tolerance["rmsd"]
        and abs(peak_shift) <= tolerance["peak_shift_khz"]
        and (
            not shifts.size
            or float(np.max(np.abs(shifts))) <= tolerance["singularity_khz"]
        )
    )
    return Comparison(
        name=name,
        n_points=int(x.size),
        rmsd=rmsd,
        max_deviation=max_dev,
        correlation=correlation,
        peak_shift_khz=float(peak_shift),
        singularity_shifts_khz=shifts,
        reference_singularities_mhz=reference_peaks,
        simulated_singularities_mhz=simulated_peaks,
        scale=float(scale),
        offset=float(offset),
        passed=bool(passed),
        notes=notes,
    )


def run_manifest(path: str | Path, divisions: int = 60) -> list[Comparison]:
    """Run every case in a validation manifest."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    tolerance = manifest.get("tolerance", {})
    results = []
    for index, case in enumerate(manifest.get("cases", [])):
        name = case.get("name", f"case {index + 1}")
        data_path = (path.parent / case["file"]).resolve()
        experiment = parse_experiment(case["experiment"])
        sites = parse_sites(case["sites"])
        reference = read_ascii(
            data_path,
            unit=case.get("unit", "MHz"),
            reference=case.get("reference", experiment.reference_frequency),
            column=case.get("column", 1),
        )
        if "window_mhz" in case:
            low, high = case["window_mhz"]
            reference = reference.crop(low, high)
        results.append(
            compare(
                reference,
                sites,
                experiment,
                name=name,
                divisions=case.get("divisions", divisions),
                tolerance={**tolerance, **case.get("tolerance", {})},
            )
        )
    return results
