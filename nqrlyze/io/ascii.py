"""Two-column text spectra: frequency then intensity.

Deliberately tolerant -- comment lines (``#``, ``%``, ``;``, ``//``), blank
lines, a text header, and whitespace, comma, semicolon or tab separators are all
accepted, which covers what QUEST, TopSpin and most plotting tools export.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from ..spectrum import Spectrum

__all__ = ["read_ascii", "write_ascii"]

_COMMENT = ("#", "%", ";", "//", "!")
_SPLIT = re.compile(r"[\s,;]+")

_UNIT_TO_MHZ = {
    "mhz": 1.0,
    "khz": 1e-3,
    "hz": 1e-6,
    "ghz": 1e3,
}


def read_ascii(
    path: str | os.PathLike,
    unit: str = "MHz",
    reference: float = 0.0,
    column: int = 1,
) -> Spectrum:
    """Read a two-column spectrum.

    Parameters
    ----------
    unit
        Unit of the first column: ``MHz``, ``kHz``, ``Hz``, ``GHz``, or ``ppm``.
        ``ppm`` and offset units are converted to absolute MHz using
        ``reference``.
    reference
        Reference frequency in MHz.  Required for ``ppm``; for ``kHz``/``Hz`` it
        is treated as an offset origin when non-zero.
    column
        Which column holds the intensity (0-based); the first column is the axis.
    """
    path = Path(path)
    axis: list[float] = []
    values: list[float] = []
    for line in path.read_text(errors="replace").splitlines():
        text = line.strip()
        if not text or text.startswith(_COMMENT):
            continue
        parts = [p for p in _SPLIT.split(text) if p]
        if len(parts) <= column:
            continue
        try:
            x = float(parts[0])
            y = float(parts[column])
        except ValueError:
            continue  # header row
        axis.append(x)
        values.append(y)

    if not axis:
        raise ValueError(f"{path}: no numeric data found")

    x = np.asarray(axis)
    key = unit.strip().lower()
    if key == "ppm":
        if reference == 0:
            raise ValueError("a ppm axis needs a reference frequency in MHz")
        freq = reference * (1.0 + x * 1e-6)
    elif key in _UNIT_TO_MHZ:
        freq = x * _UNIT_TO_MHZ[key]
        if key in ("khz", "hz") and reference != 0:
            freq = reference + freq
    else:
        raise ValueError(f"unknown unit {unit!r}")

    return Spectrum(freq, np.asarray(values), reference, {"source": str(path)})


def write_ascii(
    path: str | os.PathLike,
    spectrum: Spectrum,
    unit: str = "MHz",
    header: str = "",
) -> None:
    """Write a two-column spectrum, highest frequency first (the NMR convention)."""
    key = unit.strip().lower()
    if key == "ppm":
        axis = spectrum.ppm
    elif key == "khz":
        axis = spectrum.khz
    elif key in _UNIT_TO_MHZ:
        axis = spectrum.freq_mhz / _UNIT_TO_MHZ[key]
    else:
        raise ValueError(f"unknown unit {unit!r}")

    lines = [f"# {line}" for line in header.splitlines() if header]
    lines.append(f"# {unit}\tintensity")
    for x, y in zip(axis[::-1], spectrum.intensity[::-1]):
        lines.append(f"{x:.10g}\t{y:.10g}")
    Path(path).write_text("\n".join(lines) + "\n")
