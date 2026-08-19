"""Reader for processed Bruker data (``pdata/N/1r``).

Frequency axis
--------------
TopSpin stores the ppm value of the *leftmost* point in ``OFFSET`` and the
processed spectral width in ``SW_p`` (Hz) over ``SI`` points, referenced to
``SF`` (MHz).  Point ``i`` counted from the left therefore sits at::

    ppm[i]      = OFFSET - i * SW_p / (SI * SF)
    nu[i] / MHz = SF + (OFFSET * SF - i * SW_p / SI) * 1e-6

(``OFFSET * SF`` is in Hz because one ppm of ``SF`` MHz is exactly ``SF`` Hz.)
This is the same convention nmrglue uses, and it is what makes co-adding
sub-spectra recorded at different transmitter offsets exact.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from ..spectrum import Spectrum

__all__ = ["read_jcamp_parameters", "read_bruker", "read_bruker_series", "find_pdata"]

_ENTRY = re.compile(r"^##\$?(.+?)=\s*(.*)$")
_ARRAY = re.compile(r"^\(\s*\d+\s*\.\.\s*\d+\s*\)\s*$")


def _coerce(text: str):
    text = text.strip()
    if text.startswith("<") and text.endswith(">"):
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def read_jcamp_parameters(path: str | os.PathLike) -> dict:
    """Parse a Bruker ``acqus``/``procs`` style JCAMP-DX parameter file."""
    path = Path(path)
    params: dict = {}
    lines = path.read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        match = _ENTRY.match(lines[i])
        i += 1
        if not match:
            continue
        key, value = match.group(1).strip(), match.group(2).strip()
        if _ARRAY.match(value):
            # Values continue on the following lines until the next '##'.
            chunks: list[str] = []
            while i < len(lines) and not lines[i].startswith("##"):
                chunks.append(lines[i])
                i += 1
            text = " ".join(chunks)
            if "<" in text:
                params[key] = re.findall(r"<([^>]*)>", text)
            else:
                params[key] = [_coerce(tok) for tok in text.split()]
        else:
            params[key] = _coerce(value)
    return params


def find_pdata(path: str | os.PathLike, procno: int | str = 1) -> Path:
    """Resolve any of ``.../1r``, ``.../pdata/1`` or an experiment directory."""
    path = Path(path)
    if path.is_file():
        return path.parent
    if (path / "1r").is_file():
        return path
    candidate = path / "pdata" / str(procno)
    if (candidate / "1r").is_file():
        return candidate
    pdata = path / "pdata"
    if pdata.is_dir():
        for child in sorted(pdata.iterdir(), key=lambda p: p.name):
            if (child / "1r").is_file():
                return child
    raise FileNotFoundError(f"no processed Bruker data (1r) found under {path}")


def _read_binary(path: Path, params: dict) -> np.ndarray:
    dtype_code = int(params.get("DTYPP", 0))
    byte_order = int(params.get("BYTORDP", 0))
    endian = "<" if byte_order == 0 else ">"
    dtype = np.dtype(f"{endian}f8") if dtype_code == 2 else np.dtype(f"{endian}i4")
    raw = np.frombuffer(path.read_bytes(), dtype=dtype).astype(float)
    return raw * 2.0 ** float(params.get("NC_proc", 0))


def read_bruker(
    path: str | os.PathLike,
    procno: int | str = 1,
    imaginary: bool = False,
) -> Spectrum:
    """Read one processed 1D Bruker spectrum.

    Parameters
    ----------
    path
        Experiment directory, ``pdata/N`` directory, or the ``1r`` file itself.
    procno
        Processing number to use when ``path`` is an experiment directory.
    imaginary
        Read ``1i`` instead of ``1r``.
    """
    pdata = find_pdata(path, procno)
    procs = read_jcamp_parameters(pdata / "procs")

    binary = pdata / ("1i" if imaginary else "1r")
    if not binary.is_file():
        raise FileNotFoundError(f"{binary} not found")
    intensity = _read_binary(binary, procs)

    size = int(procs.get("SI", intensity.size))
    intensity = intensity[:size]
    if intensity.size < size:
        raise ValueError(
            f"{binary} holds {intensity.size} points but SI is {size}"
        )

    sf = float(procs["SF"])
    sw_p = float(procs["SW_p"])
    offset = float(procs["OFFSET"])
    index = np.arange(size, dtype=float)
    freq_mhz = sf + (offset * sf - index * sw_p / size) * 1e-6

    meta: dict = {
        "source": str(pdata),
        "procs": procs,
        "SF": sf,
        "SW_p": sw_p,
        "OFFSET": offset,
        "SI": size,
    }
    # acqus lives two levels up from pdata/N.
    acqus_path = pdata.parent.parent / "acqus"
    if acqus_path.is_file():
        acqus = read_jcamp_parameters(acqus_path)
        meta["acqus"] = acqus
        for key in ("SFO1", "BF1", "O1", "NS", "TD", "NUC1", "P", "D"):
            if key in acqus:
                meta[key] = acqus[key]

    return Spectrum(freq_mhz, intensity, reference=sf, meta=meta)


def read_bruker_series(
    paths, procno: int | str = 1, scale_by_scans: bool = False
) -> list[Spectrum]:
    """Read several Bruker experiments, e.g. the pieces of a stepped-frequency set.

    With ``scale_by_scans`` each piece is divided by its ``NS`` so that
    sub-spectra measured with different numbers of scans can be co-added on a
    common intensity footing.
    """
    spectra = []
    for path in paths:
        spec = read_bruker(path, procno)
        if scale_by_scans:
            scans = float(spec.meta.get("NS", 0) or 0)
            if scans <= 0:
                raise ValueError(f"{path}: NS missing, cannot scale by scans")
            spec = Spectrum(
                spec.freq_mhz, spec.intensity / scans, spec.reference, spec.meta
            )
        spectra.append(spec)
    return spectra
