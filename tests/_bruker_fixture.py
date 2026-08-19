"""Build a synthetic processed Bruker dataset on disk, for tests and examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_bruker_1d(
    root: Path,
    intensity: np.ndarray,
    sf: float,
    sw_p: float,
    offset: float,
    sfo1: float | None = None,
    ns: int = 16,
    nc_proc: int = 0,
    big_endian: bool = False,
    procno: int = 1,
) -> Path:
    """Write ``root/pdata/<procno>/{1r,procs}`` plus ``root/acqus``."""
    pdata = root / "pdata" / str(procno)
    pdata.mkdir(parents=True, exist_ok=True)
    size = intensity.size

    scaled = np.asarray(intensity, dtype=float) / 2.0**nc_proc
    dtype = np.dtype(">i4" if big_endian else "<i4")
    (pdata / "1r").write_bytes(np.round(scaled).astype(dtype).tobytes())

    (pdata / "procs").write_text(
        "##TITLE= synthetic\n"
        "##JCAMPDX= 5.0\n"
        f"##$SI= {size}\n"
        f"##$SW_p= {sw_p!r}\n"
        f"##$SF= {sf!r}\n"
        f"##$OFFSET= {offset!r}\n"
        f"##$NC_proc= {nc_proc}\n"
        f"##$BYTORDP= {1 if big_endian else 0}\n"
        "##$DTYPP= 0\n"
        "##END=\n"
    )
    (root / "acqus").write_text(
        "##TITLE= synthetic\n"
        f"##$SFO1= {(sfo1 if sfo1 is not None else sf)!r}\n"
        f"##$BF1= {sf!r}\n"
        "##$O1= 0.0\n"
        f"##$NS= {ns}\n"
        f"##$TD= {2 * size}\n"
        "##$NUC1= <35Cl>\n"
        "##$P= (0..3)\n"
        "2.5 0 0 0\n"
        "##END=\n"
    )
    return root
