"""End-to-end demonstration on synthetic data -- no measurement required.

Writes a stepped-frequency 27Al dataset in Bruker format, joins the pieces,
fits Cq, eta, the isotropic shift and both broadenings from a poor starting
guess, and saves the overlay.

    python examples/demo.py [output-directory]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from _bruker_fixture import write_bruker_1d  # noqa: E402

from nqrlyze import Experiment, Site, coadd, fit, read_bruker_series, simulate
from nqrlyze.fit import FitParameter

NUCLEUS = "27Al"
FIELD = 11.7449
TRUTH = Site(cq=5.2, eta=0.42, iso=62.0, lorentz=0.0012, gauss=0.0020, label="Al1")
PIECE_OFFSETS_KHZ = (-28.0, -8.0, 12.0)
SUB_SPECTRUM_WIDTH_HZ = 30_000.0
SIZE = 4096


def write_pieces(root: Path, experiment: Experiment) -> list[Path]:
    """Fake a stepped-frequency experiment: three overlapping Bruker datasets."""
    sf = experiment.larmor
    rng = np.random.default_rng(11)
    paths = []
    # One intensity scale for every piece, as a real experiment would have:
    # normalising each piece to its own maximum would blow up the noise in the
    # pieces that only catch the tail of the pattern.
    whole = np.linspace(sf - 0.05, sf + 0.03, 8000)
    peak = simulate(whole, TRUTH, experiment, divisions=70).max()
    for index, centre_khz in enumerate(PIECE_OFFSETS_KHZ):
        offset_ppm = (
            (centre_khz * 1e-3 + SUB_SPECTRUM_WIDTH_HZ / 2e6) / sf * 1e6
        )
        axis = np.arange(SIZE, dtype=float)
        freq = np.sort(sf + (offset_ppm * sf - axis * SUB_SPECTRUM_WIDTH_HZ / SIZE) * 1e-6)
        clean = simulate(freq, TRUTH, experiment, divisions=70)
        noisy = 4.0e5 * clean / peak + rng.normal(0, 2.0e3, SIZE)
        path = root / f"expt{10 + index}"
        write_bruker_1d(
            path, noisy[::-1], sf=sf, sw_p=SUB_SPECTRUM_WIDTH_HZ,
            offset=offset_ppm, ns=512,
        )
        paths.append(path)
    return paths


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path("demo-output")
    out.mkdir(parents=True, exist_ok=True)

    experiment = Experiment.from_nucleus(NUCLEUS, field=FIELD, transitions="ct")
    print(f"{NUCLEUS} at {FIELD} T -> Larmor {experiment.larmor:.4f} MHz")
    print(f"truth: Cq = {TRUTH.cq} MHz, eta = {TRUTH.eta}, iso = {TRUTH.iso} ppm, "
          f"L = {TRUTH.lorentz * 1e3} kHz, G = {TRUTH.gauss * 1e3} kHz\n")

    paths = write_pieces(out, experiment)
    print(f"wrote {len(paths)} Bruker sub-spectra to {out}")

    pieces = read_bruker_series(paths, scale_by_scans=True)
    for path, piece in zip(paths, pieces):
        print(f"  {path.name}: {len(piece)} points, "
              f"{piece.freq_mhz[0]:.5f} to {piece.freq_mhz[-1]:.5f} MHz")

    data = coadd(pieces, mode="mean").normalized()
    print(f"joined -> {len(data)} points, "
          f"{data.freq_mhz[0]:.5f} to {data.freq_mhz[-1]:.5f} MHz\n")

    start = [Site(cq=2.0, eta=0.9, iso=0.0, lorentz=0.005, gauss=0.005, label="Al1")]
    free = [
        FitParameter(0, "cq", 0.5, 12.0),
        FitParameter(0, "eta", 0.0, 1.0),
        FitParameter(0, "iso", -200.0, 300.0),
        FitParameter(0, "lorentz", 0.0, 0.02),
        FitParameter(0, "gauss", 0.0, 0.02),
    ]
    print("fitting from a deliberately poor starting guess "
          "(Cq = 2.0, eta = 0.9, iso = 0)...")
    result = fit(data, start, experiment, free, baseline_order=1)
    print()
    print(result.report())

    # Lorentzian and Gaussian widths are strongly correlated: what a lineshape
    # really constrains is the total Voigt width, and only high signal-to-noise
    # data splits it into the two components.
    voigt = lambda l, g: 0.5346 * l + np.sqrt(0.2166 * l**2 + g**2)
    best = result.sites[0]
    print(f"\n  total Voigt FWHM: fitted {voigt(best.lorentz, best.gauss) * 1e3:.3f} kHz"
          f"  vs true {voigt(TRUTH.lorentz, TRUTH.gauss) * 1e3:.3f} kHz")

    try:
        from nqrlyze.plotting import plot_fit

        plot_fit(
            result,
            unit="ppm",
            reference=experiment.larmor,
            title=f"{NUCLEUS} central transition, {FIELD} T",
            path=out / "fit.png",
        )
        print(f"\nwrote {out / 'fit.png'}")
    except ImportError:
        print("\n(matplotlib not installed -- skipping the plot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
