"""Command line interface.

``nqrlyze`` works from JSON job files so a fit is reproducible: the file states
the nucleus, the field, where the data lives, the starting values and what is
allowed to vary, and the result is written back out in the same form.

    nqrlyze gui
    nqrlyze template job > job.json
    nqrlyze coadd expt/1 expt/2 expt/3 --mode mean --out combined.txt
    nqrlyze fit job.json --plot fit.png --out result.json
    nqrlyze validate quest_references/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .coadd import coadd
from .config import (
    build_axis,
    dump_job,
    load_job,
    load_spectrum,
    parse_experiment,
    parse_sites,
)
from .constants import NUCLEI, get_nucleus
from .fit import default_parameters, fit
from .io import read_ascii, read_bruker_series, write_ascii
from .simulate import simulate, simulate_sites
from .spectrum import Spectrum
from .validate import MANIFEST_TEMPLATE, run_manifest

JOB_TEMPLATE = {
    "experiment": {
        "nucleus": "27Al",
        "field": 11.7449,
        "transitions": "ct",
    },
    "data": {
        "format": "bruker",
        "paths": ["expt/10", "expt/11"],
        "procno": 1,
        "coadd": {"mode": "mean", "scale_by_scans": True},
        "window_ppm": [-400, 400],
        "normalize": True,
    },
    "sites": [
        {"label": "Al1", "cq": 5.0, "eta": 0.5, "iso": 60.0,
         "lorentz": 0.002, "gauss": 0.001}
    ],
    "fit": {
        "baseline_order": 0,
        "global_search": True,
        "divisions": 50,
        "free": [
            {"site": 0, "name": "cq", "lower": 0.5, "upper": 12.0},
            {"site": 0, "name": "eta", "lower": 0.0, "upper": 1.0},
            {"site": 0, "name": "iso", "lower": -200.0, "upper": 300.0},
            {"site": 0, "name": "lorentz", "lower": 0.0, "upper": 0.02},
            {"site": 0, "name": "gauss", "lower": 0.0, "upper": 0.02},
        ],
    },
}


def _spectrum_from_job(job, path):
    if "data" not in job:
        raise SystemExit(f"{path}: no 'data' block to fit")
    return load_spectrum(job["data"], Path(path).parent)


def _tie_broadening(share: bool):
    if not share:
        return None

    def tie(sites):
        for site in sites[1:]:
            site.lorentz = sites[0].lorentz
            site.gauss = sites[0].gauss
        return sites

    return tie


def command_template(args) -> int:
    payload = JOB_TEMPLATE if args.kind == "job" else MANIFEST_TEMPLATE
    text = json.dumps(payload, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


def command_info(args) -> int:
    if not args.nucleus:
        for symbol, nuc in sorted(NUCLEI.items(), key=lambda kv: kv[1].spin):
            print(
                f"{symbol:>6}  I = {nuc.spin:<4}  gamma/2pi = {nuc.gamma:>9.4f} MHz/T"
                f"  Q = {nuc.quadrupole_moment:>9.2f} mb"
            )
        return 0
    nuc = get_nucleus(args.nucleus)
    print(f"{nuc.symbol}:  I = {nuc.spin}   gamma/2pi = {nuc.gamma} MHz/T"
          f"   Q = {nuc.quadrupole_moment} mb")
    if args.field:
        print(f"  at {args.field} T:  Larmor = {nuc.larmor(args.field):.6f} MHz")
    if args.larmor:
        print(f"  at {args.larmor} MHz:  field = {nuc.field(args.larmor):.6f} T")
    if args.vzz is not None:
        print(f"  Vzz = {args.vzz} a.u.  ->  Cq = {nuc.vzz_to_cq(args.vzz):.6f} MHz")
    if args.cq is not None:
        print(f"  Cq = {args.cq} MHz  ->  Vzz = {nuc.cq_to_vzz(args.cq):.6f} a.u.")
    return 0


def command_coadd(args) -> int:
    if args.format == "bruker":
        pieces = read_bruker_series(
            args.paths, args.procno, scale_by_scans=args.scale_by_scans
        )
    else:
        pieces = [
            read_ascii(p, args.unit, args.reference) for p in args.paths
        ]
    combined = coadd(
        pieces,
        mode=args.mode,
        dx_mhz=args.step,
        normalize_each=args.normalize_each,
    )
    print(
        f"combined {len(pieces)} spectra -> {len(combined)} points, "
        f"{combined.freq_mhz[0]:.6f} to {combined.freq_mhz[-1]:.6f} MHz "
        f"({args.mode})"
    )
    if args.out:
        write_ascii(args.out, combined, unit=args.out_unit,
                    header=f"co-added ({args.mode}) from {len(pieces)} spectra")
        print(f"wrote {args.out}")
    if args.plot:
        from .plotting import plot_spectra

        plot_spectra(
            pieces + [combined],
            labels=[f"piece {i + 1}" for i in range(len(pieces))] + ["combined"],
            unit=args.out_unit,
            reference=combined.reference,
            title="stepped-frequency co-addition",
            path=args.plot,
        )
        print(f"wrote {args.plot}")
    return 0


def command_simulate(args) -> int:
    job = load_job(args.job)
    experiment = job["experiment"]
    sites = job["sites"]
    if "axis" in job:
        x = build_axis(job["axis"], experiment.reference_frequency)
    elif "data" in job:
        x = _spectrum_from_job(job, args.job).freq_mhz
    else:
        raise SystemExit(f"{args.job}: needs an 'axis' or a 'data' block")

    y = simulate(x, sites, experiment, divisions=args.divisions)
    spectrum = Spectrum(x, y, experiment.reference_frequency)
    print(
        f"simulated {len(sites)} site(s) on {x.size} points, "
        f"{x[0]:.6f} to {x[-1]:.6f} MHz"
    )
    if args.out:
        write_ascii(args.out, spectrum, unit=args.unit, header="nqrlyze simulation")
        print(f"wrote {args.out}")
    if args.plot:
        from .plotting import plot_spectra

        plot_spectra(
            [spectrum],
            labels=["simulation"],
            unit=args.unit,
            reference=experiment.reference_frequency,
            title="nqrlyze simulation",
            path=args.plot,
        )
        print(f"wrote {args.plot}")
    return 0


def command_fit(args) -> int:
    job = load_job(args.job)
    experiment = job["experiment"]
    sites = job["sites"]
    spectrum = _spectrum_from_job(job, args.job)
    settings = dict(job.get("fit", {}))
    free = settings.pop("free", None) or default_parameters(sites)
    share = settings.pop("share_broadening", len(sites) > 1)
    if args.no_global:
        settings["global_search"] = False
    if args.divisions:
        settings["divisions"] = args.divisions

    print(f"fitting {len(spectrum)} points, {len(sites)} site(s), "
          f"{len(free)} free parameter(s)")
    result = fit(
        spectrum,
        sites,
        experiment,
        free,
        tie=_tie_broadening(share),
        verbose=args.verbose,
        **settings,
    )
    print(result.report())

    if args.correlations:
        print("\ncorrelation matrix:")
        names = result.parameter_names
        width = max(len(n) for n in names) + 1
        print(" " * width + "".join(f"{n[:8]:>9}" for n in names))
        for row, name in enumerate(names):
            cells = "".join(f"{result.correlation[row, col]:>9.3f}"
                            for col in range(len(names)))
            print(f"{name:<{width}}{cells}")

    if args.out:
        dump_job(args.out, experiment, result.sites)
        print(f"\nwrote {args.out}")
    if args.spectrum_out:
        write_ascii(
            args.spectrum_out,
            Spectrum(result.freq_mhz, result.model, experiment.reference_frequency),
            unit=args.unit,
            header="nqrlyze best fit",
        )
        print(f"wrote {args.spectrum_out}")
    if args.plot:
        from .plotting import plot_fit

        plot_fit(
            result,
            unit=args.unit,
            reference=experiment.reference_frequency,
            title=f"{experiment.nucleus or f'I = {experiment.spin}'} fit",
            path=args.plot,
        )
        print(f"wrote {args.plot}")
    return 0 if result.success else 1


def command_gui(args) -> int:
    from .webapp import serve

    return serve(host=args.host, port=args.port, open_browser=not args.no_browser)


def command_validate(args) -> int:
    results = run_manifest(args.manifest, divisions=args.divisions)
    if not results:
        print("manifest contains no cases")
        return 1
    for comparison in results:
        print(comparison.report())
    failed = [c for c in results if not c.passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} cases within tolerance")
    if failed:
        print(
            "\nA failing case is a convention mismatch, not necessarily a bug.\n"
            "  shifted peak, correct width   -> reference/offset convention\n"
            "  correct peak, wrong width     -> Cq, eta or Larmor frequency\n"
            "  positions match, shape does not -> transition selection or rf averaging\n"
            "See docs/conventions.md for what nqrlyze assumes."
        )
    return 0 if not failed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nqrlyze",
        description="Fit exact quadrupolar NMR and NQR powder patterns.",
    )
    parser.add_argument("--version", action="version", version=f"nqrlyze {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser("template", help="print a starter JSON file")
    template.add_argument("kind", choices=["job", "manifest"])
    template.add_argument("-o", "--out")
    template.set_defaults(func=command_template)

    info = subparsers.add_parser("info", help="nuclear data and unit conversions")
    info.add_argument("nucleus", nargs="?")
    info.add_argument("--field", type=float, help="magnetic field in tesla")
    info.add_argument("--larmor", type=float, help="Larmor frequency in MHz")
    info.add_argument("--vzz", type=float, help="convert Vzz in a.u. to Cq")
    info.add_argument("--cq", type=float, help="convert Cq in MHz to Vzz")
    info.set_defaults(func=command_info)

    combine = subparsers.add_parser(
        "coadd", help="join stepped-frequency sub-spectra onto one axis"
    )
    combine.add_argument("paths", nargs="+")
    combine.add_argument("--format", choices=["bruker", "ascii"], default="bruker")
    combine.add_argument("--mode", choices=["sum", "mean", "skyline"], default="mean")
    combine.add_argument("--procno", default=1)
    combine.add_argument("--step", type=float, help="output step in MHz")
    combine.add_argument("--unit", default="MHz", help="input unit for ascii")
    combine.add_argument("--reference", type=float, default=0.0)
    combine.add_argument("--out-unit", default="MHz")
    combine.add_argument("--scale-by-scans", action="store_true")
    combine.add_argument("--normalize-each", action="store_true")
    combine.add_argument("-o", "--out")
    combine.add_argument("--plot")
    combine.set_defaults(func=command_coadd)

    sim = subparsers.add_parser("simulate", help="simulate from a job file")
    sim.add_argument("job")
    sim.add_argument("--divisions", type=int, default=50)
    sim.add_argument("--unit", default="MHz")
    sim.add_argument("-o", "--out")
    sim.add_argument("--plot")
    sim.set_defaults(func=command_simulate)

    fitter = subparsers.add_parser("fit", help="fit a spectrum from a job file")
    fitter.add_argument("job")
    fitter.add_argument("--divisions", type=int)
    fitter.add_argument("--no-global", action="store_true",
                        help="skip the global search (starting values are good)")
    fitter.add_argument("--unit", default="ppm")
    fitter.add_argument("--correlations", action="store_true")
    fitter.add_argument("-o", "--out", help="write the fitted parameters as a job file")
    fitter.add_argument("--spectrum-out", help="write the best-fit spectrum")
    fitter.add_argument("--plot")
    fitter.add_argument("-v", "--verbose", action="store_true")
    fitter.set_defaults(func=command_fit)

    gui = subparsers.add_parser(
        "gui", help="open the interactive simulate-and-fit interface in a browser"
    )
    gui.add_argument("--port", type=int, default=8765)
    gui.add_argument("--host", default="127.0.0.1",
                     help="loopback by default; do not expose this on a public interface")
    gui.add_argument("--no-browser", action="store_true")
    gui.set_defaults(func=command_gui)

    validator = subparsers.add_parser(
        "validate", help="check agreement against QUEST reference spectra"
    )
    validator.add_argument("manifest")
    validator.add_argument("--divisions", type=int, default=60)
    validator.set_defaults(func=command_validate)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"nqrlyze: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
