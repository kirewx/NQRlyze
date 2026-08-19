"""Overlay and residual plots.

matplotlib is an optional dependency; importing this module without it raises a
clear error rather than failing at draw time.

The fit overlay is the standard two-panel figure: data and model share one axis,
the residual sits underneath on its own, and nothing is ever drawn on a second
y-scale.  Site components use a fixed, colourblind-safe order (checked for
deuteranopic and tritanopic separation), assigned per site and never cycled, so
a site keeps its colour when another one is added or removed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .fit import FitResult
from .spectrum import Spectrum

__all__ = ["plot_fit", "plot_spectra", "SITE_COLORS"]

#: Fixed categorical order for site components.
SITE_COLORS = ("#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#0072B2")

_DATA = "#3d3d3d"
_MODEL = "#0072B2"
_RESIDUAL = "#6e6e6e"
_MUTED = "#9a9a9a"


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plotting needs matplotlib: pip install 'nqrlyze[plot]'"
        ) from exc
    return plt


def _axis(freq_mhz: np.ndarray, unit: str, reference: float):
    unit_key = unit.strip().lower()
    if unit_key == "ppm":
        if reference == 0:
            raise ValueError("a ppm axis needs a non-zero reference frequency")
        return (freq_mhz - reference) / reference * 1e6, "shift / ppm", True
    if unit_key == "khz":
        return (freq_mhz - reference) * 1e3, "offset / kHz", True
    if unit_key == "mhz":
        return freq_mhz.copy(), "frequency / MHz", False
    raise ValueError(f"unknown unit {unit!r}; use ppm, kHz or MHz")


def _tidy(ax, invert: bool):
    ax.grid(True, color="#e4e4e4", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bdbdbd")
    ax.tick_params(colors="#5a5a5a", labelsize=9)
    if invert:
        ax.invert_xaxis()


def plot_fit(
    result: FitResult,
    unit: str = "ppm",
    reference: float | None = None,
    title: str = "",
    show_components: bool | None = None,
    path: str | None = None,
    figsize: tuple[float, float] = (7.5, 5.5),
    dpi: int = 150,
):
    """Two-panel overlay: data, model and components above, residual below.

    Returns the matplotlib figure.  Pass ``path`` to save it.
    """
    plt = _pyplot()
    if reference is None:
        reference = float(np.median(result.freq_mhz))
        if unit.strip().lower() == "ppm":
            raise ValueError("a ppm axis needs an explicit reference frequency")
    x, xlabel, invert = _axis(result.freq_mhz, unit, reference)

    if show_components is None:
        show_components = result.components.shape[0] > 1

    figure, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=figsize,
        dpi=dpi,
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
    )

    top.plot(x, result.data, color=_DATA, linewidth=1.1, label="experiment")
    top.plot(x, result.model, color=_MODEL, linewidth=1.4, label="fit")
    if show_components:
        for index, component in enumerate(result.components):
            site = result.sites[index]
            label = site.label or f"site {index + 1}"
            share = (
                100.0 * result.amplitudes[index] / np.sum(result.amplitudes)
                if np.sum(result.amplitudes) > 0
                else float("nan")
            )
            top.plot(
                x,
                component + result.baseline,
                color=SITE_COLORS[index % len(SITE_COLORS)],
                linewidth=1.1,
                alpha=0.9,
                label=f"{label} · Cq {site.cq:.3f} MHz, η {site.eta:.3f}"
                f" ({share:.0f} %)",
            )
    if np.any(result.baseline != 0):
        top.plot(
            x, result.baseline, color=_MUTED, linewidth=0.9, linestyle="--",
            label="baseline",
        )

    top.set_ylabel("intensity", fontsize=9, color="#5a5a5a")
    top.legend(frameon=False, fontsize=8, loc="upper right")
    if title:
        top.set_title(title, fontsize=11, color="#1f1f1f", loc="left", pad=10)
    _tidy(top, invert)

    bottom.axhline(0.0, color=_MUTED, linewidth=0.8)
    bottom.plot(x, result.residual, color=_RESIDUAL, linewidth=0.9)
    bottom.set_ylabel("residual", fontsize=9, color="#5a5a5a")
    bottom.set_xlabel(xlabel, fontsize=9, color="#5a5a5a")
    _tidy(bottom, False)
    if invert:
        bottom.set_xlim(top.get_xlim())

    figure.align_ylabels()
    if path:
        figure.savefig(path, bbox_inches="tight", facecolor="white")
    return figure


def plot_spectra(
    spectra: Sequence[Spectrum],
    labels: Sequence[str] | None = None,
    unit: str = "MHz",
    reference: float | None = None,
    title: str = "",
    offset: float = 0.0,
    path: str | None = None,
    figsize: tuple[float, float] = (7.5, 4.2),
    dpi: int = 150,
):
    """Overlay several spectra, optionally stacked by ``offset``."""
    plt = _pyplot()
    spectra = list(spectra)
    if not spectra:
        raise ValueError("no spectra to plot")
    if labels is None:
        labels = [
            spec.meta.get("source", f"spectrum {i + 1}") for i, spec in enumerate(spectra)
        ]
    if reference is None:
        reference = spectra[0].reference

    figure, ax = plt.subplots(figsize=figsize, dpi=dpi)
    invert = False
    for index, (spec, label) in enumerate(zip(spectra, labels)):
        x, xlabel, invert = _axis(spec.freq_mhz, unit, reference)
        ax.plot(
            x,
            spec.intensity + index * offset,
            color=SITE_COLORS[index % len(SITE_COLORS)] if len(spectra) > 1 else _DATA,
            linewidth=1.1,
            label=str(label),
        )
    ax.set_xlabel(xlabel, fontsize=9, color="#5a5a5a")
    ax.set_ylabel("intensity", fontsize=9, color="#5a5a5a")
    if len(spectra) > 1:
        ax.legend(frameon=False, fontsize=8)
    if title:
        ax.set_title(title, fontsize=11, color="#1f1f1f", loc="left", pad=10)
    _tidy(ax, invert)
    if path:
        figure.savefig(path, bbox_inches="tight", facecolor="white")
    return figure
