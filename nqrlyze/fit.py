"""Automatic fitting of quadrupolar powder patterns.

The fit is *separable*.  Site amplitudes and baseline coefficients enter the
model linearly, so at every trial of the non-linear parameters they are solved
exactly by bounded linear least squares rather than searched over.  Only the
parameters that genuinely bend the lineshape -- ``Cq``, ``eta``, the shift and
its anisotropy, and the two broadenings -- are left to the optimiser.  That is
what makes an automatic fit converge from a crude starting guess: the hardest
part of a quadrupolar fit is normally the interplay between amplitude and
width, and here amplitude is never guessed.

Finding the basin comes first, and there are three gears:

* ``global_search=True`` runs ``differential_evolution`` over the bounded
  parameters on a coarse grid and a decimated copy of the data.  Use it when
  there is no starting guess worth the name.
* ``restarts=N`` instead refines from ``N`` Latin-hypercube starting points on
  that same coarse problem and keeps the best.  Much cheaper than a global
  search and enough to escape the usual trap, where a slightly wrong ``Cq``
  survives by inflating the broadening to cover its mistake.
* neither: refine from exactly the values given.

Whichever ran, ``least_squares`` (trust region reflective, bounded) then
polishes the winner on the full data and fine grid, and produces the Jacobian.

Uncertainties are the usual ``s^2 (J^T J)^-1`` estimates, computed at the
solution over *all* parameters including the linear ones.  They describe the
precision of the fit, not systematic error; correlated parameters such as ``Cq``
and ``eta`` are better judged from the correlation matrix, which is reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import differential_evolution, least_squares, lsq_linear
from scipy.stats import qmc

from .simulate import Experiment, Site, simulate_sites
from .spectrum import Spectrum

__all__ = [
    "FitParameter",
    "FitResult",
    "default_parameters",
    "build_model",
    "fit",
]

#: Site attributes that may be fitted, with the bounds used when none are given.
PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "cq": (0.0, 100.0),
    "eta": (0.0, 1.0),
    "iso": (-10000.0, 10000.0),
    "span": (0.0, 5000.0),
    "skew": (-1.0, 1.0),
    "alpha": (0.0, 360.0),
    "beta": (0.0, 180.0),
    "gamma": (0.0, 360.0),
    "lorentz": (0.0, 10.0),
    "gauss": (0.0, 10.0),
}


@dataclass
class FitParameter:
    """One free parameter: an attribute of one site, with bounds."""

    site: int
    name: str
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self):
        if self.name not in PARAMETER_BOUNDS:
            raise ValueError(
                f"{self.name!r} is not fittable; choose from "
                f"{', '.join(sorted(PARAMETER_BOUNDS))}"
            )
        default_low, default_high = PARAMETER_BOUNDS[self.name]
        if self.lower is None:
            self.lower = default_low
        if self.upper is None:
            self.upper = default_high
        if self.lower >= self.upper:
            raise ValueError(f"{self.name}: lower bound must be below upper bound")

    @property
    def key(self) -> tuple[int, str]:
        return (self.site, self.name)


def default_parameters(
    sites: Sequence[Site],
    fit_shift: bool = True,
    fit_csa: bool = False,
    shared_broadening: bool = True,
    cq_max: float | None = None,
) -> list[FitParameter]:
    """A sensible free-parameter set: ``Cq``, ``eta``, broadening, and options.

    ``shared_broadening`` fits one Lorentzian and one Gaussian width on the
    first site only; the caller is expected to copy them to the others via
    ``tie``.  Set it to ``False`` to give every site its own widths.
    """
    params: list[FitParameter] = []
    for index, site in enumerate(sites):
        upper = cq_max if cq_max is not None else max(4.0 * max(site.cq, 0.25), 1.0)
        params.append(FitParameter(index, "cq", 0.0, upper))
        params.append(FitParameter(index, "eta", 0.0, 1.0))
        if fit_shift:
            params.append(FitParameter(index, "iso"))
        if fit_csa:
            params.append(FitParameter(index, "span"))
            params.append(FitParameter(index, "skew"))
            params.append(FitParameter(index, "beta"))
            params.append(FitParameter(index, "alpha"))
            params.append(FitParameter(index, "gamma"))
        if not shared_broadening or index == 0:
            params.append(FitParameter(index, "lorentz"))
            params.append(FitParameter(index, "gauss"))
    return params


def _baseline_columns(x: np.ndarray, order: int) -> np.ndarray:
    """Chebyshev-style polynomial columns on a normalised axis, ``(n, order + 1)``."""
    if order < 0:
        return np.zeros((x.size, 0))
    span = x[-1] - x[0]
    t = np.zeros_like(x) if span == 0 else 2.0 * (x - x[0]) / span - 1.0
    return np.polynomial.chebyshev.chebvander(t, order)


@dataclass
class FitResult:
    """Everything the fit produced."""

    sites: list[Site]
    amplitudes: np.ndarray
    baseline_coefficients: np.ndarray
    freq_mhz: np.ndarray
    data: np.ndarray
    model: np.ndarray
    components: np.ndarray
    baseline: np.ndarray
    residual: np.ndarray
    parameters: list[FitParameter]
    uncertainties: dict[tuple[int, str], float]
    amplitude_uncertainties: np.ndarray
    correlation: np.ndarray
    parameter_names: list[str]
    rmsd: float
    r_squared: float
    reduced_chi_squared: float
    n_evaluations: int
    success: bool
    message: str

    def uncertainty(self, site: int, name: str) -> float | None:
        return self.uncertainties.get((site, name))

    def report(self) -> str:
        """A human-readable summary."""
        lines = [
            f"fit {'converged' if self.success else 'did NOT converge'}: {self.message}",
            f"  points {self.data.size}   RMSD {self.rmsd:.5g}"
            f"   R^2 {self.r_squared:.6f}"
            f"   reduced chi^2 {self.reduced_chi_squared:.5g}"
            f"   {self.n_evaluations} evaluations",
        ]
        total = float(np.sum(self.amplitudes))
        for index, site in enumerate(self.sites):
            share = 100.0 * self.amplitudes[index] / total if total > 0 else float("nan")
            name = site.label or f"site {index + 1}"
            lines.append(f"  {name}  ({share:.1f} % of total intensity)")
            for attr, unit, fmt in (
                ("cq", "MHz", "{:.5f}"),
                ("eta", "", "{:.4f}"),
                ("iso", "ppm", "{:.2f}"),
                ("span", "ppm", "{:.2f}"),
                ("skew", "", "{:.3f}"),
                ("alpha", "deg", "{:.1f}"),
                ("beta", "deg", "{:.1f}"),
                ("gamma", "deg", "{:.1f}"),
                ("lorentz", "kHz", "{:.3f}"),
                ("gauss", "kHz", "{:.3f}"),
            ):
                value = getattr(site, attr)
                if attr in ("lorentz", "gauss"):
                    value = value * 1e3
                if value == 0 and attr in ("span", "skew", "alpha", "beta", "gamma"):
                    continue
                error = self.uncertainty(index, attr)
                if error is not None and attr in ("lorentz", "gauss"):
                    error = error * 1e3
                shown = fmt.format(value)
                if error is not None and math.isfinite(error):
                    shown += " +- " + fmt.format(error)
                fixed = "" if error is not None else "   (fixed)"
                lines.append(f"      {attr:<8} {shown} {unit}{fixed}")
        return "\n".join(lines)


def build_model(
    freq_mhz: np.ndarray,
    sites: Sequence[Site],
    experiment: Experiment,
    baseline_order: int,
    divisions: int,
) -> np.ndarray:
    """Design matrix: one column per site, then the baseline columns."""
    unit_sites = [replace(site, weight=1.0) for site in sites]
    components = simulate_sites(
        freq_mhz, unit_sites, experiment, divisions=divisions
    )
    return np.concatenate(
        [components.T, _baseline_columns(np.asarray(freq_mhz, float), baseline_order)],
        axis=1,
    )


def _solve_linear(
    design: np.ndarray, target: np.ndarray, n_sites: int, non_negative: bool
):
    """Bounded linear least squares for amplitudes and baseline coefficients."""
    scale = np.max(np.abs(design), axis=0)
    scale[scale == 0] = 1.0
    scaled = design / scale
    if non_negative:
        lower = np.concatenate(
            [np.zeros(n_sites), np.full(design.shape[1] - n_sites, -np.inf)]
        )
        upper = np.full(design.shape[1], np.inf)
        solution = lsq_linear(scaled, target, bounds=(lower, upper), method="trf")
        coefficients = solution.x
    else:
        coefficients = np.linalg.lstsq(scaled, target, rcond=None)[0]
    return coefficients / scale


def _apply(sites: list[Site], parameters: Sequence[FitParameter], values) -> list[Site]:
    out = [replace(site) for site in sites]
    for parameter, value in zip(parameters, values):
        setattr(out[parameter.site], parameter.name, float(value))
    return out


def fit(
    spectrum: Spectrum,
    sites: Sequence[Site],
    experiment: Experiment,
    parameters: Sequence[FitParameter] | None = None,
    baseline_order: int = 0,
    non_negative: bool = True,
    tie: Callable[[list[Site]], list[Site]] | None = None,
    weights: np.ndarray | None = None,
    global_search: bool = True,
    restarts: int = 0,
    coarse_divisions: int = 16,
    coarse_points: int = 700,
    divisions: int = 50,
    max_iterations: int = 120,
    popsize: int = 15,
    seed: int | None = 0,
    verbose: bool = False,
) -> FitResult:
    """Fit a powder spectrum.

    Parameters
    ----------
    spectrum
        Experimental data on an absolute MHz axis.
    sites
        Starting values.  Every attribute not named in ``parameters`` is held
        fixed at the value given here.
    experiment
        Nucleus, Larmor frequency and observed transitions.
    parameters
        Free parameters; defaults to :func:`default_parameters`.
    baseline_order
        Degree of the polynomial baseline, or ``-1`` for none.
    non_negative
        Constrain site amplitudes to be non-negative.  Almost always right.
    tie
        Hook applied to the site list after the free parameters are written and
        before simulation -- use it to share a broadening or an ``eta`` between
        sites.
    weights
        Per-point weights (``1 / sigma``).  Defaults to uniform.
    global_search
        Run the differential-evolution stage first.  Turn it off when the
        starting values are already close.
    restarts
        When ``global_search`` is off, refine from this many extra
        Latin-hypercube starting points on the coarse problem and keep the
        best.  A handful is usually enough to escape a local minimum that a
        single refinement would settle into, at a fraction of the cost.
    coarse_divisions, divisions
        Powder grid used during the global search and the final refinement.
        The refinement grid sets the systematic floor of the fit: at
        ``divisions = 50`` the simulated lineshape is converged to a few parts
        in a thousand of the peak height, so pushing it higher only pays off for
        data with a signal-to-noise ratio better than that.
    coarse_points
        The global search runs on the data decimated to about this many points.
        It only has to find the right basin, and a coarse copy does that just as
        well for a fraction of the cost.
    """
    sites = [replace(site) for site in sites]
    if parameters is None:
        parameters = default_parameters(sites)
    parameters = list(parameters)
    seen = set()
    for parameter in parameters:
        if parameter.site >= len(sites) or parameter.site < 0:
            raise ValueError(f"parameter refers to missing site {parameter.site}")
        if parameter.key in seen:
            raise ValueError(f"parameter {parameter.key} listed twice")
        seen.add(parameter.key)

    x = np.asarray(spectrum.freq_mhz, dtype=float)
    y = np.asarray(spectrum.intensity, dtype=float)
    if weights is None:
        weight = np.ones_like(y)
    else:
        weight = np.asarray(weights, dtype=float)
        if weight.shape != y.shape:
            raise ValueError("weights must match the spectrum length")
        if np.any(weight < 0):
            raise ValueError("weights must be non-negative")

    n_sites = len(sites)
    lower = np.array([p.lower for p in parameters], dtype=float)
    upper = np.array([p.upper for p in parameters], dtype=float)
    start = np.array(
        [float(getattr(sites[p.site], p.name)) for p in parameters], dtype=float
    )
    start = np.clip(start, lower, upper)

    counter = {"n": 0}

    def residual(values, grid_divisions, xa=x, ya=y, wa=weight):
        counter["n"] += 1
        trial = _apply(sites, parameters, values)
        if tie is not None:
            trial = tie(trial)
        design = build_model(xa, trial, experiment, baseline_order, grid_divisions)
        coefficients = _solve_linear(
            design * wa[:, None], ya * wa, n_sites, non_negative
        )
        return (design @ coefficients - ya) * wa

    if global_search and parameters:
        stride = max(1, x.size // max(coarse_points, 2))
        xc, yc, wc = x[::stride], y[::stride], weight[::stride]

        def cost(values):
            r = residual(values, coarse_divisions, xc, yc, wc)
            return float(np.dot(r, r))

        de = differential_evolution(
            cost,
            bounds=list(zip(lower, upper)),
            maxiter=max_iterations,
            popsize=popsize,
            seed=seed,
            polish=False,
            init="sobol",
            tol=1e-4,
            mutation=(0.4, 1.0),
            recombination=0.8,
            disp=verbose,
        )
        start = np.clip(de.x, lower, upper)

    elif restarts > 0 and parameters:
        stride = max(1, x.size // max(coarse_points, 2))
        xc, yc, wc = x[::stride], y[::stride], weight[::stride]

        def probe(values):
            trial = least_squares(
                residual,
                np.clip(values, lower + 1e-9, upper - 1e-9),
                bounds=(lower, upper),
                args=(coarse_divisions, xc, yc, wc),
                method="trf",
                x_scale="jac",
                # A coarse finite-difference step on purpose.  When a trial
                # pattern does not overlap the data at all the residual is flat,
                # the Jacobian comes back numerically zero and the solver stops
                # on its first step; a step small enough for polishing never
                # escapes that, while a percent-level one does.
                diff_step=1e-2,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
                max_nfev=80 * len(parameters),
            )
            return float(np.dot(trial.fun, trial.fun)), trial.x

        sampler = qmc.LatinHypercube(d=len(parameters), seed=seed)
        draws = lower + sampler.random(restarts) * (upper - lower)
        best_cost, start = probe(start)
        for candidate in draws:
            cost, position = probe(candidate)
            if cost < best_cost:
                best_cost, start = cost, position

    if parameters:
        # Nudge off the bounds so the trust-region solver can move in both
        # directions on the first step.
        step = 1e-6 * (upper - lower)
        polished = least_squares(
            residual,
            np.clip(start, lower + step, upper - step),
            bounds=(lower, upper),
            args=(divisions,),
            method="trf",
            x_scale="jac",
            diff_step=1e-4,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            verbose=2 if verbose else 0,
        )
        best = polished.x
        success, message = bool(polished.success), str(polished.message)
    else:
        best = start
        success, message = True, "no free parameters"

    final_sites = _apply(sites, parameters, best)
    if tie is not None:
        final_sites = tie(final_sites)
    design = build_model(x, final_sites, experiment, baseline_order, divisions)
    coefficients = _solve_linear(
        design * weight[:, None], y * weight, n_sites, non_negative
    )
    components = design[:, :n_sites] * coefficients[:n_sites]
    baseline = design[:, n_sites:] @ coefficients[n_sites:]
    model = components.sum(axis=1) + baseline
    resid = model - y

    for index in range(n_sites):
        final_sites[index].weight = float(coefficients[index])

    errors, correlation, names = _uncertainties(
        x, y, weight, final_sites, experiment, parameters, coefficients,
        baseline_order, divisions, n_sites, non_negative, resid, tie,
    )

    degrees = max(y.size - len(parameters) - design.shape[1], 1)
    weighted = resid * weight
    chi2 = float(np.dot(weighted, weighted))
    denominator = float(np.sum((y - y.mean()) ** 2))
    return FitResult(
        sites=final_sites,
        amplitudes=coefficients[:n_sites],
        baseline_coefficients=coefficients[n_sites:],
        freq_mhz=x,
        data=y,
        model=model,
        components=components.T,
        baseline=baseline,
        residual=resid,
        parameters=parameters,
        uncertainties={
            p.key: errors[i] for i, p in enumerate(parameters)
        },
        amplitude_uncertainties=errors[len(parameters) : len(parameters) + n_sites],
        correlation=correlation,
        parameter_names=names,
        rmsd=float(np.sqrt(np.mean(resid**2))),
        r_squared=1.0 - float(np.sum(resid**2)) / denominator if denominator > 0 else float("nan"),
        reduced_chi_squared=chi2 / degrees,
        n_evaluations=counter["n"],
        success=success,
        message=message,
    )


def _uncertainties(
    x, y, weight, sites, experiment, parameters, coefficients,
    baseline_order, divisions, n_sites, non_negative, resid, tie,
):
    """Standard errors and correlations from a numerical Jacobian at the optimum."""
    names = [f"site{p.site + 1}.{p.name}" for p in parameters]
    names += [f"site{i + 1}.amplitude" for i in range(n_sites)]
    names += [f"baseline.c{i}" for i in range(len(coefficients) - n_sites)]
    n_total = len(names)

    def predict(values, linear):
        trial = _apply(sites, parameters, values)
        if tie is not None:
            trial = tie(trial)
        design = build_model(x, trial, experiment, baseline_order, divisions)
        return design @ linear

    base_values = np.array(
        [float(getattr(sites[p.site], p.name)) for p in parameters]
    )
    jacobian = np.zeros((y.size, n_total))
    for i, parameter in enumerate(parameters):
        step = 1e-5 * max(abs(base_values[i]), abs(parameter.upper - parameter.lower))
        if step == 0:
            step = 1e-8
        shifted = base_values.copy()
        shifted[i] += step
        jacobian[:, i] = (
            predict(shifted, coefficients) - predict(base_values, coefficients)
        ) / step
    design = build_model(x, sites, experiment, baseline_order, divisions)
    jacobian[:, len(parameters) :] = design
    jacobian *= weight[:, None]

    degrees = max(y.size - n_total, 1)
    variance = float(np.sum((resid * weight) ** 2)) / degrees
    try:
        covariance = variance * np.linalg.pinv(jacobian.T @ jacobian)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        covariance = np.full((n_total, n_total), np.nan)
    diagonal = np.diag(covariance).copy()
    diagonal[diagonal < 0] = np.nan
    errors = np.sqrt(diagonal)
    scale = np.where(np.isfinite(errors) & (errors > 0), errors, np.nan)
    correlation = covariance / np.outer(scale, scale)
    return errors, correlation, names
