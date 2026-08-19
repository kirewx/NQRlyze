# NQRlyze

Automatic fitting of exact quadrupolar NMR and NQR powder patterns — `Cq`, `eta` and
Lorentzian/Gaussian broadening — from a measured spectrum.

The simulator follows the physics of
[QUEST](https://brycelab.ca/software.html) (Perras, Widdifield & Bryce, *Solid
State Nucl. Magn. Reson.* **2012**, 45–46, 36): the combined Zeeman and
quadrupolar Hamiltonian is diagonalised **exactly**, with no perturbation
expansion, so one code path covers high-field NMR, the intermediate regime where
neither interaction dominates, and pure NQR. Powder averaging uses the same
Alderman–Solum–Grant interpolation that makes QUEST fast.

On top of that sits the part QUEST does not provide: an optimiser that finds
**Cq**, **eta** and the **Lorentzian/Gaussian broadening** from a measured
spectrum, reads Bruker data directly, and joins stepped-frequency sub-spectra
first.

## Install

```bash
pip install -e .          # add [plot] for figures, [dev] for the test suite
```

Requires Python ≥ 3.10, NumPy and SciPy. No MATLAB, no QUEST installation.

## The interface

```bash
nqrlyze gui
```

Opens a local page at `http://127.0.0.1:8765`. Every parameter has a slider next
to its number, and the simulation follows as you drag — so you get close by eye,
tick the parameters you want fitted, and press **Fit**. Nothing is uploaded
anywhere: the server is the standard library, bound to loopback, and the page
makes no external requests.

Press **Synthesise** to make a noisy spectrum from the parameters on screen and
fit that, which is the quickest way to see the whole thing work without touching
your own data. You can also drop a two-column text file onto the page, or point
it at Bruker directories — several at once to co-add them first.

## Fitting in one command

```bash
nqrlyze template job > job.json   # edit it
nqrlyze fit job.json --plot fit.png -o result.json
```

```
fit converged: `xtol` termination condition is satisfied.
  points 2400   RMSD 0.0099   R^2 0.998744   1912 evaluations
  site 1  (100.0 % of total intensity)
      cq       5.20403 +- 0.00125 MHz
      eta      0.4213 +- 0.0006
      iso      62.05 +- 0.04 ppm
      lorentz  1.215 +- 0.020 kHz
      gauss    1.962 +- 0.020 kHz
```

That run started from `Cq = 2.0, eta = 0.9, iso = 0` — a deliberately poor
guess — against data simulated with `Cq = 5.2, eta = 0.42, iso = 62`.

## What it fits

| | |
|---|---|
| **Cq and eta** | the point of the exercise |
| **Lorentzian and Gaussian broadening** | independent FWHM in MHz, giving a true Voigt profile |
| **Isotropic shift** | in ppm, inside the Hamiltonian |
| **Multiple sites** | each with its own parameters; amplitudes are *solved*, not searched |
| **CSA** | span, skew and the ZYZ Euler angles relating the shift and EFG tensors |
| **Baseline and scale** | polynomial of any order, solved exactly at every iteration |

Amplitudes and baseline enter the model linearly, so at each trial of the
non-linear parameters they are obtained by bounded linear least squares rather
than searched over. Only the parameters that genuinely bend the lineshape are
left to the optimiser, which is what lets an automatic fit converge from a crude
starting guess.

Finding the right basin has three gears, and the middle one is usually what you
want:

| | when | cost on a typical CT pattern |
|---|---|---|
| `global_search=True` | no usable starting guess | ~60 s |
| `restarts=8` | you got close by eye or by slider | ~6 s |
| neither | you are already at the answer | ~3 s |

The trap `restarts` exists for is specific and common: a slightly wrong `Cq`
survives by inflating the broadening to cover its mistake, and a single
refinement settles there quite happily. In one measured case a plain local fit
stopped at `Cq = 4.85 MHz` (R² 0.982) where the truth was 5.2; eight restarts
reached 5.21 (R² 0.998) in a tenth of the time a global search needed. Worse, if
the starting pattern does not overlap the data at all, the residual is flat, the
Jacobian is numerically zero and a local fit does not move at all — restarts
escape that too. `least_squares` then polishes the winner and yields the
Jacobian for the error estimates.

## Stepped-frequency data

Wideline patterns are recorded piecewise. Because every spectrum is carried on
an absolute MHz axis, joining the pieces is exact:

```bash
nqrlyze coadd expt/10 expt/11 expt/12 --mode mean --scale-by-scans \
    -o combined.txt --plot pieces.png
```

`mean` divides by how many pieces cover each point, so uneven overlap does not
produce fake intensity; `sum` is plain co-addition; `skyline` is the traditional
display but distorts intensities and should not be fitted. A job file can do the
same inline, so the fit runs straight off the raw experiment directories:

```json
"data": {
  "format": "bruker",
  "paths": ["expt/10", "expt/11", "expt/12"],
  "coadd": {"mode": "mean", "scale_by_scans": true},
  "window_ppm": [-400, 400],
  "normalize": true
}
```

## From Python

```python
import numpy as np
from nqrlyze import Experiment, Site, fit, read_bruker, simulate

experiment = Experiment.from_nucleus("27Al", field=11.7449, transitions="ct")
data = read_bruker("expt/10").normalized()

result = fit(data, [Site(cq=4.0, eta=0.5, iso=60.0, lorentz=0.002)], experiment)
print(result.report())
print(result.sites[0].cq, result.uncertainty(0, "cq"))
```

## Agreement with QUEST

This is a reimplementation, so agreement is something to **demonstrate**, not to
assume. Two layers, both described in [`docs/conventions.md`](docs/conventions.md):

1. **Closed-form checks that need no QUEST**, run by `pytest`: exact NQR
   frequencies, forbidden transitions, the high-field limit against analytic
   second-order theory, the isotropic second-order shift, the analytic pattern
   edges, and the interpolated powder average against a 400 000-orientation
   brute-force one. These pin down the Hamiltonian and the units.
2. **Your own QUEST exports.** Simulate a handful of cases in QUEST, export them
   as two-column text, list them in a manifest, and run

   ```bash
   nqrlyze template manifest > references/manifest.json
   nqrlyze validate references/manifest.json
   ```

   Each case reports RMSD, correlation, and the shift of every powder
   singularity, with scale and offset fitted out first. `docs/conventions.md`
   has a table mapping each failure symptom to the convention that causes it.

Every convention `nqrlyze` uses — the Hamiltonian, the frequency axis, the
intensity averaging, the shift and Euler angle definitions, the FWHM units — is
stated explicitly in that document, because a disagreement between two correct
programs is almost always a convention and almost never the physics.

## Two things the fit cannot tell you

- **A single NQR line cannot separate `Cq` from `eta`.** For I = 3/2 only
  `Cq sqrt(1 + eta²/3)` is observable. The fit will look perfect and report the
  two as essentially perfectly correlated. Use `cq_range_for_line()` to bracket
  the search, and break the degeneracy with a second transition, a second
  isotope, or a field-dependent measurement.
- **A fitted Gaussian width is a lump sum** of instrumental broadening, dipolar
  coupling and any distribution of `Cq` — not a measurement of any one of them.

## Layout

```
nqrlyze/
  hamiltonian.py   exact Zeeman + quadrupolar (+ CSA) Hamiltonian
  powder.py        Alderman-Solum-Grant grids, brute-force reference grid
  lineshape.py     triangle interpolation, Fourier-domain broadening
  simulate.py      Site / Experiment, multi-site simulation
  fit.py           separable least squares, global search, uncertainties
  coadd.py         joining stepped-frequency sub-spectra
  analytic.py      closed forms used to prove the simulator right
  validate.py      comparison against QUEST reference spectra
  io/bruker.py     processed Bruker data (1r / procs / acqus)
  webapp.py        the local GUI: stdlib HTTP server, no framework
  static/          the single self-contained page it serves
  cli.py           nqrlyze gui | simulate | fit | coadd | validate | info | template
examples/demo.py   end-to-end run on synthetic data, no measurement needed
docs/conventions.md
```

## Tests

```bash
python -m pytest            # ~90 tests, a few minutes (the fits are real fits)
```
