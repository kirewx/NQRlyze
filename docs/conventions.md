# Conventions, and how to prove NQRlyze agrees with QUEST

Two programs that solve the same physics can still disagree, and when they do it
is almost never the physics — it is a convention. This document states every
convention `nqrlyze` uses, and then gives a procedure for confirming, on your
own data, that QUEST and `nqrlyze` produce the same spectrum.

## 1. What is actually computed

For each crystallite orientation the full Hamiltonian is built and diagonalised.
There is no perturbation expansion anywhere, so the same code path serves
high-field NMR, the intermediate regime, and pure NQR.

Working in the **principal axis system of the electric field gradient**, with
the magnetic field direction `n` expressed in that frame, all in MHz:

```
H/h = -nu_L (n · I)
      + Cq / (4 I (2I - 1)) · [ 3 Iz² - I(I+1) + eta (Ix² - Iy²) ]
      - nu_ref · 1e-6 · (n · delta · n) · (n · I)
```

Rotating the field into the EFG frame rather than the tensor into the lab frame
is a choice of bookkeeping, not of physics: it needs only two angles per
crystallite and leaves the quadrupolar term in its natural form.

| Symbol | Meaning | Units |
|---|---|---|
| `nu_L` | Larmor frequency, `\|gamma\| B0 / 2pi`. `0` gives NQR | MHz |
| `Cq` | `e Q Vzz / h` | MHz |
| `eta` | `(Vxx - Vyy) / Vzz`, with `\|Vzz\| >= \|Vyy\| >= \|Vxx\|` | — |
| `delta` | chemical shift tensor in the EFG frame | ppm |
| `nu_ref` | frequency of 0 ppm | MHz |

Notes that matter in practice:

- **The sign of `gamma` is irrelevant.** A powder pattern is invariant under
  `nu_L -> -nu_L`, so `nqrlyze` takes `nu_L >= 0`. (Proof: a pi rotation about
  `y` maps `H(nu_L, n)` onto `H(-nu_L, (nx, -ny, nz))`, and the powder average
  covers both orientations.) There is a test for this.
- **The chemical shift is in the Hamiltonian, not added afterwards.** It uses
  the IUPAC sign convention — larger `delta` means higher frequency — and it
  vanishes identically when `nu_ref = 0`, which is the correct behaviour for
  NQR, where there is no reference frequency and "ppm" is meaningless.
- **`Cq` may be negative.** No powder observable distinguishes the sign of `Cq`
  from `eta -> eta` alone, so fits bound it at `Cq >= 0` by default.

## 2. Frequencies

A transition between eigenstates `i < j` (sorted by ascending energy) sits at
`E_j - E_i`, the energy absorbed, and is always positive.

**All axes inside `nqrlyze` are absolute frequencies in MHz.** This is the only
convention that serves NMR and NQR equally: a Bruker axis converts to it
exactly, and in NQR there is nothing to take an offset from. Conversions are
provided, and are exactly these:

```
ppm  = (nu - nu_ref) / nu_ref · 1e6
kHz  = (nu - nu_ref) · 1e3
```

**If you compare against a QUEST export, this is the first thing to get right.**
QUEST plots an offset from the Larmor frequency; a two-column export therefore
usually needs `"unit": "kHz"` together with the reference frequency, not
`"unit": "MHz"`. A pattern of the right width sitting at the wrong place is
nearly always this and nothing more.

## 3. Intensities

For a transition `i -> j` with `u = <i| I |j>` (a complex 3-vector), the powder
transition probability is

- **with a field** — the rf field is perpendicular to `B0`, and averaging over
  its direction in that plane gives `(|u|² - |u·n|²) / 2`;
- **without a field** (NQR) — there is no distinguished axis, so the average is
  over the whole sphere: `|u|² / 3`.

`rf_average="auto"` picks by whether `nu_L` is zero.

Two consequences worth knowing:

- Intensities are **not** renormalised per site. Two sites of the same isotope
  have the same intrinsic transition probability, so a site's fitted amplitude
  is its population. Pass `normalize=True` if you want unit area per site
  instead.
- At zero field the levels are Kramers doublets, so pairing every level with
  every other produces same-energy pairs at zero frequency. These are not
  resonances and their intensity is suppressed automatically when `nu_L = 0`.

## 4. The chemical shift tensor

Herzfeld–Berger, with `d11 >= d22 >= d33`:

```
iso  = (d11 + d22 + d33) / 3
span = d11 - d33                >= 0
skew = 3 (d22 - iso) / span     in [-1, 1]
```

The Euler angles `(alpha, beta, gamma)` are **ZYZ in degrees** and rotate the
shift PAS into the EFG PAS, `delta_EFG = R delta_PAS Rᵀ` with
`R = Rz(alpha) Ry(beta) Rz(gamma)`.

Euler-angle conventions are the single most common source of disagreement
between NMR programs — ZYZ against ZXZ, active against passive, and which frame
rotates into which are all live choices. If a CSA case disagrees while the pure
quadrupolar cases pass, this is where to look, and `beta` is the angle to test
first because it alone changes the pattern for an axially symmetric tensor.

## 5. Powder averaging

The Alderman–Solum–Grant interpolation scheme, the same approach QUEST uses:
one octant of the sphere is tessellated by projecting a triangular grid from an
octahedron face, and each triangle contributes a continuous band of intensity
whose density is triangular between its three corner frequencies. Sharp
singularities therefore appear with a few hundred orientations rather than a few
million.

Symmetry decides how many octants are needed. The Zeeman-plus-quadrupolar
Hamiltonian is invariant under each pi rotation about an EFG principal axis and,
by time reversal, under `n -> -n`, so one octant suffices. A shift tensor tilted
away from the EFG frame breaks the pi-rotation symmetry but not `n -> -n`, so
four octants are used then, and eight are never required. This is automatic.

`divisions` controls the grid: it carries `(d+1)(d+2)/2` orientations and `d²`
triangles per octant. Measured against a converged reference:

| `divisions` | orientations | worst deviation |
|---|---|---|
| 14 | 120 | 2 % of peak |
| 20 | 231 | 1 % |
| 34 | 630 | 0.6 % |
| 50 | 1326 | 0.3 % |
| 70 | 2556 | 0.1 % |

The default of 50 for fitting is a deliberate compromise: **the grid sets the
systematic floor of a fit.** Raising `divisions` only buys accuracy for data
whose signal-to-noise is better than the corresponding row.

## 6. Broadening

Lorentzian and Gaussian broadening are applied as convolutions in the Fourier
domain using the exact analytic transforms, so there is no truncated-kernel
error:

```
Lorentzian, FWHM wL:   exp(-pi · wL · |t|)
Gaussian,   FWHM wG:   exp(-pi² · wG² · t² / (4 ln 2))
```

Both widths are **FWHM in MHz** — not half-widths, not standard deviations, and
not Hz. Applying both gives a true Voigt profile. A factor-of-two disagreement
in a fitted width almost always means the other program quotes a half-width.

Intensity from outside the plotted window is handled exactly for the histogram,
so a fit window may be narrower than the pattern. The convolution itself is
computed on a margin of 50 Lorentzian widths, which leaves the far Lorentzian
tail truncated at a few parts in 10⁵ of the peak.

## 7. Confirming agreement with QUEST

Two independent layers.

### Layer 1 — closed-form checks, no QUEST needed

These run in the test suite (`pytest`) and pin down the Hamiltonian, the units
and the frequency conventions against results that do not come from this code:

| Check | Agreement |
|---|---|
| I = 3/2 NQR: `nu = (Cq/2) sqrt(1 + eta²/3)` | exact to 1e-9 MHz |
| I = 5/2 NQR at `eta = 0`: lines at `3Cq/20`, `6Cq/20`, intensity ratio 1.6 | exact |
| `dm = 2` transitions forbidden at `eta = 0` | < 1e-12 |
| Central transition vs. analytic second-order theory at high field | < 0.2 % of the second-order spread |
| Isotropic second-order shift `-(nu_Q²/30 nu_L)(I(I+1) - 3/4)(1 + eta²/3)` | 0.2 % |
| Powder pattern edges vs. analytic second-order extremes | < 0.2 kHz |
| ASG lineshape vs. a 400 000-orientation brute-force average | < 2 % of peak |
| Powder pattern invariant under `nu_L -> -nu_L` | exact |

If all of these pass — and they do — then the physics is right, and the only
thing left that could differ from QUEST is a convention.

### Layer 2 — QUEST's own output

Which is what the `validate` command is for.

```bash
nqrlyze template manifest > references/manifest.json
```

1. In QUEST, simulate several cases spanning what you actually measure:
   different spins, small and large `Cq`, `eta` near 0 and near 1, one
   high-field case and one NQR case, and — if you use them — one with CSA.
2. Export each as a two-column text file into `references/`.
3. Edit the manifest so each case names its file, its units, and the exact
   parameters QUEST was given.
4. Run it:

```bash
nqrlyze validate references/manifest.json
```

Each case reports RMSD as a fraction of peak height, the correlation, the shift
of the maximum, and the shift of every powder singularity. Scale and offset are
fitted out first, because QUEST's vertical scale carries no physics.

**Singularity positions are the sharp test.** They depend only on frequencies —
not on intensities, not on broadening, not on the powder grid — so they isolate
the Hamiltonian and the axis convention from everything else.

Reading a failure:

| Symptom | Almost certainly |
|---|---|
| Pattern shifted, width correct | axis convention: offset-from-Larmor vs. absolute, or the wrong `reference` |
| Pattern scaled in frequency | `Cq`, `eta` or `nu_L` mismatch — check `nu_L` against `gamma B0` first |
| Singularities match, intensities do not | transition selection (`"ct"` vs `"all"`) or rf averaging |
| Width off by exactly 2 | one program quotes FWHM, the other half-width |
| Only the CSA cases fail | Euler angle convention — start with `beta` |
| Everything slightly soft | raise `divisions`; compare at 70 before concluding anything |

A failing case is a specification to fix, not necessarily a bug: once the
convention is identified it is a one-line change in the manifest or in the job
file, and the point of the harness is to make that identification mechanical
rather than a matter of squinting at two plots.

### What this does not prove

Agreement with QUEST is agreement on the *forward* calculation. It says nothing
about whether a fitted parameter is physically meaningful. Two limits are worth
stating plainly:

- **A single NQR line cannot separate `Cq` from `eta`.** For I = 3/2 only the
  combination `Cq sqrt(1 + eta²/3)` is observable, and fitting both to one
  resonance is ill-posed however good the fit looks. The result reports the two
  as essentially perfectly correlated; use `cq_range_for_line` to bracket, and
  measure a second transition, a second isotope, or a field-dependent spectrum
  to break the degeneracy.
- **Broadening absorbs disorder.** A fitted Gaussian width is a lump sum of
  instrumental broadening, dipolar coupling and a distribution of `Cq`; it is
  not a measurement of any one of them.
