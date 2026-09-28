# QUEST-style NQR controls in the GUI — design

Date: 2026-08-19
Status: approved

## Problem

Three related shortcomings in the web GUI (`nqrlyze/static/index.html`), reported
while trying to simulate Zeeman-perturbed ³⁵Cl NQR:

1. **The NQR checkbox forbids Zeeman-perturbed NQR.** Ticking it forces
   `larmor = 0` and disables both the Field and Larmor inputs, so a small
   Zeeman perturbation on top of the quadrupolar interaction cannot be entered.
   The backend (`eigen_transitions`, exact diagonalisation) already handles any
   Larmor value; the restriction is purely frontend.
2. **The default settings produce an invisible spectrum in NQR mode.** With the
   NQR box ticked (larmor 0) and the default transition selection
   `central only`, the simulated total is identically zero — the central
   transition carries no observable intensity at zero field with η = 0. The
   page draws nothing and gives no hint why. Verified against
   `/api/simulate`: `spin 2.5, larmor 0, transitions ct → ymax 0`;
   `transitions all → ymax 1`.
3. **The window is entered as From/To (MHz).** The user wants QUEST-style
   controls: centre frequency plus spectral width.

## Decisions (user-confirmed)

- NQR checkbox stays but becomes a **preset, not a lock** ("Checkbox
  entsperren").
- The window panel shows **Centre + Spectral width everywhere** (NMR and NQR),
  replacing From/To.
- Spectral width is entered in **kHz**.

## Design

All changes are in `nqrlyze/static/index.html` plus tests and README; the HTTP
API is untouched.

### 1. NQR checkbox as preset (Zeeman-perturbed NQR)

- Ticking **NQR**: stores the current field value, sets field and larmor inputs
  to 0, switches the Transitions select to `all`, switches the axis unit to
  MHz — and leaves Field and Larmor **editable**. Typing a non-zero field or
  Larmor while NQR is ticked is Zeeman-perturbed NQR.
- Field (T) ↔ Larmor (MHz) stay coupled through γ in both directions, exactly
  as today.
- Unticking restores the stored pre-NQR field value (and derived Larmor).
- Payload change: `experimentPayload()` sends the *actual* `S.larmor` always
  (no more `S.nqr ? 0 : S.larmor`), and additionally sends `reference: 0` when
  NQR is ticked, so the frequency axis stays absolute MHz / kHz-from-zero.
  `parse_experiment` already accepts `reference`.
- The ppm unit button is disabled while NQR is ticked (reference ≈ 0 makes ppm
  meaningless); kHz remains available as offset from 0, i.e. absolute kHz.

### 2. Window panel: Centre + Spectral width

- Replace the `From` / `To` rows with:
  - **Centre** — MHz, number input.
  - **Width** — kHz, number input.
  - **Points** and the **Auto** button stay as they are.
- Internal state keeps `S.win.low/high` as the source of truth sent to the
  server; the two inputs are a pure view:
  `low = centre − width·1e-3/2`, `high = centre + width·1e-3/2`.
- Editing either input sets `S.win.auto = false` and re-simulates (as From/To
  do today). Auto-window responses and loaded data fill Centre/Width from the
  returned `low`/`high` via `syncWindowInputs()`.
- Validation: width must be > 0; a non-positive width shows the existing error
  banner (the server would reject `high <= low` anyway).

### 3. Invisible-spectrum hint

- Frontend-only guard in `simulate()`: if the returned total trace is
  identically zero, show a status hint instead of silently drawing a flat
  line: *"no observable intensity — at zero field the central transition
  carries none; select transitions: all"*. Generic wording (it can also happen
  with `satellites` at high field edge cases), with the NQR-specific sentence
  only when larmor is 0.

### 4. Tests and docs

- `tests/test_webapp.py` additions (pytest, against the handler directly like
  the existing tests):
  - Zeeman-perturbed NQR: `spin 1.5, larmor 0.5, reference 0, transitions all`
    → non-trivial spectrum that differs from the zero-field one (Zeeman
    splitting is present).
  - Regression documentation: `larmor 0, transitions ct, eta 0` → total is
    identically zero (this is physics, not a bug; the GUI now says so).
- README: update the GUI paragraph to mention centre/width entry and
  Zeeman-perturbed NQR via a small field with the NQR box ticked.

## Out of scope

- No change to the CLI, job files, or `/api/*` request/response shapes.
- No new backend route; the zero-intensity hint is computed client-side.
- No attempt to make `transitions: ct` meaningful at zero field.

## Error handling

- Width ≤ 0 or non-finite centre: keep last valid window, show error banner.
- Everything else falls through to the existing error path (server 400 →
  banner).

## Testing strategy

- Python: the two new webapp tests above; full suite must stay green
  (91 passing today).
- Manual: tick NQR with defaults → peak visible at Cq/2·√(1+η²/3); enter
  0.1 T → satellite splitting appears; window follows centre/width edits.
