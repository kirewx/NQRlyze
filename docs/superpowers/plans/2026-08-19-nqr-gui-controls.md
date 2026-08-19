# QUEST-style NQR GUI Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unlock Zeeman-perturbed NQR in the web GUI, replace the From/To window with QUEST-style Centre (MHz) + Spectral width (kHz), and explain zero-intensity spectra instead of drawing nothing.

**Architecture:** All functional changes live in the single self-contained page `nqrlyze/static/index.html` (stdlib server, no build step). The HTTP API is untouched; `parse_experiment` already accepts `reference`. Python tests pin the backend behaviour the GUI relies on and smoke-test the page structure. Spec: `docs/superpowers/specs/2026-08-19-nqr-gui-controls-design.md`.

**Tech Stack:** Vanilla JS in one HTML file; Python 3.13 venv at `.venv`; pytest. Run tests with `.venv\Scripts\python.exe -m pytest` (Windows). The dev server (`.venv\Scripts\nqrlyze.exe gui --no-browser`, port 8765) serves the page from the repo (editable install), so a browser reload shows edits.

**Context for the engineer:**
- `Experiment.larmor` (MHz) is the Zeeman term; `0` means pure NQR. `Experiment.reference` is the frequency of 0 ppm; sending `reference: 0` keeps the axis absolute. The backend diagonalises exactly, so *any* larmor value is valid — the GUI is the only thing blocking Zeeman-perturbed NQR today.
- Verified backend facts this plan builds on: `spin 2.5, larmor 0, transitions "ct", eta 0` → total spectrum identically zero (the central transition carries no intensity at zero field); `transitions "all"` → normal peak. This is physics, not a bug.
- In the page, `S` is the single state object; `S.win.low/high` (MHz) are what `/api/simulate` receives; `simulate()` re-runs on every control change; `syncWindowInputs()` writes state back into the window inputs.

---

### Task 1: Backend regression tests (Zeeman-perturbed NQR, zero-intensity CT)

These tests pin *existing* backend behaviour that the GUI changes rely on, so they are expected to PASS immediately. If either fails, stop — the design's assumptions are wrong.

**Files:**
- Modify: `tests/test_webapp.py` (append after `test_simulate_honours_an_explicit_window`, line 87)

- [ ] **Step 1: Add the two tests**

```python
def test_zeeman_perturbed_nqr_splits_the_line():
    """A small Larmor term on top of the quadrupole interaction is a valid
    experiment: the degenerate +-m levels split and the pattern widens."""
    site = [{"cq": 72.0, "eta": 0.1, "lorentz": 0.001}]
    pure = api_simulate({"experiment": {"spin": 1.5, "larmor": 0.0, "reference": 0,
                                        "transitions": "all"}, "sites": site})
    zeeman = api_simulate({"experiment": {"spin": 1.5, "larmor": 0.5, "reference": 0,
                                          "transitions": "all"}, "sites": site})
    assert max(pure["total"]["y"]) == pytest.approx(1.0, abs=1e-9)
    assert max(zeeman["total"]["y"]) == pytest.approx(1.0, abs=1e-9)
    assert zeeman["reference"] == 0.0
    assert (zeeman["high"] - zeeman["low"]) > (pure["high"] - pure["low"])


def test_pure_nqr_central_transition_has_no_intensity():
    """At zero field the central transition carries nothing.  The GUI defaults
    used to combine (NQR, transitions=ct) silently into this empty spectrum;
    it now switches to "all" and explains a flat result."""
    model = {"experiment": {"spin": 2.5, "larmor": 0.0, "transitions": "ct"},
             "sites": [{"cq": 1.0, "eta": 0.0, "lorentz": 0.001}]}
    assert max(api_simulate(model)["total"]["y"]) == 0.0
    model["experiment"]["transitions"] = "all"
    assert max(api_simulate(model)["total"]["y"]) == pytest.approx(1.0, abs=1e-9)
```

- [ ] **Step 2: Run them — expect PASS (they pin current behaviour)**

Run: `.venv\Scripts\python.exe -m pytest tests/test_webapp.py -k "zeeman or central_transition" -v`
Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_webapp.py
git commit -m "test: pin Zeeman-perturbed NQR and zero-intensity CT backend behaviour"
```

---

### Task 2: Window panel as Centre (MHz) + Spectral width (kHz)

**Files:**
- Modify: `nqrlyze/static/index.html` (Window section HTML ~lines 206–217; `syncWindowInputs` ~line 646; input wiring ~lines 764–767)
- Test: `tests/test_webapp.py`

- [ ] **Step 1: Write the failing page-structure test** (append to `tests/test_webapp.py`)

```python
def test_page_has_quest_style_window_controls():
    """The window is entered as centre frequency + spectral width, not From/To."""
    html = (STATIC / "index.html").read_text()
    assert 'id="win-centre"' in html and 'id="win-width"' in html
    assert 'id="win-low"' not in html and 'id="win-high"' not in html
```

- [ ] **Step 2: Run it — expect FAIL**

Run: `.venv\Scripts\python.exe -m pytest tests/test_webapp.py::test_page_has_quest_style_window_controls -v`
Expected: FAIL on `'id="win-centre"' in html`.

- [ ] **Step 3: Replace the From/To rows in the HTML**

In `nqrlyze/static/index.html`, replace

```html
          <div class="row"><label for="win-low">From</label>
            <div class="ctl"><input type="number" id="win-low" class="wide" step="0.001"><span class="unit">MHz</span></div></div>
          <div class="row"><label for="win-high">To</label>
            <div class="ctl"><input type="number" id="win-high" class="wide" step="0.001"><span class="unit">MHz</span></div></div>
```

with

```html
          <div class="row"><label for="win-centre">Centre</label>
            <div class="ctl"><input type="number" id="win-centre" class="wide" step="0.001"><span class="unit">MHz</span></div></div>
          <div class="row"><label for="win-width">Width</label>
            <div class="ctl"><input type="number" id="win-width" class="wide" step="1" min="0"><span class="unit">kHz</span></div></div>
```

- [ ] **Step 4: Make `syncWindowInputs` a centre/width view of `low`/`high`**

Replace the existing function

```js
function syncWindowInputs(){
  document.getElementById("win-low").value  = S.win.low  == null ? "" : (+S.win.low).toFixed(5);
  document.getElementById("win-high").value = S.win.high == null ? "" : (+S.win.high).toFixed(5);
  document.getElementById("win-points").value = S.win.points;
}
```

with

```js
function syncWindowInputs(){
  const has = S.win.low != null && S.win.high != null;
  document.getElementById("win-centre").value = has ? ((S.win.low + S.win.high) / 2).toFixed(5) : "";
  document.getElementById("win-width").value  = has ? ((S.win.high - S.win.low) * 1e3).toFixed(1) : "";
  document.getElementById("win-points").value = S.win.points;
}
```

- [ ] **Step 5: Rewire the inputs**

Replace

```js
  ["win-low","win-high"].forEach(id => document.getElementById(id).oninput = e => {
    S.win.auto = false;
    S.win[id === "win-low" ? "low" : "high"] = +e.target.value; scheduleSim();
  });
```

with

```js
  const applyCentreWidth = () => {
    const centre = +document.getElementById("win-centre").value;
    const width  = +document.getElementById("win-width").value * 1e-3;   /* kHz -> MHz */
    if(!isFinite(centre) || !isFinite(width) || width <= 0){
      showError("spectral width must be a positive number of kHz"); return;
    }
    S.win.auto = false;
    S.win.low = centre - width / 2; S.win.high = centre + width / 2;
    scheduleSim();
  };
  ["win-centre","win-width"].forEach(id => document.getElementById(id).oninput = applyCentreWidth);
```

`S.win.low/high` stay the source of truth sent to the server; the Auto button and `setData` already refill them and call `syncWindowInputs()`, so nothing else changes.

- [ ] **Step 6: Run the test — expect PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/test_webapp.py::test_page_has_quest_style_window_controls tests/test_webapp.py::test_page_is_self_contained -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add nqrlyze/static/index.html tests/test_webapp.py
git commit -m "feat(gui): QUEST-style window entry as centre frequency + spectral width"
```

---

### Task 3: NQR checkbox as preset — Zeeman-perturbed NQR

**Files:**
- Modify: `nqrlyze/static/index.html` (`experimentPayload` ~line 335; nqr onchange ~line 751; `setUnit` guard ~line 705)
- Test: `tests/test_webapp.py`

- [ ] **Step 1: Write the failing page-structure test** (append to `tests/test_webapp.py`)

```python
def test_page_keeps_field_editable_in_nqr_mode():
    """The NQR box is a preset, not a lock: it must not disable the field and
    Larmor inputs (that is what forbade Zeeman-perturbed NQR)."""
    html = (STATIC / "index.html").read_text()
    assert '"larmor").disabled' not in html
    assert '"field").disabled' not in html
    assert "no observable intensity" in html
```

- [ ] **Step 2: Run it — expect FAIL**

Run: `.venv\Scripts\python.exe -m pytest tests/test_webapp.py::test_page_keeps_field_editable_in_nqr_mode -v`
Expected: FAIL on `'"larmor").disabled' not in html`.

- [ ] **Step 3: Send the real Larmor, plus `reference: 0` in NQR mode**

Replace

```js
const experimentPayload = () => ({
  spin:S.spin, larmor:S.nqr?0:S.larmor, transitions:S.transitions,
});
```

with

```js
const experimentPayload = () => {
  const exp = {spin:S.spin, larmor:S.larmor, transitions:S.transitions};
  if(S.nqr) exp.reference = 0;          /* keep the axis absolute: ppm is meaningless near zero field */
  return exp;
};
```

(The two `reference:S.nqr?0:S.larmor` occurrences in the load/drop handlers are already correct and stay.)

- [ ] **Step 4: Rewrite the NQR checkbox handler**

Replace

```js
  document.getElementById("nqr").onchange = e => {
    S.nqr = e.target.checked;
    document.getElementById("larmor").disabled = S.nqr;
    document.getElementById("field").disabled = S.nqr;
    if(S.nqr) setUnit("MHz");
    invalidateFit(); S.win.auto = true; simulate();
  };
```

with

```js
  document.getElementById("nqr").onchange = e => {
    S.nqr = e.target.checked;
    const ppm = document.querySelector('[data-unit="ppm"]');
    if(S.nqr){
      /* Preset, not a lock: zero the field but leave both inputs live, so a
         small field typed afterwards gives Zeeman-perturbed NQR.  Switch to
         "all" because the central transition carries nothing at zero field. */
      S.prevField = S.field;
      S.field = 0; S.larmor = 0;
      document.getElementById("field").value = 0;
      document.getElementById("larmor").value = 0;
      S.transitions = "all";
      document.getElementById("transitions").value = "all";
      ppm.disabled = true;
      setUnit("MHz");
    }else{
      S.field = S.prevField != null ? S.prevField : 11.7449;
      S.larmor = (S.gamma||0) * S.field;
      document.getElementById("field").value = S.field;
      document.getElementById("larmor").value = S.larmor.toFixed(4);
      ppm.disabled = false;
    }
    invalidateFit(); S.win.auto = true; simulate();
  };
```

The existing `field`/`larmor` `oninput` handlers already keep the two coupled through `S.gamma` and re-simulate; they now simply keep working while NQR is ticked.

- [ ] **Step 5: Guard `setUnit` against the disabled ppm unit**

Replace

```js
function setUnit(u){
  S.unit = u;
```

with

```js
function setUnit(u){
  if(u === "ppm" && S.nqr) return;      /* ppm needs a non-zero reference */
  S.unit = u;
```

- [ ] **Step 6: Explain identically-zero spectra instead of drawing nothing**

In `simulate()`, replace

```js
    if(S.win.auto){ S.win.low = r.low; S.win.high = r.high; syncWindowInputs(); }
    clearError(); draw();
```

with

```js
    if(S.win.auto){ S.win.low = r.low; S.win.high = r.high; syncWindowInputs(); }
    if(r.total.y.some(v => v > 0)){ clearError(); }
    else{
      showError("no observable intensity in this window" + (S.larmor ? "" :
        " — at zero field the central transition carries none; set transitions to “all”"));
    }
    draw();
```

- [ ] **Step 7: Run the tests — expect PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/test_webapp.py -v`
Expected: all pass (including `test_page_keeps_field_editable_in_nqr_mode` and `test_page_is_self_contained`).

- [ ] **Step 8: Commit**

```bash
git add nqrlyze/static/index.html tests/test_webapp.py
git commit -m "feat(gui): NQR checkbox becomes a preset - Zeeman-perturbed NQR and a zero-intensity hint"
```

---

### Task 4: Manual verification in the browser

The dev server may already be running on port 8765 (`.venv\Scripts\nqrlyze.exe gui --no-browser`); it serves the page from the repo, so a plain reload picks up the edits. Start it if it is not running.

- [ ] **Step 1: Reload `http://127.0.0.1:8765` and check the window panel**

Expected: the Window card shows **Centre [MHz]** and **Width [kHz]** (plus Points and Auto), no From/To. The default ²⁷Al simulation looks unchanged; editing Centre shifts the visible window, editing Width zooms.

- [ ] **Step 2: Tick NQR with a spin-3/2 nucleus**

Choose e.g. ³⁵Cl, set Cq = 72 MHz (Site panel), tick **NQR**.
Expected: Transitions flips to "all", ppm button is greyed out, a single line appears near 36 MHz (Cq/2·√(1+η²/3)) — this is the previously-invisible case.

- [ ] **Step 3: Zeeman-perturb it**

With NQR still ticked, type `0.1` into Field (T).
Expected: Larmor updates via γ, the line splits into a Zeeman pattern, and the auto window widens. Setting Field back to 0 restores the single line.

- [ ] **Step 4: Check the zero-intensity hint**

With NQR ticked and Field 0, set Transitions back to "central only".
Expected: the status line shows *"no observable intensity in this window — at zero field the central transition carries none; set transitions to “all”"* instead of a silent flat trace.

- [ ] **Step 5: Untick NQR**

Expected: Field returns to its previous value, ppm re-enabled, normal NMR pattern back.

If any step fails, fix before proceeding — do not adjust the expected behaviour to match a bug.

---

### Task 5: README update and full suite

**Files:**
- Modify: `README.md` ("The interface" section, ~lines 31–47)

- [ ] **Step 1: Extend the GUI paragraph**

In `README.md`, replace

```markdown
Opens a local page at `http://127.0.0.1:8765`. Every parameter has a slider next
to its number, and the simulation follows as you drag — so you get close by eye,
tick the parameters you want fitted, and press **Fit**. Nothing is uploaded
anywhere: the server is the standard library, bound to loopback, and the page
makes no external requests.
```

with

```markdown
Opens a local page at `http://127.0.0.1:8765`. Every parameter has a slider next
to its number, and the simulation follows as you drag — so you get close by eye,
tick the parameters you want fitted, and press **Fit**. Nothing is uploaded
anywhere: the server is the standard library, bound to loopback, and the page
makes no external requests.

The frequency window is entered QUEST-style, as a centre frequency (MHz) and a
spectral width (kHz). Ticking **NQR** zeroes the field and switches to all
transitions — but the field and Larmor inputs stay live, so typing a small
field on top gives Zeeman-perturbed NQR, computed by the same exact
diagonalisation as everything else.
```

- [ ] **Step 2: Run the full test suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 95 passed (91 existing + 4 new; takes ~10 minutes — the fits are real fits).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: describe centre/width window entry and Zeeman-perturbed NQR"
```
