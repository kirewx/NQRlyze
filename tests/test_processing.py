"""Phase and baseline correction, and the imaginary part that phasing needs."""

import numpy as np
import pytest

from _bruker_fixture import write_bruker_1d
from nqrlyze.coadd import coadd
from nqrlyze.io import read_ascii, read_bruker, read_bruker_series
from nqrlyze.processing import analytic_signal, auto_phase0, baseline, phase, process
from nqrlyze.spectrum import Spectrum

X = np.linspace(34.0, 36.0, 4001)


def _lorentzian(x=X, centre=35.0, width=0.02):
    """Complex Lorentzian: absorption real part, dispersion imaginary part."""
    z = 1.0 / (width / 2 - 1j * (x - centre))
    return z * (width / 2)


def test_phase_round_trip():
    clean = _lorentzian()
    distorted = phase(X, clean, 50.0, -120.0, pivot_mhz=35.3)
    recovered = phase(X, distorted, -50.0, 120.0, pivot_mhz=35.3)
    assert np.allclose(recovered, clean)
    assert not np.allclose(distorted.real, clean.real, atol=0.1)


def test_first_order_phase_vanishes_at_the_pivot():
    clean = _lorentzian()
    rotated = phase(X, clean, 0.0, 300.0, pivot_mhz=35.0)
    at_pivot = np.argmin(np.abs(X - 35.0))
    assert rotated[at_pivot] == pytest.approx(clean[at_pivot])


def test_auto_phase0_recovers_a_zero_order_error():
    clean = _lorentzian()
    for error in (-150.0, -37.0, 0.0, 64.0, 170.0):
        found = auto_phase0(phase(X, clean, error, 0.0))
        assert ((found + error + 180.0) % 360.0 - 180.0) == pytest.approx(0.0, abs=0.5)


def test_hilbert_transform_rebuilds_the_dispersion():
    """A real-only absorption line gets its dispersion back, so it can still be
    rephased (away from the ends, where the transform wraps round)."""
    clean = _lorentzian(width=0.01)
    rebuilt = analytic_signal(clean.real)
    inner = slice(800, -800)
    assert np.max(np.abs(rebuilt.imag[inner] - clean.imag[inner])) < 0.02


def test_baseline_removes_a_polynomial_under_the_signal():
    ramp = 0.3 + 0.8 * (X - 35.0) - 1.5 * (X - 35.0) ** 2
    line = _lorentzian(width=0.01).real
    corrected = process(X, line + ramp, baseline_order=2, baseline_edge=0.2)
    assert np.max(np.abs(corrected - line)) < 0.01
    assert np.allclose(baseline(X, ramp, order=-1), 0.0)
    with pytest.raises(ValueError):
        baseline(X, ramp, order=1, edge=0.8)


def test_process_leaves_an_uncorrected_spectrum_alone():
    real = _lorentzian().real
    assert np.array_equal(process(X, real), real)


def test_ascii_reads_a_third_column_as_imaginary(tmp_path):
    z = _lorentzian()
    path = tmp_path / "complex.txt"
    np.savetxt(path, np.column_stack([X, z.real, z.imag]))
    spectrum = read_ascii(path)
    assert np.allclose(spectrum.imag, z.imag)
    np.savetxt(path, np.column_stack([X, z.real]))
    assert read_ascii(path).imag is None


def test_bruker_reads_1i_alongside_1r(tmp_path):
    root = write_bruker_1d(tmp_path / "1", np.arange(64.0) * 100, 35.0, 1e5, 0.0)
    (root / "pdata" / "1" / "1i").write_bytes(
        (np.arange(64) * -7).astype("<i4").tobytes()
    )
    spectrum = read_bruker(root)
    # Stored ascending; Bruker files run from high frequency to low.
    assert np.allclose(spectrum.imag, (np.arange(64) * -7.0)[::-1])
    scaled = read_bruker_series([root], scale_by_scans=True)[0]
    assert np.allclose(scaled.imag, spectrum.imag / 16)


def test_coadd_combines_imaginary_parts():
    z = _lorentzian()
    pieces = [Spectrum(X[:2500], z.real[:2500], 0.0, imag=z.imag[:2500]),
              Spectrum(X[1500:], z.real[1500:], 0.0, imag=z.imag[1500:])]
    joined = coadd(pieces, mode="mean")
    expected = np.interp(joined.freq_mhz, X, z.imag)
    assert np.allclose(joined.imag, expected, atol=1e-6)
    assert coadd(pieces, mode="skyline").imag is None


def test_spectrum_keeps_imag_through_its_helpers():
    z = _lorentzian()
    spectrum = Spectrum(X[::-1], z.real[::-1], 0.0, imag=z.imag[::-1])
    assert np.allclose(spectrum.imag, z.imag)
    assert spectrum.crop(34.5, 35.5).imag.size == spectrum.crop(34.5, 35.5).intensity.size
    assert np.allclose(spectrum.normalized().imag, z.imag / np.max(np.abs(z.real)))
    with pytest.raises(ValueError):
        Spectrum(X, z.real, 0.0, imag=z.imag[:10])
