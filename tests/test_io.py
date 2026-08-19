"""Reading Bruker and text spectra, and joining sub-spectra."""

import numpy as np
import pytest

from _bruker_fixture import write_bruker_1d
from nqrlyze.coadd import coadd, common_axis
from nqrlyze.io import read_ascii, read_bruker, read_bruker_series, write_ascii
from nqrlyze.io.bruker import find_pdata, read_jcamp_parameters
from nqrlyze.spectrum import Spectrum

SF, SW_P, OFFSET, SI = 49.0525, 1.0e6, 400.0, 4096


def _dataset(tmp_path, name="1", **kwargs):
    settings = dict(sf=SF, sw_p=SW_P, offset=OFFSET, ns=16)
    settings.update(kwargs)
    intensity = np.zeros(kwargs.pop("size", SI))
    intensity[1000] = 1e6
    return write_bruker_1d(tmp_path / name, intensity, **settings)


def test_bruker_frequency_axis_follows_the_topspin_convention(tmp_path):
    spectrum = read_bruker(_dataset(tmp_path))
    assert len(spectrum) == SI
    # Highest frequency is the leftmost point, at OFFSET ppm.
    assert spectrum.freq_mhz[-1] == pytest.approx(SF + OFFSET * SF * 1e-6)
    assert spectrum.ppm[-1] == pytest.approx(OFFSET, abs=1e-9)
    # Point i sits at OFFSET - i * SW_p / (SI * SF) ppm.
    descending = spectrum.ppm[::-1]
    for i in (0, 1, 1000, SI - 1):
        assert descending[i] == pytest.approx(OFFSET - i * SW_P / (SI * SF), abs=1e-9)
    assert spectrum.reference == pytest.approx(SF)


def test_bruker_reads_acquisition_parameters(tmp_path):
    spectrum = read_bruker(_dataset(tmp_path))
    assert spectrum.meta["NS"] == 16
    assert spectrum.meta["NUC1"] == "35Cl"
    assert spectrum.meta["P"][0] == 2.5  # array-valued parameter


@pytest.mark.parametrize("big_endian", [False, True])
@pytest.mark.parametrize("nc_proc", [0, 3, -2])
def test_bruker_respects_byte_order_and_scaling(tmp_path, big_endian, nc_proc):
    root = tmp_path / f"d{big_endian}{nc_proc}"
    intensity = np.zeros(512)
    intensity[100] = 8192.0
    write_bruker_1d(
        root, intensity, sf=SF, sw_p=SW_P, offset=OFFSET,
        nc_proc=nc_proc, big_endian=big_endian,
    )
    spectrum = read_bruker(root)
    assert spectrum.intensity.max() == pytest.approx(8192.0)


def test_find_pdata_accepts_any_level(tmp_path):
    root = _dataset(tmp_path)
    direct = read_bruker(root).intensity
    assert np.array_equal(read_bruker(root / "pdata" / "1").intensity, direct)
    assert np.array_equal(read_bruker(root / "pdata" / "1" / "1r").intensity, direct)
    with pytest.raises(FileNotFoundError):
        find_pdata(tmp_path / "nothing-here")


def test_jcamp_parser_handles_strings_arrays_and_numbers(tmp_path):
    path = tmp_path / "acqus"
    path.write_text(
        "##TITLE= x\n##$SI= 4096\n##$SW= 1.25e5\n##$NUC1= <27Al>\n"
        "##$AMP= (0..3)\n1 2 3 4\n##$SOLVENT= <none>\n##END=\n"
    )
    params = read_jcamp_parameters(path)
    assert params["SI"] == 4096 and isinstance(params["SI"], int)
    assert params["SW"] == pytest.approx(1.25e5)
    assert params["NUC1"] == "27Al"
    assert params["AMP"] == [1, 2, 3, 4]
    assert params["SOLVENT"] == "none"


def test_ascii_round_trip(tmp_path):
    x = np.linspace(49.0, 49.5, 200)
    original = Spectrum(x, np.sin(x * 40), reference=49.25)
    path = tmp_path / "s.txt"
    write_ascii(path, original, unit="MHz", header="a comment\nand another")
    reread = read_ascii(path, unit="MHz")
    assert np.allclose(reread.freq_mhz, original.freq_mhz)
    assert np.allclose(reread.intensity, original.intensity)


def test_ascii_accepts_messy_files(tmp_path):
    path = tmp_path / "messy.txt"
    path.write_text(
        "# exported by something\n"
        "frequency\tintensity\n"
        "\n"
        "% another comment\n"
        "10.0, 1.0\n"
        "10.1; 2.0\n"
        "10.2   3.0\n"
    )
    spectrum = read_ascii(path, unit="MHz")
    assert len(spectrum) == 3
    assert np.allclose(spectrum.intensity, [1.0, 2.0, 3.0])


def test_ascii_unit_conversion(tmp_path):
    path = tmp_path / "ppm.txt"
    path.write_text("100.0 1.0\n0.0 2.0\n-100.0 3.0\n")
    spectrum = read_ascii(path, unit="ppm", reference=130.0)
    assert spectrum.ppm[-1] == pytest.approx(100.0)
    assert spectrum.freq_mhz[-1] == pytest.approx(130.0 * (1 + 100e-6))
    with pytest.raises(ValueError):
        read_ascii(path, unit="ppm", reference=0.0)


def test_spectrum_is_stored_ascending():
    x = np.linspace(10.0, 9.0, 50)
    spectrum = Spectrum(x, np.arange(50.0))
    assert spectrum.freq_mhz[0] < spectrum.freq_mhz[-1]
    assert spectrum.intensity[0] == 49.0


def test_coadd_reconstructs_overlapping_pieces():
    truth = lambda x: np.exp(-(((x - 10.0) / 0.3) ** 2)) + 0.6 * np.exp(
        -(((x - 10.9) / 0.2) ** 2)
    )
    pieces = [
        Spectrum(np.linspace(c - 0.5, c + 0.5, 501), truth(np.linspace(c - 0.5, c + 0.5, 501)))
        for c in (9.6, 10.2, 10.8, 11.4)
    ]
    axis = np.linspace(9.7, 11.3, 1601)
    for mode in ("mean", "skyline"):
        combined = coadd(pieces, mode=mode, dx_mhz=1e-3)
        got = np.interp(axis, combined.freq_mhz, combined.intensity)
        assert np.max(np.abs(got - truth(axis))) < 1e-4
    # Plain summation double counts the overlaps, by design.
    summed = coadd(pieces, mode="sum", dx_mhz=1e-3)
    got = np.interp(axis, summed.freq_mhz, summed.intensity)
    assert np.max(got - truth(axis)) > 0.5


def test_coadd_spans_every_piece():
    pieces = [
        Spectrum(np.linspace(1.0, 2.0, 11), np.ones(11)),
        Spectrum(np.linspace(5.0, 6.0, 101), np.ones(101)),
    ]
    axis = common_axis(pieces)
    assert axis[0] == pytest.approx(1.0)
    assert axis[-1] <= 6.0 and axis[-1] > 6.0 - 0.02
    combined = coadd(pieces, mode="sum")
    assert combined.intensity[len(combined) // 2] == 0.0  # the gap stays empty


def test_coadd_scales_by_scans(tmp_path):
    for index, scans in enumerate((4, 64)):
        intensity = np.zeros(256)
        intensity[128] = 1000.0 * scans
        write_bruker_1d(
            tmp_path / f"e{index}", intensity, sf=SF, sw_p=1e5,
            offset=OFFSET, ns=scans,
        )
    pieces = read_bruker_series(
        [tmp_path / "e0", tmp_path / "e1"], scale_by_scans=True
    )
    assert pieces[0].intensity.max() == pytest.approx(pieces[1].intensity.max())


def test_coadd_rejects_bad_input():
    with pytest.raises(ValueError):
        coadd([])
    with pytest.raises(ValueError):
        coadd([Spectrum(np.linspace(0, 1, 10), np.ones(10))], mode="nope")
    with pytest.raises(ValueError):
        coadd([Spectrum(np.linspace(0, 1, 10), np.ones(10))], weights=[1.0, 2.0])
