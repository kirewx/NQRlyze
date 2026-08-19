"""nqrlyze -- automatic fitting of exact quadrupolar NMR and NQR powder patterns.

Simulation follows the same physics as QUEST (Perras, Widdifield and Bryce,
*Solid State Nucl. Magn. Reson.* **2012**, 45-46, 36): the combined Zeeman and
quadrupolar Hamiltonian is diagonalised exactly, with no perturbation expansion,
so the whole range from high-field NMR to zero-field NQR is covered by one code
path.  Powder averaging uses the Alderman-Solum-Grant interpolation scheme.

On top of that sits the part QUEST does not provide: an optimiser that finds
``Cq``, ``eta`` and the Lorentzian/Gaussian broadening from a measured spectrum,
reads Bruker data directly, and joins stepped-frequency sub-spectra first.
"""

from .analytic import (
    cq_range_for_line,
    nqr_frequencies,
    nu_q,
    second_order_ct_isotropic_shift,
    second_order_ct_shift,
)
from .coadd import coadd
from .constants import NUCLEI, get_nucleus
from .fit import FitParameter, FitResult, default_parameters, fit
from .hamiltonian import eigen_transitions
from .io import read_ascii, read_bruker, read_bruker_series, write_ascii
from .powder import asg_grid, spiral_grid
from .simulate import (
    Experiment,
    Site,
    khz_axis,
    ppm_axis,
    simulate,
    simulate_sites,
    suggest_window,
)
from .spectrum import Spectrum
from .validate import compare, find_singularities, run_manifest

__version__ = "0.1.0"

__all__ = [
    "Experiment",
    "FitParameter",
    "FitResult",
    "NUCLEI",
    "Site",
    "Spectrum",
    "asg_grid",
    "coadd",
    "cq_range_for_line",
    "compare",
    "default_parameters",
    "eigen_transitions",
    "find_singularities",
    "fit",
    "get_nucleus",
    "khz_axis",
    "nqr_frequencies",
    "nu_q",
    "ppm_axis",
    "read_ascii",
    "read_bruker",
    "read_bruker_series",
    "run_manifest",
    "second_order_ct_isotropic_shift",
    "second_order_ct_shift",
    "simulate",
    "simulate_sites",
    "spiral_grid",
    "suggest_window",
    "write_ascii",
]
