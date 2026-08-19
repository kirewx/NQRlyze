"""Physical constants and nuclear data.

The nuclear table is a convenience only: ``Cq`` is a *fitted* parameter, so the
quadrupole moment ``Q`` matters solely for converting between ``Cq`` and a
computed electric field gradient ``Vzz``.  Gyromagnetic ratios are needed to
turn a field strength into a Larmor frequency.

Values follow the IUPAC recommendations (Harris et al., Pure Appl. Chem. 2001 /
2008) and Pyykko's quadrupole moment compilation (Mol. Phys. 2008).
"""

from __future__ import annotations

from dataclasses import dataclass

# Elementary charge divided by Planck's constant, Hz/V.
E_OVER_H = 2.417_989_242e14

# 1 atomic unit of electric field gradient, V/m^2.
AU_EFG = 9.717_362_4424e21

# 1 millibarn, m^2.
MILLIBARN = 1e-31

#: ``Cq``/MHz = ``EFG_TO_CQ`` * ``Vzz``/a.u. * ``Q``/mb
EFG_TO_CQ = E_OVER_H * AU_EFG * MILLIBARN / 1e6  # ~0.2349647


@dataclass(frozen=True)
class Nucleus:
    """A quadrupolar isotope."""

    symbol: str
    spin: float
    gamma: float
    """Gyromagnetic ratio / 2*pi, in MHz/T (signed)."""
    quadrupole_moment: float
    """Nuclear electric quadrupole moment Q, in millibarn (signed)."""

    def larmor(self, field: float) -> float:
        """Larmor frequency in MHz for a magnetic field in tesla (always >= 0).

        The sign of ``gamma`` does not affect a powder lineshape -- the
        Zeeman-plus-quadrupolar powder pattern is invariant under
        ``nu_L -> -nu_L`` -- so the magnitude is what the simulator wants.
        """
        return abs(self.gamma) * field

    def field(self, larmor: float) -> float:
        """Magnetic field in tesla giving a Larmor frequency in MHz."""
        return larmor / abs(self.gamma)

    def vzz_to_cq(self, vzz_au: float) -> float:
        """Convert Vzz in atomic units to Cq in MHz."""
        return EFG_TO_CQ * vzz_au * self.quadrupole_moment

    def cq_to_vzz(self, cq_mhz: float) -> float:
        """Convert Cq in MHz to Vzz in atomic units."""
        return cq_mhz / (EFG_TO_CQ * self.quadrupole_moment)


def _n(symbol, spin, gamma, q):
    return Nucleus(symbol, spin, gamma, q)


NUCLEI: dict[str, Nucleus] = {
    n.symbol: n
    for n in [
        _n("2H", 1.0, 6.536_0, 2.860),
        _n("6Li", 1.0, 6.266_1, -0.808),
        _n("7Li", 1.5, 16.548_3, -40.10),
        _n("9Be", 1.5, -5.983_9, 52.88),
        _n("10B", 3.0, 4.575_4, 84.59),
        _n("11B", 1.5, 13.663_0, 40.59),
        _n("14N", 1.0, 3.077_7, 20.44),
        _n("17O", 2.5, -5.774_2, -25.58),
        _n("21Ne", 1.5, -3.363_1, 101.55),
        _n("23Na", 1.5, 11.268_8, 104.0),
        _n("25Mg", 2.5, -2.608_3, 199.4),
        _n("27Al", 2.5, 11.103_1, 146.6),
        _n("33S", 1.5, 3.271_7, -67.8),
        _n("35Cl", 1.5, 4.176_5, -81.65),
        _n("37Cl", 1.5, 3.476_5, -64.35),
        _n("39K", 1.5, 1.989_3, 58.5),
        _n("41K", 1.5, 1.092_2, 71.1),
        _n("43Ca", 3.5, -2.869_7, -40.8),
        _n("45Sc", 3.5, 10.359_1, -220.0),
        _n("47Ti", 2.5, -2.404_1, 302.0),
        _n("49Ti", 3.5, -2.404_8, 247.0),
        _n("51V", 3.5, 11.213_3, -52.0),
        _n("53Cr", 1.5, -2.411_5, -150.0),
        _n("55Mn", 2.5, 10.576_3, 330.0),
        _n("59Co", 3.5, 10.077_0, 420.0),
        _n("61Ni", 1.5, -3.805_5, 162.0),
        _n("63Cu", 1.5, 11.318_3, -220.0),
        _n("65Cu", 1.5, 12.127_6, -204.0),
        _n("67Zn", 2.5, 2.669_4, 150.0),
        _n("69Ga", 1.5, 10.247_8, 171.0),
        _n("71Ga", 1.5, 13.020_8, 107.0),
        _n("73Ge", 4.5, -1.489_7, -196.0),
        _n("75As", 1.5, 7.315_0, 314.0),
        _n("79Br", 1.5, 10.704_2, 313.0),
        _n("81Br", 1.5, 11.538_4, 262.0),
        _n("85Rb", 2.5, 4.126_4, 276.0),
        _n("87Rb", 1.5, 13.981_1, 133.5),
        _n("87Sr", 4.5, -1.852_5, 305.0),
        _n("91Zr", 2.5, -3.974_8, -176.0),
        _n("93Nb", 4.5, 10.452_3, -320.0),
        _n("95Mo", 2.5, -2.787_4, -22.0),
        _n("99Ru", 2.5, -1.957_1, 79.0),
        _n("105Pd", 2.5, -1.957_0, 660.0),
        _n("115In", 4.5, 9.385_6, 810.0),
        _n("119Sn", 0.5, -15.966_0, 0.0),
        _n("121Sb", 2.5, 10.255_1, -543.0),
        _n("123Sb", 3.5, 5.553_2, -692.0),
        _n("127I", 2.5, 8.577_8, -696.0),
        _n("131Xe", 1.5, 3.535_9, -114.0),
        _n("133Cs", 3.5, 5.623_4, -3.43),
        _n("135Ba", 1.5, 4.258_2, 160.0),
        _n("137Ba", 1.5, 4.763_4, 245.0),
        _n("139La", 3.5, 6.061_2, 200.0),
        _n("181Ta", 3.5, 5.162_7, 3170.0),
        _n("185Re", 2.5, 9.717_0, 2180.0),
        _n("187Re", 2.5, 9.817_0, 2070.0),
        _n("189Os", 1.5, 3.353_6, 856.0),
        _n("197Au", 1.5, 0.729_0, 547.0),
        _n("201Hg", 1.5, -1.788_7, 387.0),
        _n("209Bi", 4.5, 6.962_8, -516.0),
    ]
}


def get_nucleus(symbol: str) -> Nucleus:
    """Look up an isotope by symbol, e.g. ``"35Cl"`` or ``"Cl35"``."""
    key = symbol.strip().replace("-", "")
    if key in NUCLEI:
        return NUCLEI[key]
    # Accept "Cl35" as well as "35Cl".
    digits = "".join(c for c in key if c.isdigit())
    letters = "".join(c for c in key if c.isalpha())
    if digits and letters:
        swapped = f"{digits}{letters.capitalize()}"
        if swapped in NUCLEI:
            return NUCLEI[swapped]
    raise KeyError(
        f"unknown nucleus {symbol!r}; known: {', '.join(sorted(NUCLEI))}"
    )
