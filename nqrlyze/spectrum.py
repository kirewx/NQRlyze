"""A measured or simulated spectrum on an absolute frequency axis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Spectrum"]


@dataclass
class Spectrum:
    """Intensity against absolute frequency in MHz, stored ascending.

    ``reference`` is the frequency of 0 ppm (Bruker ``SF``), used only to build
    a ppm scale; it is ``0`` for pure NQR data, where ppm is meaningless.
    """

    freq_mhz: np.ndarray
    intensity: np.ndarray
    reference: float = 0.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.freq_mhz = np.asarray(self.freq_mhz, dtype=float)
        self.intensity = np.asarray(self.intensity, dtype=float)
        if self.freq_mhz.shape != self.intensity.shape:
            raise ValueError("frequency and intensity must have the same shape")
        if self.freq_mhz.size and self.freq_mhz[0] > self.freq_mhz[-1]:
            self.freq_mhz = self.freq_mhz[::-1].copy()
            self.intensity = self.intensity[::-1].copy()

    def __len__(self) -> int:
        return self.freq_mhz.size

    @property
    def ppm(self) -> np.ndarray:
        if self.reference == 0:
            raise ValueError("no reference frequency: this spectrum has no ppm scale")
        return (self.freq_mhz - self.reference) / self.reference * 1e6

    @property
    def khz(self) -> np.ndarray:
        """Offset from the reference in kHz."""
        return (self.freq_mhz - self.reference) * 1e3

    def crop(self, low_mhz: float, high_mhz: float) -> "Spectrum":
        """Restrict to a frequency window (inclusive)."""
        mask = (self.freq_mhz >= low_mhz) & (self.freq_mhz <= high_mhz)
        if not np.any(mask):
            raise ValueError(f"no points between {low_mhz} and {high_mhz} MHz")
        return Spectrum(
            self.freq_mhz[mask], self.intensity[mask], self.reference, dict(self.meta)
        )

    def normalized(self) -> "Spectrum":
        """Scale so the largest absolute intensity is 1."""
        peak = np.max(np.abs(self.intensity))
        scale = 1.0 / peak if peak > 0 else 1.0
        return Spectrum(
            self.freq_mhz, self.intensity * scale, self.reference, dict(self.meta)
        )

    def resample(self, freq_mhz: np.ndarray) -> "Spectrum":
        """Linearly interpolate onto a new axis, zero outside the current range."""
        freq_mhz = np.asarray(freq_mhz, dtype=float)
        values = np.interp(
            freq_mhz, self.freq_mhz, self.intensity, left=0.0, right=0.0
        )
        return Spectrum(freq_mhz, values, self.reference, dict(self.meta))
