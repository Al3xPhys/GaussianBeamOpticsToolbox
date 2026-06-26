"""
geometry.py
===========
Station and link geometry definitions for FSO simulation.
Handles ground, airborne, and space-based terminals.
"""

from dataclasses import dataclass
import numpy as np

# Karman line — above this, no atmosphere
ATMOSPHERE_CEILING_M = 100_000.0


@dataclass
class Station:
    """
    Represents a terminal node in an FSO link.

    Parameters
    ----------
    name : str
        Human-readable label.
    altitude_m : float
        Altitude above sea level in metres. Use 0 for ground station.
    aperture_radius_m : float
        Aperture radius in metres (used for both Tx and Rx).
    pointing_jitter_urad : float
        RMS pointing jitter in microradians. Statistical pointing loss model.
        Set to 0 to disable.
    pointing_offset_urad : float
        Fixed static pointing offset in microradians. Deterministic pointing loss.
        Set to 0 to disable.
    """

    name: str
    altitude_m: float
    aperture_radius_m: float
    pointing_jitter_urad: float = 0.0
    pointing_offset_urad: float = 0.0

    @property
    def is_space(self) -> bool:
        """True if station is above the Karman line."""
        return self.altitude_m >= ATMOSPHERE_CEILING_M

    @property
    def is_ground(self) -> bool:
        return self.altitude_m < ATMOSPHERE_CEILING_M

    def __str__(self):
        kind = "space" if self.is_space else "ground/airborne"
        return (
            f"Station '{self.name}' [{kind}]  "
            f"alt={self.altitude_m/1e3:.1f} km  "
            f"aperture={self.aperture_radius_m*100:.1f} cm"
        )


@dataclass
class LinkGeometry:
    """
    Geometric description of an FSO link between two stations.

    Parameters
    ----------
    tx : Station
        Transmitting station.
    rx : Station
        Receiving station.
    elevation_angle_deg : float
        Elevation angle in degrees at the lower station (or Tx if both at same
        altitude). 90 = zenith, 0 = horizon.
        For sat-to-sat links above the atmosphere, atmospheric effects are
        skipped entirely regardless of this value.

    Derived properties
    ------------------
    link_distance_m       : total slant range (flat-Earth approx)
    atm_path_length_m     : slant path length through the atmosphere
    vacuum_path_length_m  : remaining path in vacuum
    has_atmosphere        : whether any part of the path is atmospheric
    """

    tx: Station
    rx: Station
    elevation_angle_deg: float

    def __post_init__(self):
        if not (0 < self.elevation_angle_deg <= 90):
            raise ValueError("Elevation angle must be in (0, 90] degrees.")

    @property
    def zenith_angle_rad(self) -> float:
        return np.radians(90.0 - self.elevation_angle_deg)

    @property
    def link_distance_m(self) -> float:
        """Slant range. Flat-Earth: d = delta_h / sin(elevation)."""
        delta_h = abs(self.rx.altitude_m - self.tx.altitude_m)
        if delta_h == 0:
            raise ValueError(
                "Both stations at the same altitude. For horizontal links "
                "use LinkBudget directly with free_space() propagation."
            )
        return delta_h / np.sin(np.radians(self.elevation_angle_deg))

    @property
    def has_atmosphere(self) -> bool:
        """True if any part of the link path passes through the atmosphere."""
        return min(self.tx.altitude_m, self.rx.altitude_m) < ATMOSPHERE_CEILING_M

    @property
    def atm_bottom_m(self) -> float:
        return min(self.tx.altitude_m, self.rx.altitude_m)

    @property
    def atm_top_m(self) -> float:
        return min(max(self.tx.altitude_m, self.rx.altitude_m), ATMOSPHERE_CEILING_M)

    @property
    def atm_path_length_m(self) -> float:
        """Slant path length through atmosphere in metres."""
        if not self.has_atmosphere:
            return 0.0
        delta_h = self.atm_top_m - self.atm_bottom_m
        return delta_h / np.sin(np.radians(self.elevation_angle_deg))

    @property
    def vacuum_path_length_m(self) -> float:
        return self.link_distance_m - self.atm_path_length_m

    @property
    def lower_station(self) -> Station:
        return self.tx if self.tx.altitude_m <= self.rx.altitude_m else self.rx

    @property
    def upper_station(self) -> Station:
        return self.rx if self.tx.altitude_m <= self.rx.altitude_m else self.tx

    def summary(self):
        print(f"  Tx : {self.tx}")
        print(f"  Rx : {self.rx}")
        print(f"  Elevation angle      : {self.elevation_angle_deg:.1f} deg")
        print(f"  Total link distance  : {self.link_distance_m/1e3:.1f} km")
        print(f"  Atmospheric path     : {self.atm_path_length_m/1e3:.1f} km")
        print(f"  Vacuum path          : {self.vacuum_path_length_m/1e3:.1f} km")
        print(f"  Has atmosphere       : {self.has_atmosphere}")
