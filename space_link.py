"""
space_link.py
=============
Space-aware FSO link pipeline. Handles ground-to-ground, ground-to-space,
space-to-ground, and sat-to-sat links from a unified interface.

Builds on LinkBudget internally — adds geometry awareness, slant-path
turbulence via the Hufnagel-Valley model, and pointing loss from Station
parameters.

Example — ground to LEO
-----------------------
    from space_link import SpaceLink
    from geometry import Station, LinkGeometry
    from beam_class import Beam
    import components as comp

    ground = Station("Ground (Tenerife)", altitude_m=2400, aperture_radius_m=0.15,
                     pointing_jitter_urad=2.0)
    sat    = Station("LEO sat", altitude_m=500_000, aperture_radius_m=0.10)
    geom   = LinkGeometry(tx=ground, rx=sat, elevation_angle_deg=30)
    beam   = Beam.from_beam_waist(waist_radius=5e-6, wavelength=1550e-9, power=1.0)

    link    = SpaceLink(beam, geom)
    results = link.run()
    results.summary()
    rx_power = comp.receiver(results.final_beam, sat.aperture_radius_m)
"""

import numpy as np
from beam_class import Beam
from geometry import LinkGeometry, ATMOSPHERE_CEILING_M
from link_budget import LinkBudget, LinkResults
import transfer_matrices as tm
import atmospherics as atm
from atmospherics.hufnagel_valley import integrated_cn2


class SpaceLink:
    """
    Space-aware FSO link pipeline.

    Automatically builds the correct propagation chain based on link geometry:
      - Free-space propagation (full link distance)
      - Slant-path HV turbulence (if path crosses atmosphere)
      - Pointing loss (static and/or jitter) from Station parameters,
        applied *after* propagation so the correct receiver-plane spot size is used
      - Sat-to-sat links above atmosphere skip all atmospheric effects entirely

    Custom segments can be added via add_segment() before calling run().

    Parameters
    ----------
    beam : Beam
        Fully conditioned Tx beam (after all Tx optics).
    geometry : LinkGeometry
        Link geometry (stations + elevation angle).
    wind_speed : float
        HV model rms wind speed in m/s. Default 21 m/s (HV 5/7).
    ground_cn2 : float
        HV model ground-level Cn2 in m^(-2/3). Default 1.7e-14 (HV 5/7).
    scintillation_availability : float or None
        Availability target for scintillation fade margin (e.g. 0.999).
        Set to None to skip scintillation calculation.
    """

    def __init__(
        self,
        beam: Beam,
        geometry: LinkGeometry,
        wind_speed: float = atm.hufnagel_valley.HV57_WIND_SPEED,
        ground_cn2: float = atm.hufnagel_valley.HV57_GROUND_Cn2,
        scintillation_availability: float = 0.999,
    ):
        self._beam = beam
        self.geometry = geometry
        self.wind_speed = wind_speed
        self.ground_cn2 = ground_cn2
        self.scintillation_availability = scintillation_availability
        self._extra_segments = []

    def add_segment(self, label, fn):
        """Add a custom Beam->Beam segment to the chain."""
        self._extra_segments.append((label, fn))
        return self

    def _effective_cn2_for_scintillation(self):
        """
        Computes an effective path-averaged Cn2 for the scintillation model
        from the HV Cn2 integral divided by the atmospheric path length.
        This gives a single Cn2 value that, used with the atm path length,
        reproduces the correct Rytov variance for the slant path.
        """
        geom = self.geometry
        cn2_integral = integrated_cn2(
            geom.atm_bottom_m, geom.atm_top_m, self.wind_speed, self.ground_cn2
        )
        # Divide by slant path length to get an effective Cn2
        # (the scintillation model will multiply by distance internally)
        if geom.atm_path_length_m > 0:
            return cn2_integral / geom.atm_path_length_m
        return 0.0

    def _build_link(self) -> LinkBudget:
        geom = self.geometry
        link = LinkBudget(self._beam)

        # ── Free space propagation ────────────────────────────────────────────
        d = geom.link_distance_m
        link.add_propagation(
            f"Free space ({d/1e3:.0f} km)", lambda b: b.propagate(tm.free_space(d))
        )

        # ── Atmospheric effects ───────────────────────────────────────────────
        if geom.has_atmosphere:
            link.add_atmosphere(
                "Slant turbulence (HV model)",
                lambda b: atm.hufnagel_valley.slant_turbulence(
                    b,
                    geom,
                    wind_speed=self.wind_speed,
                    ground_cn2=self.ground_cn2,
                ),
            )

        # ── Pointing losses (applied after propagation — uses Rx-plane spot) ──
        # Pointing applies whether or not there's atmosphere
        if geom.tx.pointing_offset_urad > 0:
            offset = geom.tx.pointing_offset_urad
            link.add_segment(
                f"Tx pointing offset ({offset:.1f} µrad)",
                lambda b: atm.pointing.pointing_loss_static(b, offset, d),
            )

        if geom.tx.pointing_jitter_urad > 0:
            jitter = geom.tx.pointing_jitter_urad
            link.add_segment(
                f"Tx pointing jitter ({jitter:.1f} µrad rms)",
                lambda b: atm.pointing.pointing_loss_jitter(b, jitter, d),
            )

        # ── User-added custom segments ────────────────────────────────────────
        for label, fn in self._extra_segments:
            link.add_segment(label, fn)

        return link

    def run(self) -> "SpaceLinkResults":
        link = self._build_link()
        base_results = link.run()

        # Scintillation uses path-averaged effective Cn2 over the atm path length
        scint_stats = None
        if (
            self.geometry.has_atmosphere
            and self.scintillation_availability is not None
            and self.geometry.atm_path_length_m > 0
        ):
            propagated_beam = base_results.states[1].beam  # after free-space step
            effective_cn2 = self._effective_cn2_for_scintillation()
            try:
                scint_stats = atm.hufnagel_valley.scintillation(
                    propagated_beam,
                    distance=self.geometry.atm_path_length_m,
                    Cn2=effective_cn2,
                    availability=self.scintillation_availability,
                )
            except Exception:
                scint_stats = None

        return SpaceLinkResults(base_results, self.geometry, scint_stats)


class SpaceLinkResults:
    """Results from a SpaceLink run, with space-specific reporting."""

    def __init__(
        self, base_results: LinkResults, geometry: LinkGeometry, scint_stats: dict
    ):
        self.base = base_results
        self.geometry = geometry
        self.scint_stats = scint_stats

    @property
    def final_beam(self) -> Beam:
        return self.base.final_beam

    @property
    def states(self):
        return self.base.states

    def summary(self):
        print("=" * 75)
        print("  SPACE FSO LINK BUDGET")
        print("=" * 75)
        self.geometry.summary()
        print("-" * 75)
        self.base.summary()

        if self.scint_stats:
            print()
            print("  Scintillation (slant path, effective Cn2):")
            print(
                f"    Rytov variance       : {self.scint_stats['rytov_variance']:.4f}"
            )
            print(
                f"    Scintillation index  : {self.scint_stats['scintillation_index']:.4f}"
            )
            print(f"    Regime               : {self.scint_stats['regime']}")
            print(
                f"    Fade margin ({self.scint_stats['availability']*100:.1f}%) : "
                f"{self.scint_stats['fade_margin_db']:.2f} dB"
            )

        if not self.geometry.has_atmosphere:
            print()
            print("  Note: sat-to-sat link — atmospheric effects skipped.")
        print("=" * 75)

    def plot(self, **kwargs):
        return self.base.plot(**kwargs)
