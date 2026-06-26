"""
link_budget.py
==============
Central pipeline for FSO link simulation. Chains optical and atmospheric
segments together, logging beam state at each step.

Example
-------
    from link_budget import LinkBudget
    from beam_class import Beam
    import transfer_matrices as tm
    import atmospherics as atm
    import components as comp

    beam = Beam.from_beam_waist(waist_radius=5e-6, wavelength=1550e-9, power=0.15)

    link = LinkBudget(beam)
    link.add_optic("Collimating lens",    lambda b: b.propagate(tm.free_space(0.01))
                                                     .propagate(tm.thin_lens(0.01))
                                                     .propagate_to_waist())
    link.add_optic("Beam expander 2x",   lambda b: comp.beam_expander(b, f1=0.03, f2=0.06))
    link.add_propagation("Free space 1km", lambda b: b.propagate(tm.free_space(1000)))
    link.add_atmosphere("Kim fog",        lambda b: atm.kim_fog(b, distance=1000, visibility=1000))
    link.add_atmosphere("Turbulence",     lambda b: atm.turbulence(b, distance=1000, Cn2=1e-15))

    results = link.run()
    results.summary()
    results.plot()

    rx_power = comp.receiver(results.final_beam, receiver_radius=0.15)
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from beam_class import Beam

# ---------------------------------------------------------------------------
# BeamState — snapshot of beam properties at one step
# ---------------------------------------------------------------------------


@dataclass
class BeamState:
    label: str
    step_index: int
    waist_m: float
    power_w: float
    M2: float
    beam: Beam  # full Beam object for downstream use

    @property
    def power_dbm(self):
        if self.power_w <= 0:
            return -np.inf
        return 10 * np.log10(self.power_w / 1e-3)

    @property
    def waist_mm(self):
        return self.waist_m * 1e3

    def __str__(self):
        return (
            f"[{self.step_index:02d}] {self.label:<35s} "
            f"waist={self.waist_m*1e3:8.4f} mm  "
            f"power={self.power_dbm:8.2f} dBm  "
            f"M²={self.M2:.3f}"
        )


# ---------------------------------------------------------------------------
# LinkResults — returned by LinkBudget.run()
# ---------------------------------------------------------------------------


class LinkResults:
    def __init__(self, states: List[BeamState]):
        self.states = states

    @property
    def final_beam(self) -> Beam:
        return self.states[-1].beam

    @property
    def labels(self):
        return [s.label for s in self.states]

    @property
    def powers_dbm(self):
        return np.array([s.power_dbm for s in self.states])

    @property
    def waists_m(self):
        return np.array([s.waist_m for s in self.states])

    @property
    def M2_values(self):
        return np.array([s.M2 for s in self.states])

    def summary(self):
        """Prints a table of beam state at each step."""
        print("=" * 75)
        print("  FSO LINK BUDGET SUMMARY")
        print("=" * 75)
        for state in self.states:
            print(state)
        print("=" * 75)
        print(f"  Total power loss: {self.powers_dbm[0] - self.powers_dbm[-1]:.2f} dB")
        print("=" * 75)

    def plot(self, figsize=(12, 8)):
        """
        Plots beam waist, power (dBm), and M² as a function of link step.
        Returns the matplotlib Figure object.
        """
        indices = [s.step_index for s in self.states]
        labels = self.labels

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Beam waist
        axes[0].plot(
            indices, self.waists_m * 1e3, marker="o", linewidth=1.5, markersize=4
        )
        axes[0].set_ylabel("Beam waist (mm)")
        axes[0].set_title("Beam evolution through link")
        axes[0].grid(True, alpha=0.4)

        # Power
        axes[1].plot(
            indices,
            self.powers_dbm,
            marker="o",
            linewidth=1.5,
            markersize=4,
            color="tab:orange",
        )
        axes[1].set_ylabel("Power (dBm)")
        axes[1].grid(True, alpha=0.4)

        # M²
        axes[2].plot(
            indices,
            self.M2_values,
            marker="o",
            linewidth=1.5,
            markersize=4,
            color="tab:green",
        )
        axes[2].set_ylabel("M²")
        axes[2].set_xlabel("Link step")
        axes[2].grid(True, alpha=0.4)

        # X-axis tick labels
        axes[2].set_xticks(indices)
        axes[2].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# LinkBudget — main pipeline class
# ---------------------------------------------------------------------------


class LinkBudget:
    """
    Chains optical and atmospheric segments, logging beam state at each step.

    Each segment is a callable (Beam -> Beam). Add segments with:
        add_optic()       — optical components (lenses, apertures, mirrors)
        add_propagation() — free-space propagation
        add_atmosphere()  — atmospheric effects (fog, turbulence, rain, etc.)
        add_segment()     — generic fallback

    All three are aliases for the same underlying mechanism; the distinction
    is purely documentary.
    """

    def __init__(self, initial_beam: Beam):
        self._initial_beam = initial_beam
        self._segments: List[tuple[str, Callable]] = []

    def add_segment(self, label: str, fn: Callable[[Beam], Beam]):
        """Add a generic segment. fn must accept a Beam and return a Beam."""
        self._segments.append((label, fn))
        return self  # allow chaining

    def add_optic(self, label: str, fn: Callable[[Beam], Beam]):
        """Add an optical component (lens, aperture, expander, etc.)."""
        return self.add_segment(label, fn)

    def add_propagation(self, label: str, fn: Callable[[Beam], Beam]):
        """Add a free-space propagation step."""
        return self.add_segment(label, fn)

    def add_atmosphere(self, label: str, fn: Callable[[Beam], Beam]):
        """Add an atmospheric effect (fog, turbulence, rain, scintillation, etc.)."""
        return self.add_segment(label, fn)

    def run(self) -> LinkResults:
        """
        Runs the link budget simulation.

        Returns:
            LinkResults: Object containing beam state at every step.
        """
        beam = self._initial_beam
        states = [
            BeamState(
                label="Source",
                step_index=0,
                waist_m=beam.get_waist(),
                power_w=beam.power,
                M2=beam.M2,
                beam=beam,
            )
        ]

        for i, (label, fn) in enumerate(self._segments, start=1):
            beam = fn(beam)
            states.append(
                BeamState(
                    label=label,
                    step_index=i,
                    waist_m=beam.get_waist(),
                    power_w=beam.power,
                    M2=beam.M2,
                    beam=beam,
                )
            )

        return LinkResults(states)

    def __len__(self):
        return len(self._segments)

    def __repr__(self):
        return f"LinkBudget({len(self._segments)} segments)"
