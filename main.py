"""
main.py
=======
Example FSO link budget using the refactored simulation pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt

from beam_class import Beam
import transfer_matrices as tm
import atmospherics as atm
import components as comp
from link_budget import LinkBudget
from atmospherics.scintillation import scintillation

# ── Link parameters ──────────────────────────────────────────────────────────
WAVELENGTH = 1550e-9  # m
BEAM_WAIST = 30e-2  # m
TX_POWER = 1.0  # W
DISTANCE = 1000.0  # m
Cn2 = 1e-15  # m^(-2/3), moderate turbulence
VISIBILITY = 1000.0  # m, light fog
RECEIVER_RADIUS = 0.15  # m
M2 = 1.1  # beam quality factor
# ─────────────────────────────────────────────────────────────────────────────


def build_link(distance, visibility, Cn2):
    """Construct and run a single FSO link budget."""
    beam = Beam.from_beam_waist(
        waist_radius=BEAM_WAIST, wavelength=WAVELENGTH, power=TX_POWER, M2=M2
    )

    link = LinkBudget(beam)

    link.add_optic(
        "Beam expander (f1=30mm, f2=60mm)",
        lambda b: comp.beam_expander(b, f1=0.03, f2=0.06),
    )

    # link.add_optic(
    #     "Aperture stop (r=0.05m)", lambda b: comp.aperture_stop(b, aperture_radius=0.05)
    # )

    # Free-space propagation
    link.add_propagation(
        f"Free space ({distance:.0f} m)", lambda b: b.propagate(tm.free_space(distance))
    )

    # Atmospheric effects
    link.add_atmosphere(
        f"Kim fog (V={visibility:.0f} m)",
        lambda b: atm.kim_fog(b, distance=distance, visibility=visibility),
    )

    link.add_atmosphere(
        f"Turbulence (Cn2={Cn2:.0e})",
        lambda b: atm.turbulence(b, distance=distance, Cn2=Cn2),
    )

    return link.run()


# ── Single link example ───────────────────────────────────────────────────────
results = build_link(DISTANCE, VISIBILITY, Cn2)
results.summary()

# Receiver power and detector performance
rx_power_w = comp.receiver(results.final_beam, RECEIVER_RADIUS)
detector = comp.Detector(responsivity=0.8, bandwidth=1e9)
print(f"\nReceiver aperture radius : {RECEIVER_RADIUS} m")
print(
    f"Received optical power   : {rx_power_w*1e6:.3f} µW  ({10*np.log10(rx_power_w/1e-3):.2f} dBm)"
)
print(f"SNR                      : {detector.snr_db(rx_power_w):.1f} dB")
print(f"BER                      : {detector.ber(rx_power_w):.2e}")
print(
    f"Receiver sensitivity     : {detector.sensitivity(1e-9)*1e6:.3f} µW  ({10*np.log10(detector.sensitivity(1e-9)/1e-3):.2f} dBm)"
)

results.plot()
plt.show()


# ── Distance sweep across fog conditions ─────────────────────────────────────
visibility_conditions = {
    "Clear (10km)": 10000,
    "Light fog (1km)": 1000,
    "Moderate fog (500m)": 500,
    "Dense fog (200m)": 200,
}
distances = np.linspace(1, 500000, 100)

fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(11, 8))

for label, vis in visibility_conditions.items():
    rx_powers_dbm = []
    rx_bers = []

    for d in distances:
        # slowly decrease cn2 with distance to simulate decreasing turbulence as it passes through the atmosphere using Hufnagel-Valley model
        Cn2 = (
            5.94e-53 * (21 / 27) ** 2 * d**10 * np.exp(-d / 1000)
            + 2.7e-16 * np.exp(-d / 1500)
            + 1.7e-14 * np.exp(-d / 100)
        )
        res = build_link(d, vis, Cn2)
        rx_power = comp.receiver(res.final_beam, RECEIVER_RADIUS)
        rx_power = max(rx_power, 1e-15)

        scint = scintillation(res.final_beam, distance=d, Cn2=Cn2, availability=0.999)
        rx_power_dbm = 10 * np.log10(rx_power / 1e-3) - scint["fade_margin_db"]

        rx_powers_dbm.append(rx_power_dbm)
        rx_bers.append(detector.ber(rx_power))

    ax1.plot(
        distances, rx_powers_dbm, label=label, linewidth=1.5, marker="o", markersize=2
    )
    ax2.semilogy(
        distances, rx_bers, label=label, linewidth=1.5, marker="o", markersize=2
    )

ax1.axhline(y=-40, color="r", linestyle="--", label="Sensitivity threshold (−40 dBm)")
ax1.set_title("Received power vs distance (with scintillation fade margin)")
ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("Effective received power (dBm)")
ax1.grid(True, alpha=0.4)
ax1.legend()

ax2.axhline(y=1e-9, color="r", linestyle="--", label="BER = 1e-9")
ax2.set_title("BER vs distance")
ax2.set_xlabel("Distance (m)")
ax2.set_ylabel("BER")
ax2.grid(True, alpha=0.4)
ax2.legend()

plt.tight_layout()
plt.show()
