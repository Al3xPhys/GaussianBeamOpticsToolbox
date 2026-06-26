import numpy as np
import matplotlib.pyplot as plt
from beam_class import Beam
import transfer_matrices as tm
import atmospherics as atm
import components as comp
from geometry import LinkGeometry, Station
from space_link import SpaceLink

WAVELENGTH = 1550e-9  # m
BEAM_WAIST = 5e-6  # m
TX_POWER = 1.0  # W
M2 = 1.1  # beam quality factor


def make_tx_beam(
    waist_radius=1e-3,
    wavelength=1550e-9,
    power=1.0,
    M2=1.0,
    f1=0.03,
    f2=0.06,
    f_collimating=0.01,
):
    """Create a Gaussian beam at the transmitter."""
    beam = Beam.from_beam_waist(
        waist_radius=waist_radius, wavelength=wavelength, power=power, M2=M2
    )
    beam = comp.collimating_lens(
        beam, focal_length=f_collimating
    )  # 10 mm collimating lens
    beam = comp.beam_expander(beam, f1=f1, f2=f2)  # 2x beam expander
    return beam


# ground to LEO scenario

# ground station
ground_station = Station(
    "Ground",
    altitude_m=2400,
    aperture_radius_m=0.15,
    pointing_jitter_urad=2.0,
    pointing_offset_urad=0.5,
)
# low earth orbit satellite
leo_satellite = Station(
    "LEO Satellite",
    altitude_m=500_000,
    aperture_radius_m=0.10,
)

# make the link geometry for a ground to LEO satellite link at 45 degrees elevation
geom_leo = LinkGeometry(tx=ground_station, rx=leo_satellite, elevation_angle_deg=45)

# make the transmitter beam with a collimating lens and a 4x beam expander
beam = make_tx_beam(
    waist_radius=BEAM_WAIST,
    wavelength=WAVELENGTH,
    power=TX_POWER,
    M2=M2,
    f1=0.05,
    f2=0.2,
    f_collimating=0.025,
)

# create the space link object
link_leo = SpaceLink(beam, geom_leo, scintillation_availability=0.999)
results_leo = link_leo.run()
results_leo.summary()


rx_power_leo = comp.receiver(results_leo.final_beam, leo_satellite.aperture_radius_m)
detector = comp.Detector(responsivity=0.8, bandwidth=1e9)
print(
    f"\n  Received power : {rx_power_leo*1e9:.3f} nW  "
    f"({10*np.log10(max(rx_power_leo,1e-30)/1e-3):.1f} dBm)"
)
print(f"  SNR            : {detector.snr_db(rx_power_leo):.1f} dB")
print(f"  BER            : {detector.ber(rx_power_leo):.2e}")
