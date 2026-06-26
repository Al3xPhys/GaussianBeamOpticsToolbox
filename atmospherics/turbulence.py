import numpy as np
from beam_class import Beam


def turbulence(beam, distance, Cn2):
    """
    Applies turbulence-induced beam spreading using the Fried parameter r0.

    The effective beam waist after turbulence is:
        w_turb = w(z) * sqrt(1 + (2*w(z)/r0)^(5/3))

    where r0 = 0.185 * (lambda^2 / (Cn2 * distance))^(3/5) is the Fried coherence length.

    Parameters:
        beam (Beam): Input beam.
        distance (float): Propagation distance in metres.
        Cn2 (float): Refractive index structure parameter in m^(-2/3).
            Typical values:
                ~1e-17  : very weak turbulence
                ~1e-15  : moderate turbulence
                ~1e-13  : strong turbulence

    Returns:
        Beam: New beam with enlarged waist due to turbulence.
    """
    if distance <= 1.0:
        return beam  # Negligible turbulence effect at very short distances
    if Cn2 < 0:
        raise ValueError("Refractive index structure parameter must be non-negative.")
    if Cn2 == 0:
        return beam

    r0 = 0.185 * (beam.wavelength**2 / (Cn2 * distance)) ** (3 / 5)
    g = np.sqrt(1 + (2 * beam.get_waist() / r0) ** (5 / 3))
    w_turb = beam.get_waist() * g
    return Beam.from_beam_waist(
        waist_radius=w_turb, wavelength=beam.wavelength, power=beam.power, M2=beam.M2
    )
