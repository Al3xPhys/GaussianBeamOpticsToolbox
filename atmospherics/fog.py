import numpy as np
from beam_class import Beam


def beer_lambert(beam, distance, alpha):
    """
    Applies the Beer-Lambert law to the beam over a given distance.

    Parameters:
        beam (Beam): Input beam.
        distance (float): Propagation distance in metres.
        alpha (float): Absorption coefficient in 1/m.

    Returns:
        Beam: New beam with reduced power.
    """
    if distance < 0:
        raise ValueError("Distance must be non-negative.")
    if alpha < 0:
        raise ValueError("Absorption coefficient must be non-negative.")
    new_power = beam.power * np.exp(-alpha * distance)
    return Beam(q=beam.q, wavelength=beam.wavelength, power=new_power, M2=beam.M2)


def kim_fog(beam, distance, visibility, sizing_parameter_q=None):
    """
    Applies the Kim model for fog attenuation to the beam.

    Parameters:
        beam (Beam): Input beam.
        distance (float): Propagation distance in metres.
        visibility (float): Meteorological visibility in metres.
        sizing_parameter_q (float, optional): Kim model q parameter. Auto-selected from
            visibility if not provided.

    Returns:
        Beam: New beam with reduced power.
    """
    if distance < 0:
        raise ValueError("Distance must be non-negative.")
    if visibility <= 0:
        raise ValueError("Visibility must be positive.")

    if sizing_parameter_q is None:
        if visibility <= 500:
            sizing_parameter_q = 0
        elif visibility <= 1000:
            sizing_parameter_q = 0.5
        elif visibility <= 3000:
            sizing_parameter_q = 1.6
        elif visibility <= 6000:
            sizing_parameter_q = 0.9
        else:
            sizing_parameter_q = 0.5

    alpha = (3.912 / visibility) * (beam.wavelength / 550e-9) ** (-sizing_parameter_q)
    new_power = beam.power * np.exp(-alpha * distance)
    return Beam(q=beam.q, wavelength=beam.wavelength, power=new_power, M2=beam.M2)


def kruse_haze(beam, distance, visibility):
    """
    Applies the Kruse model for haze/aerosol attenuation.

    Parameters:
        beam (Beam): Input beam.
        distance (float): Propagation distance in metres.
        visibility (float): Meteorological visibility in metres.

    Returns:
        Beam: New beam with reduced power.
    """
    if distance < 0:
        raise ValueError("Distance must be non-negative.")
    if visibility <= 0:
        raise ValueError("Visibility must be positive.")

    wavelength_um = beam.wavelength * 1e6  # convert to microns for Kruse formula
    if visibility > 6000:
        q = 1.6
    elif visibility > 1000:
        q = 1.3
    else:
        q = 0.585 * visibility ** (1 / 3)

    alpha = (3.912 / visibility) * (wavelength_um / 0.55) ** (-q)
    new_power = beam.power * np.exp(-alpha * distance)
    return Beam(q=beam.q, wavelength=beam.wavelength, power=new_power, M2=beam.M2)
