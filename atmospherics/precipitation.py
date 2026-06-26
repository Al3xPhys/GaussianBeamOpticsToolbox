import numpy as np
from beam_class import Beam


def rain_attenuation(beam, distance, rain_rate):
    """
    Applies optical rain attenuation using empirical coefficients for near-IR wavelengths.
    Uses the Al Naboulsi model for optical wavelengths rather than microwave coefficients.

    Parameters:
        beam (Beam): Input beam.
        distance (float): Propagation distance in metres.
        rain_rate (float): Rain rate in mm/h.

    Returns:
        Beam: New beam with reduced power.

    Notes:
        Specific attenuation (dB/km) ≈ 1.076 * rain_rate^0.67 for optical wavelengths,
        derived from Carbonneau (1998) for FSO links.
    """
    if distance < 0:
        raise ValueError("Distance must be non-negative.")
    if rain_rate < 0:
        raise ValueError("Rain rate must be non-negative.")
    if rain_rate == 0:
        return beam

    # Carbonneau (1998) optical specific attenuation in dB/km
    attenuation_db_per_km = 1.076 * rain_rate**0.67
    attenuation_db = attenuation_db_per_km * (distance / 1000)
    new_power = beam.power * 10 ** (-attenuation_db / 10)
    return Beam(q=beam.q, wavelength=beam.wavelength, power=new_power, M2=beam.M2)


def snow_attenuation(beam, distance, snowfall_rate, wet=False):
    """
    Applies snow attenuation using the Kim snow model.

    Parameters:
        beam (Beam): Input beam.
        distance (float): Propagation distance in metres.
        snowfall_rate (float): Snowfall rate in mm/h (liquid water equivalent).
        wet (bool): If True, uses wet snow coefficients; otherwise dry snow.

    Returns:
        Beam: New beam with reduced power.

    Notes:
        Uses Kim et al. (2001) model:
          - Dry snow: attenuation (dB/km) = a * snowfall_rate^b, a=5.42e-5*lambda+5.4958824, b=1.0
            simplified to a=7.0 dB/km per mm/h for 1550nm
          - Wet snow: roughly 3x dry snow attenuation
        Ref: Kim et al., "Wireless optical transmission of fast Ethernet, FDDI, ATM,
             and ESCON protocol data using the TerraLink laser communication system", 1998.
    """
    if distance < 0:
        raise ValueError("Distance must be non-negative.")
    if snowfall_rate < 0:
        raise ValueError("Snowfall rate must be non-negative.")
    if snowfall_rate == 0:
        return beam

    wavelength_nm = beam.wavelength * 1e9
    if wet:
        # Wet snow - larger, wetter flakes, higher attenuation
        a = 0.000102 * wavelength_nm + 3.7855072
        b = 0.72
    else:
        # Dry snow
        a = 0.0000542 * wavelength_nm + 5.4958824
        b = 1.38

    attenuation_db_per_km = a * snowfall_rate**b
    attenuation_db = attenuation_db_per_km * (distance / 1000)
    new_power = beam.power * 10 ** (-attenuation_db / 10)
    return Beam(q=beam.q, wavelength=beam.wavelength, power=new_power, M2=beam.M2)
