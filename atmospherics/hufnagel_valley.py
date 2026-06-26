"""
atmospherics/hufnagel_valley.py
================================
Hufnagel-Valley turbulence model for ground-to-space and space-to-ground links.

The HV model gives Cn2 as a function of altitude h (metres):

    Cn2(h) = 0.00594 * (v/27)^2 * (1e-5*h)^10 * exp(-h/1000)
           + 2.7e-16 * exp(-h/1500)
           + C0 * exp(-h/100)

Slant-path beam spreading uses the extended Huygens-Fresnel (EHF) model,
which gives the long-term beam radius at the receiver as:

    W_LT^2 = W(L)^2 + 4*L^2 / (k^2 * rho_0^2)

where W(L) is the diffraction-limited spot radius at distance L, k is the
wavenumber, and rho_0 is the spherical-wave coherence radius:

    rho_0 = ( 0.546 * k^2 * sec(zeta)
              * integral( Cn2(h) * (1 - h/H)^(5/3) dh, 0, H ) )^(-3/5)

The (1 - h/H)^(5/3) weighting factor is what makes the slant path physically
correct — it down-weights high-altitude turbulence and means the dense
boundary-layer turbulence near the ground dominates, as observed.

References
----------
Andrews & Phillips, "Laser Beam Propagation through Random Media", 2nd ed.
    SPIE Press, 2005. Chapter 12 (uplink/downlink).
"""

import numpy as np
from scipy.integrate import quad
from beam_class import Beam

# ── HV 5/7 standard parameters ───────────────────────────────────────────────
HV57_WIND_SPEED = 21.0  # m/s
HV57_GROUND_Cn2 = 1.7e-14  # m^(-2/3)


def hv_cn2(h, wind_speed=HV57_WIND_SPEED, ground_cn2=HV57_GROUND_Cn2):
    """
    Hufnagel-Valley Cn2 profile as a function of altitude.

    Parameters
    ----------
    h : float or array
        Altitude above sea level in metres.
    wind_speed : float
        RMS wind speed in m/s. Default 21 m/s (HV 5/7).
    ground_cn2 : float
        Ground-level Cn2 in m^(-2/3). Default 1.7e-14 (HV 5/7).

    Returns
    -------
    float or array : Cn2(h) in m^(-2/3).
    """
    h = np.asarray(h, dtype=float)
    term1 = 0.00594 * (wind_speed / 27) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000)
    term2 = 2.7e-16 * np.exp(-h / 1500)
    term3 = ground_cn2 * np.exp(-h / 100)
    return term1 + term2 + term3


def integrated_cn2(
    h_bottom, h_top, wind_speed=HV57_WIND_SPEED, ground_cn2=HV57_GROUND_Cn2
):
    """
    Numerically integrates Cn2(h) from h_bottom to h_top (unweighted).
    Used for the effective Cn2 passed to the scintillation model.

    Returns
    -------
    float : integral of Cn2(h) dh in m^(1/3).
    """
    result, _ = quad(hv_cn2, h_bottom, h_top, args=(wind_speed, ground_cn2), limit=200)
    return result


def _spherical_wave_coherence_radius(
    wavelength,
    zenith_angle_rad,
    h_bottom,
    h_top,
    wind_speed=HV57_WIND_SPEED,
    ground_cn2=HV57_GROUND_Cn2,
):
    """
    Spherical-wave coherence radius rho_0 for a slant path.

    The EHF model uses a (1 - h/H)^(5/3) path weighting that accounts for
    the changing Fresnel zone size along the slant path. For an uplink
    (transmitter at h_bottom, receiver at h_top = H):

        rho_0 = ( 0.546 * k^2 * sec(zeta)
                  * integral( Cn2(h) * (1 - (h-h_bottom)/delta_h)^(5/3) dh ) )^(-3/5)

    Parameters
    ----------
    wavelength : float
        Optical wavelength in metres.
    zenith_angle_rad : float
        Zenith angle in radians at the lower station.
    h_bottom : float
        Altitude at base of atmospheric path in metres.
    h_top : float
        Altitude at top of atmospheric path in metres.
    wind_speed : float
        HV model wind speed in m/s.
    ground_cn2 : float
        HV model ground-level Cn2 in m^(-2/3).

    Returns
    -------
    float : rho_0 in metres.
    """
    k = 2 * np.pi / wavelength
    sec_zeta = 1.0 / np.cos(zenith_angle_rad)
    delta_h = h_top - h_bottom

    def integrand(h):
        xi = (h - h_bottom) / delta_h  # normalised path coordinate, 0 at Tx, 1 at Rx
        weight = (1.0 - xi) ** (5 / 3)  # EHF uplink weighting
        return hv_cn2(h, wind_speed, ground_cn2) * weight

    integral, _ = quad(integrand, h_bottom, h_top, limit=200)
    rho_0 = (0.546 * k**2 * sec_zeta * integral) ** (-3 / 5)
    return rho_0


def fried_parameter(
    wavelength,
    zenith_angle_rad,
    h_bottom,
    h_top,
    wind_speed=HV57_WIND_SPEED,
    ground_cn2=HV57_GROUND_Cn2,
):
    """
    Fried coherence length r0 for a slant path (plane-wave approximation).

    r0 = ( 0.423 * k^2 * sec(zeta) * integral(Cn2(h) dh) )^(-3/5)

    This is the standard r0 used for imaging/AO calculations. For beam
    spreading, use slant_turbulence() which uses the EHF rho_0 instead.

    Returns
    -------
    float : r0 in metres.
    """
    k = 2 * np.pi / wavelength
    sec_zeta = 1.0 / np.cos(zenith_angle_rad)
    cn2_integral = integrated_cn2(h_bottom, h_top, wind_speed, ground_cn2)
    return (0.423 * k**2 * sec_zeta * cn2_integral) ** (-3 / 5)


def slant_turbulence(
    beam, geometry, wind_speed=HV57_WIND_SPEED, ground_cn2=HV57_GROUND_Cn2
):
    """
    Applies turbulence-induced beam spreading along a slant atmospheric path
    using the extended Huygens-Fresnel (EHF) long-term beam radius model.

    The long-term beam radius at the receiver is:

        W_LT^2 = W(L)^2 + 4*L^2 / (k^2 * rho_0^2)

    where:
        W(L)  = diffraction-limited beam radius at distance L (from beam.get_waist())
        L     = total link distance
        k     = 2*pi/lambda
        rho_0 = spherical-wave coherence radius (EHF weighted integral)

    The turbulence term 4L^2/(k^2*rho_0^2) is the additional spreading due
    to atmospheric phase distortion. When rho_0 >> beam size the correction
    is negligible; when rho_0 << beam size the turbulence dominates.

    Parameters
    ----------
    beam : Beam
        Beam at the receiver plane (after free-space propagation to distance L).
    geometry : LinkGeometry
        Link geometry object.
    wind_speed : float
        HV model rms wind speed in m/s. Default 21 m/s (HV 5/7).
    ground_cn2 : float
        HV model ground-level Cn2 in m^(-2/3). Default 1.7e-14 (HV 5/7).

    Returns
    -------
    Beam : Beam with enlarged waist due to turbulence, same power and wavelength.
    """
    if not geometry.has_atmosphere:
        return beam

    rho_0 = _spherical_wave_coherence_radius(
        wavelength=beam.wavelength,
        zenith_angle_rad=geometry.zenith_angle_rad,
        h_bottom=geometry.atm_bottom_m,
        h_top=geometry.atm_top_m,
        wind_speed=wind_speed,
        ground_cn2=ground_cn2,
    )

    k = 2 * np.pi / beam.wavelength
    L = geometry.link_distance_m
    W_L = beam.get_waist()  # diffraction-limited spot at receiver

    W_LT = np.sqrt(W_L**2 + 4 * L**2 / (k**2 * rho_0**2))

    return Beam.from_beam_waist(
        waist_radius=W_LT,
        wavelength=beam.wavelength,
        power=beam.power,
        M2=beam.M2,
    )
