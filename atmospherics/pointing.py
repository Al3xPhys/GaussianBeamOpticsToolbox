"""
atmospherics/pointing.py
========================
Pointing loss models for FSO links.

Two models:

1. Static offset (deterministic)
   A fixed angular misalignment. Power fraction at receiver:
       eta = exp(-2 * (r_offset / w_rx)^2)
   where r_offset = theta_offset * distance is the lateral displacement
   at the receiver, and w_rx is the beam radius at the receiver plane.

2. RMS jitter (statistical)
   Gaussian random jitter with rms sigma_j (radians). Mean power fraction:
       <eta> = 1 / (1 + 2*(sigma_j * distance / w_rx)^2)
   This is the closed-form mean of the Gaussian-beam / Gaussian-jitter integral.

Both take an explicit link distance so the angular error is correctly mapped
to a lateral displacement at the receiver plane.
"""

import numpy as np
from beam_class import Beam


def pointing_loss_static(beam, offset_urad, link_distance_m):
    """
    Deterministic pointing loss from a fixed angular offset.

    Parameters
    ----------
    beam : Beam
        Beam at the *receiver plane* (i.e. after free-space propagation),
        so beam.get_waist() gives the spot size at the receiver.
    offset_urad : float
        Angular pointing offset in microradians.
    link_distance_m : float
        Total link distance in metres (used to convert angle to lateral offset).

    Returns
    -------
    Beam : Beam with reduced power.
    """
    if offset_urad == 0.0:
        return beam

    theta = offset_urad * 1e-6  # rad
    r_off = theta * link_distance_m  # lateral offset at Rx (m)
    w_rx = beam.get_waist()  # beam radius at Rx plane
    eta = np.exp(-2 * (r_off / w_rx) ** 2)
    return Beam(
        q=beam.q, wavelength=beam.wavelength, power=beam.power * eta, M2=beam.M2
    )


def pointing_loss_jitter(beam, jitter_urad, link_distance_m):
    """
    Statistical pointing loss from Gaussian rms angular jitter.

    Parameters
    ----------
    beam : Beam
        Beam at the *receiver plane*.
    jitter_urad : float
        RMS pointing jitter in microradians.
    link_distance_m : float
        Total link distance in metres.

    Returns
    -------
    Beam : Beam with mean power reduced by jitter loss.
    """
    if jitter_urad == 0.0:
        return beam

    sigma_j = jitter_urad * 1e-6  # rad
    sigma_r = sigma_j * link_distance_m  # rms lateral displacement at Rx (m)
    w_rx = beam.get_waist()
    eta = 1.0 / (1.0 + 2 * (sigma_r / w_rx) ** 2)
    return Beam(
        q=beam.q, wavelength=beam.wavelength, power=beam.power * eta, M2=beam.M2
    )


def pointing_loss_db_static(beam, offset_urad, link_distance_m):
    """Returns static pointing loss in dB (positive = loss)."""
    b_out = pointing_loss_static(beam, offset_urad, link_distance_m)
    if b_out.power <= 0:
        return np.inf
    return -10 * np.log10(b_out.power / beam.power)


def pointing_loss_db_jitter(beam, jitter_urad, link_distance_m):
    """Returns mean jitter pointing loss in dB (positive = loss)."""
    b_out = pointing_loss_jitter(beam, jitter_urad, link_distance_m)
    if b_out.power <= 0:
        return np.inf
    return -10 * np.log10(b_out.power / beam.power)
