import numpy as np
from beam_class import Beam
import transfer_matrices


def aperture_stop(beam, aperture_radius):
    """
    Hard aperture mid-path. Clips power to what passes through the aperture,
    and increases M2 slightly due to truncation.

    Parameters:
        beam (Beam): Input beam.
        aperture_radius (float): Aperture radius in metres.

    Returns:
        Beam: Beam with clipped power and increased M2.
    """
    P_capture = beam.power * (
        1 - np.exp(-2 * (aperture_radius / beam.get_waist()) ** 2)
    )
    truncation_ratio = aperture_radius / beam.get_waist()
    new_M2 = beam.M2 * (1 + 0.5 * np.exp(-2 * truncation_ratio**2))
    return Beam(q=beam.q, wavelength=beam.wavelength, power=P_capture, M2=new_M2)


def collimating_lens(beam, focal_length):
    """
    Propagates the beam to the lens (placed one focal length from current waist),
    applies the thin lens, then propagates to the new waist.

    Parameters:
        beam (Beam): Input beam, assumed to be at its waist.
        focal_length (float): Focal length of the collimating lens in metres.

    Returns:
        Beam: Collimated beam at its new waist.
    """
    beam = beam.propagate(transfer_matrices.free_space(focal_length))
    beam = beam.propagate(transfer_matrices.thin_lens(focal_length))
    return beam.propagate_to_waist()


def beam_expander(beam, f1, f2):
    """
    Passes the beam through an afocal beam expander (two thin lenses, d = f1 + f2).
    Magnification M = -f2/f1; beam waist scales by |f2/f1|.

    Parameters:
        beam (Beam): Input beam, assumed to be at its waist.
        f1 (float): Focal length of the first (input) lens in metres.
        f2 (float): Focal length of the second (output) lens in metres.

    Returns:
        Beam: Expanded beam at its new waist.
    """
    beam = beam.propagate(transfer_matrices.beam_expander(f1, f2))
    return beam.propagate_to_waist()
