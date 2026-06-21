import numpy as np
from beam_class import Beam

def aperture_stop(beam, aperture_radius):
    """ This function calculates loss to power falling onto a finite radius aperture mid-path, beam continues onwards"""
    P_total = beam.power
    P_capture = P_total * (1 - np.exp(-2 * (aperture_radius / beam.get_waist())**2))
    truncation_ratio = aperture_radius / beam.get_waist()
    new_M2 = beam.M2 * (1 + 0.5 * (np.exp(-2 * truncation_ratio**2)))
    return Beam(q=beam.q, wavelength=beam.wavelength, power=P_capture, M2=new_M2)

def receiver(beam, receiver_radius):
    """ This function calculates captured power in Watts, at a receiver at the end of a path. """
    P_total = beam.power
    P_capture = P_total * (1 - np.exp(-2 * (receiver_radius / beam.get_waist())**2))
    return P_capture
