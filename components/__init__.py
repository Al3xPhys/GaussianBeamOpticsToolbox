"""
components
==========
Optical components and receiver models for FSO link simulation.

Usage
-----
    from components import aperture_stop, collimating_lens, beam_expander, receiver, Detector
"""

from components.optics import aperture_stop, collimating_lens, beam_expander
from components.detector import receiver, Detector

__all__ = [
    "aperture_stop",
    "collimating_lens",
    "beam_expander",
    "receiver",
    "Detector",
]
