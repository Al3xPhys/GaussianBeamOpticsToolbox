import numpy as np

def free_space(distance):
    """
    Returns the transfer matrix for free space propagation over a given distance.

    Parameters:
    distance (float): The distance of free space propagation.

    Returns:
    numpy.ndarray: The 2x2 transfer matrix for free space propagation.
    """
    return np.array([[1, distance], [0, 1]])

def thin_lens(focal_length):
    """
    Returns the transfer matrix for a thin lens with a given focal length.

    Parameters:
    focal_length (float): The focal length of the thin lens.

    Returns:
    numpy.ndarray: The 2x2 transfer matrix for a thin lens.
    """
    return np.array([[1, 0], [-1/focal_length, 1]])

def beam_expander(f1, f2, d=None):
    """
    Returns the transfer matrix for a beam expander consisting of two lenses with focal lengths f1 and f2 separated by distance d.

    Parameters:
    f1 (float): The focal length of the first lens.
    f2 (float): The focal length of the second lens.
    d (float): The distance between the two lenses (keep none to make beam expander afocal).

    Returns:
    numpy.ndarray: The 2x2 transfer matrix for the beam expander.
    """
    # Transfer matrix for the first lens
    if d is None:
        # If distance is not provided, calculate it to make the system afocal
        d = f1 + f2  # For an afocal system, the distance should be the sum of the focal lengths
    lens1 = thin_lens(f1)
    # Transfer matrix for free space propagation between the lenses
    free_space_between = free_space(d)
    # Transfer matrix for the second lens
    lens2 = thin_lens(f2)
    # Total transfer matrix is the product of the three matrices
    return lens2 @ free_space_between @ lens1
