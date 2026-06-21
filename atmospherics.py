import numpy as np 
import warnings
from scipy.special import erfinv, gamma, kv
from scipy.integrate import cumulative_trapezoid, trapezoid
from beam_class import Beam

def beer_lambert(beam, distance, alpha):
    """
    Applies the Beer-Lambert law to the beam over a given distance with a specified absorption coefficient.

    Parameters:
    beam (Beam): The Gaussian beam instance.
    distance (float): The distance over which the beam propagates.
    alpha (float): The absorption coefficient (in 1/m).

    Returns:
    Beam: A new Beam instance with reduced power due to absorption.
    """
    if not isinstance(beam, Beam):
        raise ValueError("Input must be an instance of the Beam class.")
    if distance < 0:
        raise ValueError("Distance must be non-negative.")
    if alpha < 0:
        raise ValueError("Absorption coefficient must be non-negative.")
    # Calculate the new power after absorption
    new_power = beam.power * np.exp(-alpha * distance)
    # Create a new Beam instance with the same q parameter and wavelength, but reduced power
    return Beam(q=beam.q, wavelength=beam.wavelength, power=new_power, M2=beam.M2)

def kim_fog(beam, distance, visibility, sizing_parameter_q=None):
    """
    Applies the Kim model for fog attenuation to the beam over a given distance with a specified visibility.

    Parameters:
    beam (Beam): The Gaussian beam instance.
    distance (float): The distance over which the beam propagates.
    visibility (float): The visibility in meters.

    Returns:
    Beam: A new Beam instance with reduced power due to fog attenuation.
    """
    if not isinstance(beam, Beam):
        raise ValueError("Input must be an instance of the Beam class.")
    if distance < 0:
        raise ValueError("Distance must be non-negative.")
    if visibility <= 0:
        raise ValueError("Visibility must be positive.")
    
    if sizing_parameter_q is None:
        if visibility <= 500:
            sizing_parameter_q = 0
        elif 500 < visibility <= 1000:
            sizing_parameter_q = 0.5
        elif 1000 < visibility <= 3000:
            sizing_parameter_q = 1.6
        elif 3000 < visibility <= 6000:
            sizing_parameter_q = 0.9
        else:
            sizing_parameter_q = 0.5

    # Calculate the attenuation coefficient using the Kim model
    alpha = (3.912 / visibility) * (beam.wavelength / 550e-9) ** (-sizing_parameter_q)
    # Calculate the new power after fog attenuation
    new_power = beam.power * np.exp(-alpha * distance)
    # Create a new Beam instance with the same q parameter and wavelength, but reduced power
    return Beam(q=beam.q, wavelength=beam.wavelength, power=new_power, M2=beam.M2)

def turbulence(beam, distance, Cn2):
    """
    Applies turbulence effects to the beam over a given distance with a specified refractive index structure parameter.

    Parameters:
    beam (Beam): The Gaussian beam instance.
    distance (float): The distance over which the beam propagates.
    Cn2 (float): The refractive index structure parameter (in m^(-2/3)).

    Returns:
    Beam: A new Beam instance with modified M2 due to turbulence effects.
    """
    if not isinstance(beam, Beam):
        raise ValueError("Input must be an instance of the Beam class.")
    if distance <= 1.0:
        return beam  # No turbulence effect for zero or negative distance
    if Cn2 < 0:
        raise ValueError("Refractive index structure parameter must be non-negative.")
    # calc r0 = 0.185 * (lambda^2 / (Cn2 * distance))^(3/5)
    r0 = 0.185 * (beam.wavelength**2 / (Cn2 * distance))**(3/5)
    # Calculate gamma = (2 * beam.get_waist() / r0)**(5/3)
    g = np.sqrt(1 + (2*beam.get_waist() / r0)**(5/3))
    w_turb =  beam.get_waist() * g
    # Create a new Beam instance with the same q parameter and wavelength, but modified M2
    return Beam.from_beam_waist(waist_radius=w_turb, wavelength=beam.wavelength, power=beam.power, M2=beam.M2)

def rain_attenuation(beam, distance, rain_rate, a=0.0001, b=0.8):
    """
    Applies rain attenuation to the beam over a given distance with a specified rain rate.

    Parameters:
    beam (Beam): The Gaussian beam instance.
    distance (float): The distance over which the beam propagates.
    rain_rate (float): The rain rate in mm/h.

    Returns:
    Beam: A new Beam instance with reduced power due to rain attenuation.
    """
    if not isinstance(beam, Beam):
        raise ValueError("Input must be an instance of the Beam class.")
    if distance < 0:
        raise ValueError("Distance must be non-negative.")
    if rain_rate < 0:
        raise ValueError("Rain rate must be non-negative.")
    
    # Calculate the specific attenuation using the ITU-R model
    k = a * (rain_rate ** b)  # Example Marshall-Palmer empirical formula for specific attenuation
    alpha = k * distance
    # Calculate the new power after rain attenuation
    new_power = beam.power * np.exp(-alpha)
    # Create a new Beam instance with the same q parameter and wavelength, but reduced power
    return Beam(q=beam.q, wavelength=beam.wavelength, power=new_power, M2=beam.M2)

def scintillation(beam, distance, Cn2, availability=None):
    """
    Applies scintillation effects to the beam over a given distance with a specified refractive index structure parameter.

    Parameters:
    beam (Beam): The Gaussian beam instance.
    distance (float): The distance over which the beam propagates.
    Cn2 (float): The refractive index structure parameter (in m^(-2/3)).

    Returns:
    dict: A dictionary containing scintillation statistics, 
            including Rytov variance, scintillation index, 
            and optionally fade threshold and fade margin if availability is specified.
    """
    if not isinstance(beam, Beam):
        raise ValueError("Input must be an instance of the Beam class.")
    if distance <= 0:
        raise ValueError("Distance must be positive.")
    if Cn2 < 0:
        raise ValueError("Refractive index structure parameter must be non-negative.")
    
    # Calculate the scintillation index using the Rytov approximation
    k = 2 * np.pi / beam.wavelength
    sigma_R2 = 1.23 * Cn2 * k**(7/6) * distance**(11/6)
    term1 = 0.49 * sigma_R2 / (1 + 1.11 * sigma_R2**(6/7))**(7/6)
    term2 = 0.51 * sigma_R2 / (1 + 0.69 * sigma_R2**(6/7))**(5/6)
    sigma_ln2 = term1 + term2
    if sigma_R2 < 1.0:
        # Weak turbulence regime, use log-normal distribution
        scintillation_index = np.exp(sigma_ln2) - 1
        mu_ln = -sigma_ln2 / 2  # Mean of the log-normal distribution to 1
        if availability is None:
            I_values = np.linspace(0.001, 5, 1000)
            pdf = (1 / (I_values * np.sqrt(sigma_ln2) * np.sqrt(2*np.pi))) * \
                    np.exp(-(np.log(I_values) - mu_ln)**2 / (2 * sigma_ln2))
            cdf = np.cumsum(pdf) * (I_values[1] - I_values[0])
            return {
                'rytov_variance': sigma_R2,
                'scintillation_index': scintillation_index,
                'intensity_values': I_values,
                'pdf': pdf,
                'cdf': cdf
            }
        else:
            # return targeted stats as dict
            fade_threshold = np.exp(mu_ln + np.sqrt(2 * sigma_ln2) * erfinv(2*(1-availability) - 1) * np.sqrt(2))
            fade_margin_db = -10 * np.log10(fade_threshold)
            return {
                'rytov_variance': sigma_R2,
                'scintillation_index': scintillation_index,
                'fade_threshold': fade_threshold,
                'fade_margin_db': fade_margin_db,
                'availability': availability
            } 
    elif sigma_R2 >= 1.0:
        warnings.warn("Rytov variance >= 1, strong turbulence regime — log-normal model may not be accurate, consider gamma-gamma distribution")
        # gamma-gamma uses:
        alpha = 1 / (np.exp(term1) - 1)
        beta  = 1 / (np.exp(term2) - 1)
        scintillation_index = 1/alpha + 1/beta + 1/(alpha*beta)
        I_values = np.linspace(1e-6, 10, 10000)
        pdf = (2 * (alpha*beta)**((alpha+beta)/2) / (gamma(alpha) * gamma(beta))) * \
                I_values**((alpha+beta)/2 - 1) * \
                kv(alpha - beta, 2 * np.sqrt(alpha * beta * I_values))
        cdf = cumulative_trapezoid(pdf, I_values, initial=0)
        if availability is None:
            return {
                'rytov_variance': sigma_R2,
                'scintillation_index': scintillation_index,
                'intensity_values': I_values,
                'pdf': pdf,
                'cdf': cdf
            }
        else:
            fade_threshold = I_values[np.searchsorted(cdf, 1 - availability)]
            fade_margin_db = -10 * np.log10(fade_threshold)
            return {
                'rytov_variance': sigma_R2,
                'scintillation_index': scintillation_index,
                'fade_threshold': fade_threshold,
                'fade_margin_db': fade_margin_db,
                'availability': availability
            } 