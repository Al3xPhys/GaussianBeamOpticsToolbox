import numpy as np
import warnings
from scipy.special import erfinv, gamma, kv
from scipy.integrate import cumulative_trapezoid
from beam_class import Beam


def scintillation(beam, distance, Cn2, availability=None):
    """
    Computes scintillation statistics for the beam.

    Uses the log-normal model in weak turbulence (sigma_R^2 < 1) and the
    gamma-gamma model in moderate-to-strong turbulence (sigma_R^2 >= 1).

    Parameters:
        beam (Beam): Input beam.
        distance (float): Propagation distance in metres.
        Cn2 (float): Refractive index structure parameter in m^(-2/3).
        availability (float, optional): Link availability target (e.g. 0.999 for 99.9%).
            If provided, returns fade margin instead of full PDF/CDF.

    Returns:
        dict with keys:
            Always present:
                'rytov_variance'      : sigma_R^2
                'scintillation_index' : SI = var(I) / <I>^2
                'regime'              : 'weak' or 'strong'
            If availability is None:
                'intensity_values'    : I array (normalised)
                'pdf'                 : probability density
                'cdf'                 : cumulative distribution
            If availability is specified:
                'fade_threshold'      : intensity threshold for given availability
                'fade_margin_db'      : fade margin in dB
                'availability'        : availability target
    """
    if not isinstance(beam, Beam):
        raise ValueError("Input must be an instance of the Beam class.")
    if distance <= 0:
        raise ValueError("Distance must be positive.")
    if Cn2 < 0:
        raise ValueError("Refractive index structure parameter must be non-negative.")

    k = 2 * np.pi / beam.wavelength
    sigma_R2 = 1.23 * Cn2 * k ** (7 / 6) * distance ** (11 / 6)

    term1 = 0.49 * sigma_R2 / (1 + 1.11 * sigma_R2 ** (6 / 7)) ** (7 / 6)
    term2 = 0.51 * sigma_R2 / (1 + 0.69 * sigma_R2 ** (6 / 7)) ** (5 / 6)
    sigma_ln2 = term1 + term2

    if sigma_R2 < 1.0:
        regime = "weak"
        scintillation_index = np.exp(sigma_ln2) - 1
        mu_ln = -sigma_ln2 / 2  # normalised so <I> = 1

        if availability is None:
            I_values = np.linspace(0.001, 5, 1000)
            pdf = (1 / (I_values * np.sqrt(sigma_ln2) * np.sqrt(2 * np.pi))) * np.exp(
                -((np.log(I_values) - mu_ln) ** 2) / (2 * sigma_ln2)
            )
            cdf = np.cumsum(pdf) * (I_values[1] - I_values[0])
            return {
                "rytov_variance": sigma_R2,
                "scintillation_index": scintillation_index,
                "regime": regime,
                "intensity_values": I_values,
                "pdf": pdf,
                "cdf": cdf,
            }
        else:
            fade_threshold = np.exp(
                mu_ln
                + np.sqrt(2 * sigma_ln2)
                * erfinv(2 * (1 - availability) - 1)
                * np.sqrt(2)
            )
            fade_margin_db = -10 * np.log10(fade_threshold)
            return {
                "rytov_variance": sigma_R2,
                "scintillation_index": scintillation_index,
                "regime": regime,
                "fade_threshold": fade_threshold,
                "fade_margin_db": fade_margin_db,
                "availability": availability,
            }

    else:
        regime = "strong"
        warnings.warn(
            f"Rytov variance = {sigma_R2:.2f} >= 1: strong turbulence regime — "
            "log-normal model not valid, using gamma-gamma distribution."
        )
        alpha = 1 / (np.exp(term1) - 1)
        beta = 1 / (np.exp(term2) - 1)
        scintillation_index = 1 / alpha + 1 / beta + 1 / (alpha * beta)

        I_values = np.linspace(1e-6, 10, 10000)
        pdf = (
            (2 * (alpha * beta) ** ((alpha + beta) / 2) / (gamma(alpha) * gamma(beta)))
            * I_values ** ((alpha + beta) / 2 - 1)
            * kv(alpha - beta, 2 * np.sqrt(alpha * beta * I_values))
        )
        cdf = cumulative_trapezoid(pdf, I_values, initial=0)

        if availability is None:
            return {
                "rytov_variance": sigma_R2,
                "scintillation_index": scintillation_index,
                "regime": regime,
                "intensity_values": I_values,
                "pdf": pdf,
                "cdf": cdf,
            }
        else:
            fade_threshold = I_values[np.searchsorted(cdf, 1 - availability)]
            fade_margin_db = -10 * np.log10(fade_threshold)
            return {
                "rytov_variance": sigma_R2,
                "scintillation_index": scintillation_index,
                "regime": regime,
                "fade_threshold": fade_threshold,
                "fade_margin_db": fade_margin_db,
                "availability": availability,
            }
