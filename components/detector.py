import numpy as np
from beam_class import Beam


def receiver(beam, receiver_radius):
    """
    Calculates optical power captured by a circular receiver aperture at the end of the link.
    Uses the Gaussian beam power integral.

    Parameters:
        beam (Beam): Beam at the receiver plane.
        receiver_radius (float): Receiver aperture radius in metres.

    Returns:
        float: Captured optical power in Watts.
    """
    return beam.power * (1 - np.exp(-2 * (receiver_radius / beam.get_waist()) ** 2))


class Detector:
    """
    Simple photodetector model. Converts received optical power to photocurrent,
    then computes SNR and BER for OOK (on-off keying) modulation.

    Parameters:
        responsivity (float): Detector responsivity in A/W. Default 0.8 A/W (typical InGaAs at 1550nm).
        dark_current (float): Dark current in Amperes. Default 5 nA.
        load_resistance (float): Load/transimpedance resistance in Ohms. Default 50 Ω.
        temperature (float): Operating temperature in Kelvin. Default 290 K.
        bandwidth (float): Receiver electrical bandwidth in Hz. Default 1 GHz.
        excess_noise_factor (float): APD excess noise factor (set to 1 for PIN). Default 1.0.
    """

    def __init__(
        self,
        responsivity=0.8,
        dark_current=5e-9,
        load_resistance=50.0,
        temperature=290.0,
        bandwidth=1e9,
        excess_noise_factor=1.0,
    ):
        self.responsivity = responsivity
        self.dark_current = dark_current
        self.load_resistance = load_resistance
        self.temperature = temperature
        self.bandwidth = bandwidth
        self.excess_noise_factor = excess_noise_factor

    def photocurrent(self, optical_power):
        """Returns signal photocurrent in Amperes."""
        return self.responsivity * optical_power

    def shot_noise_variance(self, optical_power):
        """Shot noise variance (A^2). Includes dark current contribution."""
        q = 1.602e-19  # electron charge
        I_sig = self.photocurrent(optical_power)
        return (
            2
            * q
            * (I_sig + self.dark_current)
            * self.excess_noise_factor
            * self.bandwidth
        )

    def thermal_noise_variance(self):
        """Johnson-Nyquist thermal noise variance (A^2)."""
        k_B = 1.381e-23  # Boltzmann constant
        return 4 * k_B * self.temperature * self.bandwidth / self.load_resistance

    def snr(self, optical_power):
        """
        Electrical SNR for OOK: SNR = I_sig^2 / (sigma_shot^2 + sigma_thermal^2).

        Parameters:
            optical_power (float): Received optical power in Watts.

        Returns:
            float: Linear SNR (not in dB).
        """
        if optical_power <= 0:
            return 0.0
        I_sig = self.photocurrent(optical_power)
        noise = self.shot_noise_variance(optical_power) + self.thermal_noise_variance()
        return I_sig**2 / noise

    def ber(self, optical_power):
        """
        Bit error rate for OOK with direct detection.
        BER = 0.5 * erfc(sqrt(SNR) / (2*sqrt(2)))

        This assumes equal probability of 0 and 1 and threshold set at half the
        signal level (optimum for shot-noise limited, equal-power 0/1 assumption).

        Parameters:
            optical_power (float): Received optical power in Watts.

        Returns:
            float: BER (dimensionless, in [0, 0.5]).
        """
        from scipy.special import erfc

        snr_val = self.snr(optical_power)
        if snr_val <= 0:
            return 0.5
        Q = np.sqrt(snr_val) / 2
        return 0.5 * erfc(Q / np.sqrt(2))

    def sensitivity(self, target_ber=1e-9):
        """
        Estimates receiver sensitivity (minimum detectable power) for a target BER.
        Solved numerically by bisection over optical power.

        Parameters:
            target_ber (float): Target BER. Default 1e-9.

        Returns:
            float: Minimum optical power in Watts for the target BER.
        """
        from scipy.optimize import brentq

        f = lambda P: self.ber(P) - target_ber
        try:
            return brentq(f, 1e-15, 1e-1)
        except ValueError:
            return np.nan

    def snr_db(self, optical_power):
        """Returns SNR in dB."""
        return 10 * np.log10(self.snr(optical_power))

    def __repr__(self):
        return (
            f"Detector(responsivity={self.responsivity} A/W, "
            f"dark_current={self.dark_current:.2e} A, "
            f"bandwidth={self.bandwidth:.2e} Hz)"
        )
