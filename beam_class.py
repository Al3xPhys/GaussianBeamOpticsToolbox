import numpy as np

class Beam():
    def __init__(self, q, wavelength=1550e-09, power=1.0, M2=1.0):
        self.wavelength = wavelength
        self.q = q
        self.power = power
        self.M2 = M2

    @property
    def rayleigh_range(self):
        """ Calculates the Rayleigh range z_R from the complex beam parameter q.
        The Rayleigh range is given by z_R = Im(q)."""
        return np.imag(self.q)
    
    @property
    def waist_radius(self):
        """ Calculates the waist radius w_0 from the complex beam parameter q.
        The waist radius is given by w_0 = sqrt(λ * Im(q) / π)."""
        return np.sqrt(self.wavelength * self.rayleigh_range / np.pi)
    
    @property
    def peak_intensity(self):
        """ Calculates the peak intensity I_0 of the Gaussian beam.
        The peak intensity is given by I_0 = 2 * P / (π * w_0^2), where P is the power of the beam."""
        return 2 * self.power / (np.pi * self.waist_radius**2)
    
    @property
    def current_position(self):
        return np.real(self.q)

    
    @classmethod
    def from_beam_waist(cls, waist_radius, wavelength=1550e-09, power=1.0, M2=1.0):
        """ Initializes a Beam instance using the waist radius and wavelength.
        The complex beam parameter q is calculated as q = i * z_R, where z_R is the Rayleigh range."""
        if wavelength is None:
            raise ValueError("Wavelength must be provided to calculate the Rayleigh range.")
        rayleigh_range = np.pi * waist_radius**2 / wavelength
        q = 1j * rayleigh_range
        return cls(q=q, wavelength=wavelength, power=power, M2=M2)
    
    def get_waist(self, z=None):
        """ Calculates the beam waist radius w(z) at a given position z along the propagation direction.
        The beam waist radius is given by w(z) = w_0 * sqrt(1 + (M2 * z / z_R)^2)."""
        z = self.current_position if z is None else z
        return self.waist_radius * np.sqrt(1 + (self.M2 * (z / self.rayleigh_range))**2)
    
    def get_intensity(self, r, z):
        """ Calculates the intensity I(r, z) of the Gaussian beam at a given position z and radial distance r.
        The intensity is given by I(r, z) = I_0 * exp(-2 * r^2 / w(z)^2), where I_0 is the peak intensity."""
        w_z = self.get_waist(z)
        return self.peak_intensity * np.exp(-2 * r**2 / w_z**2)
    
    def get_cartesian_intensity(self, x, y, z):
        """ Calculates the intensity I(x, y, z) of the Gaussian beam at a given position z and Cartesian coordinates (x, y).
        The intensity is given by I(x, y, z) = I_0 * exp(-2 * (x^2 + y^2) / w(z)^2), where I_0 is the peak intensity."""
        r = np.sqrt(x**2 + y**2)
        return self.get_intensity(r, z)
    
    def propagate(self, transfer_matrix):
        """ Propagates the beam through an optical system represented by a transfer matrix.
        The complex beam parameter q is transformed according to the ABCD law: q_out = (A * q_in + B) / (C * q_in + D)."""
        A, B, C, D = transfer_matrix.flatten()
        q_out = (A * self.q + B) / (C * self.q + D)
        return Beam(q=q_out, wavelength=self.wavelength, power=self.power, M2=self.M2)
    
    def propagate_to_waist(self):
        """ Propagates the beam to its waist position, where the beam radius is minimum.
        The waist position is given by z = -Re(q), and the transfer matrix for free space propagation
        is used to propagate the beam to this position."""
        free_space_matrix = np.array([[1, -self.current_position], [0, 1]])
        propagated_beam = self.propagate(free_space_matrix)
        return propagated_beam

    def __str__(self):
        return f"Beam(wavelength={self.wavelength}, waist_radius={self.waist_radius}, power={self.power}, M2={self.M2})"