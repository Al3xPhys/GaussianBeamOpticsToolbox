from beam_class import Beam
import transfer_matrices
import atmospherics
import components
import numpy as np
import matplotlib.pyplot as plt

w0 = 5e-6 # Waist radius in meters
wavelength = 1550e-9 # Wavelength in meters
beam_power = 0.150
receiver_sens_threshold = -40  # Receiver sensitivity threshold in dBm



beam = Beam.from_beam_waist(waist_radius=w0, wavelength=wavelength, power=beam_power)
beam_after_scintillation = atmospherics.scintillation(beam, distance=3000, Cn2=20e-15, availability=None)
print(f"last cdf value: {beam_after_scintillation['pdf']}")

fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(beam_after_scintillation['intensity_values'], beam_after_scintillation['pdf'], label='PDF', linewidth=1, marker='o', markersize=2)
ax2.plot(beam_after_scintillation['intensity_values'], beam_after_scintillation['cdf'], label='CDF', linewidth=1, marker='o', markersize=2)
ax1.set_title('PDF of Intensity due to Scintillation')
ax1.set_xlabel('Intensity (normalized)')
ax1.set_ylabel('Probability Density Function (PDF)')
ax1.axvline(x=1.0, color='r', linestyle='--', label='Mean intensity')
ax1.grid()
ax2.set_title('CDF of Intensity due to Scintillation')
ax2.set_xlabel('Intensity (normalized)')
ax2.set_ylabel('Cumulative Distribution Function (CDF)')
ax2.axhline(y=0.999, color='r', linestyle='--', label='99.9% availability')
ax2.grid()
plt.tight_layout()
plt.show()

# # collimating thin lens, focal length 10mm
# beam = beam.propagate(transfer_matrices.free_space(0.01))  # Propagate to the new waist after the lens
# beam = beam.propagate(transfer_matrices.thin_lens(focal_length=0.01))
# beam_collimated = beam.propagate_to_waist()  # Propagate to the new waist after the lens
# print(f"collimated beam waist: {beam_collimated.waist_radius:.6e}")
# # expand the beam using a beam expander with focal lengths f1 = 0.03 m and f2 = 0.06 m
# expanded_beam = beam_collimated.propagate(transfer_matrices.beam_expander(f1=0.03, f2=0.06))
# expanded_beam = expanded_beam.propagate_to_waist()  # Propagate to the new waist after the expander

# #propagate the beam through free space for a distance of 1000 m
# propagation_distance = 1000
# propagated_beam = expanded_beam.propagate(transfer_matrices.free_space(propagation_distance))
# print(f"Inital beam M2: {expanded_beam.M2:.6e}")
# print(f"Inital beam waist: {expanded_beam.get_waist():.6e} m")
# print(f"propagated_beam.current_position: {propagated_beam.current_position:.6e} m")
# print(f"propagated_beam.q: {propagated_beam.q}")
# print(f"propagated_beam.get_waist(): {propagated_beam.get_waist():.6e} m")
# # apply turbulence effects to the beam over the propagation distance with a specified refractive index structure parameter Cn2
# beam_with_turbulence = atmospherics.turbulence(propagated_beam, distance=propagation_distance, Cn2=1e-15)
# print(f"Beam M2 after turbulence: {beam_with_turbulence.M2:.6e}")

# Looping through range of distances with atmospheric effects to do link budget analysis
visibility_conditions = {"Clear (10km)": 10000, "Light Fog (1km)": 1000, "Moderate Fog (500m)": 500, "Dense Fog (200m)": 200}
distances = np.linspace(0.001, 5000, 100)  # Distances
captured_powers = {}  # Reset captured powers for each visibility condition
beam_waists = {}  # Reset beam waists for each visibility condition
captured_fractions = {}  # Reset captured fractions for each visibility condition
for label, visibility in visibility_conditions.items():
    captured_powers[label] = []  # Initialize list for captured powers for this visibility condition
    beam_waists[label] = []  # Initialize list for beam waists for this visibility condition
    captured_fractions[label] = []  # Initialize list for captured fractions for this visibility condition
    for distance in distances:
        #New beam
        beam = Beam.from_beam_waist(waist_radius=w0, wavelength=wavelength, power=beam_power)

        # collimating thin lens, focal length 10mm
        beam = beam.propagate(transfer_matrices.free_space(0.01)) 
        beam = beam.propagate(transfer_matrices.thin_lens(focal_length=0.01))
        beam_collimated = beam.propagate_to_waist()  # Propagate to the new waist after the lens

        # expand the beam using a beam expander with focal lengths f1 = 0.03 m and f2 = 0.06 m
        expanded_beam = beam_collimated.propagate(transfer_matrices.beam_expander(f1=0.03, f2=0.06))
        expanded_beam = expanded_beam.propagate_to_waist()  # Propagate to the new waist after the expander

        # Propagate the beam through free space for the given distance
        propagated_beam = expanded_beam.propagate(transfer_matrices.free_space(distance))

        # Apply atmospheric effects (fog) to the beam
        beam_after_fog = atmospherics.kim_fog(propagated_beam, distance=distance, visibility=visibility)

        beam_after_turbulence = atmospherics.turbulence(beam_after_fog, distance=distance, Cn2=1e-15)

        # Calculate the power captured by a receiver with a radius of 0.15 m (150 mm)
        receiver_radius = 0.15
        captured_power = components.receiver(beam_after_turbulence, receiver_radius)
        captured_power = max(captured_power, 1e-12)  # Ensure captured power is not zero to avoid log(0)
        beam_waist = beam_after_turbulence.get_waist()
        #convert to Watts to dBm
        captured_power_dbm = 10 * np.log10(captured_power / 1e-3)  # Convert to dBm
        scint = atmospherics.scintillation(beam_after_turbulence, distance=distance, Cn2=1e-15, availability=0.999)
        effective_power_dbm = captured_power_dbm - scint['fade_margin_db']  # Adjust for fade margin
        
        # calc. captured fraction of beam waist area captured by receiver
        receiver_area = np.pi * receiver_radius**2
        beam_area = np.pi * beam_waist**2
        captured_fraction = receiver_area / beam_area


        # print(f"Distance: {distance:.2f} m, Captured Power: {captured_power:.6e} W, Captured Power: {captured_power_dbm:.2f} dBm")
        captured_powers[label].append(effective_power_dbm)
        beam_waists[label].append(beam_waist)
        captured_fractions[label].append(captured_fraction)


# Plotting the results for each visibility condition on the same graph
fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(10, 8))
for label, visibility in visibility_conditions.items():
    ax1.plot(distances, list(captured_powers[label]), label=f'Visibility: {visibility} m', linewidth=1, marker='o', markersize=2)
    ax2.plot(distances, list(captured_fractions[label]), label=f'Visibility: {visibility} m', linewidth=1, marker='o', markersize=2)
ax1.axhline(y=receiver_sens_threshold, color='r', linestyle='--', label='Receiver Sensitivity Threshold (-40 dBm)')
ax1.set_title('Captured Power vs Distance with Atmospheric Effects')
ax1.set_xlabel('Distance (m)')
ax1.set_ylabel('Captured Power (dBm)')
ax1.grid()
ax1.legend()
ax2.set_title('Beam Waist vs Distance with Atmospheric Effects')
ax2.set_xlabel('Distance (m)')
ax2.set_ylabel('Beam Waist (m)')
ax2.grid()
ax2.legend()


plt.tight_layout()
plt.show()



# #Propagate the beam through free space for a distance of 0.05 m
# propagation_distance = 0.05
# propagated_beam = beam.propagate(transfer_matrices.free_space(propagation_distance))
# print(f"Initial beam width: {beam.get_waist():.6e} m")        # w0 = 70µm
# print(f"After {propagation_distance} m: {propagated_beam.get_waist():.6e} m")       # should be much larger


# # put beam through a thin lens with focal length f = 0.03 m (30 mm)
# beam_after_lens = propagated_beam.propagate(transfer_matrices.thin_lens(0.03))
# beam_at_new_waist = beam_after_lens.propagate_to_waist()
# print(f"Beam waist after lens: {beam_at_new_waist.get_waist():.6e} m")  # should be smaller than before the lens

# # Now put the beam through a beam expander with focal lengths f1 = 0.03 m and f2 = 0.06 m
# expanded_beam = beam_at_new_waist.propagate(transfer_matrices.beam_expander(f1=0.03, f2=0.06))
# expanded_beam = expanded_beam.propagate_to_waist()
# print(f"Input to expander: {beam_at_new_waist.get_waist():.6e} m")   # ~94µm
# print(f"After expander: {expanded_beam.get_waist():.6e} m")           # ~188µm

# # Now let's put the expanded beam through fog with a visibility of 1000 m over a distance of 1000 m
# print(f"Initial beam power: {beam.power:.6e} W")  # should be 1.0 W
# beam_through_fog = atmospherics.kim_fog(expanded_beam, distance=1000, visibility=1000)
# print(f"Beam power after 1 km of fog: {beam_through_fog.power:.6e} W")  # should be less than 1.0 W
# print(f"Beam waist after fog: {beam_through_fog.get_waist():.6e} m")  # should be the same as before the fog

# # Now let's put the beam through an aperture stop with a radius of 0.0001 m (100 µm)
# aperture_radius = 0.0001
# beam_after_aperture = components.aperture_stop(beam_through_fog, aperture_radius)
# print(f"Beam waist after aperture stop: {beam_after_aperture.get_waist():.6e} m")  # should be the same as before the aperture
# print(f"Beam power after aperture stop: {beam_after_aperture.power:.6e} W")  # should be less than the power after fog

# # Now let's calculate the power captured by a receiver with a radius of 0.0001 m (100 µm)
# receiver_radius = 0.0001
# captured_power = components.receiver(beam_after_aperture, receiver_radius)
# print(f"Power captured by receiver: {captured_power:.6e} W")