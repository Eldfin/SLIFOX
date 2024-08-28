import h5py
import matplotlib.pyplot as plt
from SLIFOX import fit_image_stack, get_image_peak_pairs, pick_data, plot_data_pixels,\
                    map_number_of_peaks, map_peak_distances, map_peak_amplitudes, \
                    map_peak_widths, map_directions, map_direction_significances, write_fom
from SLIFOX.filters import apply_filter
import os

# Settings
dataset_path = "pyramid/00"
data_file_path = "/home/user/workspace/SLI_Data.h5"
output_file_path = "output/output.h5"
area = None  # [x_left, x_right, y_top, y_bot]
randoms = 0 # number of random pixels to pick from data (0 equals full data)
distribution = "wrapped_cauchy"
num_processes = 2
#pre_filter = ["gauss", 1]

# Pick the SLI measurement data
data, indices = pick_data(data_file_path, dataset_path, area = area, randoms = randoms)

# Optional: Filter the data before processing
#data = apply_filter(data, pre_filter)

# Optional: Write the picked data array to a HDF5 file
#with h5py.File("input.h5", "w") as h5f:
#    group = h5f.create_group(dataset_path)
#    group.create_dataset("data", data=data)
#    group.create_dataset("indices", data=indices)

# Fit the picked data
image_params, image_peaks_mask = fit_image_stack(data, fit_height_nonlinear = True, 
                                threshold = 1000, distribution = distribution,
                                n_steps_fit = 5, n_steps_height = 10, n_steps_mu = 10, 
                                n_steps_scale = 10, refit_steps = 0, init_fit_filter = None, 
                                method="leastsq", num_processes = num_processes)

# Optional: Write the output to a HDF5 file
directory = os.path.dirname(output_file_path)
if not os.path.exists(directory):
    os.makedirs(directory)
with h5py.File(output_file_path, "w") as h5f:
    group = h5f.create_group(dataset_path)
    group.create_dataset("params", data=image_params)
    group.create_dataset("peaks_mask", data=image_peaks_mask)
    group.create_dataset("indices", data=indices)

# Optional: Pick SLIF output data (if already fitted)
#image_params, _ = pick_data(output_file_path, dataset_path + "/params")
#image_peaks_mask, _ = pick_data(output_file_path, dataset_path + "/peaks_mask")
    
# Find the peak pairs (directions)
image_peak_pairs = get_image_peak_pairs(data, image_params, image_peaks_mask, min_distance = 20,
                            distribution = distribution, only_mus = False, num_processes = num_processes,
                            amplitude_threshold = 3000, rel_amplitude_threshold = 0.1, 
                            gof_threshold = 0.5, significance_threshold = 0.3, 
                            significance_weights = [1, 1], angle_threshold = 20, 
                            num_attempts = 100000, search_radius = 50)

# Use best pairs of all possible pairs
best_image_peak_pairs = image_peak_pairs[:, :, 0, :, :]

# Optional: Plot the picked data (only recommended for small image areas)
#plot_data_pixels(data, image_params, image_peaks_mask, best_image_peak_pairs, 
#                        distribution = distribution, indices = indices, directory = "plots")

# Calculate the nerve fiber directions and save direction map in directory
image_mus = image_params[:, :, 1::3]
image_directions = map_directions(best_image_peak_pairs, image_mus, directory = "maps")

# Map the significance of the directions
image_direction_sig = map_direction_significances(data, best_image_peak_pairs, image_params, 
                                image_peaks_mask, distribution = distribution, 
                                weights = [1, 1])

# Optional: Map the threshold filtered direction images 
# map_direction_significance can also be called with specific amplitude / gof thresholds beforehand
#map_directions(best_image_peak_pairs, image_mus, directiory = "maps", 
#                image_direction_sig = image_direction_sig, significance_threshold = 0.8)


# Create the fiber orientation map (fom) using the two direction files (for max 4 peaks)
direction_files = ["maps/dir_1.tiff", "maps/dir_2.tiff"]
write_fom(direction_files, "direction_maps")

# Create map for the number of peaks
map_number_of_peaks(data, image_params, image_peaks_mask, distribution = "wrapped_cauchy", 
                            amplitude_threshold = 3000, rel_amplitude_threshold = 0.1, 
                            gof_threshold = 0.5, only_mus = False)

# Create map for the distance between two paired peaks (of 2 peak pixels)
map_peak_distances(data, image_params, image_peaks_mask, distribution = "wrapped_cauchy", 
                            amplitude_threshold = 3000, rel_amplitude_threshold = 0.1, 
                            gof_threshold = 0.5, only_mus = False)

# Create map for the mean amplitudes
map_peak_amplitudes(data, image_params, image_peaks_mask, distribution = "wrapped_cauchy", 
                            amplitude_threshold = 3000, rel_amplitude_threshold = 0.1, 
                            gof_threshold = 0.5, only_mus = False)

# Create map for the mean peak widths
map_peak_widths(data, image_params, image_peaks_mask, distribution = "wrapped_cauchy", 
                            amplitude_threshold = 3000, rel_amplitude_threshold = 0.1, 
                            gof_threshold = 0.5)  