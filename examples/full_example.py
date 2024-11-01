import h5py
import matplotlib.pyplot as plt
from SLIFOX import fit_image_stack, get_image_peak_pairs, pick_data, plot_data_pixels,\
                    map_number_of_peaks, map_peak_distances, map_mean_peak_amplitudes, \
                    map_mean_peak_widths, map_directions, map_direction_significances, map_fom, \
                    get_sig_peaks_mask
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
                                n_steps_height = 10, n_steps_mu = 10, n_steps_scale = 10, 
                                n_steps_fit = 3, min_steps_diff = 5,
                                refit_steps = 0, init_fit_filter = None, max_fit_peaks = 4,
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
image_peak_pairs = get_image_peak_pairs(data, image_params, image_peaks_mask, method = "neighbor",
                            min_distance = 20, distribution = distribution, 
                            only_mus = False, num_processes = num_processes,
                            amplitude_threshold = 1000, rel_amplitude_threshold = 0.1, 
                            gof_threshold = 0.2, significance_threshold = 0, 
                            significance_weights = [1, 1], significance_sens = [1, 1],
                            nb_angle_threshold = 5, 
                            max_attempts = 50, search_radius = 50, max_paired_peaks = 6)

# Use best pairs of all possible pairs
best_image_peak_pairs = image_peak_pairs[:, :, 0, :, :]

# Optional: Plot the picked data (only recommended for small image areas)
#plot_data_pixels(data, image_params, image_peaks_mask, best_image_peak_pairs, 
#                        distribution = distribution, indices = indices, directory = "plots")

# Calculate the nerve fiber directions and save direction map in directory
image_mus = image_params[:, :, 1::3]
image_directions = map_directions(best_image_peak_pairs, image_mus, directory = "maps",
                                    exclude_lone_peaks = True)

# Map the significance of the directions
image_direction_sig = map_direction_significances(data, best_image_peak_pairs, image_params, 
                                image_peaks_mask, distribution = distribution, 
                                weights = [1, 1], sens = [1, 1], num_processes = num_processes)

# Optional: Map the threshold filtered direction images 
# (when pairs filtered with same threshold redundant)
# map_direction_significance can also be called with specific amplitude / gof thresholds beforehand
#map_directions(best_image_peak_pairs, image_mus, directiory = "maps",  exclude_lone_peaks = True,
#                image_direction_sig = image_direction_sig, significance_threshold = 0.8)

# Create the fiber orientation map (fom) using the two direction files (for max 4 peaks)
map_fom(image_directions, directory = "maps")

# Create a mask for the significant peaks
image_sig_peaks_mask = get_sig_peaks_mask(image_stack = image_stack, image_params = image_params, 
                            image_peaks_mask = image_peaks_mask,
                            distribution = distribution, 
                            amplitude_threshold = amplitude_threshold, 
                            rel_amplitude_threshold = rel_amplitude_threshold, 
                            gof_threshold = gof_threshold, only_mus = only_mus)

# Create map for the number of peaks (will be saved as .tiff in "maps" directory)
image_num_peaks = map_number_of_peaks(image_sig_peaks_mask = image_sig_peaks_mask, directory = "maps")

# Create map for the distance between two paired peaks
map_peak_distances(image_params = image_params, only_mus = False, deviation = False,
                            image_peak_pairs = best_image_peak_pairs, directory = "maps",
                            num_processes = num_processes, only_peaks_count = -1,
                            image_sig_peaks_mask = image_sig_peaks_mask,
                            image_num_peaks = image_num_peaks)

# Create map for the mean amplitudes
map_mean_peak_amplitudes(image_stack = data, image_params = image_params, 
                            image_peaks_mask = image_peaks_mask, distribution = "wrapped_cauchy", 
                            only_mus = False, directory = "maps",
                            image_sig_peaks_mask = image_sig_peaks_mask)

# Create map for the mean peak widths
map_mean_peak_widths(image_stack = data, image_params = image_params, 
                            image_peaks_mask = image_peaks_mask, distribution = "wrapped_cauchy", 
                            directory = "maps", image_sig_peaks_mask = image_sig_peaks_mask)  