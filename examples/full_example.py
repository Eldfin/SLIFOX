import h5py
import matplotlib.pyplot as plt
from SLIF import fit_image_stack, calculate_peak_pairs, calculate_directions, pick_data, plot_data_pixels
import os

# Settings
dataset_path = "pyramid/00"
distribution = "wrapped_cauchy"
data_path = "SLI_Data.h5"
output_directory = ""
output_filename = "output.h5"
area = None  # [x_left, x_right, y_top, y_bot]
randoms = 0 # number of random pixels to pick from data (0 equals full data)

# Pick the SLI measurement data
data, indices = pick_data(data_path, dataset_path, area = area, randoms = randoms)

# Optional: Write the picked data array to a HDF5 file
#with h5py.File("input.h5", "w") as h5f:
#    group = h5f.create_group(dataset_path)
#    group.create_dataset("data", data=data)
#    group.create_dataset("indices", data=indices)

# Fit the picked data
output_params, output_peaks_mask = fit_image_stack(data, fit_height_nonlinear = True, 
                                threshold = 1000,
                                n_steps_fit = 10, n_steps_height = 10, n_steps_mu = 10, 
                                n_steps_scale = 10, refit_steps = 1, init_fit_filter = None, 
                                method="leastsq")

# Optional: Write the output to a HDF5 file
if output_directory != "" and not os.path.exists(output_directory):
    os.makedirs(output_directory)

with h5py.File(output_directory + "/" + output_filename, "w") as h5f:
    group = h5f.create_group(dataset_path)
    group.create_dataset("params", data=output_params)
    group.create_dataset("peaks_mask", data=output_peaks_mask)
    group.create_dataset("indices", data=indices)

# Optional: Pick SLIF output data
#output_params, _ = pick_data(output_filename, dataset_path + "/params", area = None, randoms = 0)
#output_peaks_mask, _ = pick_data(output_filename, dataset_path + "/peaks_mask", area = None, randoms = 0)
    
# Calculate the peak pairs (directions)
peak_pairs = calculate_peak_pairs(data, output_params, output_peaks_mask, distribution)

# Optional: Plot the picked data
plot_data_pixels(data, output_params, output_peaks_mask, peak_pairs, distribution, indices, 
                        directory = "plots")

# Calculate the nerve fiber directions and save direction map in directory
output_mus = output_params[:, :, 1::3]
directions = calculate_directions(peak_pairs, output_mus, directory = "direction_maps")