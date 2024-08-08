import h5py
import matplotlib.pyplot as plt
from SLIF import find_image_peaks, get_image_peak_pairs, calculate_directions, pick_data, plot_data_pixels
import os

# Settings
dataset_path = "pyramid/00"
data_file_path = "/home/user/workspace/SLI_Data.h5"
output_file_path = "output/output.h5"
area = None  # [x_left, x_right, y_top, y_bot]
randoms = 0 # number of random pixels to pick from data (0 equals full data)
distribution = "wrapped_cauchy"

# Pick the SLI measurement data
data, indices = pick_data(data_file_path, dataset_path, area = area, randoms = randoms)

# Optional: Write the picked data array to a HDF5 file
#with h5py.File("input.h5", "w") as h5f:
#    group = h5f.create_group(dataset_path)
#    group.create_dataset("data", data=data)
#    group.create_dataset("indices", data=indices)

# Find the peaks from the picked data
image_mus, image_peaks_mask = find_image_peaks(data, threshold = 1000, init_fit_filter = None, 
                        only_peaks_count = -1, max_peaks = 4, num_processes = 2)


# Optional: Write the output to a HDF5 file
directory = os.path.dirname(output_file_path)
if not os.path.exists(directory):
    os.makedirs(directory)
with h5py.File(output_file_path, "w") as h5f:
    group = h5f.create_group(dataset_path)
    group.create_dataset("mus", data=image_mus)
    group.create_dataset("peaks_mask", data=image_peaks_mask)
    group.create_dataset("indices", data=indices)

# Optional: Pick SLIF output data (if already fitted)
#image_mus, _ = pick_data(output_file_path, dataset_path + "/mus")
#image_peaks_mask, _ = pick_data(output_file_path, dataset_path + "/peaks_mask")
    
# Find the peak pairs (directions)
image_peak_pairs = get_image_peak_pairs(data, image_mus, image_peaks_mask, 
                            distribution = distribution, only_mus = True, num_processes = 2,
                            significance_threshold = 0.9, significance_weights = [1, 1],
                            angle_threshold = 30 * np.pi / 180, num_attempts = 100000, 
                            search_radius = 100)

# Use best pairs of all possible pairs
best_image_peak_pairs = image_peak_pairs[:, :, 0, :, :]

# Calculate the nerve fiber directions and save direction map in directory
directions = calculate_directions(best_image_peak_pairs, image_mus, directory = "direction_maps")

# Create the fiber orientation map (fom) using the two direction files (for max 4 peaks)
direction_files = ["direction_maps/dir_1.tiff", "direction_maps/dir_2.tiff"]
write_fom(direction_files, "direction_maps")