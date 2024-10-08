import h5py
import matplotlib.pyplot as plt
from SLIFOX import find_image_peaks, pick_data
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
                        only_peaks_count = -1, num_processes = 2)


# Optional: Write the output to a HDF5 file
directory = os.path.dirname(output_file_path)
if not os.path.exists(directory):
    os.makedirs(directory)
with h5py.File(output_file_path, "w") as h5f:
    group = h5f.create_group(dataset_path)
    group.create_dataset("mus", data=image_mus)
    group.create_dataset("peaks_mask", data=image_peaks_mask)
    group.create_dataset("indices", data=indices)