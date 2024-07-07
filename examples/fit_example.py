import h5py
from SLIF import fit_image_stack, pick_data
import os

# Settings
dataset_path = "pyramid/00"
distribution = "wrapped_cauchy"
data_path = "SLI_Data.h5"
output_directory = ""
output_filename = "output.h5"
area = None
randoms = 0

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

# Write the output to a HDF5 file
if output_directory != "" and not os.path.exists(output_directory):
    os.makedirs(output_directory)

with h5py.File(output_directory + output_filename, "w") as h5f:
    group = h5f.create_group(dataset_path)
    group.create_dataset("params", data=output_params)
    group.create_dataset("peaks_mask", data=output_peaks_mask)
    group.create_dataset("indices", data=indices)