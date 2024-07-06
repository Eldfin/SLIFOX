# SLIF
Scattered Light Imaging Fitter
Finds the peaks in Scattered Light Imaging measurement data, fits a given distribution to this peaks and calculates the found nerve fiber directions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)

## Installation

### Clone the repository

```bash
git clone https://github.com/Eldfin/SLIF.git
```

### Dependencies
Install the dependencies if necesarry.

```bash
cd your-repository
pip install fcmaes, h5py, imageio, lmfit, matplotlib, numba, numpy, scipy, pympi-pypi, PyQt5
```

## Usage
### Example
```python
import h5py
import matplotlib.pyplot as plt
from SLIF import fit_image_stack, calculate_peak_pairs, calculate_directions, pick_data, plot_pixels

# Settings
dataset_path = "pyramid/00"
distribution = "wrapped_cauchy"
data_path = "input_p0_x(5500, 5650)_y(3000,3150).h5"
output_directory = ""
output_filename = "output_p0_x(5500, 5650)_y(3000,3150).h5"
area = None
random = 0

# Pick the SLI measurement data
data, indices = pick_data(data_path, dataset_path, area = None, randoms = 0)

# Optional: Write the picked data array to a HDF5 file
#with h5py.File("input.h5", "w") as h5f:
#    group = h5f.create_group(dataset_path)
#    group.create_dataset("data", data=data)
#    group.create_dataset("indices", data=indices)

# Fit the picked data
output_params, output_peaks_mask = fit_image_stack(data, fit_height_nonlinear = True, 
                                n_steps_fit = 10, n_steps_height = 10, n_steps_mu = 10, 
                                n_steps_scale = 10, refit_steps = 1, init_fit_filter = None, 
                                method="leastsq")

# Optional: Write the output to a HDF5 file
if output_directory != "" and not os.path.exists(output_directory):
    os.makedirs(output_directory)

with h5py.File(output_directory + output_filename, "w") as h5f:
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
#plot_data_pixels(data, output_params, output_peaks_mask, peak_pairs, distribution, indices, directory = "plots")

# Calculate the nerve fiber directions and save direction map in dictionary
directions = calculate_directions(peak_pairs, output_mus, dictionary = "direction_maps")
```

## Features







