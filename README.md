# SLIF
Scattered Light Imaging Fitter  
Finds the peaks in Scattered Light Imaging measurement data, fits a given distribution to this peaks and calculates the found nerve fiber directions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)

## Installation

Installed Python version >= 3.12.4 is necessary.
Operating system should be Linux.

### Clone the repository

```bash
git clone https://github.com/Eldfin/SLIF.git
```

### Create a virtual environment
```bash
cd SLIF
python -m venv .venv
```

### Activate the virtual environment
* On Windows:
```bash
.venv\Scripts\activate
```
* On macOS and Linux:
```bash
source .venv/bin/activate
```

### Install the SLIF package
```bash
pip install .
```


## Features
- Finds the number of peaks and the peak positions in a SLI intensity profile.
- Fit a distribution to the SLI intensity profile to refine the found peak positions.
- Calculates the nerve fiber directions of a SLI intensity profile (in particular for overlapping nerve fibers) by searching for similar directions in neighbouring pixels.


## Usage

### Functions

#### `fit_pixel_stack`
##### Description
Fits the data of one pixel.

##### Parameters
- `angles`: np.ndarray (n, )
    Array that stores the angles at which the intensities are measured.
- `intensities`: np.ndarray (n, )
    The measured intensities of the pixel.
- `intensities_err`: np.ndarray (n, )
    The error (standard deviation) of the corresponding measured intensities of the pixel.
- `distribution`: "wrapped_cauchy", "von_mises", or "wrapped_laplace"
    The name of the distribution.
- `n_steps_height`: int
    Number of variations in height to search for best initial guess.
- `n_steps_mu`: int
    Number of variations in mu to search for best initial guess.
- `n_steps_scale`: int
    Number of variations in scale to search for best initial guess.
- `n_steps_fit`: int
    Number of initial guesses to pick for fitting (starting from best inital guesses).
- `fit_height_nonlinear`: boolean
    Whether to include the heights in the nonlinear fitting or not.
- `refit_steps`: int
    Number that defines how often the fitting process should be repeated with the result
    as new initial guess.
- `init_fit_filter`: None or list
    List that defines which filter to apply before the first fit. 
    This filter will be applied on the intensities before doing anything and
    will be remove after one fit is done. Then the normal fitting process starts with
    this result as initial guess.
    First value of the list is a string with:
    "fourier", "gauss", "uniform", "median", "moving_average", or "savgol".
    The following one to two values are the params for this filter (scipy docs).
- `method`: string
    Defines which fitting method to use. Can be anything from the methods in lmfit.minimize
    and additionally "biteopt" (Derivative-Free Global Optimization Method).
- `only_peaks_count`: int
    Defines a filter for found peaks, so that if the count of found peaks is not equal that number,
    the function return the same as if no peaks are found.
- `max_peaks`: int
    Defines the maximum number of peaks that should be returned from the 
    total found peaks (starting from highest peak).
- `max_peak_hwhm`: float
    Estimated maximum peak half width at half maximum.
- `min_peak_hwhm`: float
    Estimated minimum peak half width at half maximum.
- `mu_range`: float
    Range of mu (regarding estimated maximum and minimum bounds around true mu).
- `scale_range`: float
    Range of scale (regarding estimated maximum and minimum bounds around true scale).


##### Returns
- `best_parameters`: np.ndarray (m, )
    Array which stores the best found parameters.
- `best_redchi`: float
    Calculated Chi2 of the model with the found parameters and given data.
- `peaks_mask`: np.ndarray (n_peaks, n)
    Array that stores the indices of the measurements that corresponds (mainly) to a peak.

#### `fit_image_stack`
##### Description
Fits the data of a full image stack.

##### Parameters
- `image_stack`: np.ndarray (n, m, p)
    Array that stores the p intensity measurements for every pixel in the (n*m sized) image
- `distribution`: "wrapped_cauchy", "von_mises", or "wrapped_laplace"
    The name of the distribution.
- `fit_height_nonlinear`: boolean
    Whether to include the heights in the nonlinear fitting or not.
- `threshold`: int
    Threshold value. If the mean intensity of one pixel is lower than that threshold value,
    the pixel will not be evaluated.
- `n_steps_height`: int
    Number of variations in height to search for best initial guess.
- `n_steps_mu`: int
    Number of variations in mu to search for best initial guess.
- `n_steps_scale`: int
    Number of variations in scale to search for best initial guess.
- `n_steps_fit`: int
    Number of initial guesses to pick for fitting (starting from best inital guesses).
- `refit_steps`: int
    Number that defines how often the fitting process should be repeated with the result
    as new initial guess.
- `init_fit_filter`: None or list
    List that defines which filter to apply before the first fit. 
    This filter will be applied on the intensities before doing anything and
    will be removed after one fit is done. Then the normal fitting process starts with
    this result as initial guess.
    First value of the list is a string with:
    "fourier", "gauss", "uniform", "median", "moving_average", or "savgol".
    The following one to two values are the params for this filter (scipy docs).
- `method`: string
    Defines which fitting method to use. Can be anything from the methods in lmfit.minimize
    and additionally "biteopt (Derivative-Free Global Optimization Method).
- `only_peaks_count`: int
    Defines a filter for found peaks, so that if the count of found peaks is not equal that number,
    the function returns the same as if no peaks are found.
- `max_peaks`: int
    Defines the maximum number of peaks that should be returned from the 
    total found peaks (starting from highest peak).
- `max_peak_hwhm`: float
    Estimated maximum peak half width at half maximum.
- `min_peak_hwhm`: float
    Estimated minimum peak half width at half maximum.
- `mu_range`: float
    Range of mu (regarding estimated maximum and minimum bounds around true mu).
- `scale_range`: float
    Range of scale (regarding estimated maximum and minimum bounds around true scale).
- `num_processes`: int
    Number that defines in how many sub-processes the fitting process should be split into.
      
##### Returns
- `best_parameters`: np.ndarray (m, )
        Array which stores the best found parameters.
- `best_redchi`: float
    Calculated Chi2 of the model with the found parameters and given data.
- `peaks_mask`: np.ndarray (n_peaks, n)
    Array that stores the indices of the measurements that corresponds (mainly) to a peak.


#### `find_image_peaks`
##### Description
Finds the peaks of an image stack using only the peak finder (no fitting).

##### Parameters
- `image_stack`: np.ndarray (n, m, p)
    Array that stores the p intensity measurements for every pixel in the (n*m sized) image
- `threshold`: int
    Threshold value. If the mean intensity of one pixel is lower than that threshold value,
    the pixel will not be evaluated.
- `init_fit_filter`: None or list
    List that defines which filter to apply before the first fit. 
    This filter will be applied on the intensities before doing anything and
    will be remove after one fit is done. Then the normal fitting process starts with
    this result as initial guess.
    First value of the list is a string with:
    "fourier", "gauss", "uniform", "median", "moving_average", or "savgol".
    The following one to two values are the params for this filter (scipy docs).
- `only_peaks_count`: int
    Defines a filter for found peaks, so that if the count of found peaks is not equal that number,
    the function return the same as if no peaks are found.
- `max_peaks`: int
    Defines the maximum number of peaks that should be returned from the 
    total found peaks (starting from highest peak).
- `max_peak_hwhm`: float
    Estimated maximum peak half width at half maximum.
- `min_peak_hwhm`: float
    Estimated minimum peak half width at half maximum.
- `mu_range`: float
    Range of mu (regarding estimated maximum and minimum bounds around true mu).
- `scale_range`: float
    Range of scale (regarding estimated maximum and minimum bounds around true scale).
- `num_processes`: int
    Number that defines in how many sub-processes the fitting process should be split into.
      
##### Returns
- `peaks_mus`: np.ndarray (n, m, max_peaks)
        Array which stores the best found parameters for every pixel (of n*m pixels).
- `peaks_mask`: np.ndarray (n, m, max_peaks, p)
    Array that stores the indices of the measurements that corresponds (mainly) to a peak,
    for every pixel (of n*m pixels).


#### `plot_data_pixels`
##### Description
Plots all the intensity profiles of the pixels of given data.

##### Parameters
- `data`: np.ndarray (n, m, p)
    The image stack containing the measured intensities.
    n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
- `output_params`: np.ndarray (n, m, q)
    The output of fitting the image stack, which stores the parameters of the full fitfunction.
    q = 3 * n_peaks + 1, is the number of parameters (max 19 for 6 peaks).
- `output_peaks_mask`: np.ndarray (n, m, n_peaks, p)
    The mask defining which of the p-measurements corresponds to one of the peaks.
    The first two dimensions are the image dimensions.
- `peak_pairs`: np.ndarray (n, m, 3, 2)
    The peak pairs for every pixel, where the fourth dimension contains both peak numbers of
    a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the third dimension
    is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
    The first two dimensions are the image dimensions.
- `only_mus`: bool
    Defines if only the mus (for every pixel) are given in the output_params.
- `distribution`: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
    The name of the distribution.
- `indices`: np.ndarray (n, m, 2)
    The array storing the x- `and` y-coordinate of one pixel (if plotted data != full data).
- `directory`: string
    The directory path where the plots should be written to.
        
##### Returns
- `None`

#### `calculate_peak_pairs`
##### Description
Calculates all the peak_pairs for a whole image stack..

##### Parameters
- `image_stack`: np.ndarray (n, m, p)
    The image stack containing the measured intensities.
    n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
- `output_params`: np.ndarray (n, m, q)
    The output of fitting the image stack, which stores the parameters of the full fitfunction.
    q = 3 * max_peaks + 1, is the number of parameters (max 19 for 6 peaks).
- `output_peaks_mask`: np.ndarray (n, m, max_peaks, p)
    The mask defining which of the p-measurements corresponds to one of the peaks.
    The first two dimensions are the image dimensions.
- `distribution`: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
    The name of the distribution.
- `only_mus`: bool
    Defines if only the mus (for every pixel) are given in the output_params.
- `num_processes`: int
    Defines the number of processes to split the task into.
        
##### Returns
- `peak_pairs`: np.ndarray (n, m, max_peaks // 2, 2)
    The peak pairs for every pixel, where the fourth dimension contains both peak numbers of
    a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the third dimension
    is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
    The first two dimensions are the image dimensions.


#### `calculate_directions`
##### Description
Calculates the directions from given peak_pairs.

##### Parameters
- `peak_pairs`: np.ndarray (n, m, max_peaks, 2)
    The peak pairs for every pixel, where the fourth dimension contains both peak numbers of
    a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the third dimension
    is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
    The first two dimensions are the image dimensions.
- `output_mus`: np.ndarray (n, m, max_peaks)
    The mus (centers) of the found (max_peaks) peaks for everyone of the (n * m) pixels.
- `directory`: string
    The directory path defining where direction images should be writen to.
    If None, no images will be writen.

##### Returns
- `directions`: (n, m, max_peaks // 2)
    The calculated directions for everyoe of the (n * m) pixels.
    Max 3 directions (for 6 peaks).

### Examples

#### Fit a single pixel
```python
import numpy as np
import h5py
from SLIF import fit_pixel_stack, show_pixel
from SLIF.signal_filters import apply_filter

# Settings:
data_file_path = "/home/user/workspace/SLI_Data.h5"
dataset_path = "pyramid/02"
pixel = [6311, 3300]
distribution = "wrapped_cauchy"
#pre_filter = ["gauss", 2]


# Pick the pixel data
with h5py.File(data_file_path, "r") as h5f:

    intensities = h5f[dataset_path][pixel[0], pixel[1]]
    intensities_err = np.sqrt(intensities)

    # Apply filter
    #intensities = apply_filter(intensities, pre_filter)
    #intensities_err = apply_filter(intensities_err, pre_filter)


# Calculate the angles
angles = np.linspace(0, 2*np.pi, num=len(intensities), endpoint=False)

# Fit the pixel
best_parameters, r_chi2, peaks_mask = fit_pixel_stack(angles, intensities, intensities_err, 
                                                fit_height_nonlinear = True, refit_steps = 1,
                                                n_steps_height = 10, n_steps_mu = 10, n_steps_scale = 10,
                                                n_steps_fit = 10, init_fit_filter = None, 
                                                method="leastsq", distribution = distribution)

print("Optimized parameters:", best_parameters)
print("r_chi2: ", r_chi2)

# Show the pixel
show_pixel(intensities, intensities_err, best_parameters, peaks_mask, distribution)

```

#### Fit image stack
```python
import h5py
from SLIF import fit_image_stack, pick_data
import os

# Settings
dataset_path = "pyramid/00"
data_file_path = "/home/user/workspace/SLI_Data.h5"
output_directory = "output"
output_filename = "output.h5"
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

# Fit the picked data
output_params, output_peaks_mask = fit_image_stack(data, fit_height_nonlinear = True, 
                                threshold = 1000, distribution = distribution,
                                n_steps_fit = 5, n_steps_height = 10, n_steps_mu = 10, 
                                n_steps_scale = 10, refit_steps = 0, init_fit_filter = None, 
                                method="leastsq", num_processes = 2)

# Write the output to a HDF5 file
if output_directory != "" and not os.path.exists(output_directory):
    os.makedirs(output_directory)

with h5py.File(output_directory + output_filename, "w") as h5f:
    group = h5f.create_group(dataset_path)
    group.create_dataset("params", data=output_params)
    group.create_dataset("peaks_mask", data=output_peaks_mask)
    group.create_dataset("indices", data=indices)
```

#### Only find peaks (without fitting)
```python
import h5py
import matplotlib.pyplot as plt
from SLIF import find_image_peaks, calculate_peak_pairs, calculate_directions, pick_data, plot_data_pixels
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
output_mus, output_peaks_mask = find_image_peaks(data, threshold = 1000, init_fit_filter = None, 
                        only_peaks_count = -1, max_peaks = 4, num_processes = 2)


# Optional: Write the output to a HDF5 file
directory = os.path.dirname(output_file_path)
if not os.path.exists(directory):
    os.makedirs(directory)
with h5py.File(output_file_path, "w") as h5f:
    group = h5f.create_group(dataset_path)
    group.create_dataset("mus", data=output_mus)
    group.create_dataset("peaks_mask", data=output_peaks_mask)
    group.create_dataset("indices", data=indices)

# Optional: Pick SLIF output data (if already fitted)
#output_mus, _ = pick_data(output_file_path, 
#                                dataset_path + "/mus", area = None, randoms = 0)
#output_peaks_mask, _ = pick_data(output_file_path, 
#                               dataset_path + "/peaks_mask", area = None, randoms = 0)
    
# Calculate the peak pairs (directions)
peak_pairs = calculate_peak_pairs(data, output_mus, output_peaks_mask, only_mus = True)

# Optional: Plot the picked data
#plot_data_pixels(data, output_mus, output_peaks_mask, peak_pairs, indices = indices, 
#                        directory = "plots", only_mus = True)

# Calculate the nerve fiber directions and save direction map in directory
#directions = calculate_directions(peak_pairs, output_mus, directory = "direction_maps")
```

#### Fit data and calculate directions
```python
import h5py
import matplotlib.pyplot as plt
from SLIF import fit_image_stack, calculate_peak_pairs, calculate_directions, pick_data, plot_data_pixels
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

# Fit the picked data
output_params, output_peaks_mask = fit_image_stack(data, fit_height_nonlinear = True, 
                                threshold = 1000, distribution = distribution,
                                n_steps_fit = 5, n_steps_height = 10, n_steps_mu = 10, 
                                n_steps_scale = 10, refit_steps = 0, init_fit_filter = None, 
                                method="leastsq", num_processes = 2)

# Optional: Write the output to a HDF5 file
directory = os.path.dirname(output_file_path)
if not os.path.exists(directory):
    os.makedirs(directory)
with h5py.File(output_file_path, "w") as h5f:
    group = h5f.create_group(dataset_path)
    group.create_dataset("params", data=output_params)
    group.create_dataset("peaks_mask", data=output_peaks_mask)
    group.create_dataset("indices", data=indices)

# Optional: Pick SLIF output data (if already fitted)
#output_params, _ = pick_data(output_file_path, 
#                                dataset_path + "/params", area = area, randoms = randoms)
#output_peaks_mask, _ = pick_data(output_file_path, 
#                               dataset_path + "/peaks_mask", area = area, randoms = randoms)
    
# Calculate the peak pairs (directions)
peak_pairs = calculate_peak_pairs(data, output_params, output_peaks_mask, distribution)

# Optional: Plot the picked data
plot_data_pixels(data, output_params, output_peaks_mask, peak_pairs, 
                        distribution = distribution, indices = indices, directory = "plots")

# Calculate the nerve fiber directions and save direction map in directory
output_mus = output_params[:, :, 1::3]
directions = calculate_directions(peak_pairs, output_mus, directory = "direction_maps")
```








