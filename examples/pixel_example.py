import numpy as np
import h5py
from SLIF import fit_pixel_stack, show_pixel
from SLIF.signal_filters import apply_filter

# Settings:
filename = "SLI_Data.h5"
dataset_path = "pyramid/02"
pixel = [1250, 734]
distribution = "wrapped_cauchy"
#pre_filter = ["gauss", 2]


# Pick the pixel data
with h5py.File(filename, "r") as h5f:

    intensities = h5f[dataset_path][pixel[0]][pixel[1]]
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
