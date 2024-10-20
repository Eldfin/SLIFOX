import numpy as np
from numba import njit
import matplotlib as plt
import pymp
from .utils import angle_distance
from .fitter import fit_image_stack, full_fitfunction, find_image_peaks
from itertools import combinations, chain
from tqdm import tqdm

def get_distance_deviations(image_stack, fit_SLI = False, num_pixels = 0, 
                            distribution = "wrapped_cauchy", threshold = 1000, num_processes = 2):
    """
    Gets the deviations from 180 degree distance between two peaks pixels.
    Usefull to analyze how many flat nerve fibers are in the data
    and if more than 2 peaks can be paired to 180 degree pairs.
    
    Parameters:
    - image_stack: np.ndarray (n, m, p)
        SLI measurement data.
    - fit_SLI: boolean
        Defines whether to fit the data to find the peak centers or just to use the peak finder.
    - num_pixels: int
        The number of random pixels to pick from the data.
    - distribution: "wrapped_cauchy", "von_mises", or "wrapped_laplace"
        The name of the distribution.
    - threshold: int
        Threshold value. If the mean intensity of one pixel is lower than that threshold value,
        the pixel will not be evaluated.
    - num_processes: int
        Number that defines in how many sub-processes the fitting process should be split into.

    Returns:
    - distance_deviations: np.ndarray (q, )
        Virtual direction value of the PLI measurement.
    - full_pixel_mask: np.ndarray (n, m)
        Boolean array defining which pixels of the "image_stack" are used.

    """

    if num_pixels > 0:
        # Pick random pixels
        x_indices = np.random.randint(0, image_stack.shape[0], num_pixels)
        y_indices = np.random.randint(0, image_stack.shape[1], num_pixels)
        data = image_stack[x_indices, y_indices].reshape((num_pixels, 1, image_stack.shape[-1]))
    else:
        data = np.copy(image_stack)
    
    if fit_SLI:
        # Fit all pixels with exactly 2 peaks
        output_params, output_peaks_mask = fit_image_stack(data, threshold = threshold,
                        n_steps_height = 10, n_steps_mu = 10, n_steps_scale = 5,
                            n_steps_fit = 5, refit_steps = 0,
                            init_fit_filter = None, method = "leastsq", 
                            only_peaks_count = 2, max_peaks = 2,
                            num_processes = num_processes)

        # Get the mus and flatten the image dimensions
        output_mus = output_params[:, :, 1::3]
    else:
        output_mus, output_peaks_mask = find_image_peaks(data, threshold = 1000, 
                        only_peaks_count = 2, max_peaks = 2, num_processes = 2)

    angles = np.linspace(0, 2*np.pi, num=image_stack.shape[2], endpoint = False)
    pixel_mask = np.zeros(data.shape[:2], dtype = np.bool_)
    full_pixel_mask = np.zeros(image_stack.shape[:2], dtype = np.bool_)
    
    for i, pixel in enumerate(np.ndindex(data.shape[:2])):
        intensities = data[pixel]
        peaks_mask = output_peaks_mask[pixel]
        if not np.any(peaks_mask): continue
        
        if fit_SLI:
            params = output_params[pixel]
            model_y = full_fitfunction(angles, params, distribution)
            peaks_gof = calculate_peaks_gof(intensities, model_y, peaks_mask, method = "nrmse")
            if np.all(peaks_gof > 0.8):
                pixel_mask[pixel] = True
                if num_pixels > 0:
                    full_pixel_mask[x_indices[i], y_indices[i]] = True
                else:
                    full_pixel_mask[pixel] = True
        else:
            peaks_mus = output_mus[pixel]
            # global maximum has to be around one mu
            index_maximum = np.argmax(intensities)
            min_dist_max = np.min(np.abs(angle_distance(peaks_mus, angles[index_maximum])))
            if (peaks_mask[0, index_maximum] or peaks_mask[1, index_maximum]) \
                    and min_dist_max < 20 * np.pi / 180:
                pixel_mask[pixel] = True
                if num_pixels > 0:
                    full_pixel_mask[x_indices[i], y_indices[i]] = True
                else:
                    full_pixel_mask[pixel] = True

    mus = output_mus[pixel_mask]

    distance_deviations = np.pi - np.abs(angle_distance(mus[:, 0], mus[:, 1]))

    return distance_deviations, full_pixel_mask