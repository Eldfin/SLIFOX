import numpy as np
from numba import njit
import matplotlib as plt
from .utils import angle_distance
from .SLIF import fit_image_stack, full_fitfunction, find_image_peaks

def add_birefringence(
    dir_1,
    ret_1,
    dir_2,
    ret_2,
    symmetric=True,
):
    """Add birefringence from 1 to 2"""
    dir_1 = np.asarray(dir_1)
    ret_1 = np.asarray(ret_1)
    dir_2 = np.asarray(dir_2)
    ret_2 = np.asarray(ret_2)

    if symmetric:
        delta_1 = np.arcsin(ret_1)
        delta_2 = np.arcsin(ret_2)
        real_part_1, im_part_1 = mod2cplx(dir_1, np.sin(delta_1 / 2) * np.cos(delta_2 / 2))
        real_part_2, im_part_2 = mod2cplx(dir_2, np.sin(delta_2 / 2) * np.cos(delta_1 / 2))
        dir_new, ret_new = cplx2mod(real_part_1 + real_part_2, im_part_1 + im_part_2)
        ret_new = np.sin(np.arcsin(ret_new) * 2)
    else:
        delta_1 = np.arcsin(ret_1)
        delta_2 = np.arcsin(ret_2)
        real_part_1, im_part_1 = mod2cplx(dir_1, np.sin(delta_1) * np.cos(delta_2))
        real_part_2, im_part_2 = mod2cplx(dir_2, np.cos(delta_1 / 2) ** 2 * np.sin(delta_2))
        real_part_3, im_part_3 = mod2cplx(
            2 * dir_1 - dir_2, -np.sin(delta_1 / 2) ** 2 * np.sin(delta_2)
        )
        dir_new, ret_new = cplx2mod(
            real_part_1 + real_part_2 + real_part_3,
            im_part_1 + im_part_2 + im_part_3,
        )

    return dir_new, ret_new

@njit(cache = True, fastmath = True)
def cplx2mod(real_part, im_part, scale=2.0):
    """Convert complex number to direction and retardation"""
    retardation = np.sqrt(real_part**2 + im_part**2)
    direction = np.arctan2(im_part, real_part) / scale
    
    return direction, retardation

@njit(cache = True, fastmath = True)
def mod2cplx(direction, retardation, scale=2.0):
    """Convert direction and retardation to complex number"""
    im_part = retardation * np.sin(scale * direction)
    real_part = retardation * np.cos(scale * direction)
    return real_part, im_part

def get_distance_deviations(image_stack, fit_SLI = False, num_pixels = 0, 
                            distribution = "wrapped_cauchy", threshold = 1000, num_processes = 2):

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
