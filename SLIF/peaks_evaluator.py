import numpy as np
import h5py
from numba import njit
from .SLIF import full_fitfunction, angle_distance
from .wrapped_distributions import distribution_pdf
from collections import deque
import os
import imageio
import pymp
from tqdm import tqdm
import multiprocessing

@njit(cache = True, fastmath = True)
def calculate_peaks_gof(intensities, model_y, peaks_mask, method = "nrmse"):
    """
    Calculate the goodness-of-fit for given peaks.

    Parameters:
    - intensities: np.ndarray (n, )
        The measured intensities (y-data) from SLI with n-measurements of one pixel.
    - model_y: np.ndarray (n, )
        The fitted (model) intensities with n-measurements of one pixel
    - peaks_mask: np.ndarray (m, n)
        m is the number of peaks.
        The mask defining which of the n-measurements corresponds to one of the m-peaks.

    Returns:
    - peaks_gof: np.ndarray (m, )
        The goodness-of-fit values, one value for each of the m-peaks
    """

    number_of_peaks = len(peaks_mask)
    if number_of_peaks == 0:
        return np.zeros(1)
    peaks_gof = np.empty(number_of_peaks)
    for peak_number in range(number_of_peaks):

        mask = peaks_mask[peak_number]
        peak_intensities = intensities[mask]
        peak_model_y = model_y[mask]

        if np.all(peak_model_y < 1):
            peaks_gof[peak_number] = 0
        elif method == "nrmse":
            # normalized peak data
            intensity_range = np.max(intensities) - np.min(intensities)
            peak_intensities = peak_intensities / intensity_range
            peak_model_y = peak_model_y / intensity_range

            # calculate normalized root mean squared error (NRMSE)
            residuals = peak_intensities - peak_model_y
            peak_nrmse = np.sqrt(np.mean(residuals**2))
            peaks_gof[peak_number] = max(1 - peak_nrmse, 0)
        elif method == "r2":
            ss_res = np.sum((peak_intensities - peak_model_y) ** 2)
            ss_tot = np.sum((peak_intensities - np.mean(peak_intensities)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            peaks_gof[peak_number] = max(r2, 0)
        elif method == "mae":
            intensity_range = np.max(intensities) - np.min(intensities)
            peak_intensities = peak_intensities / intensity_range
            peak_model_y = peak_model_y / intensity_range
            mae = np.mean(np.abs(peak_intensities - peak_model_y))
            peaks_gof[peak_number] = max(1 - mae, 0)

    return peaks_gof

import numpy as np

@njit(cache = True, fastmath = True)
def _find_closest_true_pixel_numba(mask, start_pixel):
    rows, cols = mask.shape
    visited = np.zeros_like(mask, dtype = np.bool_)
    queue = np.empty((rows * cols, 2), dtype = np.int32)
    queue_idx = 0
    queue[queue_idx] = start_pixel
    queue_idx += 1

    front = 0
    end = queue_idx

    while front != end:
        r, c = queue[front]
        front = (front + 1) % (rows * cols)
        
        if mask[r, c]:
            return (r, c)
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                visited[nr, nc] = True
                queue[queue_idx] = (nr, nc)
                queue_idx = (queue_idx + 1) % (rows * cols)
                end = (end + 1) % (rows * cols)
    
    return (-1, -1)


def _find_closest_true_pixel(mask, start_pixel):
    """
    Finds the closest true pixel for a given 2d-mask and a start_pixel.

    Parameters:
    - mask: np.ndarray (n, m)
        The boolean mask defining which pixels are False or True.
    - start_pixel: tuple
        The x- and y-coordinates of the start_pixel.

    Returns:
    - closest_true_pixel: tuple
        The x- and y-coordinates of the closest true pixel.
    """

    rows, cols = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    queue = deque([start_pixel])

    while queue:
        r, c = queue.popleft()
        
        if mask[r, c]:
            return (r, c)
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))
    
    # When no true pixel in the mask, return (-1, -1)
    return (-1, -1)


def calculate_peak_pairs(image_stack, output_params, output_peaks_mask, 
                            distribution = "wrapped_cauchy", only_mus = False, num_processes = 2):
    """
    Calculates all the peak_pairs for a whole image stack.

    Parameters:
    - image_stack: np.ndarray (n, m, p)
        The image stack containing the measured intensities.
        n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
    - output_params: np.ndarray (n, m, q)
        The output of fitting the image stack, which stores the parameters of the full fitfunction.
        q = 3 * n_peaks + 1, is the number of parameters (max 19 for 6 peaks).
    - output_peaks_mask: np.ndarray (n, m, n_peaks, p)
        The mask defining which of the p-measurements corresponds to one of the peaks.
        The first two dimensions are the image dimensions.
    - distribution: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
        The name of the distribution.
    - only_mus: bool
        Defines if only the mus (for every pixel) are given in the output_params.
    - num_processes: int
        Defines the number of processes to split the task into.

    Returns:
    - peak_pairs: np.ndarray (n, m, 3, 2)
        The peak pairs for every pixel, where the fourth dimension contains both peak numbers of
        a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the third dimension
        is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
        The first two dimensions are the image dimensions.
    """

    n_rows = image_stack.shape[0]
    n_cols = image_stack.shape[1]
    total_pixels = n_rows * n_cols
    flattened_stack = image_stack.reshape((total_pixels, image_stack.shape[2]))
    flattened_params = output_params.reshape((total_pixels, output_params.shape[2]))
    flattened_peaks_mask = output_peaks_mask.reshape((total_pixels, 
                                            output_peaks_mask.shape[2], output_peaks_mask.shape[3]))

    peak_pairs = pymp.shared.array((total_pixels, 3, 2), dtype = np.int16)
    with pymp.Parallel(num_processes) as p:
        for i in p.range(peak_pairs.shape[0]):
            peak_pairs[i, :] = -1

    direction_mask = pymp.shared.array((n_rows, n_cols), dtype = np.bool_)

    if not only_mus:
        output_mus = output_params[:, :, 1::3]
    else:
        output_mus = output_params

    # Initialize the progress bar
    pbar = tqdm(total = total_pixels, desc = "Calculating Peak pairs", smoothing = 0, leave = True)
    shared_counter = pymp.shared.array(1, dtype = int)
    shared_counter[0] = 0

    # First calculate for one and two peak pixels, then 3, then 4:
    for peak_iteration in range(2, 5):
        with pymp.Parallel(num_processes) as p:
            for i in p.range(total_pixels):
                intensities = flattened_stack[i]
                angles = np.linspace(0, 2*np.pi, num=len(intensities), endpoint=False)
                params = flattened_params[i]
                peaks_mask = flattened_peaks_mask[i]

                if not only_mus:
                    heights = params[0:-1:3]
                    scales = params[2::3]
                    mus = params[1::3]
                    mus = mus[heights >= 1]
                    num_peaks = len(mus)
                else:
                    mus = params
                    if not np.any(peaks_mask[0] != 0):
                        num_peaks = 0
                
                # Update progress bar
                shared_counter[0] += 1
                pbar.update(shared_counter[0] - pbar.n)

                if num_peaks == 0: 
                    continue

                global_amplitude = np.max(intensities) - np.min(intensities)
                if not only_mus:
                    offset = params[-1]
                    params = params[:(3 * len(mus))]
                    params = np.append(params, offset)

                    model_y = full_fitfunction(angles, params, distribution)
                    peaks_gof = calculate_peaks_gof(intensities, model_y, peaks_mask, method = "r2")

                    scales = scales[heights >= 1]
                    peaks_gof = peaks_gof[heights >= 1]
                    heights = heights[heights >= 1]
                    amplitudes = np.empty(len(heights))
                    rel_amplitudes = np.empty(len(heights))
                    for k in range(len(heights)):
                        amplitudes[k] = heights[k] * distribution_pdf(0, 0, scales[k], distribution)
                        rel_amplitudes[k] = 2 * amplitudes[k] \
                                    - full_fitfunction(mus[k], params, distribution) + params[-1]
                    # Only consider peaks with height not zero and amplitude over 15% of global amplitude
                    # and gof over 0.5
                    condition = (heights >= 1) & (amplitudes > 0.2 * global_amplitude) & (peaks_gof > 0.5) \
                        & (amplitudes > 0.2 * np.max(amplitudes)) & (rel_amplitudes > 0.05 * global_amplitude)
                else:
                    amplitudes = np.empty(len(mus))
                    for k in range(len(mus)):
                        amplitudes[k] = np.max(intensities[peaks_mask[k]])
                    condition = (amplitudes > 0.2 * global_amplitude) & (amplitudes > 0.2 * np.max(amplitudes))
                mus = mus[condition]
                sig_peak_indices = condition.nonzero()[0]
                num_sig_peaks = len(mus)
                if (num_sig_peaks != 1 and num_sig_peaks != peak_iteration) or num_sig_peaks == 0: 
                    continue
                if num_sig_peaks == 1:
                    peak_pairs[i, 0] = sig_peak_indices[0], -1
                    continue
                elif num_sig_peaks == 2:
                    peak_pairs[i, 0] = sig_peak_indices
                    direction_mask[i // n_cols, i % n_cols] = True
                    continue
                elif num_sig_peaks >= 3:
                    # Get closest pixel with a defined direction (and minimum 2 peaks)

                    mask = np.copy(direction_mask)
                    while True:

                        row, col = _find_closest_true_pixel(mask, (i // n_cols, i % n_cols))
                        if row == -1 and col == -1: break
                        other_mus = output_mus[row, col]

                        best_direction = -1
                        best_pair = [-1, -1]
                        min_direction_distance = np.inf
                        for peak_pair in peak_pairs[row * n_cols + col]:
                            if peak_pair[0] == -1 or peak_pair[1] == -1: continue
                            distance = angle_distance(other_mus[peak_pair[0]], other_mus[peak_pair[1]])
                            direction = (other_mus[peak_pair[0]] + distance / 2) % (np.pi)
                            for index in range(num_sig_peaks):
                                for index2 in range(index + 1, num_sig_peaks):
                                    distance = angle_distance(mus[index], mus[index2])
                                    test_direction = (mus[index] + distance / 2) % (np.pi)
                                    direction_distance = np.abs(angle_distance(test_direction, direction))
                                    if direction_distance < min_direction_distance:
                                        min_direction_distance = direction_distance
                                        best_pair = [sig_peak_indices[index], sig_peak_indices[index2]]
                                        best_direction = direction
                        if min_direction_distance < np.pi / 8:
                            peak_pairs[i, 0] = best_pair[0], best_pair[1]
                            direction_mask[i // n_cols, i % n_cols] = True
                            indices = np.array(range(num_sig_peaks))
                            remaining_indices = np.setdiff1d(indices, best_pair)
                            #if len(remaining_indices) == 1 and num_peaks == num_sig_peaks:
                            #    peak_pairs[i, 1] = remaining_indices[0], -1
                            if len(remaining_indices) == 2:
                                peak_pairs[i, 1] = sig_peak_indices[remaining_indices[0]], \
                                                        sig_peak_indices[remaining_indices[1]]
                            break
                        else:
                             mask[row, col] = False

    peak_pairs = np.reshape(peak_pairs, (n_rows, n_cols, 3, 2))

    return peak_pairs

def calculate_directions(peak_pairs, output_mus, directory = None):
    """
    Calculates the directions from given peak_pairs.

    Parameters:
    - peak_pairs: np.ndarray (n, m, 3, 2)
        The peak pairs for every pixel, where the fourth dimension contains both peak numbers of
        a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the third dimension
        is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
        The first two dimensions are the image dimensions.
    - output_mus: np.ndarray (n, m, n_peaks)
        The mus (centers) of the found (n_peaks) peaks for everyone of the (n * m) pixels.
    - directory: string
        The directory path defining where direction images should be writen to.
        If None, no images will be writen.

    Returns:
    - directions: (n, m, 3)
        The calculated directions for everyoe of the (n * m) pixels.
        Max 3 directions (for 6 peaks).
    """

    x_range = peak_pairs.shape[0]
    y_range = peak_pairs.shape[1]
    directions = np.full((x_range, y_range, 3), -1, dtype=np.float64)
    for x in range(x_range):
        for y in range(y_range):
            mus = output_mus[x, y]
            for k, pair in enumerate(peak_pairs[x, y]):
                if pair[0] == -1 and pair[1] == -1:
                    # No pair
                    continue
                elif pair[0] == -1 or pair[1] == -1:
                    # One peak direction
                    direction = mus[pair[pair != -1][0]] % (np.pi)
                else:
                    # Two peak direction
                    distance = angle_distance(mus[pair[0]], mus[pair[1]])
                    direction = (mus[pair[0]] + distance / 2) % (np.pi)

                directions[x, y, k] = direction

    if directory != None:
        if not os.path.exists(directory):
                os.makedirs(directory)

        for dir_n in range(directions.shape[-1]):
            imageio.imwrite(f'{directory}/dir_{dir_n + 1}.tiff', np.swapaxes(directions[:, :, dir_n], 0, 1))

    return directions

def direction_significance(peak_pair, peaks_gof, amplitudes, intensities, weights = [1, 1]):
    """
    Calculates the significance of one direction for one pixel.

    Parameters:
    - peak_pair: np.ndarray (2, )
        The peak pair containing both number of peaks (e.g. [1, 3], for peak number 1 and 3 beeing paired).
    - peaks_gof: np.ndarray (2, )
        The goodness of fit values for both peaks.
    - amplitudes: np.ndarray (2, )
        The amplitudes of the peaks.
    - intensities: np.ndarray (n, )
        The measured intensities of the pixel.
    - weights: array (2, )
        The weights for the amplitude and for the goodnes-of-fit, when calculating the significance

    Returns:
    - significance: float
        The calculated significance ranging from 0 to 1.
    """
    global_amplitude = np.max(intensities) - np.min(intensities)
    amplitude_significance = np.mean(amplitudes / global_amplitude)
    gof_significance = np.mean(peaks_gof)
    
    significance = (amplitude_significance * weights[0] + gof_significance * weights[1]) / 2

    return significance

