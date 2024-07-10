import numpy as np
import h5py
from numba import njit
from .SLIF import full_fitfunction, angle_distance
from .wrapped_distributions import distribution_pdf
from collections import deque
import os
import imageio

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

def calculate_peak_pairs(data, output_params, output_peaks_mask, distribution, only_mus = False):
    """
    Calculates all the peak_pairs for a whole image stack.

    Parameters:
    - data: np.ndarray (n, m, p)
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

    Returns:
    - peak_pairs: np.ndarray (n, m, 3, 2)
        The peak pairs for every pixel, where the fourth dimension contains both peak numbers of
        a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the third dimension
        is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
        The first two dimensions are the image dimensions.
    """

    if not only_mus:
        output_mus = output_params[:, :, 1::3]
    else:
        output_mus = output_params

    #directions = np.full((output_mus.shape[0], output_mus.shape[1], 2), -1, dtype=np.float64)
    peak_pairs = np.full((output_mus.shape[0], output_mus.shape[1], 3, 2), -1, dtype=np.int64)

    # First calculate for one and two peak pixels, then 3, then 4:
    for p in range(2, 5):
        for i in range(output_mus.shape[0]):
            for j in range(output_mus.shape[1]):
                intensities = data[i][j]
                intensities_err = np.sqrt(intensities)
                angles = np.linspace(0, 2*np.pi, num=len(intensities), endpoint=False)

                if not only_mus:
                    params = output_params[i][j]
                    heights = params[0:-1:3]
                    scales = params[2::3]
                    mus = params[1::3]
                else:
                    mus = output_mus[i][j]
                mus = mus[heights >= 1]
                num_peaks = len(mus)
                if num_peaks == 0: continue

                peaks_mask = output_peaks_mask[i][j]
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
                    amplitudes = np.empty(len(mus)
                    for k in range(len(mus))):
                        amplitudes = np.max(intensities[peaks_mask])
                    condition = (amplitudes > 0.2 * global_amplitude) & (amplitudes > 0.2 * np.max(amplitudes))
                mus = mus[condition]
                sig_peak_indices = condition.nonzero()[0]
                num_sig_peaks = len(mus)
                if (num_sig_peaks != 1 and num_sig_peaks != p) or num_sig_peaks == 0: continue
                if num_sig_peaks == 1:
                    peak_pairs[i][j][0] = sig_peak_indices[0], -1
                    continue
                elif num_sig_peaks == 2:
                    peak_pairs[i][j][0] = sig_peak_indices
                    continue
                elif num_sig_peaks >= 3:
                    # Get closest pixel with defined direction

                    # Create mask of pixels where at least 2 peaks are paired
                    mask = ((peak_pairs != -1).sum(axis = 3) >= 2).any(axis = 2)
                    #mask = (directions != -1).any(axis=2)
                    while True:

                        row, col = _find_closest_true_pixel(mask, (i, j))
                        if row == -1 and col == -1: break
                        other_mus = output_mus[row][col]

                        best_direction = -1
                        best_pair = [-1, -1]
                        min_direction_distance = np.inf
                        for peak_pair in peak_pairs[row][col]:
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
                            peak_pairs[i][j][0] = best_pair[0], best_pair[1]
                            indices = np.array(range(num_sig_peaks))
                            remaining_indices = np.setdiff1d(indices, best_pair)
                            #if len(remaining_indices) == 1 and num_peaks == num_sig_peaks:
                            #    peak_pairs[i][j][1] = remaining_indices[0], -1
                            if len(remaining_indices) == 2:
                                peak_pairs[i][j][1] = sig_peak_indices[remaining_indices[0]], \
                                                        sig_peak_indices[remaining_indices[1]]
                            break
                        else:
                             mask[row][col] = False

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
            mus = output_mus[x][y]
            for k, pair in enumerate(peak_pairs[x][y]):
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

                directions[x][y][k] = direction

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

