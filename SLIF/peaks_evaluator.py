import numpy as np
import h5py
from numba import njit
from SLIF import full_fitfunction, angle_distance
from wrapped_distributions import distribution_pdf
from collections import deque
import os
import imageio

similarity_weights = [1, 1]
dataset_path = "pyramid/00"
distribution = "wrapped_cauchy"

@njit(cache = True, fastmath = True)
def calculate_peaks_gof(intensities, model_y, peaks_mask, method = "nrmse"):
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

def calculate_peak_pairs(data, output_params, output_peaks_mask, distribution):

    output_heights = output_params[:, :, 0:-1:3]
    output_mus = output_params[:, :, 1::3]
    output_scales = output_params[:, :, 2::3]

    output_amplitudes = np.zeros(output_heights.shape)

    for i in range(output_heights.shape[0]):
        for j in range(output_heights.shape[1]):
            for k in range(output_heights.shape[2]):
                if output_heights[i][j][k] == 0:
                    output_amplitudes[i][j][k] = 0
                else:
                    output_amplitudes[i][j][k] = output_heights[i][j][k] / (np.pi * output_scales[i][j][k])

    #directions = np.full((output_mus.shape[0], output_mus.shape[1], 2), -1, dtype=np.float64)
    peak_pairs = np.full((output_mus.shape[0], output_mus.shape[1], 2, 2), -1, dtype=np.int64)

    # First calculate for one and two peak pixels, then 3, then 4:
    for p in range(2, 5):
        for i in range(output_mus.shape[0]):
            for j in range(output_mus.shape[1]):
                intensities = data[i][j]
                intensities_err = np.sqrt(intensities)
                angles = np.linspace(0, 2*np.pi, num=len(intensities), endpoint=False)

                params = output_params[i][j]
                heights = params[0:-1:3]
                scales = params[2::3]
                mus = params[1::3]
                mus = mus[heights >= 1]
                num_peaks = len(mus)
                if num_peaks == 0: continue
                offset = params[-1]
                params = params[:(3 * len(mus))]
                params = np.append(params, offset)

                model_y = full_fitfunction(angles, params, distribution)
                peaks_mask = output_peaks_mask[i][j]
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
                global_amplitude = np.max(intensities) - np.min(intensities)
                # Only consider peaks with height not zero and amplitude over 15% of global amplitude
                # and gof over 0.5
                condition = (heights >= 1) & (amplitudes > 0.2 * global_amplitude) & (peaks_gof > 0.5) \
                    & (amplitudes > 0.2 * np.max(amplitudes)) & (rel_amplitudes > 0.05 * global_amplitude)
                mus = mus[condition]
                sig_peak_indices = condition.nonzero()[0]
                num_sig_peaks = len(mus)
                if (num_sig_peaks != 1 and num_sig_peaks != p) or num_sig_peaks == 0: continue
                if num_sig_peaks == 1:
                    peak_pairs[i][j][0] = sig_peak_indices[0], -1
                    #directions[i][j] = mus
                    continue
                elif num_sig_peaks == 2:
                    peak_pairs[i][j][0] = sig_peak_indices
                    #distance = angle_distance(mus[0], mus[1])
                    #directions[i][j][0] = (mus[0] + distance / 2) % (np.pi)
                    continue
                elif num_sig_peaks == 3 or num_sig_peaks == 4:
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
                            #directions[i][j][0] = best_direction
                            peak_pairs[i][j][0] = best_pair[0], best_pair[1]
                            indices = np.array(range(num_sig_peaks))
                            remaining_indices = np.setdiff1d(indices, best_pair)
                            #if len(remaining_indices) == 1 and num_peaks == num_sig_peaks:
                            #    peak_pairs[i][j][1] = remaining_indices[0], -1
                            if len(remaining_indices) == 2:
                                peak_pairs[i][j][1] = sig_peak_indices[remaining_indices[0]], \
                                                        sig_peak_indices[remaining_indices[1]]
                                #distance = angle_distance[remaining_mus[0], remaining_mus[1]]
                                #directions[i][j][1] = (remaining_mus[0] + distance / 2) % (np.pi)
                            break
                        else:
                             mask[row][col] = False

    return peak_pairs

def calculate_directions(peak_pairs, output_mus, dictionary = None):
    x_range = peak_pairs.shape[0]
    y_range = peak_pairs.shape[1]
    directions = np.full((x_range, y_range, 2), -1, dtype=np.float64)
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

    if dictionary != None:
        if not os.path.exists(dictionary):
                os.makedirs(dictionary)

        for dir_n in range(directions.shape[-1]):
            imageio.imwrite(f'{dictionary}/dir_{dir_n + 1}.tiff', np.swapaxes(directions[:, :, dir_n], 0, 1))

    return directions

def direction_significance(peak_pair, peaks_gof, amplitudes, intensities, weights = [1, 1]):
    global_amplitude = np.max(intensities) - np.min(intensities)
    amplitude_significance = np.mean(amplitudes / global_amplitude)
    gof_significance = np.mean(peaks_gof)
    
    significance = (amplitude_significance * weights[0] + gof_significance * weights[1]) / 2

    return significance

