import numpy as np
import h5py
from numba import njit, prange
from .fitter import full_fitfunction, angle_distance
from .wrapped_distributions import distribution_pdf, wrapped_cauchy_pdf
from .utils import find_closest_true_pixel, add_birefringence, calculate_inclination, \
                    calculate_birefringence, calculate_retardation
from scipy.spatial import KDTree
import os
import imageio
import pymp
from tqdm import tqdm

#@njit(cache = True, fastmath = True)
def calculate_peaks_gof(intensities, model_y, peaks_mask, method = "r2"):
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

    # Ensure intensity dtype is sufficient for calculations
    if intensities.dtype != np.int32:
        intensities = intensities.astype(np.int32)

    peaks_gof = np.zeros(peaks_mask.shape[:-1])
    peak_intensities = np.where(peaks_mask, np.expand_dims(intensities, axis = -2), np.nan)
    peak_model_y = np.where(peaks_mask, np.expand_dims(model_y, axis = -2), np.nan)

    valid_peaks = np.any(peaks_mask, axis = -1)
    # where no valid peak set all peak intensities and model_y to zero so peaks_gof will be zero
    peak_intensities = np.where(valid_peaks[..., np.newaxis], peak_intensities, 0)
    peak_model_y = np.where(valid_peaks[..., np.newaxis], peak_model_y, 0)

    if method == "r2":
        ss_res = np.nansum((peak_intensities - peak_model_y) ** 2, axis = -1)
        mean_peak_intensities = np.nanmean(peak_intensities, axis = -1)
        ss_tot = np.nansum((peak_intensities - mean_peak_intensities[..., np.newaxis]) ** 2, axis = -1)

        # replace zeros in ss_tot with np.inf for division
        #ss_tot = np.where(ss_tot == 0, np.inf, ss_tot)
        #peaks_gof = np.where(valid_peaks, 1 - ss_res / ss_tot, 0)
        ss_tot[ss_tot == 0] = 1
        
        peaks_gof[valid_peaks] = 1 - ss_res[valid_peaks] / ss_tot[valid_peaks]
        peaks_gof = np.where(peaks_gof < 0, 0, peaks_gof)
    elif method == "nrmse":
        intensity_range = np.nanmax(intensities, axis = -1) - np.nanmin(intensities, axis = -1)
        peak_intensities = peak_intensities / intensity_range[..., np.newaxis]
        peak_model_y = peak_intensities / intensity_range[..., np.newaxis]

        # calculate normalized root mean squared error (NRMSE)
        # Where no valid peak set all residuals to zero, so peaks_gof is zero
        residuals = peak_intensities - peak_model_y
        peaks_gof = np.sqrt(np.nanmean(residuals**2, axis = -1))
    elif method == "mae":
        intensity_range = np.nanmax(intensities, axis = -1) - np.nanmin(intensities, axis = -1)
        peak_intensities = peak_intensities / intensity_range[..., np.newaxis]
        peak_model_y = peak_intensities / intensity_range[..., np.newaxis]
    
        peaks_gof = 1 - np.nanmean(np.abs(peak_intensities - peak_model_y), axis = -1)
        peaks_gof = np.where(peaks_gof < 0, 0, peaks_gof)

    return peaks_gof

@njit(cache = True, fastmath = True)
def calculate_image_peaks_gof(image_stack, image_model_y, peaks_mask, method = "nrmse"):
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

def get_image_peak_pairs(image_stack, image_params, image_peaks_mask, method = "neighbor", 
                            min_distance = 30, max_distance = 180, 
                            distribution = "wrapped_cauchy", only_mus = False, 
                            num_processes = 2, amplitude_threshold = 1000, rel_amplitude_threshold = 0.1,
                            gof_threshold = 0.5, significance_threshold = 0,
                            nb_significance_threshold = 0.9, 
                            significance_weights = [1, 1], significance_sens = [1, 1],
                            max_paired_peaks = 4,
                            nb_diff_threshold = 5, pli_diff_threshold = 5, 
                            pli_ret_diff_threshold = 1, max_attempts = 100, 
                            search_radius = 50, min_directions_diff = 20, exclude_lone_peaks = True,
                            image_num_peaks = None, image_sig_peaks_mask = None,
                            image_pli_directions = None, image_pli_retardations = None):
    """
    Finds all the peak_pairs for a whole image stack and sorts them by comparing with neighbor pixels.

    Parameters:
    - image_stack: np.ndarray (n, m, p)
        The image stack containing the measured intensities.
        n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
    - image_params: np.ndarray (n, m, q)
        The output of fitting the image stack, which stores the parameters of the full fitfunction.
        q is the number of parameters. Can also store only mus, when only_mus = True.
    - image_peaks_mask: np.ndarray (n, m, max_find_peaks, p)
        The mask defining which of the p-measurements corresponds to one of the peaks.
        The first two dimensions are the image dimensions.
    - method: string or list of strings
        Method that is used to sort the possible peak pair combinations.  
        Can be "single", "neighbor", "pli", "significance" or "random".  
        "single" will only return a combination if there is only one possible (no sorting).
        "neighbor" will sort the possible peak pair combinations by neighbouring peak pairs.
        "pli" will sort the possible peak pair combinations by given 3d-pli measurement data.
        "significance" will sort the peak pair combinations by direction significance.
        "random" will sort the peak pair combinations randomly.
        Can also be a list containing multiple methods that are used in order.
        E.g. ["neighbor", "significance"] will sort remaining combinations of a pixel by significance 
        if sorting by neighbor was not sucessfull.
    - min_distance: float or list of floats
        Defines the minimum (180 degree periodic) distance between two paired peaks in degree.
        Can also be a list to define different values for pixels with different number of peaks.
        E.g. [30, 60, 120] for [2, 3, >4] Peaks.
    - max_distance: float or list of floats
        Defines the maximum (180 degree periodic) distance between two paired peaks in degree.
        Default is 180 degree (no limit).
        Can also be a list to define different values for pixels with different number of peaks.
        E.g. [30, 60, 120] for [2, 3, >4] Peaks.
    - distribution: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
        The name of the distribution used for calculation of the goodness-of-fit for gof thresholding.
    - only_mus: bool
        Defines if only the mus (for every pixel) are given in the image_params.
    - num_processes: int
        Defines the number of processes to split the task into.
    - amplitude_threshold: float
        Peaks with a amplitude below this threshold will not be evaluated.
    - rel_amplitude_threshold: float
        Value between 0 and 1.
        Peaks with a relative amplitude (to maximum - minimum intensity of the pixel) below
        this threshold will not be evaluated.
    - gof_threshold: float
        Value between 0 and 1. If greater than 0, only fitted peaks can be paired.
        Peaks with a goodness-of-fit value below this threshold will not be evaluated.
    - significance_threshold: float
        Value between 0 and 1. Peak Pairs with peaks that have a significance
        lower than this threshold are not considered for possible pairs.
        This Value should stay low, so that the number of possible pairs is not reduced too much.
        A too high value can lead to wrong pairs, because of pairing only "good" peaks.
        See also "direction_significance" function for more info.
    - nb_significance_threshold: float
        Value between 0 and 1. Neighboring directions with a lower significance are not considered
        as match for the neighboring method. This threshold should be high, when using
        the neighbor method. Lower value will lead to faster computing times, but increased
        probability of wrong pairs.
    - significance_weights: list (2, )
        The weights for the amplitude (first value) and for the goodnes-of-fit (second value), 
        when calculating the significance.
        See also "direction_significance" function for more info.
    - significance_sens: list (2, )
        The sensitivity values for the amplitude (first value) and for the goodness-of-fit (second value),
        when calculating the significance.
    - max_paired_peaks: int
        Defines the maximum number of peaks that are paired.
        Value has to be smaller or equal the number of peaks in image_params (and max 6)
        (max_paired_peaks <= max_fit_peaks or max_find_peaks)
    - nb_diff_threshold: float
        Threshold in degrees defining when a neighboring pixel direction is used to pair peaks
        with the "neighbor" method.
    - pli_diff_threshold: float
        If the difference between the measured PLI direction and the calculated PLI direction from SLI
        is larger than this threshold, the "pli" method will not return any peak pair combination.
    - pli_ret_diff_threshold: float
        The difference between the single fiber retardations and the pli retardation must be smaller
        than this value to count the peak pair combination as valid in the "pli" method.
    - max_attempts: int
        Number defining how many times it should be attempted to find a neighboring pixel 
        with a direction difference lower than the given "nb_diff_threshold".
    - search_radius: int
        The radius within which to search for the closest pixel with a defined direction.
    - min_directions_diff: float
        Value between 0 and 180.
        If any difference between directions of a peak pair is lower than this value,
        then this peak pair combination is not considered.
    - exclude_lone_peaks: bool
        Whether to exclude lone peaks when calculating the directions for comparison.
        Since lone peak directions have a high probability to be incorrect, due to an 
        unfound peak, this value should normally stay True. This is just for the 
        comparing process, so lone peaks will still be visible in the returned peak pairs 
        with with a pair like e.g. [2, -1] for the second peak index.
    - image_num_peaks: np.ndarray (n, m)
        If the number of peaks are already calculated, they can be inserted here to speed up the process.
    - image_sig_peaks_mask: np.ndarray (n, m, max_find_peaks)
        If the significant peaks mask is already calculated, 
        it can be inserted here to speed up the process.
    - image_pli_directions: np.ndarray (n, m)
        The directions in radians (0 to pi) from a pli measurement used for the method "pli".
    - image_pli_retardations: np.ndarray (n, m)
        The retardations from a pli measurement used for the method "pli".

    Returns:
    - image_peak_pair_combs: np.ndarray (n, m, p, np.ceil(max_paired_peaks / 2), 2)
        The possible peak pair combinations for every pixel (sorted by difference to neighbors), 
        where the fifth dimension contains both peak numbers of
        a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the fourth dimension
        is the number of the peak pair (dimension has length np.ceil(num_peaks / 2)).
        The third dimension contains the different possible combinations of peak pairs,
        which has the length:
        p = math.factorial(num_peaks) // ((2 ** (num_peaks // 2)) * math.factorial(num_peaks // 2))
                so n = 3 for num_peaks = 4 and n = 15 for num_peaks = 6.
                Odd numbers of num_peaks have the same dimension size as num_peaks + 1.
        The first two dimensions are the image dimensions.
        
    """
    n_rows = image_stack.shape[0]
    n_cols = image_stack.shape[1]
    total_pixels = n_rows * n_cols
    angles = np.linspace(0, 2*np.pi, num = image_stack.shape[2], endpoint = False)

    # Convert to radians
    min_distance = np.atleast_1d(min_distance) * np.pi / 180
    max_distance = np.atleast_1d(max_distance) * np.pi / 180
    nb_diff_threshold = nb_diff_threshold * np.pi / 180
    pli_diff_threshold = pli_diff_threshold * np.pi / 180

    if image_pli_directions is not None:
        if np.max(image_pli_directions) > np.pi:
            image_pli_directions = image_pli_directions * np.pi / 180
    
    if only_mus:
        max_peaks = image_params.shape[2]
    else:
        max_peaks = int((image_params.shape[2] - 1) / 3)

    max_paired_peaks = min(max_paired_peaks, max_peaks)
    max_paired_peaks = np.clip(max_paired_peaks, 2, 6)

    if max_paired_peaks <= 2:
        max_combs = 1
    elif max_paired_peaks <= 4:
        max_combs = 3
    elif max_paired_peaks >= 5:
        max_combs = 15
    image_peak_pair_combs = pymp.shared.array((total_pixels,
                                            max_combs, 
                                            np.ceil(max_paired_peaks / 2).astype(int), 
                                            2), dtype = np.int16)

    with pymp.Parallel(num_processes) as p:
        for i in p.range(total_pixels):
            image_peak_pair_combs[i, ...] = -1

    image_peak_pair_combs = image_peak_pair_combs.reshape(n_rows, n_cols,
                                                image_peak_pair_combs.shape[1],
                                                image_peak_pair_combs.shape[2],
                                                image_peak_pair_combs.shape[3])

    if not isinstance(image_sig_peaks_mask, np.ndarray):
        image_sig_peaks_mask = get_sig_peaks_mask(image_stack = image_stack, 
                            image_params = image_params, image_peaks_mask = image_peaks_mask,
                            distribution = distribution, 
                            amplitude_threshold = amplitude_threshold, 
                            rel_amplitude_threshold = rel_amplitude_threshold, 
                            gof_threshold = gof_threshold, only_mus = only_mus)

    if not isinstance(image_num_peaks, np.ndarray):
        image_num_peaks = np.sum(image_sig_peaks_mask, axis = -1)

    if isinstance(method, str):
        method = [method]
    if "pli" in method:
        # Get amplitudes
        image_amplitudes = get_peak_amplitudes(image_stack, image_params = image_params, 
                            image_peaks_mask = image_peaks_mask, 
                            distribution = distribution, only_mus = only_mus)
        max_amp = np.max(image_amplitudes)
        norm_image_amplitudes = image_amplitudes / max_amp
        

    direction_found_mask = pymp.shared.array((n_rows, n_cols), dtype = np.bool_)

    max_sig_peaks = np.max(image_num_peaks)

    #iteration_list = np.append(np.arange(2, max_sig_peaks + 1, 2), np.arange(3, max_sig_peaks + 1, 2))
    #iteration_list = np.append(iteration_list, [1, 0])
    iteration_list = np.append(np.arange(2, max_sig_peaks + 1), [1, 0])

    pbar = None
    for iteration_index, i in enumerate(iteration_list):
        mask = (image_num_peaks == i)
        num_peaks = min(i, max_paired_peaks)
        peak_pairs_combinations = possible_pairs(num_peaks)
        num_combs = peak_pairs_combinations.shape[0]
        num_pairs = peak_pairs_combinations.shape[1]
        indices = np.argwhere(mask)
        num_pixels_iteration = np.count_nonzero(mask)

        if indices.size == 0: continue
        no_processed = True
        if i != 2:
            # Sort indices from distance to already processed indices
            processed_mask = np.isin(image_num_peaks, iteration_list[:iteration_index])
            processed_indices = np.argwhere(processed_mask)
            if processed_indices.size != 0:
                tree = KDTree(processed_indices)
                distances, _ = tree.query(indices)
                sorted_indices = np.argsort(distances)
                no_processed = False
            if not pbar is None: pbar.close()

        # Initialize the progress bar
        pbar = tqdm(total = num_pixels_iteration, 
                    desc = f'Calculating Peak pairs for {i} Peaks',
                    smoothing = 0)
        shared_counter = pymp.shared.array((num_processes, ), dtype = int)

        with pymp.Parallel(num_processes) as p:
            for process_index in p.range(num_processes):
                # divide iterations so every process starts from the lowest sorted index
                divided_iterations = range(process_index, num_pixels_iteration, num_processes)
                for j in divided_iterations:
                    if i == 2 or no_processed:
                        index = j
                    else:
                        index = sorted_indices[j]
 
                    x, y = indices[index, 0], indices[index, 1]
                    sig_peak_indices = image_sig_peaks_mask[x, y].nonzero()[0]

                    # Update progress bar
                    shared_counter[p.thread_num] += 1
                    status = np.sum(shared_counter)
                    pbar.update(status - pbar.n)

                    if num_peaks == 0:
                        continue
                    elif num_peaks == 1:
                        sig_peak_pair_combs = np.where(peak_pairs_combinations == -1, -1, 
                                                sig_peak_indices[peak_pairs_combinations])
                        significances = direction_significances(sig_peak_pair_combs[0], params, peaks_mask, 
                                    intensities, angles, weights = significance_weights,
                                    sens = significance_sens, 
                                    distribution = distribution, only_mus = only_mus,
                                    exclude_lone_peaks = False)
                        if significances[0] > significance_threshold:
                            
                            image_peak_pair_combs[x, y, 
                                        :sig_peak_pair_combs.shape[0],
                                        :sig_peak_pair_combs.shape[1]] = sig_peak_pair_combs
                        continue

                    params = image_params[x, y]
                    peaks_mask = image_peaks_mask[x, y]
                    intensities = image_stack[x, y]
                    num_found_peaks = np.count_nonzero(np.any(peaks_mask, axis = -1))
                    min_distance_val = min_distance[min(i - 2, len(min_distance) - 1)]
                    max_distance_val = max_distance[min(i - 2, len(max_distance) - 1)]

                    if not only_mus:
                        mus = params[1::3]
                    else:
                        mus = params

                    valid_combs_mask = np.ones(num_combs, dtype = np.bool_)
                    valid_pairs_mask = np.ones((num_combs, num_pairs), dtype = np.bool_)
                    direction_combs = np.full((num_combs, num_pairs), -1, 
                                                dtype = np.float64)
                    comb_significances = np.full((num_combs, num_pairs), -1, dtype = np.float64)
                    num_unvalid_differences = np.zeros(num_combs, dtype = int)
                    unvalid_dir_indices_1 = np.full((num_combs, num_pairs), -1)
                    unvalid_dir_indices_2 = np.full((num_combs, num_pairs), -1)

                    for k in range(num_combs):
                        peak_pairs = np.where(peak_pairs_combinations[k] == -1, -1, 
                                                        sig_peak_indices[peak_pairs_combinations[k]])

                        # Check if a pair has a smaller/larger distance than min/max distance
                        for pair_index, pair in enumerate(peak_pairs):
                            if np.any(pair == -1): continue
                            distance = np.abs(angle_distance(mus[pair[0]], mus[pair[1]]))
                            if distance < min_distance_val or distance > max_distance_val:
                                if num_peaks == num_found_peaks:
                                    valid_combs_mask[k] = False
                                else:
                                    valid_pairs_mask[k, pair_index] = False

                        if not np.any(valid_pairs_mask[k]):
                            valid_combs_mask[k] = False
                            continue
                        else:
                            # if every valid pair is a lone peak the comb is unvalid
                            valid_pairs = peak_pairs[valid_pairs_mask[k]]
                            if np.all(np.any(valid_pairs == -1, axis = -1)):
                                valid_combs_mask[k] = False
                                
                        if not valid_combs_mask[k]:
                            continue

                        peak_pairs = peak_pairs[valid_pairs_mask[k]]
                        significances = direction_significances(peak_pairs, params, peaks_mask, 
                                    intensities, angles, weights = significance_weights,
                                    sens = significance_sens, 
                                    distribution = distribution, only_mus = only_mus,
                                    exclude_lone_peaks = exclude_lone_peaks)

                        comb_significances[k, :len(significances)] = significances

                        if np.all(significances < significance_threshold): 
                            valid_combs_mask[k] = False
                            continue

                        directions = peak_pairs_to_directions(peak_pairs, mus, 
                                                            exclude_lone_peaks = exclude_lone_peaks)
                        # Filter directions with low significance out
                        directions = directions[significances >= significance_threshold]
                        directions = directions[directions != -1]
                    
                        # If any differences between directions are below min_directions_diff,
                        # its not a valid peak pair combination (if found peaks = peaks to pair)
                        #differences = np.abs(directions[:, np.newaxis] - directions)
                        #differences = differences[differences != 0]
                        if len(directions) > 1:
                            # Calculate the differences between every directions
                            differences = np.abs(angle_distance(directions[:, np.newaxis], 
                                                    directions[np.newaxis, :], wrap = np.pi))
                            dir_indices_1, dir_indices_2 = np.triu_indices(len(directions), k=1)
                            differences = differences[dir_indices_1, dir_indices_2]
                            unvalid_differences = differences < min_directions_diff * np.pi / 180
                            unvalid_differences_indices = unvalid_differences.nonzero()[0]
                            num_unvalid_differences[k] = len(unvalid_differences_indices)

                            if num_unvalid_differences[k] > 0:
                                if num_peaks == num_found_peaks:
                                    # if the number of peaks (to pair) equals the found peaks
                                    # set the whole peak pair combination to unvalid
                                    valid_combs_mask[k] = False
                                    continue
                                else:
                                    # Save unvalid directions in mask
                                    # only one of two unvalid directions will be paired later
                                    # (this with the best neighbor difference)
                                    unvalid_dir_indices_1[k, :num_unvalid_differences[k]] = \
                                        dir_indices_1[unvalid_differences_indices]
                                    unvalid_dir_indices_2[k, :num_unvalid_differences[k]] = \
                                        dir_indices_2[unvalid_differences_indices]
                                        
                        elif len(directions) == 0:
                            valid_combs_mask[k] = False

                        direction_combs[k, :len(directions)] = directions

                    if num_peaks == num_found_peaks:
                        if np.any(~valid_pairs_mask) and np.any(np.all(valid_pairs_mask, axis = -1)):
                            # if the number of peaks (to pair) equals the found peaks
                            # and any comb has only valid pairs
                            # set the whole combs with unvalid pairs to unvalid
                            unvalid_comb_indices = np.where(np.any(~valid_pairs_mask, axis = -1))[0]
                            valid_combs_mask[unvalid_comb_indices] = False

                            # same for direction difference handling
                        if np.any(num_unvalid_differences > 0) and np.any(num_unvalid_differences == 0):
                            unvalid_comb_indices = np.where(num_unvalid_differences > 0)[0]
                            valid_combs_mask[unvalid_comb_indices] = False

                    if not np.any(valid_combs_mask):
                        continue

                    sig_peak_pair_combs = np.where(peak_pairs_combinations[valid_combs_mask] == -1, -1, 
                                                sig_peak_indices[peak_pairs_combinations[valid_combs_mask]])

                    num_sig_combs = sig_peak_pair_combs.shape[0]
                    valid_pairs_mask = valid_pairs_mask[valid_combs_mask]
                    direction_combs = direction_combs[valid_combs_mask]
                    comb_significances = comb_significances[valid_combs_mask]
                    num_unvalid_differences = num_unvalid_differences[valid_combs_mask]
                    unvalid_dir_indices_1 = unvalid_dir_indices_1[valid_combs_mask]
                    unvalid_dir_indices_2 = unvalid_dir_indices_2[valid_combs_mask]

                    # Set unvalid pairs to [-1, -1] and move the to the end of pairs
                    for k in range(num_sig_combs):
                        valid_pairs = sig_peak_pair_combs[k, valid_pairs_mask[k]]
                        sig_peak_pair_combs[k, :len(valid_pairs)] = valid_pairs
                        sig_peak_pair_combs[k, len(valid_pairs):] = [-1, -1]

                    if num_sig_combs == 1 and num_unvalid_differences[0] == 0:
                        image_peak_pair_combs[x, y, 
                                        :sig_peak_pair_combs.shape[0],
                                        :sig_peak_pair_combs.shape[1]] = sig_peak_pair_combs
                        if num_peaks >= 2: 
                            direction_found_mask[x, y] = True
                        continue

                    # If a pair is in every significant combination, its a match
                    # what can only happen for more than 4 peaks and only once (for up to 6 peaks)
                    all_match_indices = None
                    if num_peaks > 4:
                        for pair in sig_peak_pair_combs[0]:
                            if np.all(pair == -1): continue
                            isin_mask = (sig_peak_pair_combs == pair).all(axis=-1)
                            if np.all(np.any(isin_mask, axis = -1)):
                                all_match_indices = isin_mask.nonzero()
                                break

                        if not all_match_indices is None:
                            # Check if a comb has only this match and a lone peak
                            # and if so use this comb and continue to next pixel
                            lone_peaks_mask = np.any(sig_peak_pair_combs == -1, axis = -1)
                            full_comb_mask = np.all(isin_mask | lone_peaks_mask, axis = -1)
                            full_comb_indices = full_comb_mask.nonzero()[0]
                            num_full_combs = len(full_comb_indices)
                            if num_full_combs > 0:
                                image_peak_pair_combs[x, y, 
                                    :num_full_combs,
                                    :sig_peak_pair_combs.shape[1]] = sig_peak_pair_combs[full_comb_indices]
                                continue

                    # Try method(s) with loop
                    for current_method in method:
                        if current_method not in [None, "None", "single", "pli", 
                                                    "neighbor", "random", "significance"]:
                            raise Exception("Method not found.")
                        if direction_found_mask[x, y]: 
                            break
                        if current_method in [None, "None", "single"]:
                            if not all_match_indices is None:
                                # if a matched pair is in every comb use it
                                index = (all_match_indices[0][0], all_match_indices[1][0])
                                image_peak_pair_combs[x, y, 0, 0] = sig_peak_pair_combs[index]
                            break
                        # Get the start_index for insertion
                        start_index = np.where(np.any(
                                    image_peak_pair_combs[x, y] != -1, axis = (1, 2)))[0] 
                        if len(start_index) > 0:
                            # another method already used
                            start_index = start_index[0] + 1
                        else:
                            # no method already used
                            start_index = 0
                        if current_method == "random":
                            np.random.shuffle(sig_peak_pair_combs[start_index:])
                            image_peak_pair_combs[x, y, 
                                        :sig_peak_pair_combs.shape[0],
                                        :sig_peak_pair_combs.shape[1]] = sig_peak_pair_combs
                            break
                        elif current_method == "significance":
                            sort_indices = np.argsort(np.max(comb_significances[start_index:], axis = -1))
                            sig_peak_pair_combs[start_index:] = \
                                        sig_peak_pair_combs[start_index:][sort_indices[::-1]]
                            image_peak_pair_combs[x, y, 
                                        :sig_peak_pair_combs.shape[0],
                                        :sig_peak_pair_combs.shape[1]] = sig_peak_pair_combs
                            break
                        elif current_method == "pli":
                            sorted_peak_pair_combs = sig_peak_pair_combs[start_index:]
                            pli_direction = image_pli_directions[x, y]
                            pli_retardation = image_pli_retardations[x, y]
                            num_sorted_combs = sorted_peak_pair_combs.shape[0]
                            sim_pli_directions = np.full(num_sorted_combs, -1, dtype = np.float64)
                            sim_pli_retardations = np.full(num_sorted_combs, -1, dtype = np.float64)
                            sim_diffs = np.full(num_sorted_combs, -1, dtype = np.float64)
                            for k, peak_pairs in enumerate(sorted_peak_pair_combs):
                                if num_unvalid_differences[start_index + k] > 0: continue
                                sim_pli_directions[k], sim_pli_retardations[k], sim_diffs[k] = \
                                    sli_to_pli_brute(peak_pairs, mus, 
                                                pli_direction, pli_retardation)

                            dir_mask = (sim_pli_directions != -1)
                            if not np.any(dir_mask): continue
                            sorted_peak_pair_combs = sorted_peak_pair_combs[dir_mask]
                            sim_pli_directions = sim_pli_directions[dir_mask]
                            sim_pli_retardations = sim_pli_retardations[dir_mask]
                            sim_diffs = sim_diffs[dir_mask]

                            dir_differences = np.abs(angle_distance(pli_direction, 
                                                            sim_pli_directions, wrap = np.pi))
                            ret_differences = np.abs(pli_retardation - sim_pli_retardations)
                            best_diff_index = np.argmin(sim_diffs)
                            
                            if np.min(dir_differences[best_diff_index]) <= pli_diff_threshold \
                                and np.min(ret_differences[best_diff_index]) <= pli_ret_diff_threshold:
                                sorting_indices = np.argsort(sim_diffs)
                                sorted_peak_pair_combs = sorted_peak_pair_combs[sorting_indices]
                                image_peak_pair_combs[x, y, 
                                            start_index:sorted_peak_pair_combs.shape[0],
                                            :sorted_peak_pair_combs.shape[1]] = sorted_peak_pair_combs

                        elif current_method == "neighbor":
                            check_mask = np.copy(direction_found_mask)
                            matched_dir_mask = np.zeros(direction_combs.shape, dtype = np.bool_)
                            matched_dir_diffs = np.full(direction_combs.shape, np.pi, dtype = np.float64)

                            # Set matched pairs in every comb as already matched
                            if not all_match_indices is None:
                                matched_dir_mask[all_match_indices] = True
                                matched_dir_diffs[all_match_indices] = 0

                            # Set lone peak pairs as already matched
                            lone_peak_pair_indices = np.where(np.any(sig_peak_pair_combs == -1, axis = -1))
                            matched_dir_mask[lone_peak_pair_indices] = True
                            matched_dir_diffs[lone_peak_pair_indices] = 0
                            num_best_combs = 0

                            for attempt in range(max_attempts):

                                neighbor_x, neighbor_y = find_closest_true_pixel(check_mask, (x, y), 
                                                                                search_radius)
                                if neighbor_x == -1 and neighbor_y == -1:
                                    # When no true pixel within radius: return no pairs
                                    break

                                neighbor_peak_pairs = image_peak_pair_combs[neighbor_x, neighbor_y, 0]
                                neighbor_params = image_params[neighbor_x, neighbor_y]
                                neighbor_peaks_mask = image_peaks_mask[neighbor_x, neighbor_y]
                                neighbor_intensities = image_stack[neighbor_x, neighbor_y]
                                if not only_mus:
                                    neighbor_mus = neighbor_params[1::3]
                                else:
                                    neighbor_mus = neighbor_params

                                neighbor_directions = peak_pairs_to_directions(neighbor_peak_pairs, 
                                                    neighbor_mus, 
                                                    exclude_lone_peaks = exclude_lone_peaks)

                                neighbor_significances = direction_significances(neighbor_peak_pairs, 
                                            neighbor_params, neighbor_peaks_mask, 
                                            neighbor_intensities, angles, 
                                            weights = significance_weights,
                                            sens = significance_sens, 
                                            distribution = distribution, only_mus = only_mus,
                                            exclude_lone_peaks = exclude_lone_peaks)

                                # Filter neighbor directions with low significance out
                                # if the number of peaks is 2 
                                # (more peak pixel directions are already filtered then by algorithm)
                                nb_num_combs = np.sum(np.any(
                                    image_peak_pair_combs[x, y] != -1, axis = (1, 2)))
                                if nb_num_combs == 1:
                                    neighbor_directions = neighbor_directions[neighbor_significances
                                                                    >= nb_significance_threshold]

                                neighbor_directions = neighbor_directions[neighbor_directions != -1]

                                if len(neighbor_directions) > 0:
                                    direction_diffs = np.empty(num_sig_combs, dtype = np.float64)
                                    dir_diff_indices = np.empty(num_sig_combs, dtype = np.int16)
                                    for k in range(num_sig_combs):
                                        directions = direction_combs[k]
                                        directions = directions[~matched_dir_mask[k]]
                                        unmatched_indices = (~matched_dir_mask[k]).nonzero()[0]

                                        differences = np.abs(angle_distance(neighbor_directions[:, np.newaxis], 
                                                    directions[np.newaxis, :], wrap = np.pi))
                                        min_diff_index = np.argmin(differences)
                                        nbd_index, min_diff_dir_index = np.unravel_index(min_diff_index, differences.shape)
                                        dir_diff_indices[k] = unmatched_indices[min_diff_dir_index]
                                        # Insert minimum difference to neighbor directions into array for sorting later
                                        direction_diffs[k] = differences[nbd_index, min_diff_dir_index]

                                    if np.min(direction_diffs) <= nb_diff_threshold:
                                        # If minimum difference to neighbor direction is 
                                        # smaller than given threshold mark the matching directions
                                        min_diff = np.min(direction_diffs)
                                        best_combs_indices = np.where(direction_diffs == min_diff)[0]
                                        num_best_combs = len(best_combs_indices)
                                        for k in best_combs_indices:
                                            matched_dir_mask[k, dir_diff_indices[k]] = True
                                            matched_dir_diffs[k, dir_diff_indices[k]] = min_diff
                                            if num_unvalid_differences[k] > 0:
                                                # If a direction difference was too low,
                                                # set the pair not so close to neighbor direction to unvalid
                                                # and mark it as matched
                                                if dir_diff_indices[k] in unvalid_dir_indices_1[k]:
                                                    matched_unvalid_dir_index = \
                                                        (unvalid_dir_indices_1[k] == dir_diff_indices[k]).nonzero()[0][0]
                                                    unvalid_dir_index = unvalid_dir_indices_2[k, matched_unvalid_dir_index]
                                                    valid_pairs_mask[k, unvalid_dir_index] = False
                                                    matched_dir_mask[k, unvalid_dir_index] = True
                                                elif dir_diff_indices[k] in unvalid_dir_indices_2[k]:
                                                    matched_unvalid_dir_index = \
                                                        (unvalid_dir_indices_2[k] == dir_diff_indices[k]).nonzero()[0][0]
                                                    unvalid_dir_index = unvalid_dir_indices_1[k, matched_unvalid_dir_index]
                                                    valid_pairs_mask[k, unvalid_dir_index] = False
                                                    matched_dir_mask[k, unvalid_dir_index] = True

                                        if np.any(np.all(matched_dir_mask, axis = -1)) \
                                            or (num_best_combs == 1 and num_peaks == num_found_peaks):
                                            # if any combination is full matched
                                            # Set unvalid pairs to [-1, -1] and move them to the end of pairs
                                            for k in range(num_sig_combs):
                                                valid_pairs = sig_peak_pair_combs[k, valid_pairs_mask[k]]
                                                sig_peak_pair_combs[k, :len(valid_pairs)] = valid_pairs
                                                sig_peak_pair_combs[k, len(valid_pairs):] = [-1, -1]

                                            # Sort combs first by number of matches (descending)
                                            # and second by the mean matched direction differences
                                            num_matches = np.count_nonzero(matched_dir_mask, axis = -1)
                                            mean_dir_diffs = np.mean(matched_dir_diffs, axis = -1)
                                            sort_indices = np.lexsort((mean_dir_diffs, -num_matches))
                                            sig_peak_pair_combs = sig_peak_pair_combs[sort_indices]

                                            # save the sorted significant peak pair combinations
                                            image_peak_pair_combs[x, y, 
                                                        :sig_peak_pair_combs.shape[0],
                                                        :sig_peak_pair_combs.shape[1]] = sig_peak_pair_combs
                                            direction_found_mask[x, y] = True
                                            break
                                        else:
                                            continue
                                
                                check_mask[neighbor_x, neighbor_y] = False
                                if attempt == max_attempts - 1 or not np.any(check_mask):
                                    # When no neighboring pixel within max_attempts had
                                    # a direction difference below the threshold
                                    # save matched pairs if any

                                    no_lone_matched_dir_mask = np.copy(matched_dir_mask)
                                    no_lone_matched_dir_mask[lone_peak_pair_indices] = False
                                    matched_comb_mask = np.any(no_lone_matched_dir_mask, axis = -1)
                                    matched_combs = sig_peak_pair_combs[matched_comb_mask]
                                    matched_indices = matched_comb_mask.nonzero()[0]

                                    for matched_index, comb_index in enumerate(matched_indices):
                                        matched_peak_pairs = \
                                            sig_peak_pair_combs[comb_index, matched_dir_mask[comb_index]]
                                        num_matched_pairs = len(matched_peak_pairs)
                                        matched_combs[matched_index, :num_matched_pairs] = \
                                                    matched_peak_pairs
                                        if "significance" in method:
                                            # sort unmatched peak pairs in matched combs by significance
                                            unmatched_peak_pairs = matched_combs[matched_index, 
                                                                        ~matched_dir_mask[comb_index]]
                                            sort_indices = np.argsort(comb_significances[comb_index, 
                                                                    ~matched_dir_mask[comb_index]])[::-1]
                                            matched_combs[matched_index, num_matched_pairs:] \
                                                 = unmatched_peak_pairs[sort_indices]
                                        else:
                                            # remove unmatched peak pairs
                                            matched_combs[matched_index, num_matched_pairs:] = -1

                                    # Sort matched combs by best direction significance
                                    matched_significances = comb_significances[matched_comb_mask]
                                    sort_indices = np.argsort(np.max(matched_significances, axis = -1))
                                    matched_combs = matched_combs[sort_indices[::-1]]
                                    matched_significances = matched_significances[sort_indices[::-1]]

                                    # Rearrange arrays for other methods
                                    num_matched_combs = len(matched_combs)
                                    unmatched_combs = sig_peak_pair_combs[~matched_comb_mask]
                                    unmatched_significances = comb_significances[~matched_comb_mask]
                                    sig_peak_pair_combs[:num_matched_combs] = matched_combs
                                    sig_peak_pair_combs[num_matched_combs:] = unmatched_combs
                                    comb_significances[:num_matched_combs] = matched_significances
                                    comb_significances[num_matched_combs:] = unmatched_significances
                                    
                                    image_peak_pair_combs[x, y, 
                                        :matched_combs.shape[0],
                                        :matched_combs.shape[1]] = matched_combs

                                    break

        # Set the progress bar to 100%
        pbar.update(pbar.total - pbar.n)
    
    return image_peak_pair_combs

@njit(cache = True, fastmath = True, parallel = True)
def peak_pairs_to_directions(peak_pairs, mus, exclude_lone_peaks = True):
    """
    Calculates the directions from given peak_pairs of a pixel.

    Parameters:
    - peak_pairs: np.ndarray (np.ceil(max_paired_peaks / 2), 2)
        Array containing both peak numbers of a pair (e.g. [1, 3], 
        which means peak 1 and peak 3 is paired). A pair with -1 defines a unpaired peak.
        The first dimension is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
    - mus: np.ndarray (max_find_peaks, )
        The center positions of the peaks.
    - exclude_lone_peaks: bool
        Whether to exclude the directions for lone peaks 
        (for peak pairs with only one number unequal -1 e.g. [2, -1]).

    Returns:
    - directions: np.ndarray (np.ceil(max_paired_peaks/2), )
        The calculated directions for every peak pair.
    """
    directions = np.empty(peak_pairs.shape[0])
    for k, pair in enumerate(peak_pairs):
        if pair[0] == -1 and pair[1] == -1:
            # No pair
            direction = -1
        elif pair[0] == -1 or pair[1] == -1:
            # One peak direction
            if exclude_lone_peaks:
                direction = -1
            else:
                direction = ((270 * np.pi / 180) - mus[pair[pair != -1][0]]) % np.pi
        else:
            # Two peak direction
            #distance = angle_distance(mus[pair[0]], mus[pair[1]])
            #direction = (mus[pair[0]] + distance / 2) % (np.pi)
            direction = ((270 * np.pi / 180) - (mus[pair[0]] + mus[pair[1]]) / 2) % np.pi

        directions[k] = direction

    return directions

@njit(cache = True, fastmath = True)
def peak_pairs_to_inclinations(peak_pairs, mus):
    """
    Placeholder function until a prober way to get the inclinations from a SLI measurement is found.
    Calculates the inclinations from given peak_pairs of a pixel.

    Parameters:
    - peak_pairs: np.ndarray (np.ceil(max_paired_peaks / 2), 2)
        Array ontaining both peak numbers of a pair (e.g. [1, 3], 
        which means peak 1 and peak 3 is paired). A pair with -1 defines a unpaired peak.
        The first dimension is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
    - mus: np.ndarray (max_find_peaks, )
        The center positions of the peaks.

    Returns:
    - inclinations: np.ndarray (np.ceil(max_paired_peaks / 2), )
        The calculated inclinations for every peak pair.
    """
    num_pairs = peak_pairs.shape[0]
    inclinations = np.empty(num_pairs)
    for i in range(num_pairs):
        current_mus = mus[peak_pairs[i]]
        distance_deviation = np.pi - np.abs(angle_distance(current_mus[0], current_mus[1]))
        inclinations[i] = distance_deviation

    return inclinations

@njit(cache = True, fastmath = True, parallel = True)
def calculate_directions(image_peak_pairs, image_mus, exclude_lone_peaks = True):
    """
    Calculates the directions from given image_peak_pairs.

    Parameters:
    - image_peak_pairs: np.ndarray (n, m, np.ceil(max_paired_peaks / 2), 2)
        The peak pairs for every pixel, where the fourth dimension contains both peak numbers of
        a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the third dimension
        is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
        The first two dimensions are the image dimensions.
    - image_mus: np.ndarray (n, m, max_find_peaks)
        The mus (centers) of the found (max_find_peaks) peaks for everyone of the (n * m) pixels.
    - exclude_lone_peaks: bool
        Whether to exclude the directions for lone peaks 
        (for peak pairs with only one number unequal -1 e.g. [2, -1]).

    Returns:
    - image_directions: (n, m, np.ceil(max_paired_peaks / 2))
        The calculated directions (in degrees) for everyoe of the (n * m) pixels.
        Max 3 directions (for 6 peaks).
    """

    x_range = image_peak_pairs.shape[0]
    y_range = image_peak_pairs.shape[1]
    max_directions = image_peak_pairs.shape[2]
    image_directions = np.full((x_range, y_range, max_directions), -1, dtype=np.float64)
    for x in prange(x_range):
        for y in prange(y_range):

            directions = peak_pairs_to_directions(image_peak_pairs[x, y], image_mus[x, y],
                                                                exclude_lone_peaks = exclude_lone_peaks)
            mask = (directions != -1)
            directions[mask] = directions[mask] * 180 / np.pi
                
            image_directions[x, y] = directions

    return image_directions

@njit(cache = True, fastmath = True)
def possible_pairs(num_peaks):
    """
    Function returning pre calculated possible peak pairs of the get_possible_pairs function.

    Parameters:
    - num_peaks: int
        The maximum number (index) of the indices to use as elements for pair combinations.

    Returns:
    - pair_combinations: (n, m, 2)
        The possible combinations of pairs.
        The size of dimensions is: 
            n = math.factorial(num_peaks) // ((2 ** (num_peaks // 2)) * math.factorial(num_peaks // 2))
                so n = 3 for num_peaks = 4 and n = 15 for num_peaks = 6.
                Odd numbers of num_peaks have the same dimension size as num_peaks + 1.
            m = np.ceil(num_peaks / 2)

    """
    if num_peaks == 1:
        possible_pairs = np.array([[[0, -1]]])
    elif num_peaks == 2:
        possible_pairs = np.array([[[0, 1]]])
    elif num_peaks == 3:
        possible_pairs = np.array([
            [[1, 2], [0, -1]],
            [[0, 2], [1, -1]],
            [[0, 1], [2, -1]]
            ])
    elif num_peaks == 4:
        possible_pairs = np.array([
            [[1, 2], [0, 3]],
            [[0, 2], [1, 3]],
            [[0, 1], [2, 3]]
            ])
    elif num_peaks == 5:
        possible_pairs = np.array([
            [[1, 2], [3, 4], [0, -1]],
            [[1, 3], [2, 4], [0, -1]],
            [[1, 4], [2, 3], [0, -1]],
            [[0, 2], [3, 4], [1, -1]],
            [[0, 3], [2, 4], [1, -1]],
            [[0, 4], [2, 3], [1, -1]],
            [[0, 1], [3, 4], [2, -1]],
            [[0, 3], [1, 4], [2, -1]],
            [[0, 4], [1, 3], [2, -1]],
            [[0, 1], [2, 4], [3, -1]],
            [[0, 2], [1, 4], [3, -1]],
            [[0, 4], [1, 2], [3, -1]],
            [[0, 1], [2, 3], [4, -1]],
            [[0, 2], [1, 3], [4, -1]],
            [[0, 3], [1, 2], [4, -1]]
            ])
    elif num_peaks == 6:
        possible_pairs = np.array([
            [[1, 2], [3, 4], [0, 5]],
            [[1, 3], [2, 4], [0, 5]],
            [[1, 4], [2, 3], [0, 5]],
            [[0, 2], [3, 4], [1, 5]],
            [[0, 3], [2, 4], [1, 5]],
            [[0, 4], [2, 3], [1, 5]],
            [[0, 1], [3, 4], [2, 5]],
            [[0, 3], [1, 4], [2, 5]],
            [[0, 4], [1, 3], [2, 5]],
            [[0, 1], [2, 4], [3, 5]],
            [[0, 2], [1, 4], [3, 5]],
            [[0, 4], [1, 2], [3, 5]],
            [[0, 1], [2, 3], [4, 5]],
            [[0, 2], [1, 3], [4, 5]],
            [[0, 3], [1, 2], [4, 5]]
            ])
    else:
        possible_pairs = np.array([[[-1, -1]]])

    return possible_pairs

def get_possible_pairs(num_peaks):
    """
    Calculates all possible combinations of pairs of numbers from 0 to num_peaks.

    Parameters:
    - num_peaks: int
        The maximum number (index) of the indices to use as elements for pair combinations.

    Returns:
    - pair_combinations: (n, m, 2)
        The possible combinations of pairs.
        The size of dimensions is: 
            n = math.factorial(num_peaks) // ((2 ** (num_peaks // 2)) * math.factorial(num_peaks // 2))
                so n = 3 for num_peaks = 4 and n = 15 for num_peaks = 6.
                Odd numbers of num_peaks have the same dimension size as num_peaks + 1.
            m = np.ceil(num_peaks / 2)

    """
    indices = list(range(num_peaks))
    all_index_pairs = []

    if num_peaks % 2 == 0:
        index_combinations = list(combinations(indices, 2))

        def is_non_overlapping(pair_set):
            flat_list = [index for pair in pair_set for index in pair]
            return len(flat_list) == len(set(flat_list))

        for pair_set in combinations(index_combinations, num_peaks // 2):
            if is_non_overlapping(pair_set):
                all_index_pairs.append(pair_set)

    else:
        # Odd case: include an unmatched index with -1
        for unmatched in indices:
            remaining_indices = [i for i in indices if i != unmatched]
            index_combinations = list(combinations(remaining_indices, 2))
            
            def is_non_overlapping(pair_set):
                flat_list = list(chain.from_iterable(pair_set))
                return len(flat_list) == len(set(flat_list))

            for pair_set in combinations(index_combinations, (num_peaks - 1) // 2):
                if is_non_overlapping(pair_set):
                    all_index_pairs.append(pair_set + ((unmatched, -1),))

    pair_combinations = np.array(all_index_pairs)
    return pair_combinations

#@njit(cache = True, fastmath = True)
def direction_significances(peak_pairs, params, peaks_mask, intensities, angles, 
                            weights = [1, 1], sens = [1, 1],
                            distribution = "wrapped_cauchy", only_mus = False,
                            exclude_lone_peaks = True):
    """
    Calculates the significances of the directions for one (fitted) pixel.
    (Old function, performance can be improved similar to get_image_direction_significances).

    Parameters:
    - peak_pairs: np.ndarray (np.ceil(max_paired_peaks / 2), 2)
        Array ontaining both peak numbers of a pair (e.g. [1, 3], 
        which means peak 1 and peak 3 is paired). The first dimension 
        is the number of the peak pair.
    - params: np.ndarray (q, )
        The output of fitting the pixel, which stores the parameters of the full fitfunction.
    - peaks_mask: np.ndarray (m, n)
        The mask defining which of the n-measurements corresponds to one of the m-peaks.
    - intensities: np.ndarray (n, )
        The measured intensities of the pixel.
    - angles: np.ndarray (n, )
        The angles at which the intensities are measured.
    - weights: list (2, )
        The weights for the amplitude (first value) and for the goodnes-of-fit (second value), 
        when calculating the significance.
    - sens: list (2, )
        The sensitivity values for the amplitude (first value) and for the goodness-of-fit (second value),
        when calculating the significance.
    - distribution: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
        The name of the distribution.
    - only_mus: bool
        Defines if only the mus are given in the params.
    - exclude_lone_peaks: bool
        Whether to exclude the directions for lone peaks 
        (for peak pairs with only one number unequal -1 e.g. [2, -1]).

    Returns:
    - significances: np.ndarray (np.ceil(max_paired_peaks / 2), )
        The calculated significance for every direction (peak-pair) ranging from 0 to 1.
    """
    global_amplitude = np.max(intensities) - np.min(intensities)
    num_directions = peak_pairs.shape[0]
    significances = np.zeros(num_directions)
    num_peaks = np.count_nonzero(np.any(peaks_mask, axis = -1))
    
    # Get indices of unpaired peaks (Note: lone peaks can be paired peaks by semantic definition)
    paired_peak_indices = np.unique(peak_pairs)
    all_peak_indices = set(range(num_peaks))
    unpaired_peak_indices = list(all_peak_indices - set(paired_peak_indices))

    amplitudes = np.zeros(num_peaks)

    if only_mus:
        # If only mus are provided, just use amplitude significance
        for i in range(num_peaks):
            peak_intensities = intensities[peaks_mask[i]]
            if len(peak_intensities) == 0: continue
            amplitudes[i] = np.max(peak_intensities) - np.min(intensities)

        for i in range(num_directions):
            peak_pair = peak_pairs[i]
            indices = peak_pair[peak_pair != -1]
            if len(indices) == 0 or (len(indices) == 1 and exclude_lone_peaks): continue
            # calculate max amplitude of unpaired peaks and subtract it from significance
            if len(unpaired_peak_indices) > 0:
                malus_amplitude = np.max(amplitudes[unpaired_peak_indices])
            else:
                malus_amplitude = 0
            significances[i] = np.mean((amplitudes[indices] - malus_amplitude)/ global_amplitude)
            significances = np.clip(significances, 0, 1)

        return significances
    
    model_y = full_fitfunction(angles, params, distribution)
    peaks_gof = calculate_peaks_gof(intensities, model_y, peaks_mask, method = "r2")
    heights = params[0:-1:3]
    scales = params[2::3]

    for i in range(num_peaks):
        if i < len(heights):
            amplitudes[i] = heights[i] * distribution_pdf(0, 0, scales[i], distribution)
        else:
            amplitudes[i] = np.max(intensities[peaks_mask[i]]) - np.min(intensities)

    if len(unpaired_peak_indices) > 0:
        malus_amplitude = np.max(amplitudes[unpaired_peak_indices])
    else:
        malus_amplitude = 0

    for i in range(num_directions):
        peak_pair = peak_pairs[i]
        if peak_pair[0] == -1 and peak_pair[1] == -1: 
            continue
        elif (peak_pair[0] == -1 or peak_pair[1] == -1):
            if exclude_lone_peaks: continue
            peak_index = peak_pair[peak_pair != -1][0]
            amplitude_significance = (amplitudes[peak_index] - malus_amplitude) / global_amplitude
            gof_significance = peaks_gof[peak_index]
        else:
            amplitude_significance = (np.mean(amplitudes[peak_pair]) - malus_amplitude) / global_amplitude
            gof_significance = np.mean(peaks_gof[peak_pair])

        significances[i] = ((weights[0] * amplitude_significance**sens[0] 
                            + weights[1] * gof_significance**sens[1]) 
                            / np.sum(weights))

    significances = np.clip(significances, 0, 1)

    return significances

def get_image_direction_significances(image_stack, image_peak_pairs, image_params, image_peaks_mask, 
                            distribution = "wrapped_cauchy", 
                            amplitude_threshold = 0, rel_amplitude_threshold = 0,
                            gof_threshold = 0,
                            weights = [1, 1], sens = [1, 1], only_mus = False, num_processes = 2):

    angles = np.linspace(0, 2*np.pi, num = image_stack.shape[2], endpoint = False)
    n_rows = image_stack.shape[0]
    n_cols = image_stack.shape[1]
    total_pixels = n_rows * n_cols

    flat_image_stack = image_stack.reshape((total_pixels, image_stack.shape[2]))
    flat_image_peak_pairs = image_peak_pairs.reshape((total_pixels, *image_peak_pairs.shape[2:]))
    flat_image_params = image_params.reshape((total_pixels, image_params.shape[2]))
    flat_image_peaks_mask = image_peaks_mask.reshape((total_pixels, *image_peaks_mask.shape[2:]))

    image_direction_sig = pymp.shared.array((total_pixels, image_peak_pairs.shape[2]))

    # Initialize the progress bar
    pbar = tqdm(total = total_pixels, 
                desc = f'Calculating direction significances',
                smoothing = 0)
    shared_counter = pymp.shared.array((num_processes, ), dtype = int)

    with pymp.Parallel(num_processes) as p:
        for i in p.range(total_pixels):

            # Update progress bar
            shared_counter[p.thread_num] += 1
            status = np.sum(shared_counter)
            pbar.update(status - pbar.n)
            
            image_direction_sig[i] = direction_significances(flat_image_peak_pairs[i], 
                        flat_image_params[i], flat_image_peaks_mask[i], flat_image_stack[i], 
                            angles, weights = weights, sens = sens, distribution = distribution, 
                            only_mus = only_mus)

    # Set the progress bar to 100%
    pbar.update(pbar.total - pbar.n)

    image_direction_sig = image_direction_sig.reshape((n_rows, n_cols, image_direction_sig.shape[1]))

    return image_direction_sig

def get_image_direction_significances_vectorized(image_stack, image_peak_pairs, image_params, 
                            image_peaks_mask, 
                            distribution = "wrapped_cauchy", 
                            amplitude_threshold = 0, rel_amplitude_threshold = 0,
                            gof_threshold = 0,
                            weights = [1, 1], sens = [1, 1],
                            only_mus = False, image_sign_peaks_mask = None):
    """
    Returns the direction significances for every pixel.
    To-Do: This function is missing "malus_amplitude",
           which should be subtracted from amplitude significance, 
           but this function is not used anyway yet.

    Parameters:
    - image_stack: np.ndarray (n, m, p)
        The image stack containing the measured intensities.
        n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
    - image_peak_pairs: np.ndarray (n, m, np.ceil(max_peaks / 2), 2)
        The peak pairs for every pixel, where the fourth dimension contains both peak numbers of
        a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the third dimension
        is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
        The first two dimensions are the image dimensions.
    - image_params: np.ndarray (n, m, q)
        The output of fitting the image stack, which stores the parameters of the full fitfunction.
        q = 3 * max_peaks + 1, is the number of parameters (max 19 for 6 peaks).
    - image_peaks_mask: np.ndarray (n, m, max_peaks, p)
        The mask defining which of the p-measurements corresponds to one of the peaks.
        The first two dimensions are the image dimensions.
    - distribution: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
        The name of the distribution.
    - amplitude_threshold: float
        Peak-Pairs with a amplitude below this threshold will not be evaluated.
    - rel_amplitude_threshold: float
        Value between 0 and 1.
        Peak-Pairs with a relative amplitude (to maximum - minimum intensity of the pixel) below
        this threshold will not be evaluated.
    - gof_threshold: float
        Value between 0 and 1.
        Peak-Pairs with a goodness-of-fit value below this threshold will not be evaluated.
    - weights: list (2, )
        The weights for the amplitude and for the goodnes-of-fit, when calculating the significance.
        First weight is for amplitude, second for goodness-of-fit.
    - sens: list (2, )
        The sensitivity values for the amplitude (first value) and for the goodness-of-fit (second value),
        when calculating the significance.
    - only_mus: boolean
        Whether only the mus are provided in image_params. If so, only amplitude_threshold is used.
    - image_sig_peaks_mask: np.ndarray (n, m, max_find_peaks)
        If significant peaks mask is already calculated, 
        it can be provided here to speed up the process.

    Returns:
    - image_direction_sig: (n, m, np.ceil(max_peaks / 2))
        The calculated significances (ranging from 0 to 1) for everyoe of the (n * m) pixels.
        Max 3 significances (shape like directions).
    """

    print("Calculating image direction significances...")

    angles = np.linspace(0, 2*np.pi, num = image_stack.shape[2], endpoint = False)
    image_global_amplitudes = np.max(image_stack, axis = -1) - np.min(image_stack, axis = -1)
    image_global_amplitudes[image_global_amplitudes == 0] = 1
    n_rows = image_stack.shape[0]
    n_cols = image_stack.shape[1]

    image_amplitudes = get_peak_amplitudes(image_stack, image_params = image_params, 
                            image_peaks_mask = image_peaks_mask, 
                            distribution = distribution, only_mus = only_mus)

    image_rel_amplitudes = get_peak_rel_amplitudes(image_stack, image_amplitudes)

    if not isinstance(image_sig_peaks_mask, np.ndarray):
        image_sig_peaks_mask = get_sig_peaks_mask(image_stack = image_stack, 
                            image_params = image_params, 
                            image_peaks_mask = image_peaks_mask,
                            distribution = distribution, 
                            amplitude_threshold = amplitude_threshold, 
                            rel_amplitude_threshold = rel_amplitude_threshold, 
                            gof_threshold = gof_threshold, only_mus = only_mus)
    
    # Set unvalid values to -2, so the calculated mean later cant be greater than 0
    image_rel_amplitudes[~image_sig_peaks_mask] = -2
    if not only_mus:
        image_peaks_gof[~image_sig_peaks_mask] = -2

    # Replace -1 values in the peak pairs array with duplicates of the other index
    # e.g. [2, -1] is replaced with [2, 2]
    image_peak_pairs_copy = np.copy(image_peak_pairs)
    mask = (image_peak_pairs_copy == -1)
    image_peak_pairs_copy[mask] = image_peak_pairs_copy[mask[:, :, :, ::-1]]

    # Convert image_peak_pairs array values into relative amplitude values
    image_rel_amplitudes = image_rel_amplitudes[np.arange(n_rows)[:, None, None, None], 
                                            np.arange(n_cols)[None, :, None, None], 
                                            image_peak_pairs_copy]

    # Set all relative amplitudes to zero where no peak is paired
    mask = np.all(image_peak_pairs_copy == -1, axis = -1)
    image_rel_amplitudes[mask] = 0

    # Calculate mean relative amplitude for every pair
    image_rel_amplitudes = np.mean(image_rel_amplitudes, axis = -1)
    image_rel_amplitudes[image_rel_amplitudes < 0] = 0

    # Do the same for gof
    if not only_mus:
        image_peaks_gof = image_peaks_gof[np.arange(n_rows)[:, None, None, None], 
                                                np.arange(n_cols)[None, :, None, None], 
                                                image_peak_pairs_copy]
        image_peaks_gof[mask] = 0
        image_peaks_gof = np.mean(image_peaks_gof, axis = -1)
        image_peaks_gof[image_peaks_gof < 0] = 0

        image_direction_sig = ((image_rel_amplitudes**sens[0] * weights[0] 
                                    + image_peaks_gof**sens[1] * weights[1]) 
                                    / np.sum(weights))
    else:
        image_direction_sig = image_rel_amplitudes

    print("Done")

    return image_direction_sig

def get_peak_amplitudes(image_stack, image_params = None, image_peaks_mask = None, 
                            distribution = "wrapped_cauchy", only_mus = False):
    if not only_mus:
        # Use params for amplitudes
        image_heights = image_params[..., 0:-1:3]
        image_scales = image_params[..., 2::3]
        image_amplitudes = image_heights * \
                            distribution_pdf(0, 0, image_scales, distribution)[..., 0]
    else:
        # Use peaks mask for amplitudes
        image_peak_intensities = np.expand_dims(image_stack, axis = 2)
        image_peak_intensities = np.where(image_peaks_mask, image_peak_intensities, 0)
        image_amplitudes = (np.max(image_peak_intensities, axis = -1)
                            - np.min(image_stack, axis = -1)[..., np.newaxis])
        valid_peaks = np.any(image_peaks_mask, axis = -1)
        image_amplitudes = np.where(valid_peaks, image_amplitudes, 0)

    return image_amplitudes

def get_peak_rel_amplitudes(image_stack, image_amplitudes):
    image_global_amplitudes = np.max(image_stack, axis = -1) - np.min(image_stack, axis = -1)
    image_global_amplitudes[image_global_amplitudes == 0] = 1
    image_rel_amplitudes = image_amplitudes / image_global_amplitudes[..., np.newaxis]

    return image_rel_amplitudes

def get_peak_gofs(image_stack, image_params, image_peaks_mask, distribution = "wrapped_cauchy"):
    image_heights = image_params[..., 0:-1:3]
    max_fit_peaks = image_heights.shape[-1]
    angles = np.linspace(0, 2*np.pi, num = image_stack.shape[-1], endpoint = False) 
    image_model_y = full_fitfunction(angles, image_params, distribution)
    image_peaks_gof = calculate_peaks_gof(image_stack, image_model_y, 
                            image_peaks_mask[..., :max_fit_peaks, :], method = "r2")

    return image_peaks_gof

def get_sig_peaks_mask(image_stack = None, image_params = None, image_peaks_mask = None, 
                            distribution = "wrapped_cauchy", 
                            amplitude_threshold = 3000, rel_amplitude_threshold = 0.1, 
                            gof_threshold = 0.5, only_mus = False):
    """
    Returns a boolean mask defining which peaks are significant, based on the given thresholds.

    Parameters:
    - image_stack: np.ndarray (n, m, p)
        The image stack containing the measured intensities.
        n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
    - image_params: np.ndarray (n, m, q)
        The output of fitting the image stack, which stores the parameters of the full fitfunction.
    - image_peaks_mask: np.ndarray (n, m, max_find_peaks, p)
        The mask defining which of the p-measurements corresponds to one of the peaks.
        The first two dimensions are the image dimensions.
    - distribution: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
        The name of the distribution.
    - amplitude_threshold: float
        Peaks with a amplitude below this threshold will not be evaluated.
    - rel_amplitude_threshold: float
        Value between 0 and 1.
        Peaks with a relative amplitude (to maximum - minimum intensity of the pixel) below
        this threshold will not be evaluated.
    - gof_threshold: float
        Value between 0 and 1.
        Peaks with a goodness-of-fit value below this threshold will not be evaluated.
    - only_mus: boolean
        Whether only the mus are provided in image_params. If so, only amplitude_threshold is used.

    Returns:
    - image_sig_peaks_mask: np.ndarray (n, m, max_find_peaks)
        A boolean mask for every peak in the image, defining whether the peak is significant.
    """
    if gof_threshold == 0 and amplitude_threshold == 0 and rel_amplitude_threshold == 0:
        image_sig_peaks_mask = np.ones(image_peaks_mask.shape[:-1], dtype = np.bool_)
    else:
        image_amplitudes = get_peak_amplitudes(image_stack, image_params = image_params, 
                            image_peaks_mask = image_peaks_mask, distribution = distribution, 
                            only_mus = only_mus)
        image_rel_amplitudes = get_peak_rel_amplitudes(image_stack, image_amplitudes)
        if gof_threshold == 0 or only_mus:
            image_sig_peaks_mask = ((image_amplitudes > amplitude_threshold)
                                    & (image_rel_amplitudes > rel_amplitude_threshold))
        else:
            image_peaks_gof = get_peak_gofs(image_stack, image_params, image_peaks_mask,
                                            distribution = distribution)

            image_sig_peaks_mask = ((image_amplitudes > amplitude_threshold)
                                        & (image_rel_amplitudes > rel_amplitude_threshold)
                                        & (image_peaks_gof > gof_threshold))

    return image_sig_peaks_mask

#@njit(cache = True, fastmath = True)
def get_number_of_peaks(image_stack = None, image_params = None, image_peaks_mask = None, 
                            distribution = "wrapped_cauchy", 
                            amplitude_threshold = 3000, rel_amplitude_threshold = 0.1, 
                            gof_threshold = 0.5, only_mus = False, 
                            image_sig_peaks_mask = None):
    """
    Returns the number of peaks for every pixel.

    Parameters:
    - image_stack: np.ndarray (n, m, p)
        The image stack containing the measured intensities.
        n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
    - image_params: np.ndarray (n, m, q)
        The output of fitting the image stack, which stores the parameters of the full fitfunction.
        q is the number of parameters.
    - image_peaks_mask: np.ndarray (n, m, max_find_peaks, p)
        The mask defining which of the p-measurements corresponds to one of the peaks.
        The first two dimensions are the image dimensions.
    - distribution: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
        The name of the distribution.
    - amplitude_threshold: float
        Peaks with a amplitude below this threshold will not be evaluated.
    - rel_amplitude_threshold: float
        Value between 0 and 1.
        Peaks with a relative amplitude (to maximum - minimum intensity of the pixel) below
        this threshold will not be evaluated.
    - gof_threshold: float
        Value between 0 and 1.
        Peaks with a goodness-of-fit value below this threshold will not be evaluated.
    - only_mus: boolean
        Whether only the mus (peak centers) are provided in the image_params.
        If so only amplitude_threshold will be used.
    - image_sig_peaks_mask: np.ndarray (n, m, max_find_peaks)
        When significant peaks mask is already calculated, 
        it can be provided here to speed up the process.

    Returns:
    - image_num_peaks: np.ndarray (n, m)
        The number of peaks for every pixel.
    """
    print("Calculating image number of peaks...")

    if not isinstance(image_sig_peaks_mask, np.ndarray):
        image_sig_peaks_mask = get_sig_peaks_mask(image_stack = image_stack, 
                                image_params = image_params, image_peaks_mask = image_peaks_mask,
                                distribution = distribution, 
                                amplitude_threshold = amplitude_threshold, 
                                rel_amplitude_threshold = rel_amplitude_threshold, 
                                gof_threshold = gof_threshold, only_mus = only_mus)

    image_num_peaks = np.sum(image_sig_peaks_mask, axis = -1)

    print("Done")

    return image_num_peaks
    
def get_peak_distances(image_stack = None, image_params = None, image_peaks_mask = None, 
                            image_peak_pairs = None,
                            distribution = "wrapped_cauchy",
                            amplitude_threshold = 3000, rel_amplitude_threshold = 0.1, 
                            gof_threshold = 0.5, only_mus = False, 
                            only_peaks_count = -1, num_processes = 2,
                            image_num_peaks = None, image_sig_peaks_mask = None):
    """
    Returns the distance between (paired) peaks for every pixel (and every direction).
    Note: This function uses pymp until now. Performance could be improved by vectorizing,
    similar to the other map creation functions.

    Parameters:
    - image_stack: np.ndarray (n, m, p)
        The image stack containing the measured intensities.
        n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
    - image_params: np.ndarray (n, m, q)
        The output of fitting the image stack, which stores the parameters of the full fitfunction.
    - image_peaks_mask: np.ndarray (n, m, max_find_peaks, p)
        The mask defining which of the p-measurements corresponds to one of the peaks.
        The first two dimensions are the image dimensions.
    - image_peak_pairs: np.ndarray (n, m, np.ceil(max_paired_peaks / 2), 2)
        The peak pairs for every pixel, where the fourth dimension contains both peak numbers of
        a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the third dimension
        is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
        The first two dimensions are the image dimensions.
        If image_peak_pairs is None, only_peaks_count is set to 2.
    - distribution: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
        The name of the distribution.
    - amplitude_threshold: float
        Peaks with a amplitude below this threshold will not be evaluated.
    - rel_amplitude_threshold: float
        Value between 0 and 1.
        Peaks with a relative amplitude (to maximum - minimum intensity of the pixel) below
        this threshold will not be evaluated.
    - gof_threshold: float
        Value between 0 and 1.
        Peaks with a goodness-of-fit value below this threshold will not be evaluated.
    - only_mus: boolean
        Whether only the mus (peak centers) are provided in the image_params.
        If so only amplitude_threshold will be used.
    - only_peaks_count: int
        Only use pixels where the number of peaks equals this number.
    - num_processes: int
        Defines the number of processes to split the task into.
    - image_num_peaks: np.ndarray (n, m)
        If already calculated the number of peaks for every pixel can be inserted here.
    - image_sig_peaks_mask: np.ndarray (n, m, max_find_peaks, p)
        If already calculated the mask for the peaks for every pixel can be inserted here.

    Returns:
    - image_distances: np.ndarray (n, m)
        The distance between paired peaks for every pixel.
    """

    print("Calculating image peak distances...")
    
    if not only_mus:
        image_mus = image_params[:, :, 1::3]
    else:
        image_mus = image_params

    if not isinstance(image_sig_peaks_mask, np.ndarray):
        image_sig_peaks_mask = get_sig_peaks_mask(image_stack = image_stack, 
                            image_params = image_params, 
                            image_peaks_mask = image_peaks_mask,
                            distribution = distribution, 
                            amplitude_threshold = amplitude_threshold, 
                            rel_amplitude_threshold = rel_amplitude_threshold, 
                            gof_threshold = gof_threshold, only_mus = only_mus)
    
    if not isinstance(image_num_peaks, np.ndarray):
        image_num_peaks = np.sum(image_sig_peaks_mask, axis = -1)

    n_rows = image_mus.shape[0]
    n_cols = image_mus.shape[1]
    total_pixels = n_rows * n_cols
    if not isinstance(image_peak_pairs, np.ndarray) or only_peaks_count == 2:
        only_peaks_count = 2
        image_distances = pymp.shared.array(total_pixels, dtype = np.float32)
    elif isinstance(image_peak_pairs, np.ndarray):
        image_distances = pymp.shared.array((total_pixels, image_peak_pairs.shape[2]), 
                                                dtype = np.float32)
        flat_image_peak_pairs = image_peak_pairs.reshape((total_pixels, *image_peak_pairs.shape[2:]))
    else:
        raise Exception("Error: When you define only_peaks_count for more than 2 peaks, " \
                        "you also have to input image_peak_pairs")

    with pymp.Parallel(num_processes) as p:
        for i in p.range(total_pixels):
            image_distances[i, ...] = -1

    flat_image_num_peaks = image_num_peaks.reshape(total_pixels)
    flat_image_sig_peaks_mask = image_sig_peaks_mask.reshape((total_pixels, image_sig_peaks_mask.shape[2]))
    flat_image_mus = image_mus.reshape((total_pixels, image_mus.shape[2]))

    # Initialize the progress bar
    pbar = tqdm(total = total_pixels, 
                desc = f'Calculating peak distances',
                smoothing = 0)
    shared_counter = pymp.shared.array((num_processes, ), dtype = int)

    with pymp.Parallel(num_processes) as p:
        for i in p.range(total_pixels):

            # Update progress bar
            shared_counter[p.thread_num] += 1
            status = np.sum(shared_counter)
            pbar.update(status - pbar.n)

            if only_peaks_count > 1 and flat_image_num_peaks[i] != only_peaks_count:
                continue
            if not np.any(flat_image_sig_peaks_mask[i]): continue
            if only_peaks_count == 2:
                sig_peak_indices = flat_image_sig_peaks_mask[i].nonzero()[0]
                image_distances[i] = np.abs(angle_distance(flat_image_mus[i, sig_peak_indices[0]], 
                                            flat_image_mus[i, sig_peak_indices[1]]))
                continue

            for j, pair in enumerate(flat_image_peak_pairs[i]):
                if np.any(pair == -1): continue
                if flat_image_sig_peaks_mask[i, pair[0]] and flat_image_sig_peaks_mask[i, pair[1]]:
                    image_distances[i, j] = np.abs(angle_distance(flat_image_mus[i, pair[0]], 
                                            flat_image_mus[i, pair[1]]))

    # Set the progress bar to 100%
    pbar.update(pbar.total - pbar.n)

    if only_peaks_count == 2:
        image_distances = image_distances.reshape((n_rows, n_cols))
    else:
        image_distances = image_distances.reshape((n_rows, n_cols, image_distances.shape[1]))

    return image_distances

def get_mean_peak_amplitudes(image_stack = None, image_params = None, image_peaks_mask = None,
                         distribution = "wrapped_cauchy", 
                        amplitude_threshold = 3000, rel_amplitude_threshold = 0.1,
                        gof_threshold = 0.5, only_mus = False,
                        image_sig_peaks_mask = None):
    """
    Returns the mean peak amplitude for every pixel.

    Parameters:
    - image_stack: np.ndarray (n, m, p)
        The image stack containing the measured intensities.
        n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
    - image_params: np.ndarray (n, m, q)
        The output of fitting the image stack, which stores the parameters of the full fitfunction.
    - image_peaks_mask: np.ndarray (n, m, max_find_peaks, p)
        The mask defining which of the p-measurements corresponds to one of the peaks.
        The first two dimensions are the image dimensions.
    - distribution: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
        The name of the distribution.
    - amplitude_threshold: float
        Peaks with a amplitude below this threshold will not be evaluated.
    - rel_amplitude_threshold: float
        Value between 0 and 1.
        Peaks with a relative amplitude (to maximum - minimum intensity of the pixel) below
        this threshold will not be evaluated.
    - gof_threshold: float
        Value between 0 and 1.
        Peaks with a goodness-of-fit value below this threshold will not be evaluated.
    - only_mus: boolean
        Whether only the mus are provided in image_params. If so, only amplitude_threshold is used.
    - image_sig_peaks_mask: np.ndarray (n, m, max_find_peaks)
        If significant peaks mask is already calculated, 
        it can be inserted here to speed up the process.


    Returns:
    - image_mean_amplitudes: np.ndarray (n, m)
        The mean amplitude for every pixel.
    """

    print("Calculating image peak amplitudes...")

    if not isinstance(image_sig_peaks_mask, np.ndarray):
        image_sig_peaks_mask = get_sig_peaks_mask(image_stack = image_stack, 
                                    image_params = image_params, 
                                    image_peaks_mask = image_peaks_mask,
                                    distribution = distribution, 
                                    amplitude_threshold = amplitude_threshold, 
                                    rel_amplitude_threshold = rel_amplitude_threshold, 
                                    gof_threshold = gof_threshold, only_mus = only_mus)

    image_mean_amplitudes = get_peak_amplitudes(image_stack, image_params = image_params, 
                            image_peaks_mask = image_peaks_mask, distribution = distribution, 
                            only_mus = only_mus)

    image_mean_amplitudes = np.where(image_sig_peaks_mask, image_mean_amplitudes, np.nan)
    all_nan_slices = np.all(~image_sig_peaks_mask, axis = -1)
    image_mean_amplitudes[all_nan_slices] = 0
    image_mean_amplitudes = np.nanmean(image_mean_amplitudes, axis = -1)

    print("Done")

    return image_mean_amplitudes

def get_mean_peak_widths(image_stack = None, image_params = None, image_peaks_mask = None, 
                            distribution = "wrapped_cauchy", 
                            amplitude_threshold = 3000, rel_amplitude_threshold = 0.1,
                            gof_threshold = 0.5, image_sig_peaks_mask = None):
    """
    Returns the mean peak width for every pixel.

    Parameters:
    - image_stack: np.ndarray (n, m, p)
        The image stack containing the measured intensities.
        n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
    - image_params: np.ndarray (n, m, q)
        The output of fitting the image stack, which stores the parameters of the full fitfunction.
    - image_peaks_mask: np.ndarray (n, m, max_find_peaks, p)
        The mask defining which of the p-measurements corresponds to one of the peaks.
        The first two dimensions are the image dimensions.
    - distribution: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
        The name of the distribution.
    - amplitude_threshold: float
        Peaks with a amplitude below this threshold will not be evaluated.
    - rel_amplitude_threshold: float
        Value between 0 and 1.
        Peaks with a relative amplitude (to maximum - minimum intensity of the pixel) below
        this threshold will not be evaluated.
    - gof_threshold: float
        Value between 0 and 1.
        Peaks with a goodness-of-fit value below this threshold will not be evaluated.
    - image_sig_peaks_mask: np.ndarray (n, m, max_find_peaks)
        If significant peaks mask is already calculated, 
        it can be inserted here to speed up the process.

    Returns:
    - image_mean_widths: np.ndarray (n, m)
        The mean amplitude for every pixel.
    """

    print("Calculating image peak widths...")

    if not isinstance(image_sig_peaks_mask, np.ndarray):
        image_sig_peaks_mask = get_sig_peaks_mask(image_stack = image_stack, image_params = image_params, 
                                    image_peaks_mask = image_peaks_mask,
                                    distribution = distribution, 
                                    amplitude_threshold = amplitude_threshold, 
                                    rel_amplitude_threshold = rel_amplitude_threshold, 
                                    gof_threshold = gof_threshold, only_mus = only_mus)

    image_mean_widths = image_params[:, :, 2::3]
    image_mean_widths = np.where(image_sig_peaks_mask, image_mean_widths, np.nan)
    all_nan_slices = np.all(~image_sig_peaks_mask, axis = -1)
    image_mean_widths[all_nan_slices] = 0
    image_mean_widths = np.nanmean(image_mean_widths, axis = -1)

    print("Done")

    return image_mean_widths

@njit(cache = True, fastmath = True)
def peak_pairs_to_amplitudes(intensities, only_mus, params, peaks_mask, distribution):
    
    if not only_mus:
        heights = params[0:-1:3]
        scales = params[2::3]
    
    max_peaks = peaks_mask.shape[0]
    amplitudes = np.zeros(max_peaks)    
    for i in range(max_peaks):
        if not only_mus and i >= len(heights):
            amplitudes = amplitudes[:len(heights)]
            break
        elif not np.any(peaks_mask[i]): continue
        elif not only_mus:
            if heights[i] > 0:
                amplitudes[i] = heights[i] * distribution_pdf(0, 0, scales[i], distribution)[0]
        else:
            peak_intensities = intensities[peaks_mask[i]]
            amplitudes[i] = np.max(peak_intensities) - np.min(intensities)
        
    return amplitudes

#@njit(cache = True, fastmath = True)
def image_sli_to_pli(image_stack, image_peak_pairs, image_params, only_mus, image_peaks_mask,
                        distribution, mu_s = 0.548, b = 0.782, amp_0 = 1.001, 
                        inclination_max = 80 * np.pi / 180):

    image_amplitudes = get_peak_amplitudes(image_stack, image_params = image_params, 
                            image_peaks_mask = image_peaks_mask, 
                            distribution = distribution, only_mus = only_mus)
    image_amplitudes = image_amplitudes / np.max(image_amplitudes)

    if not only_mus:
        image_mus = image_params[:, :, 1::3]
    else:
        image_mus = image_params

    pli_direction_image = np.full(image_mus.shape[:-1], -1, dtype = np.float64)
    pli_retardation_image = np.full(image_mus.shape[:-1], -1, dtype = np.float64)
    for x in range(image_mus.shape[0]):
        for y in range(image_mus.shape[1]):
            pli_direction_image[x, y], pli_retardation_image[x, y] = sli_to_pli(image_peak_pairs[x, y], 
                        image_mus[x, y], image_amplitudes[x, y], mu_s, b, amp_0, inclination_max)

    return pli_direction_image, pli_retardation_image

@njit(cache = True, fastmath = True)
def sli_to_pli_brute(peak_pairs, mus, pli_direction, pli_retardation):
    directions = peak_pairs_to_directions(peak_pairs, mus, exclude_lone_peaks = True)
    directions = directions[directions != -1]
    num_directions = len(directions)
    
    sim_pli_direction = -1
    sim_pli_retardation = -1
    min_diff = np.inf
    if num_directions == 1:
        # For one directions return the direction
        sim_pli_direction = directions[0]
        sim_pli_retardation = pli_retardation
        min_diff = (angle_distance(sim_pli_direction, pli_direction) / np.pi)**2
    elif num_directions == 2 or num_directions == 3:
        # Find best match by looping through retardation values (since SLI has no retardation values)
        # Possible To-Do: For low peak distances (high inclination) the retardation must be low

        test_retardations = np.linspace(0, 1, 101)
        sim_pli_retardation = -1
        sim_pli_direction = -1
        num_loops_3 = len(test_retardations) if num_directions == 3 else 1

        for i in range(num_loops_3):
            ret_3 = test_retardations[i] 
            for ret_1 in test_retardations:
                for ret_2 in test_retardations:
                    retardations = np.array([ret_1, ret_2, ret_3])
                    retardations = retardations[:num_directions]
                    total_dir, total_ret = add_birefringence(directions, retardations)
                    dir_diff = (angle_distance(total_dir, pli_direction, wrap = np.pi))
                    dir_diff = dir_diff / np.pi
                    ret_diff = (total_ret - pli_retardation)
                    diff = (dir_diff**2 + ret_diff**2) / 2
                    if diff < min_diff:
                        min_diff = diff
                        sim_pli_retardation = total_ret
                        sim_pli_direction = total_dir

    return sim_pli_direction, sim_pli_retardation, min_diff




@njit(cache = True, fastmath = True)
def sli_to_pli(peak_pairs, mus, norm_amplitudes, 
                    mu_s = 0.548, b = 0.782, amp_0 = 1.001, inclination_max = 80 * np.pi / 180):
    """
    Converts the results from a SLI measurement (peak pairs, mus, heights) to a virtual PLI measurement.
    "What would be measured in a PLI measurement for the found nerve fibers in the SLI measurement?"

    Parameters:
    - peak_pairs: np.ndarray (np.ceil(m / 2), 2)
        Array ontaining both peak numbers of a pair (e.g. [1, 3], 
        which means peak 1 and peak 3 is paired). A pair with -1 defines a unpaired peak.
        The first dimension (m equals number of peaks)
        is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
    - mus: np.ndarray (m, )
        The center positions of the peaks.
    - norm_amplitudes: np.ndarray (m, )
        The norm amplitudes of the peaks (normalized by maximum amplitude of the image).
    - mu_s: float 
        Scattering coefficient (from a fit).
    - b: float
        Constant (depending on wavelength and refractive index) in retardation formula (from fit).
    - amp_0: float
        Relative intensity before the scattering (from a fit).

    Returns:
    - pli_direction: float
        Virtual direction value of pli measurement.
    - pli_retardation: float
        Virtual retardation value of pli measurement.

    """

    directions = peak_pairs_to_directions(peak_pairs, mus, exclude_lone_peaks = True)
    directions = directions[directions != -1]
    num_directions = len(directions)
    
    if num_directions == 0:
        pli_direction = -1
    elif num_directions == 1:
        # For one directions return the direction
        pli_direction = directions[0]
    else:
        
        pair_distances = np.zeros(num_directions)
        pair_amplitudes = np.zeros(num_directions)

        for i in range(num_directions):
            peak_indices = peak_pairs[i]
            pair_distances[i] = np.abs(angle_distance(mus[peak_indices[0]], mus[peak_indices[1]]))
            pair_amplitudes[i] = (norm_amplitudes[peak_indices[0]] + norm_amplitudes[peak_indices[1]]) / 2
        
        retardations = sli_to_pli_retardation(pair_distances, pair_amplitudes, 
                                            mu_s, b, amp_0, inclination_max)

        pli_direction, pli_retardation = add_birefringence(directions, retardations)

    return pli_direction, pli_retardation

@njit(cache = True, fastmath = True)
def distance_to_inclination(distance, inclination_max):
    # Calculate inclination from peak distance 
    # and the max inclination that result in a peak (around 80 degree)
    # Assuming distance = 180 * sqrt(cos(inclination / inclination_max * (pi / 2)))
    # which ist just a heuristic estimation
    inclination = (2 * inclination_max / np.pi) * np.arccos((distance / np.pi)**2)
    return inclination

@njit(cache = True, fastmath = True)
def distance_to_retardation(distance, thickness, b, inclination_max):
    # Calculate retardation from peak distance (radians), thickness and a (fit) constant b
    # where b = refractive_indice_difference * 2pi / wavelength
    inclination = distance_to_inclination(distance, inclination_max)
    retardation = np.abs(np.sin(b * thickness * np.cos(inclination)**2))
    return retardation

@njit(cache = True, fastmath = True)
def amplitude_to_thickness(amplitude, mu_s, amp_0):
    # Calculate thickness from peak amplitude and (fit) constants mu_s and amp_0
    # (mu_s is the scattering coefficient and amp_0 the relative amplitude before scattering)
    # Assuming Lamberts beer law and a asymptotic maximum thickness
    # but this formular is just a heuristic estimation
    thickness_limit = np.pi / 2
    thickness = -1 / mu_s * np.log(1 - (amplitude / amp_0))
    thickness = thickness_limit * (1 - np.exp(-thickness))
    return thickness

@njit(cache = True, fastmath = True)
def sli_to_pli_retardation(distance, amplitude, mu_s, b, amp_0, inclination_max):
    # Calculate pli retardation from SLI parameters peak distance and (mean) peak amplitude of a pair
    thickness = amplitude_to_thickness(amplitude, mu_s, amp_0)
    retardation = distance_to_retardation(distance, thickness, b, inclination_max)
    return retardation