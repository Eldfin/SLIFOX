import numpy as np
import h5py
from numba import njit, prange
from .fitter import full_fitfunction, angle_distance
from .wrapped_distributions import distribution_pdf, wrapped_cauchy_pdf
from collections import deque
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

def _find_closest_true_pixel_new(mask, start_pixel, radius):
    """
    Finds the closest true pixel for a given 2D-mask and a start_pixel, within a given radius.

    Parameters:
    - mask: np.ndarray (n, m)
        The boolean mask defining which pixels are False or True.
    - start_pixel: tuple
        The x- and y-coordinates of the start_pixel.
    - radius: int
        The radius within which to search for the closest true pixel.

    Returns:
    - closest_true_pixel: tuple
        The x- and y-coordinates of the closest true pixel, 
        or (-1, -1) if no true pixel is found within the radius.
    """
    # Step 1: Get the coordinates of all True pixels in the mask
    true_pixel_coords = np.argwhere(mask)
    
    # If there are no true pixels, return (-1, -1)
    if true_pixel_coords.size == 0:
        return (-1, -1)

    # Step 2: Build KDTree for the True pixels
    tree = KDTree(true_pixel_coords)

    # Step 3: Query the tree for the closest pixel within the radius
    dist, idx = tree.query(start_pixel, distance_upper_bound=radius)
    
    # Step 4: If a valid pixel is found (distance is finite), return its coordinates
    if np.isinf(dist):
        return (-1, -1)
    else:
        return tuple(true_pixel_coords[idx])

def _find_closest_true_pixel(mask, start_pixel, radius):
    """
    Finds the closest true pixel for a given 2d-mask and a start_pixel within a given radius.

    Parameters:
    - mask: np.ndarray (n, m)
        The boolean mask defining which pixels are False or True.
    - start_pixel: tuple
        The x- and y-coordinates of the start_pixel.
    - radius: int
        The radius within which to search for the closest true pixel.

    Returns:
    - closest_true_pixel: tuple
        The x- and y-coordinates of the closest true pixel, or (-1, -1) if no true pixel is found.
    """
    rows, cols = mask.shape
    sr, sc = start_pixel

    visited = np.zeros_like(mask, dtype=bool)
    queue = deque([(sr, sc)])
    visited[sr, sc] = True

    while queue:
        r, c = queue.popleft()

        if mask[r, c]:
            return (r, c)

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                if abs(nr - sr) <= radius and abs(nc - sc) <= radius:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    # When no true pixel in the mask within the radius, return (-1, -1)
    return (-1, -1)

def get_image_peak_pairs(image_stack, image_params, image_peaks_mask, min_distance = 20,
                            distribution = "wrapped_cauchy", only_mus = False, num_processes = 2,
                            amplitude_threshold = 3000, rel_amplitude_threshold = 0.1,
                            gof_threshold = 0.5, significance_threshold = 0.3, 
                            significance_weights = [1, 1],
                            angle_threshold = 20, max_attempts = 10000, 
                            search_radius = 50, min_directions_diff = 20, exclude_lone_peaks = True):
    """
    Finds all the peak_pairs for a whole image stack and sorts them by comparing with neighbour pixels.

    Parameters:
    - image_stack: np.ndarray (n, m, p)
        The image stack containing the measured intensities.
        n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
    - image_params: np.ndarray (n, m, q)
        The output of fitting the image stack, which stores the parameters of the full fitfunction.
        q = 3 * max_peaks + 1, is the number of parameters (max 19 for 6 peaks).
        Can also store only mus, when only_mus = True.
    - image_peaks_mask: np.ndarray (n, m, max_peaks, p)
        The mask defining which of the p-measurements corresponds to one of the peaks.
        The first two dimensions are the image dimensions.
    - min_distance: float
        Defines the minimum distance between two paired peaks in degrees, when more than 2 peaks are present.
        If this value is close to 180 the peak finding method converges to a method where neighbours
        does not matter and only peaks with distances around 180 are paired.
    - distribution: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
        The name of the distribution.
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
        Value between 0 and 1.
        Peaks with a goodness-of-fit value below this threshold will not be evaluated.
    - significance_threshold: float
        Value between 0 and 1. Peak Pairs with peaks that have a significance
        lower than this threshold are not considered for possible pairs.
        See also "direction_significance" function for more info.
    - significance_weights: list (2, )
        The weights for the amplitude and for the goodnes-of-fit, when calculating the significance.
        See also "direction_significance" function for more info.
    - angle_threshold: float
        Threshold in degrees defining when a neighbouring pixel direction is considered as same nerve fiber.
    - max_attempts: int
        Number defining how many times it should be attempted to find a neighbouring pixel within
        the given "angle_threshold".
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

    Returns:
    - image_peak_pair_combs: np.ndarray (n, m, p, np.ceil(max_peaks / 2), 2)
        The possible peak pair combinations for every pixel (sorted by difference to neighbours), 
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
    max_peaks = image_peaks_mask.shape[2]
    n_rows = image_stack.shape[0]
    n_cols = image_stack.shape[1]
    total_pixels = n_rows * n_cols
    angles = np.linspace(0, 2*np.pi, num = image_stack.shape[2], endpoint = False)
    max_combs = 3
    if max_peaks >= 5:
        max_combs = 15
    image_peak_pair_combs = pymp.shared.array((total_pixels,
                                            max_combs, 
                                            np.ceil(max_peaks / 2).astype(int), 
                                            2), dtype = np.int16)

    with pymp.Parallel(num_processes) as p:
        for i in p.range(total_pixels):
            image_peak_pair_combs[i, :, :, :] = -1

    image_peak_pair_combs = image_peak_pair_combs.reshape(n_rows, n_cols,
                                                image_peak_pair_combs.shape[1],
                                                image_peak_pair_combs.shape[2],
                                                image_peak_pair_combs.shape[3])

    image_num_peaks, sig_image_peaks_mask = get_number_of_peaks(image_stack, image_params, image_peaks_mask, 
                            distribution = distribution, only_mus = only_mus, 
                            amplitude_threshold = amplitude_threshold,
                            rel_amplitude_threshold = rel_amplitude_threshold,
                            gof_threshold = gof_threshold)

    direction_found_mask = pymp.shared.array((n_rows, n_cols), dtype = np.bool_)

    # iterate from 2 to max_peaks and 1, 0 at last
    iteration_list = np.append(np.arange(2, max_peaks + 1), [1, 0])
    for iteration_index, i in enumerate(iteration_list):
        mask = (image_num_peaks == i)
        peak_pairs_combinations = possible_pairs(i)
        num_combs = peak_pairs_combinations.shape[0]
        indices = np.argwhere(mask)
        num_pixels_iteration = np.count_nonzero(mask)

        if indices.size == 0: continue
        no_processed = True
        if i != 2:
            processed_mask = np.isin(image_num_peaks, iteration_list[:iteration_index])
            processed_indices = np.argwhere(processed_mask)
            if processed_indices.size != 0:
                tree = KDTree(processed_indices)
                distances, _ = tree.query(indices)
                sorted_indices = np.argsort(distances)
                no_processed = False
            pbar.close()

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
                    sig_peak_indices = sig_image_peaks_mask[x, y].nonzero()[0]

                    # Update progress bar
                    shared_counter[p.thread_num] += 1
                    status = np.sum(shared_counter)
                    pbar.update(status - pbar.n)

                    if i == 0:
                        #image_peak_pair_combs[x, y] = np.array([[[-1, -1]]])
                        continue
                    elif i <= 2:
                        sig_peak_pair_combs = np.where(peak_pairs_combinations == -1, -1, 
                                                        sig_peak_indices[peak_pairs_combinations])
                        image_peak_pair_combs[x, y, 
                                        :peak_pairs_combinations.shape[0],
                                        :peak_pairs_combinations.shape[1]] \
                                    = sig_peak_pair_combs
                        direction_found_mask[x, y] = True
                        continue

                    params = image_params[x, y]
                    peaks_mask = image_peaks_mask[x, y]
                    intensities = image_stack[x, y]

                    if not only_mus:
                        mus = params[1::3]
                    else:
                        mus = params

                    valid_combs = np.ones(num_combs, dtype = np.bool_)
                    direction_combs = np.full((num_combs, image_peak_pair_combs.shape[3]), -1, dtype = np.float64)
                    for k in range(num_combs):
                        peak_pairs = np.where(peak_pairs_combinations[k] == -1, -1, 
                                                        sig_peak_indices[peak_pairs_combinations[k]])

                        # Check if any pair has a smaller distance than min_distance
                        for pair in peak_pairs:
                            if np.any(pair == -1): continue
                            distance = np.abs(angle_distance(mus[pair[0]], mus[pair[1]]))
                            if distance < min_distance * np.pi / 180:
                                valid_combs[k] = False
                                break

                        if not valid_combs[k]:
                            continue

                        significances = direction_significances(peak_pairs, params, peaks_mask, 
                                    intensities, angles, weights = significance_weights, 
                                    distribution = distribution, only_mus = only_mus)

                        if np.all(significances <= significance_threshold): 
                            valid_combs[k] = False
                            continue

                        directions = peak_pairs_to_directions(peak_pairs, mus, 
                                                            exclude_lone_peaks = exclude_lone_peaks)
                        # Filter directions with low significance out
                        directions = directions[significances > significance_threshold]

                        directions = directions[directions != -1]
                    
                        # If any differences between directions are below min_directions_diff,
                        # its not a valid peak pair combination
                        #differences = np.abs(directions[:, np.newaxis] - directions)
                        #differences = differences[differences != 0]
                        if len(directions) > 1:
                            # Calculate the differences between every directions
                            differences = np.abs(angle_distance(directions[:, np.newaxis], 
                                                    directions[np.newaxis, :], wrap = np.pi))
                            differences = differences[np.triu_indices(len(directions), k=1)]

                            if np.any(differences < min_directions_diff * np.pi / 180):
                                valid_combs[k] = False
                                continue
                        elif len(directions) == 0:
                            valid_combs[k] = False

                        direction_combs[k, :len(directions)] = directions

                    if not np.any(valid_combs):
                        continue

                    sig_peak_pair_combs = np.where(peak_pairs_combinations[valid_combs] == -1, -1, 
                                                        sig_peak_indices[peak_pairs_combinations[valid_combs]])
                    direction_combs = direction_combs[valid_combs]
                    num_sig_combs = sig_peak_pair_combs.shape[0]

                    if num_sig_combs == 1:
                        image_peak_pair_combs[x, y, 
                                        :sig_peak_pair_combs.shape[0],
                                        :sig_peak_pair_combs.shape[1]] = sig_peak_pair_combs
                        direction_found_mask[x, y] = True
                        continue

                    check_mask = np.copy(direction_found_mask)
                    num_best_combs = num_sig_combs
                    unmatched_dir_mask = np.ones(direction_combs.shape, dtype = np.bool_)
                    for attempt in range(max_attempts):
                        neighbour_x, neighbour_y = _find_closest_true_pixel(check_mask, (x, y), 
                                                                        search_radius)
                        if neighbour_x == -1 and neighbour_y == -1:
                            # When no true pixel within radius: return no pairs
                            #image_peak_pair_combs[x, y] = np.array([[[-1, -1]]])
                            break

                        neighbour_peak_pairs = image_peak_pair_combs[neighbour_x, neighbour_y, 0]
                        neighbour_params = image_params[neighbour_x, neighbour_y]
                        neighbour_peaks_mask = image_peaks_mask[neighbour_x, neighbour_y]
                        neighbour_intensities = image_stack[neighbour_x, neighbour_y]
                        if not only_mus:
                            neighbour_mus = neighbour_params[1::3]
                        else:
                            neighbour_mus = neighbour_params

                        #To-Do: Filter neighbour peaks with amplitude and gof threshold

                        neighbour_directions = peak_pairs_to_directions(neighbour_peak_pairs, 
                                            neighbour_mus, 
                                            exclude_lone_peaks = exclude_lone_peaks)

                        neighbour_significances = direction_significances(neighbour_peak_pairs, 
                                    neighbour_params, neighbour_peaks_mask, 
                                    neighbour_intensities, angles, weights = significance_weights, 
                                    distribution = distribution, only_mus = only_mus)

                        # Filter neighbour directions with low significance out
                        neighbour_directions = neighbour_directions[neighbour_significances
                                                                        > significance_threshold]

                        neighbour_directions = neighbour_directions[neighbour_directions != -1]

                        if len(neighbour_directions) == 0:
                            check_mask[neighbour_x, neighbour_y] = False
                            if attempt == max_attempts - 1 or not np.any(check_mask):
                                # When no neighbouring pixel within max_attempts had
                                # a direction difference below the threshold
                                # return no peak pairs
                                image_peak_pair_combs[x, y] = np.array([[[-1, -1]]])
                                break
                            continue

                        direction_diffs = np.empty(num_best_combs, dtype = np.float64)
                        dir_diff_indices = np.empty(num_best_combs, dtype = np.int16)
                        no_directions = False
                        for k in range(num_best_combs):
                            directions = direction_combs[k]
                            directions = directions[unmatched_dir_mask[k]]
                            directions = directions[directions != -1]

                            if len(directions) == 0:
                                no_directions = True
                                break

                            differences = np.abs(neighbour_directions[:, np.newaxis] - directions)
                            min_diff_index = np.argmin(differences)
                            _, dir_diff_indices[k] = np.unravel_index(min_diff_index, 
                                                                                    differences.shape)
                            # Insert minimum difference to neighbour directions into array for sorting later
                            direction_diffs[k] = np.min(differences)

                        if no_directions:
                            image_peak_pair_combs[x, y, 
                                            :sig_peak_pair_combs.shape[0],
                                            :sig_peak_pair_combs.shape[1]] = sig_peak_pair_combs
                            direction_found_mask[x, y] = True
                            break

                        if np.min(direction_diffs) < angle_threshold * np.pi / 180:
                            # If minimum difference to neighbour direction is smaller than given threshold
                            # Sort possible peak pairs by difference to neighbour directions
                            # and accept the result

                            sort_indices = np.argsort(direction_diffs)
                            sig_peak_pair_combs[:num_best_combs] = sig_peak_pair_combs[sort_indices]
                            direction_combs[:num_best_combs] = direction_combs[sort_indices]
                            unmatched_dir_mask[:num_best_combs] = unmatched_dir_mask[sort_indices]
                            old_num_best_combs = num_best_combs
                            num_best_combs = np.sum(direction_diffs == np.min(direction_diffs))
                            best_dir_indices = dir_diff_indices[sort_indices[:num_best_combs]]
                            for k in range(num_best_combs):
                                unmatched_dir_mask[k, best_dir_indices[k]] = False

                            direction_combs = direction_combs[:num_best_combs]
                            if num_best_combs == 1 or num_best_combs == old_num_best_combs:
                                image_peak_pair_combs[x, y, 
                                            :sig_peak_pair_combs.shape[0],
                                            :sig_peak_pair_combs.shape[1]] = sig_peak_pair_combs
                                direction_found_mask[x, y] = True
                                break
                            else:
                                continue
                        else:
                            check_mask[neighbour_x, neighbour_y] = False
                            if attempt == max_attempts - 1 or not np.any(check_mask):
                                # When no neighbouring pixel within max_attempts had
                                # a direction difference below the threshold
                                # return no peak pairs
                                image_peak_pair_combs[x, y] = np.array([[[-1, -1]]])
                                break

    return image_peak_pair_combs

@njit(cache = True, fastmath = True)
def peak_pairs_to_directions(peak_pairs, mus, exclude_lone_peaks = True):
    """
    Calculates the directions from given peak_pairs of a pixel.

    Parameters:
    - peak_pairs: np.ndarray (m // 2, 2)
        Array containing both peak numbers of a pair (e.g. [1, 3], 
        which means peak 1 and peak 3 is paired). A pair with -1 defines a unpaired peak.
        The first dimension (m equals number of peaks)
        is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
    - mus: np.ndarray (m, )
        The center positions of the peaks.
    - exclude_lone_peaks: bool
        Whether to exclude the directions for lone peaks 
        (for peak pairs with only one number unequal -1 e.g. [2, -1]).

    Returns:
    - directions: np.ndarray (m // 2, )
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
                direction = mus[pair[pair != -1][0]] % (np.pi)
        else:
            # Two peak direction
            distance = angle_distance(mus[pair[0]], mus[pair[1]])
            direction = (mus[pair[0]] + distance / 2) % (np.pi)

        directions[k] = direction

    return directions

@njit(cache = True, fastmath = True)
def peak_pairs_to_inclinations(peak_pairs, mus):
    """
    Placeholder function until a prober way to get the inclinations from a SLI measurement is found.
    Calculates the inclinations from given peak_pairs of a pixel.

    Parameters:
    - peak_pairs: np.ndarray (m // 2, 2)
        Array ontaining both peak numbers of a pair (e.g. [1, 3], 
        which means peak 1 and peak 3 is paired). A pair with -1 defines a unpaired peak.
        The first dimension (m equals number of peaks)
        is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
    - mus: np.ndarray (m, )
        The center positions of the peaks.

    Returns:
    - inclinations: np.ndarray (m // 2, )
        The calculated inclinations for every peak pair.
    """
    num_pairs = peak_pairs.shape[0]
    inclinations = np.empty(num_pairs)
    for i in range(num_pairs):
        current_mus = mus[peak_pairs[i]]
        distance_deviation = np.pi - np.abs(angle_distance(current_mus[0], current_mus[1]))
        inclinations[i] = distance_deviation

    return inclinations

def calculate_directions(image_peak_pairs, image_mus, only_peaks_count = -1, exclude_lone_peaks = True):
    """
    Calculates the directions from given image_peak_pairs.

    Parameters:
    - image_peak_pairs: np.ndarray (n, m, np.ceil(max_peaks / 2), 2)
        The peak pairs for every pixel, where the fourth dimension contains both peak numbers of
        a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the third dimension
        is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
        The first two dimensions are the image dimensions.
    - image_mus: np.ndarray (n, m, max_peaks)
        The mus (centers) of the found (max_peaks) peaks for everyone of the (n * m) pixels.
    - directory: string
        The directory path defining where direction images should be writen to.
        If None, no images will be writen.
    - only_peaks_count: int (or list of ints)
        Defines a filter for the number of peaks, so that only pixels will be processed that have
        this number of peaks.
    - exclude_lone_peaks: bool
        Whether to exclude the directions for lone peaks 
        (for peak pairs with only one number unequal -1 e.g. [2, -1]).

    Returns:
    - image_directions: (n, m, np.ceil(max_peaks / 2))
        The calculated directions for everyoe of the (n * m) pixels.
        Max 3 directions (for 6 peaks).
    """
    print("Calculating image directions...")

    x_range = image_peak_pairs.shape[0]
    y_range = image_peak_pairs.shape[1]
    max_directions = image_peak_pairs.shape[2]
    image_directions = np.full((x_range, y_range, max_directions), -1, dtype=np.float64)
    for x in range(x_range):
        for y in range(y_range):
            if only_peaks_count != -1:
                num_peaks = int(np.max(image_peak_pairs[x, y]) + 1)
                if isinstance(only_peaks_count, list):
                    if num_peaks not in only_peaks_count: continue
                else:
                    if num_peaks != only_peaks_count: continue
            
            image_directions[x, y] = peak_pairs_to_directions(image_peak_pairs[x, y], image_mus[x, y],
                                                                exclude_lone_peaks = exclude_lone_peaks)

    print("Done")

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
def direction_significances(peak_pairs, params, peaks_mask, intensities, angles, weights = [1, 1],
                            distribution = "wrapped_cauchy", only_mus = False):
    """
    Calculates the significances of the directions for one (fitted) pixel.
    (Old function, performance can be improved similar to get_image_direction_significances).

    Parameters:
    - peak_pairs: np.ndarray (np.ceil(m / 2), 2)
        Array ontaining both peak numbers of a pair (e.g. [1, 3], 
        which means peak 1 and peak 3 is paired). The first dimension (m equals number of peaks)
        is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
    - params: np.ndarray (q, )
        The output of fitting the pixel, which stores the parameters of the full fitfunction.
        q = 3 * max_peaks + 1, is the number of parameters (max 19 for 6 peaks).
    - peaks_mask: np.ndarray (m, n)
        m is the number of peaks.
        The mask defining which of the n-measurements corresponds to one of the m-peaks.
    - intensities: np.ndarray (n, )
        The measured intensities of the pixel.
    - angles: np.ndarray (n, )
        The angles at which the intensities are measured.
    - weights: list (2, )
        The weights for the amplitude and for the goodnes-of-fit, when calculating the significance
    - distribution: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
        The name of the distribution.
    - only_mus: bool
        Defines if only the mus are given in the params.

    Returns:
    - significances: np.ndarray (np.ceil(m / 2), )
        The calculated significance for every direction (peak-pair) ranging from 0 to 1.
    """
    global_amplitude = np.max(intensities) - np.min(intensities)
    num_directions = peak_pairs.shape[0]
    significances = np.zeros(num_directions)

    if only_mus:
        # If only mus are provided, just use amplitude significance
        num_peaks = peaks_mask.shape[0]
        amplitudes = np.zeros(num_peaks)
        for i in range(num_peaks):
            peak_intensities = intensities[peaks_mask[i]]
            if len(peak_intensities) == 0: continue
            amplitudes[i] = np.max(peak_intensities) - np.min(intensities)

        for i in range(num_directions):
            peak_pair = peak_pairs[i]
            indices = peak_pair[peak_pair != -1]
            if len(indices) == 0: continue
            significances[i] = np.mean(amplitudes[indices] / global_amplitude)
        
        return significances
    
    model_y = full_fitfunction(angles, params, distribution)
    peaks_gof = calculate_peaks_gof(intensities, model_y, peaks_mask, method = "r2")
    heights = params[0:-1:3]
    scales = params[2::3]

    for i in range(num_directions):
        peak_pair = peak_pairs[i]
        if peak_pair[0] == -1 and peak_pair[1] == -1: 
            continue
        elif peak_pair[0] == -1 or peak_pair[1] == -1:
            peak_index = peak_pair[peak_pair != -1][0]
            amplitude = heights[peak_index] * distribution_pdf(0, 0, scales[peak_index], distribution)
            amplitude_significance = amplitude / global_amplitude
            gof_significance = peaks_gof[peak_index]
        else:
            amplitudes = np.empty(2)
            for k in range(2):
                amplitudes[k] = heights[peak_pair[k]]  \
                                        * distribution_pdf(0, 0, scales[peak_pair[k]], distribution)
            amplitude_significance = np.mean(amplitudes / global_amplitude)
            gof_significance = np.mean(peaks_gof[peak_pair])

        significances[i] = (amplitude_significance * weights[0] + gof_significance * weights[1]) / 2

    return significances

def get_image_direction_significances(image_stack, image_peak_pairs, image_params, image_peaks_mask, 
                            distribution = "wrapped_cauchy", 
                            amplitude_threshold = 0, rel_amplitude_threshold = 0,
                            gof_threshold = 0,
                            weights = [1, 1], only_mus = False):
    """
    Returns the direction significances for every pixel.

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
    - only_mus: boolean
        Whether only the mus are provided in image_params. If so, only amplitude_threshold is used.

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

    if not only_mus:
        image_heights = image_params[:, :, 0:-1:3]
        image_scales = image_params[:, :, 2::3]
        image_amplitudes = image_heights * \
                                distribution_pdf(0, 0, image_scales, distribution)[..., 0]
        image_rel_amplitudes = image_amplitudes / image_global_amplitudes[..., np.newaxis]
        image_model_y = full_fitfunction(angles, image_params, distribution)
        image_peaks_gof = calculate_peaks_gof(image_stack, image_model_y, image_peaks_mask, method = "r2")

        image_valid_peaks_mask = ((image_amplitudes > amplitude_threshold)
                                    & (image_rel_amplitudes > rel_amplitude_threshold)
                                    & (image_peaks_gof > gof_threshold))
    else:
        image_intensities = np.expand_dims(image_stack, axis = 2)
        image_intensities = np.where(image_peaks_mask, image_intensities, 0)
        image_amplitudes = (np.max(image_intensities, axis = -1)
                                - np.min(image_stack, axis = -1)[..., np.newaxis])
        image_rel_amplitudes = image_amplitudes / image_global_amplitudes[..., np.newaxis]
        image_valid_peaks_mask = ((image_amplitudes > amplitude_threshold)
                                    & (image_rel_amplitudes > rel_amplitude_threshold))
    
    # Set unvalid values to -2, so the calculated mean later cant be greater than 0
    image_rel_amplitudes[~image_valid_peaks_mask] = -2
    if not only_mus:
        image_peaks_gof[~image_valid_peaks_mask] = -2

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

        image_direction_sig = (image_rel_amplitudes * weights[0] + image_peaks_gof * weights[1]) / 2
    else:
        image_direction_sig = image_rel_amplitudes

    print("Done")

    return image_direction_sig

@njit(cache = True, fastmath = True)
def peak_significances(intensities, angles, params, peaks_mask, distribution, only_mus,
                        significance_weights):
    global_amplitude = np.max(intensities) - np.min(intensities)
    num_peaks = peaks_mask.shape[0]

    if only_mus:
        # If only mus are provided, just use amplitude significance
        
        amplitudes = np.zeros(num_peaks)
        for i in range(num_peaks):
            peak_intensities = intensities[peaks_mask[i]]
            if len(peak_intensities) == 0:
                continue
            amplitudes[i] = np.max(peak_intensities) - np.min(peak_intensities)

        significances = amplitudes / global_amplitude
        return significances

    model_y = full_fitfunction(angles, params, distribution)
    peaks_gof = calculate_peaks_gof(intensities, model_y, peaks_mask, method = "r2")

    heights = params[0:-1:3]
    scales = params[2::3]
    rel_amplitudes = np.zeros(num_peaks)
    for i in range(num_peaks):
        if heights[i] < 1:
            continue
        amplitude = heights[i] * distribution_pdf(0, 0, scales[i], distribution)
        rel_amplitudes[i] = amplitude / global_amplitude

    significances = (rel_amplitudes * significance_weights[0] + peaks_gof * significance_weights[1]) / 2

    return significances

#@njit(cache = True, fastmath = True)
def get_number_of_peaks(image_stack, image_params, image_peaks_mask, distribution = "wrapped_cauchy", 
                            amplitude_threshold = 3000, rel_amplitude_threshold = 0.1, 
                            gof_threshold = 0.5, only_mus = False):
    """
    Returns the number of peaks for every pixel.

    Parameters:
    - image_stack: np.ndarray (n, m, p)
        The image stack containing the measured intensities.
        n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
    - image_params: np.ndarray (n, m, q)
        The output of fitting the image stack, which stores the parameters of the full fitfunction.
        q = 3 * max_peaks + 1, is the number of parameters (max 19 for 6 peaks).
    - image_peaks_mask: np.ndarray (n, m, max_peaks, p)
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

    Returns:
    - image_num_peaks: np.ndarray (n, m)
        The number of peaks for every pixel.
    - image_valid_peaks_mask: np.ndarray (n, m, max_peaks)
        Mask that stores the information, which peaks are used in the counting process.
    """
    print("Calculating image number of peaks...")

    if gof_threshold == 0 and amplitude_threshold == 0 and rel_amplitude_threshold == 0:
        # Get the number of peaks for every pixel
        image_num_peaks = np.sum(np.any(image_peaks_mask, axis=-1), axis = -1)
        image_valid_peaks_mask = np.ones(image_peaks_mask.shape[:-1], dtype = np.bool_)
   
    else:
        angles = np.linspace(0, 2*np.pi, num = image_stack.shape[2], endpoint = False) 
        image_global_amplitudes = np.max(image_stack, axis = -1) - np.min(image_stack, axis = -1)
        image_global_amplitudes[image_global_amplitudes == 0] = 1

        if not only_mus:
            image_heights = image_params[:, :, 0:-1:3]
            image_scales = image_params[:, :, 2::3]
            image_amplitudes = image_heights * \
                                distribution_pdf(0, 0, image_scales, distribution)[..., 0]
            image_rel_amplitudes = image_amplitudes / image_global_amplitudes[..., np.newaxis]
            image_model_y = full_fitfunction(angles, image_params, distribution)
            image_peaks_gof = calculate_peaks_gof(image_stack, image_model_y, image_peaks_mask, method = "r2")

            image_valid_peaks_mask = ((image_amplitudes > amplitude_threshold)
                                        & (image_rel_amplitudes > rel_amplitude_threshold)
                                        & (image_peaks_gof > gof_threshold))
        else:
            image_intensities = np.expand_dims(image_stack, axis = 2)
            image_intensities = np.where(image_peaks_mask, image_intensities, 0)
            image_amplitudes = (np.max(image_intensities, axis = -1)
                                - np.min(image_stack, axis = -1)[..., np.newaxis])
            image_rel_amplitudes = image_amplitudes / image_global_amplitudes[..., np.newaxis]
            
            image_valid_peaks_mask = ((image_amplitudes > amplitude_threshold)
                                        & (image_rel_amplitudes > rel_amplitude_threshold))

        image_num_peaks = np.sum(image_valid_peaks_mask, axis = -1)

    print("Done")

    return image_num_peaks, image_valid_peaks_mask
    
def get_peak_distances(image_stack, image_params, image_peaks_mask, distribution = "wrapped_cauchy",
                            amplitude_threshold = 3000, rel_amplitude_threshold = 0.1, 
                            gof_threshold = 0.5, only_mus = False,
                            only_peaks_count = 2):
    """
    Returns the number of peaks for every pixel.

    Parameters:
    - image_stack: np.ndarray (n, m, p)
        The image stack containing the measured intensities.
        n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
    - image_params: np.ndarray (n, m, q)
        The output of fitting the image stack, which stores the parameters of the full fitfunction.
        q = 3 * max_peaks + 1, is the number of parameters (max 19 for 6 peaks).
    - image_peaks_mask: np.ndarray (n, m, max_peaks, p)
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
    - only_peaks_count: int or list of ints
        Only use pixels where the number of peaks equals this number.

    Returns:
    - image_distances: np.ndarray (n, m)
        The distance between paired peaks for every pixel.
    """

    print("Calculating image peak distances...")
    
    if not only_mus:
        image_mus = image_params[:, :, 1::3]
    else:
        image_mus = image_params[:, :]

    image_num_peaks, sig_image_peaks_mask = get_number_of_peaks(image_stack, image_params, image_peaks_mask, 
                            distribution = distribution, only_mus = only_mus,  
                            amplitude_threshold = amplitude_threshold, 
                            rel_amplitude_threshold = rel_amplitude_threshold,
                            gof_threshold = gof_threshold)

    # Get the image mask where only defined peak counts are
    mask = (image_num_peaks == only_peaks_count)

    # Get significant image mus for these pixels
    sig_image_peaks_mask[~mask, :] = False
    sig_image_mus = image_mus[sig_image_peaks_mask]

    # Get the distance between two neighbouring significant image mus and reshape into image
    image_distances = -np.ones(mask.shape, dtype = np.float32)
    flat_mask = mask.flatten()
    image_distances = image_distances.flatten()
    sig_image_mus = sig_image_mus.reshape(-1, sig_image_mus.shape[-1])
    image_distances[flat_mask] = np.abs(angle_distance(sig_image_mus[:, 0], sig_image_mus[:, 1]))
    image_distances = image_distances.reshape(mask.shape)

    print("Done")

    return image_distances

def get_peak_amplitudes(image_stack, image_params, image_peaks_mask, distribution = "wrapped_cauchy", 
                        amplitude_threshold = 3000, rel_amplitude_threshold = 0.1,
                        gof_threshold = 0.5, only_mus = False):
    """
    Returns the mean peak amplitude for every pixel.

    Parameters:
    - image_stack: np.ndarray (n, m, p)
        The image stack containing the measured intensities.
        n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
    - image_params: np.ndarray (n, m, q)
        The output of fitting the image stack, which stores the parameters of the full fitfunction.
        q = 3 * max_peaks + 1, is the number of parameters (max 19 for 6 peaks).
    - image_peaks_mask: np.ndarray (n, m, max_peaks, p)
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
    - image_amplitudes: np.ndarray (n, m)
        The mean amplitude for every pixel.
    """

    print("Calculating image peak amplitudes...")

    angles = np.linspace(0, 2*np.pi, num = image_stack.shape[2], endpoint = False)
    image_global_amplitudes = np.max(image_stack, axis = -1) - np.min(image_stack, axis = -1)
    image_global_amplitudes[image_global_amplitudes == 0] = 1

    if not only_mus:
        image_heights = image_params[:, :, 0:-1:3]
        image_scales = image_params[:, :, 2::3]
        image_amplitudes = image_heights * \
                                distribution_pdf(0, 0, image_scales, distribution)[..., 0]
        image_rel_amplitudes = image_amplitudes / image_global_amplitudes[..., np.newaxis]
        image_model_y = full_fitfunction(angles, image_params, distribution)
        image_peaks_gof = calculate_peaks_gof(image_stack, image_model_y, image_peaks_mask, method = "r2")
        image_valid_peaks_mask = ((image_amplitudes > amplitude_threshold)
                                & (image_rel_amplitudes > rel_amplitude_threshold)
                                & (image_peaks_gof > gof_threshold))

    else:
        image_intensities = np.expand_dims(image_stack, axis = 2)
        image_intensities = np.where(image_peaks_mask, image_intensities, 0)
        image_amplitudes = (np.max(image_intensities, axis = -1)
                                - np.min(image_stack, axis = -1)[..., np.newaxis])
        image_rel_amplitudes = image_amplitudes / image_global_amplitudes[..., np.newaxis]
        image_valid_peaks_mask = ((image_amplitudes > amplitude_threshold)
                                & (image_rel_amplitudes > rel_amplitude_threshold))

    #image_amplitudes = _process_image_amplitudes(image_amplitudes, image_rel_amplitudes, 
    #                        image_peaks_gof, amplitude_threshold, 
    #                        rel_amplitude_threshold, gof_threshold, only_mus)

    
    image_amplitudes = np.where(image_valid_peaks_mask, image_amplitudes, np.nan)
    all_nan_slices = np.all(~image_valid_peaks_mask, axis = -1)
    image_amplitudes[all_nan_slices] = 0
    image_amplitudes = np.nanmean(image_amplitudes, axis = -1)

    print("Done")

    return image_amplitudes

@njit(cache = True, fastmath = True, parallel = True)
def _process_image_amplitudes(image_amplitudes, image_rel_amplitudes, image_peaks_gof,
                            amplitude_threshold, rel_amplitude_threshold, gof_threshold, only_mus):
    # Not implemented, because numba (with parallelization) for map creation not implemented yet

    n, m = image_amplitudes.shape[:2]
    mean_amplitudes = np.zeros((n, m))

    for i in prange(n):
        for j in prange(m):
            if only_mus:
                valid_peaks_mask = ((image_amplitudes[i, j] > amplitude_threshold)
                                    & (image_rel_amplitudes[i, j] > rel_amplitude_threshold)
                                    & (image_peaks_gof[i, j] > gof_threshold))
            else:
                valid_peaks_mask = ((image_amplitudes[i, j] > amplitude_threshold)
                                    & (image_rel_amplitudes[i, j] > rel_amplitude_threshold))
                
            valid_amplitudes = image_amplitudes[i,j][valid_peaks_mask]
            if valid_amplitudes.size == 0:
                mean_amplitudes[i, j] = 0
            else:
                mean_amplitudes[i, j] = np.mean(valid_amplitudes)

    return mean_amplitudes

def get_peak_widths(image_stack, image_params, image_peaks_mask, distribution = "wrapped_cauchy", 
                            amplitude_threshold = 3000, rel_amplitude_threshold = 0.1,
                            gof_threshold = 0.5):
    """
    Returns the mean peak width for every pixel.

    Parameters:
    - image_stack: np.ndarray (n, m, p)
        The image stack containing the measured intensities.
        n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
    - image_params: np.ndarray (n, m, q)
        The output of fitting the image stack, which stores the parameters of the full fitfunction.
        q = 3 * max_peaks + 1, is the number of parameters (max 19 for 6 peaks).
    - image_peaks_mask: np.ndarray (n, m, max_peaks, p)
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

    Returns:
    - image_widths: np.ndarray (n, m)
        The mean amplitude for every pixel.
    """

    print("Calculating image peak widths...")

    angles = np.linspace(0, 2*np.pi, num = image_stack.shape[2], endpoint = False) 
    image_heights = image_params[:, :, 0:-1:3]
    image_scales = image_params[:, :, 2::3]
    image_amplitudes = image_heights * \
                                distribution_pdf(0, 0, image_scales, distribution)[..., 0]
    image_global_amplitudes = np.max(image_stack, axis = -1) - np.min(image_stack, axis = -1)
    image_rel_amplitudes = image_amplitudes / image_global_amplitudes[..., np.newaxis]
    image_model_y = full_fitfunction(angles, image_params, distribution)
    image_peaks_gof = calculate_peaks_gof(image_stack, image_model_y, image_peaks_mask, method = "r2")

    image_valid_peaks_mask = ((image_amplitudes > amplitude_threshold)
                                & (image_rel_amplitudes > rel_amplitude_threshold)
                                & (image_peaks_gof > gof_threshold))

    image_scales = np.where(image_valid_peaks_mask, image_scales, np.nan)
    all_nan_slices = np.all(~image_valid_peaks_mask, axis = -1)
    image_scales[all_nan_slices] = 0
    image_scales = np.nanmean(image_scales, axis = -1)

    print("Done")

    return image_scales