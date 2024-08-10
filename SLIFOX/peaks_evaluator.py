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
from scipy.ndimage import distance_transform_edt

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
                            significance_threshold = 0.8, significance_weights = [1, 1],
                            angle_threshold = 30, num_attempts = 10000, 
                            search_radius = 100):
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
    - min_distance: float
        Defines the minimum distance between two paired peaks in degrees, when more than 2 peaks are present.
        If this value is close to 180 the peak finding method converges to a method assuming mostly 
        flat nerve fibers are present in the area and only peaks with distances around 180 are paired.
    - image_peaks_mask: np.ndarray (n, m, max_peaks, p)
        The mask defining which of the p-measurements corresponds to one of the peaks.
        The first two dimensions are the image dimensions.
    - distribution: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
        The name of the distribution.
    - only_mus: bool
        Defines if only the mus (for every pixel) are given in the image_params.
    - num_processes: int
        Defines the number of processes to split the task into.
    - significance_threshold: float
        Value between 0 and 1. Peak Pairs with peaks that have a significance
        lower than this threshold are not considered for possible pairs.
        See also "direction_significance" function for more info.
    - significance_weights: list (2, )
        The weights for the amplitude and for the goodnes-of-fit, when calculating the significance.
        See also "direction_significance" function for more info.
    - angle_threshold: float
        Threshold in degrees defining when a neighbouring pixel direction is considered as same nerve fiber.
    - num_attempts: int
        Number defining how many times it should be attempted to find a neighbouring pixel within
        the given "angle_threshold".
    - search_radius: int
        The radius within which to search for the closest pixel with a defined direction.

    Returns:
    - image_peak_pairs: np.ndarray (n, m, p, np.ceil(max_peaks / 2), 2)
        The peak pairs for every pixel, where the fifth dimension contains both peak numbers of
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
    image_peak_pairs = pymp.shared.array((total_pixels,
                                            max_combs, 
                                            np.ceil(max_peaks / 2).astype(int), 
                                            2), dtype = np.int16)

    with pymp.Parallel(num_processes) as p:
        for i in p.range(total_pixels):
            image_peak_pairs[i, :, :, :] = -1

    image_peak_pairs = image_peak_pairs.reshape(n_rows, n_cols,
                                                image_peak_pairs.shape[1],
                                                image_peak_pairs.shape[2],
                                                image_peak_pairs.shape[3])

    # Get the number of peaks for every pixel
    image_num_peaks = np.sum(np.any(image_peaks_mask, axis=-1), axis = -1)
    direction_found_mask = pymp.shared.array((n_rows, n_cols), dtype = np.bool_)

    # iterate from 2 to max_peaks and 1, 0 at last
    iteration_list = np.append(np.arange(2, max_peaks + 1), [1, 0])
    for i in iteration_list:
        mask = (image_num_peaks == i)
        peak_pairs_combinations = possible_pairs(i)
        num_combs = peak_pairs_combinations.shape[0]
        indices = mask.nonzero()
        num_pixels_iteration = np.count_nonzero(mask)
        # Initialize the progress bar
        if i != 2:
            pbar.close()
        pbar = tqdm(total = num_pixels_iteration, 
                    desc = f'Calculating Peak pairs for {i} Peaks',
                    smoothing = 0)
        shared_counter = pymp.shared.array((num_processes, ), dtype = int)

        with pymp.Parallel(num_processes) as p:
            for j in p.range(num_pixels_iteration):
                x, y = indices[0][j], indices[1][j]

                # Update progress bar
                shared_counter[p.thread_num] += 1
                status = np.sum(shared_counter)
                pbar.update(status - pbar.n)

                if i == 0:
                    #image_peak_pairs[x, y] = np.array([[[-1, -1]]])
                    continue
                elif i <= 2:
                    image_peak_pairs[x, y, 
                                    :peak_pairs_combinations.shape[0],
                                    :peak_pairs_combinations.shape[1]] = peak_pairs_combinations
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
                direction_combs = np.full((num_combs, image_peak_pairs.shape[3]), -1, dtype = np.float64)
                for k in range(num_combs):
                    peak_pairs = peak_pairs_combinations[k]

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

                    directions = peak_pairs_to_directions(peak_pairs, mus)
                    # Filter directions with low significance out
                    directions = directions[significances > significance_threshold]
                    direction_combs[k, :len(directions)] = directions

                if not np.any(valid_combs):
                    continue

                sig_peak_pair_combs = peak_pairs_combinations[valid_combs]
                direction_combs = direction_combs[valid_combs]
                num_sig_combs = sig_peak_pair_combs.shape[0]

                if num_sig_combs == 1:
                    image_peak_pairs[x, y, 
                                    :sig_peak_pair_combs.shape[0],
                                    :sig_peak_pair_combs.shape[1]] = sig_peak_pair_combs
                    direction_found_mask[x, y] = True
                    continue

                check_pixels = np.copy(direction_found_mask)
                num_best_combs = num_sig_combs
                best_nb_indices = np.full(num_sig_combs, -1)
                best_dir_indices = np.full(num_sig_combs, -1)
                first_neighbour = (-1, -1)
                for attempt in range(num_attempts):
                    neighbour_x, neighbour_y = _find_closest_true_pixel(check_pixels, (x, y), search_radius)
                    if neighbour_x == -1 and neighbour_y == -1:
                        # When no true pixel within radius: return no pairs
                        #image_peak_pairs[x, y] = np.array([[[-1, -1]]])
                        break

                    neighbour_peak_pairs = image_peak_pairs[neighbour_x, neighbour_y, 0]
                    neighbour_params = image_params[neighbour_x, neighbour_y]
                    neighbour_peaks_mask = image_peaks_mask[neighbour_x, neighbour_y]
                    neighbour_intensities = image_stack[neighbour_x, neighbour_y]
                    if not only_mus:
                        neighbour_mus = neighbour_params[1::3]
                    else:
                        neighbour_mus = neighbour_params

                    neighbour_directions = peak_pairs_to_directions(neighbour_peak_pairs, neighbour_mus)

                    neighbour_significances = direction_significances(neighbour_peak_pairs, 
                                neighbour_params, neighbour_peaks_mask, 
                                neighbour_intensities, angles, weights = significance_weights, 
                                distribution = distribution, only_mus = only_mus)

                    # Filter directions with low significance out
                    neighbour_directions = neighbour_directions[neighbour_significances
                                                                    > significance_threshold]

                    if len(neighbour_directions) == 0:
                        check_pixels[neighbour_x, neighbour_y] = False
                        continue

                    direction_diffs = np.empty(num_best_combs, dtype = np.float64)
                    nb_diff_indices = np.empty(num_best_combs, dtype = np.int16)
                    dir_diff_indices = np.empty(num_best_combs, dtype = np.int16)
                    for k in range(num_best_combs):
                        directions = direction_combs[k]
                        directions = directions[directions != -1]
                        
                        # Remove already matched directions
                        if first_neighbour == (neighbour_x, neighbour_y) and best_nb_indices[k] != -1:
                            neighbour_directions = np.delete(neighbour_directions, best_nb_indices[k])
                        if best_dir_indices[k] != -1:
                            directions = np.delete(directions, best_dir_indices[k])

                        differences = np.abs(neighbour_directions[:, np.newaxis] - directions)
                        min_diff_index = np.argmin(differences)
                        nb_diff_indices[k], dir_diff_indices[k] = np.unravel_index(min_diff_index, 
                                                                                differences.shape)
                        # Insert minimum difference to neighbour directions into array for sorting later
                        direction_diffs[k] = np.min(differences)

                    if np.min(direction_diffs) < angle_threshold * np.pi / 180:
                        # If minimum difference to neighbour direction is smaller than given threshold
                        # Sort possible peak pairs by difference to neighbour directions
                        # and accept the result

                        sort_indices = np.argsort(direction_diffs)
                        sig_peak_pair_combs[:num_best_combs] = sig_peak_pair_combs[sort_indices]
                        old_num_best_combs = num_best_combs
                        num_best_combs = np.sum(direction_diffs == np.min(direction_diffs))
                        best_nb_indices = nb_diff_indices[sort_indices[:num_best_combs]]
                        best_dir_indices = dir_diff_indices[sort_indices[:num_best_combs]]
                        direction_combs = direction_combs[:num_best_combs]
                        if first_neighbour == (-1, -1):
                            first_neighbour = (neighbour_x, neighbour_y)
                        else:
                            check_pixels[neighbour_x, neighbour_y] = False

                        if num_best_combs == 1 or num_best_combs == old_num_best_combs:
                            image_peak_pairs[x, y, 
                                        :sig_peak_pair_combs.shape[0],
                                        :sig_peak_pair_combs.shape[1]] = sig_peak_pair_combs
                            direction_found_mask[x, y] = True
                            break
                    else:
                        check_pixels[neighbour_x, neighbour_y] = False
                        if attempt == num_attempts - 1 or not np.any(check_pixels):
                            # When no neighbouring pixel within num_attempts had
                            # a direction difference below the threshold
                            # return no peak pairs
                            image_peak_pairs[x, y] = np.array([[[-1, -1]]])
                            break

    return image_peak_pairs

@njit(cache = True, fastmath = True)
def peak_pairs_to_directions(peak_pairs, mus):
    """
    Calculates the directions from given peak_pairs of a pixel.

    Parameters:
    - peak_pairs: np.ndarray (m // 2, 2)
        Array ontaining both peak numbers of a pair (e.g. [1, 3], 
        which means peak 1 and peak 3 is paired). A pair with -1 defines a unpaired peak.
        The first dimension (m equals number of peaks)
        is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
    - mus: np.ndarray (m, )
        The center positions of the peaks.

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

def calculate_directions(image_peak_pairs, image_mus, directory = None, only_peaks_count = -1):
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
    - only_peaks_count: int
        Defines a filter for the number of peaks, so that only pixels will be processed that have
        this number of peaks.

    Returns:
    - directions: (n, m, np.ceil(max_peaks / 2))
        The calculated directions for everyoe of the (n * m) pixels.
        Max 3 directions (for 6 peaks).
    """

    x_range = image_peak_pairs.shape[0]
    y_range = image_peak_pairs.shape[1]
    max_directions = image_peak_pairs.shape[2]
    directions = np.full((x_range, y_range, max_directions), -1, dtype=np.float64)
    for x in range(x_range):
        for y in range(y_range):
            if only_peaks_count != -1:
                num_peaks = int(np.max(image_peak_pairs[x, y]) + 1)
                if isinstance(only_peaks_count, list):
                    if num_peaks in only_peaks_count: continue
                else:
                    if num_peaks != only_peaks_count: continue
            
            directions[x, y] = peak_pairs_to_directions(image_peak_pairs[x, y], image_mus[x, y])

    if directory != None:
        if not os.path.exists(directory):
                os.makedirs(directory)

        for dir_n in range(max_directions):
            write_direction = np.swapaxes(directions[:, :, dir_n], 0, 1)
            write_direction[write_direction != -1] = write_direction[write_direction != -1] * 180 / np.pi
            imageio.imwrite(f'{directory}/dir_{dir_n + 1}.tiff', write_direction)

    return directions

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

@njit(cache = True, fastmath = True)
def direction_significances(peak_pairs, params, peaks_mask, intensities, angles, weights = [1, 1],
                            distribution = "wrapped_cauchy", only_mus = False):
    """
    Calculates the significances of the directions for one (fitted) pixel.

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
            amplitudes[i] = np.max(peak_intensities) - np.min(peak_intensities)

        for i in range(num_directions):
            peak_pair = peak_pairs[i]
            indices = peak_pair[peak_pair != -1]
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

def image_direction_significances(image_stack, image_peak_pairs, image_params, image_peaks_mask, 
                                directory = None, distribution = "wrapped_cauchy",
                                weights = [1, 1]):
    """
    Calculates the significances of all found directions from given "image_peak_pairs".

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
    - directory: string
        The directory path defining where the significance image should be writen to.
        If None, no image will be writen.
    - distribution: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
        The name of the distribution.
    - weights: list (2, )
        The weights for the amplitude and for the goodnes-of-fit, when calculating the significance

    Returns:
    - significances: (n, m, np.ceil(max_peaks / 2))
        The calculated significances (ranging from 0 to 1) for everyoe of the (n * m) pixels.
        Max 3 significances (shape like directions).
    """

    angles = np.linspace(0, 2*np.pi, num = image_stack.shape[2], endpoint = False)
    x_range = image_peak_pairs.shape[0]
    y_range = image_peak_pairs.shape[1]
    max_significances = image_peak_pairs.shape[2]
    significances = np.zeros((x_range, y_range, max_significances), dtype = np.float64)
    for x in range(x_range):
        for y in range(y_range):
            peak_pairs = image_peak_pairs[x, y]
            params = image_params[x, y]
            peaks_mask = image_peaks_mask[x, y]
            intensities = image_stack[x, y]
            
            significances[x, y] = pixel_significances(peak_pairs, params, peaks_mask, 
                                    intensities, angles, weights = weights,
                                    distribution = distribution)
            
    if directory != None:
        if not os.path.exists(directory):
                os.makedirs(directory)

        for dir_n in range(max_significances):
            imageio.imwrite(f'{directory}/dir_{dir_n + 1}_sig.tiff', 
                                np.swapaxes(significances[:, :, dir_n], 0, 1))

    return significances

