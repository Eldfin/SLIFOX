import numpy as np
from numba import njit
import matplotlib as plt
import pymp
from .utils import angle_distance
from .peaks_evaluator import calculate_peaks_gof, peak_pairs_to_directions, \
                            peak_pairs_to_inclinations, direction_significances
from .SLIF import fit_image_stack, full_fitfunction, find_image_peaks
from itertools import combinations, chain
from tqdm import tqdm

@njit(cache = True, fastmath = True)
def calculate_inclination(thickness, birefringence, wavelength, retardation):
    inclination = np.arccos(np.sqrt(np.arcsin(retardation) * wavelength 
                                / (2 * np.pi * thickness * birefringence)))

    return inclination

@njit(cache = True, fastmath = True)
def calculate_retardation(thickness, birefringence, wavelength, inclination):
    retardation = np.abs(np.sin(2 * np.pi * thickness * birefringence * np.cos(inclination)**2 
                                    / wavelength))

    return retardation

@njit(cache = True, fastmath = True)
def calculate_thicknesses(t_rel, birefringence, wavelength, relative_thicknesses):
    thicknesses = t_rel * wavelength / (4 * birefringence * (1 + relative_thicknesses))

    return thicknesses

@njit(cache = True, fastmath = True)
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

@njit(cache = True, fastmath = True)
def SLI_to_PLI(peak_pairs, mus, heights, SLI_inclination = False):
    """
    Converts the results from a SLI measurement (peak pairs, mus, heights) to a virtual PLI measurement.
    "What whould be measured in a PLI measurement for the found nerve fibers in the SLI measurement?"

    Parameters:
    - peak_pairs: np.ndarray (m // 2, 2)
        Array ontaining both peak numbers of a pair (e.g. [1, 3], 
        which means peak 1 and peak 3 is paired). A pair with -1 defines a unpaired peak.
        The first dimension (m equals number of peaks)
        is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
    - mus: np.ndarray (m, )
        The center positions of the peaks.
    - heights: np.ndarray (m, )
        The height values of the peaks.
    - SLI_inclination: boolean
        Use the found inclination in the SLI measurement to produce a virtual PLI retardation?
        Default is False, because SLI inclination could not be determined reliable to the point of writing.

    Returns:
    - new_dir: float
        Virtual direction value of the PLI measurement.
    - new_ret: float
        Virtual retardation value of the PLI measurement.
        Does not store usable information when not using SLI inclination (SLI_inclination = False).

    """

    directions = peak_pairs_to_directions(peak_pairs, mus)
    num_directions = len(directions)
    
    if num_directions == 0:
        return -1, 0
    elif num_directions == 1:
        # For one directions return the direction
        return directions[0], 0
    else:
        first_height = np.max(heights[peak_pairs[0]])
        second_height = np.max(heights[peak_pairs[1]])
        relative_height =  min(first_height, second_height) / max(first_height, second_height)
    
        if SLI_inclination:
            #dummy t_rel, birefringence, wavelenght
            t_rel, birefringence, wavelength = 1, 1, 1
            inclinations = peak_pairs_to_inclinations(peak_pairs, mus)
            relative_thicknesses = np.array([relative_height, 1 / relative_height])
            thicknesses = calculate_thicknesses(t_rel, birefringence, wavelength, relative_thicknesses)
            retardations = calculate_retardation(thicknesses, birefringence, wavelength, inclinations)
        else:
            retardations = np.array([relative_height, 1 - relative_height])
        
        new_dir, new_ret = add_birefringence(directions[0], retardations[0], 
                                                directions[1], retardations[1])
        if num_directions == 3:
            # For three directions:
            # To-Do: better handling than again add_birefringence
            first_height = np.max(heights[peak_pairs[0]]) + np.max(heights[peak_pairs[1]])
            second_height = np.max(heights[peak_pairs[2]])
            relative_height =  min(first_height, second_height) / max(first_height, second_height)
            retardations = np.array([relative_height, 1 - relative_height])
            new_dir, new_ret = add_birefringence(new_dir, retardations[0], 
                                                    directions[2], retardations[1])

    return new_dir, new_ret

@njit(cache = True, fastmath = True)
def peak_pairs_by_PLI(PLI_direction, mus, heights, params, peaks_mask, intensities, angles,
                        significance_weights = [1, 1], distribution = "wrapped_cauchy",
                        significance_threshold = 0.8):
    """
    Sort all possible peak pairs from lowest to highest difference to the PLI direction measurement.

    Parameters:
    - PLI_direction: float
        The PLI direction value of the pixel.
    - mus: np.ndarray (m, )
        The center positions of the peaks.
    - heights: np.ndarray (m, )
        The height values of the peaks.

    Returns:
    - sorted_peak_pairs: (n, m, 2)
        The possible combinations of pairs sorted from lowest to highest difference to the PLI direction.
        The size of dimensions is: 
            n = math.factorial(num_peaks) // ((2 ** (num_peaks // 2)) * math.factorial(num_peaks // 2))
                so n = 3 for num_peaks = 4 and n = 15 for num_peaks = 6.
                Odd numbers of num_peaks have the same dimension size as num_peaks + 1.
            m = np.ceil(num_peaks / 2)

    """
    # Only use peaks not zero (significant)
    sig_peak_indices = (heights > 1).nonzero()[0]
    mus = mus[sig_peak_indices]
    heights = heights[sig_peak_indices]

    num_peaks = len(mus)
    if num_peaks == 0:
        return np.array([[[-1, -1]]])

    peak_pairs_combinations = possible_pairs(num_peaks)
    best_dir_diff, best_ret_diff = np.inf, np.inf
    best_pair_index = -1
    num_pairs = peak_pairs_combinations.shape[0]
    SLI_directions = np.empty(num_pairs)
    for i in range(num_pairs):
        peak_pairs = peak_pairs_combinations[i]

        significances = direction_significances(peak_pairs, params, peaks_mask, intensities, angles, 
                                weights = significance_weights, distribution = distribution)

        if np.mean(significances) < significance_threshold:
            # If mean direction significance is below threshold store it as negative direction
            # -2.1 pi - 1 for difference handling later
            SLI_directions[i] = -2.1 * np.pi - 1 + np.mean(significances)
        else:
            SLI_directions[i], _ = SLI_to_PLI(peak_pairs, mus, heights)

    if np.all(SLI_directions < 0):
        # If every possible pair has a bad significance
        # return same as if no peak pair
        return np.full(peak_pairs_combinations.shape, -1)

    # Convert SLI direction from radians to angles, because PLI direction is in angles
    SLI_directions = SLI_directions * 180 / np.pi

    direction_diffs = np.abs(PLI_direction - SLI_directions)
    sorting_indices = np.argsort(direction_diffs)
    sorted_peak_pairs = peak_pairs_combinations[sorting_indices]

    # Convert the significant indices back to total indices
    for i in range(sorted_peak_pairs.shape[0]):
        for j in range(sorted_peak_pairs.shape[1]):
            for k in range(sorted_peak_pairs.shape[2]):
                if sorted_peak_pairs[i, j, k] != -1:
                    sorted_peak_pairs[i, j, k] = sig_peak_indices[sorted_peak_pairs[i, j, k]]

    return sorted_peak_pairs

def image_peak_pairs_by_PLI(PLI_directions, image_mus, image_heights, 
                            SLI_data, image_params, image_peaks_mask,
                            num_processes = 2, significance_weights = [1, 1],
                            significance_threshold = 0.8, distribution = "wrapped_cauchy"):
    """
    Get the sorted peak pairs of a whole image by using PLI.

    Parameters:
    - PLI_directions: np.ndarray (i, j)
        The PLI directions value of the pixels.
    - image_mus: np.ndarray (i, j, m )
        The center positions of the peaks for every pixel.
    - heights: np.ndarray (i, j, m )
        The height values of the peaks for every pixel.

    Returns:
    - image_peak_pairs: (i, j, n, m, 2)
        The possible combinations of pairs sorted from lowest to highest difference to the PLI direction.
        The size of dimensions is:
            i, j = image dimensions 
            n = math.factorial(num_peaks) // ((2 ** (num_peaks // 2)) * math.factorial(num_peaks // 2))
                so n = 3 for num_peaks = 4 and n = 15 for num_peaks = 6.
                Odd numbers of num_peaks have the same dimension size as num_peaks + 1.
            m = np.ceil(num_peaks / 2)

    """

    max_peaks = image_mus.shape[2]
    n_rows = PLI_directions.shape[0]
    n_cols = PLI_directions.shape[1]
    total_pixels = n_rows * n_cols
    image_mus = image_mus.reshape((total_pixels, image_mus.shape[2]))
    image_heights = image_heights.reshape((total_pixels, image_heights.shape[2]))
    PLI_directions = PLI_directions.reshape(total_pixels)
    SLI_data = SLI_data.reshape((total_pixels, SLI_data.shape[2]))
    image_params = image_params.reshape((total_pixels, image_params.shape[2]))
    image_peaks_mask = image_peaks_mask.reshape((total_pixels, image_peaks_mask.shape[2],
                                                    image_peaks_mask.shape[3]))
    angles = np.linspace(0, 2*np.pi, num = SLI_data.shape[1], endpoint = False)

    max_combs = 3
    if max_peaks == 6:
        max_combs = 15
    image_peak_pairs = pymp.shared.array((total_pixels, 
                                            max_combs, 
                                            np.ceil(max_peaks / 2).astype(int), 
                                            2), dtype = np.int16)

    with pymp.Parallel(num_processes) as p:
        for i in p.range(total_pixels):
            image_peak_pairs[i, :, :] = -1

    # Initialize the progress bar
    num_tasks = total_pixels
    pbar = tqdm(total = num_tasks, desc = "Calculating Peak pairs", smoothing = 0)
    shared_counter = pymp.shared.array((num_processes, ), dtype = int)

    with pymp.Parallel(num_processes) as p:
        for i in p.range(image_peak_pairs.shape[0]):
            image_peak_pairs[i, :] = -1

    with pymp.Parallel(num_processes) as p:
        for i in p.range(total_pixels):
            intensities = SLI_data[i]
            image_peak_pairs[i] = peak_pairs_by_PLI(PLI_directions[i], image_mus[i], image_heights[i],
                        image_params[i], image_peaks_mask[i], intensities, angles,
                        significance_weights = significance_weights, distribution = distribution,
                        significance_threshold = significance_threshold)

            # Update progress bar
            shared_counter[p.thread_num] += 1
            status = np.sum(shared_counter)
            pbar.update(status - pbar.n)

    image_peak_pairs = image_peak_pairs.reshape((n_rows, n_cols, 
                                                image_peak_pairs.shape[1], 
                                                image_peak_pairs.shape[2],
                                                image_peak_pairs.shape[3]))

    return image_peak_pairs

@njit(cache = True, fastmath = True)
def possible_pairs(num_peaks):
    # Function returning pre calculated value of the get_possible_pairs function
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

