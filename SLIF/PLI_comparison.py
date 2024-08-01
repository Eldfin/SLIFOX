import numpy as np
from numba import njit
import matplotlib as plt
from .utils import angle_distance
from .peaks_evaluator import calculate_peaks_gof, peak_pairs_to_directions, peak_pairs_to_inclinations
from .SLIF import fit_image_stack, full_fitfunction, find_image_peaks
from itertools import combinations

def calculate_inclination(thickness, birefringence, wavelength, retardation):
    inclination = np.arccos(np.sqrt(np.arcsin(retardation) * wavelength 
                                / (2 * np.pi * thickness * birefringence)))

    return inclination

def calculate_retardation(thickness, birefringence, wavelength, inclination):
    retardation = np.abs(np.sin(2 * np.pi * thickness * birefringence * np.cos(inclination)**2 
                                    / wavelength))

    return retardation

def calculate_thicknesses(t_rel, birefringence, wavelength, relative_thicknesses):
    thicknesses = t_rel * wavelength / (4 * birefringence * (1 + relative_thicknesses))

    return thicknesses

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

def SLI_to_PLI(peak_pairs, mus, heights, SLI_inclination = False):
    """
    Converts the results from a SLI measurement (peak pairs, mus, heights) to a virtual PLI measurement.
    "What whould be measured in a PLI measurement for the found nerve fibers in the SLI measurement?"

    Parameters:
    - peak_pairs: float
        The PLI direction value of the pixel.
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
    relative_height = np.max(heights[peak_pairs[0]]) / np.max(heights[peak_pairs[1]])

    if SLI_inclination:
        inclinations = peak_pairs_to_inclinations(peak_pairs, mus)
        relative_thicknesses = np.array(relative_height, 1 / relative_height)
        thicknesses = calculate_thicknesses(t_rel, birefringence, wavelength, relative_thicknesses)
        retardations = calculate_retardation(thicknesses, birefringence, wavelength, inclinations)
    else:
        retardations = np.array(relative_height, 1 / relative_height)

    new_dir, new_ret = add_birefringence(directions[0], retardations[0], didirections[1], retardations[1])

    return new_dir, new_ret

def sort_peak_pairs_by_PLI(PLI_direction, mus, heights):
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

    peak_pairs_combinations = get_possible_pairs(len(mus))
    best_dir_diff, best_ret_diff = np.inf, np.inf
    best_pair_index = -1
    num_pairs = peak_pairs_combinations.shape[0]
    SLI_directions = np.empty(num_pairs)
    for i in range(num_pairs):
        peak_pairs = peak_pairs_combinations[i]

        SLI_directions[i], _ = SLI_to_PLI(peak_pairs, mus, heights)

    direction_diffs = np.abs(PLI_direction - SLI_directions)
    sorting_indices = np.argsort(direction_diffs)
    sorted_peak_pairs = peak_pairs_combinations[sorting_indices]

    return sorted_peak_pairs


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

