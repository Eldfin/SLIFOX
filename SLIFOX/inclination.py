import numpy as np
from scipy.ndimage import median_filter
from numba import njit

#@njit(cache = True, fastmath = True)
def calculate_inclination_lm(retardation, retardation_max_lm):

    inclination = np.arccos(np.sqrt(np.arcsin(retardation) / np.arcsin(retardation_max_lm)))

    return inclination

#@njit(cache = True, fastmath = True)
def calculate_inclination_hm(retardation, retardation_max_hm, transmittance, 
                                transmittance_hm, transmittance_lm):

    inclination = np.arccos(np.sqrt(np.arcsin(retardation) / np.arcsin(retardation_max_hm)
                                * np.log(transmittance_lm / transmittance_hm) 
                                / np.log(transmittance_lm / transmittance)))
    
    return inclination

#@njit(cache = True, fastmath = True)
def classify_regions(retardations, transmittances, 
                ret_threshold, trans_threshold, trans_background):
    lm_mask = (retardations <= ret_threshold) & (transmittances >= trans_threshold) \
                    & (transmittances <= trans_background)
    hm_mask = (retardations > ret_threshold) | (transmittances < trans_threshold)
    background_mask = transmittances > trans_background

    return lm_mask, hm_mask, background_mask

#@njit(cache = True, fastmath = True)
def find_max_curvature_bin(histogram, bins, find_range = 0, borders = [-1, -1]):
    # Find the point (bin) of max curvature
    # between the peak value and the peak value + find_range * FWHM
    # or between borders if not equal [-1, -1]

    bin_midpoints = (bins[:-1] + bins[1:]) / 2.0
    if borders[0] != -1 and borders[1] != -1:
        if borders[0] > borders[1]:
            condition = (bin_midpoints >= borders[1]) & (bin_midpoints <= borders[0])
        else:
            condition = (bin_midpoints >= borders[0]) & (bin_midpoints <= borders[1])
    else:
        if find_range == 0:
            condition = (bin_midpoints <= 1)
        else:
            # Find the peak in the histogram
            # find_range is in fwhm units
            peak_index = np.argmax(histogram)
            peak_value = bin_midpoints[peak_index]

            # Compute FWHM
            half_max = np.max(histogram) / 2.0
            indices = np.nonzero(histogram >= half_max)[0]
            if len(indices) <= 2:
                hwhm = bin_midpoints[1] - bin_midpoints[0]
            else:
                hwhm = bin_midpoints[indices[-1]] - bin_midpoints[peak_index]
            fwhm = 2 * hwhm

            border = peak_value + find_range * fwhm
            if border > peak_value:
                condition = (bin_midpoints >= peak_value) & (bin_midpoints <= border)
            else:
                condition = (bin_midpoints >= border) & (bin_midpoints <= peak_value)

    bins_indices = np.nonzero(condition)[0]
    second_gradient = np.gradient(np.gradient(histogram[bins_indices]))
    threshold_bin_index = bins_indices[np.argmax(second_gradient)]
    #threshold = bins[threshold_bin_index]

    return threshold_bin_index


def compute_thresholds(retardations, transmittances):

    # Apply circular median filter with radius 5 on transmittance
    transmittances = median_filter(transmittances, size = (5, 5))

    ret_threshold_bin_index = -1
    trans_background_bin_index = -1
    for n_bins in [64, 128, 256]:
        hist_ret, bins_ret = np.histogram(retardations, bins = n_bins, range = (0, 1))
        hist_trans, bins_trans = np.histogram(transmittances, bins = n_bins, range = (0, 1))

        if ret_threshold_bin_index == -1:
            ret_threshold_bin_index = find_max_curvature_bin(hist_ret, bins_ret, find_range = 20)
            trans_background_bin_index = find_max_curvature_bin(hist_trans, bins_trans, find_range = -10)
        else:
            left_border = bins_ret[2 * (ret_threshold_bin_index - 1)]
            right_border = bins_ret[2 * (ret_threshold_bin_index + 1)]
            ret_threshold_bin_index = find_max_curvature_bin(hist_ret, bins_ret, 
                                            borders = [left_border, right_border])

            left_border = bins_trans[2 * (trans_background_bin_index - 1)]
            right_border = bins_trans[2 * (trans_background_bin_index + 1)]
            trans_background_bin_index = find_max_curvature_bin(hist_trans, bins_trans, 
                                            borders = [left_border, right_border])

    ret_threshold = bins_ret[ret_threshold_bin_index]
    trans_background = bins_trans[trans_background_bin_index]

    # Get the average transmittance value of the top region of the retardation values
    sorted_indices = np.argsort(retardations, axis=None)[::-1]
    region_size = int(0.01 * retardations.size)
    top_indices = sorted_indices[:region_size]
    top_indices_2d = np.unravel_index(top_indices, retardations.shape)
    top_transmittance_values = transmittances[top_indices_2d]
    trans_rmax = np.mean(top_transmittance_values)

    # transmittance threshold as point of maximum curvature betweenn trans_rmax and trans_background
    # does not work as mentioned in paper because this equals trans_background

    #trans_threshold_bin = find_max_curvature_bin(hist_trans, bins_trans, 
    #                                        borders = [trans_rmax, trans_background])
    #trans_threshold = bins_trans[trans_threshold_bin]

    trans_threshold = trans_rmax + bins_trans[1]

    return ret_threshold, trans_background, trans_threshold

#@njit(cache = True, fastmath = True)
def calculate_pixel_inclination(retardation, transmittance, 
                                retardation_max_lm, retardation_max_hm, 
                                transmittance_hm, transmittance_lm, region = "background"):
    
    if region == "background":
        # Is in background
        inclination = 0
    elif region == "hm":
        # Is in high myelination (hm) region
        inclination = calculate_inclination_hm(retardation, retardation_max_hm, transmittance, 
                                transmittance_hm, transmittance_lm)
    elif region == "lm":
        # Is in low myelination (lm) region
        inclination = calculate_inclination_lm(retardation, retardation_max_lm)
    else:
        inclination = 0

    return inclination

def calculate_image_inclinations(retardations, transmittances):
    # Normalize transmittances to [0, 1]
    norm_transmittances = (transmittances - np.min(transmittances)) \
                    / (np.max(transmittances) - np.min(transmittances))

    ret_threshold, trans_background, trans_threshold = compute_thresholds(retardations, norm_transmittances)

    lm_mask, hm_mask, background_mask = classify_regions(retardations, norm_transmittances, 
                ret_threshold, trans_threshold, trans_background)

    retardation_max_lm = np.max(retardations[lm_mask])
    retardation_max_hm = np.max(retardations[hm_mask])
    transmittance_hm = np.mean(transmittances[hm_mask])
    transmittance_lm = np.mean(transmittances[lm_mask])

    inclinations = np.zeros(retardations.shape)

    for pixel in np.ndindex(retardations.shape):
        retardation = retardations[pixel]
        transmittance = transmittances[pixel]

        if lm_mask[pixel]:
            region = "lm"
        elif hm_mask[pixel]:
            region = "hm"
        else:
            region = "background"

        inclinations[pixel] = calculate_pixel_inclination(retardation, transmittance, 
                                retardation_max_lm, retardation_max_hm, 
                                transmittance_hm, transmittance_lm, region = region)

    return inclinations
