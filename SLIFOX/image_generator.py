import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from .fitter import full_fitfunction
from .peaks_evaluator import calculate_peaks_gof, get_number_of_peaks, get_image_direction_significances, \
                                get_peak_distances, get_mean_peak_amplitudes, get_mean_peak_widths, \
                                    calculate_directions
from .wrapped_distributions import distribution_pdf
from .utils import angle_distance
import imageio

#Paul Tol's Bright Color Palette for color blindness
default_colormap = np.array([
    (102, 204, 238),  # Cyan
    (170, 51, 119),   # Purple
    (204, 187, 68),   # Yellow
    (34, 136, 51),    # Green
    (68, 119, 170),   # Blue
    (238, 102, 119),  # Red
    (187, 187, 187)  # Grey
])
norm_default_colormap = default_colormap / 255

def normalize_to_rgb(array, value_range = [None, None], colormap = "viridis"):
    # Normalizes an 2d-array using a colormap
    # if min (or max) in range is None, it will be the min (or max) of the image
    
    cmap = plt.get_cmap(colormap)
    if value_range[0] == None:
        value_range[0] = np.min(array[array > 0])
    if value_range[1] == None:
        value_range[1] = np.max(array)
    normalized_image = np.clip(array, value_range[0], value_range[1])
    normalized_image = (normalized_image - value_range[0]) / (value_range[1] - value_range[0])
    image = cmap(normalized_image)[:, :, :3] * 255  # Apply colormap and convert to 0-255 range
    image = image.astype(np.uint8)
    
    # Set values below normalization range to black and above to white
    image[array < value_range[0]] = [0, 0, 0]
    image[array > value_range[1]] = [255, 255, 255]

    return image

def alternating_vline(x, ax=None, colors=['blue', 'red'], num_segments=100, **kwargs):
    """
    Plots a vline at position x with alternating colors.
    """
    if ax is None:
        ax = plt.gca()
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)
    y_vals = np.linspace(ymin, ymax, num_segments + 1)
    for i in range(num_segments):
        ax.vlines(x, y_vals[i], y_vals[i+1], color=colors[i % len(colors)], **kwargs)

def plot_peaks_gof(peaks_gof, heights, mus, scales, 
                        distribution, peaks_mask, angles, gof_weight = 1):
    """
    Visualizes the goodnes-of-fit with the amplitudes of the peaks in the current plot.

    Parameters:
    - peaks_gof: np.ndarray (n_peaks, )
        The goodness of fit values for the peaks.
    - peaks_mask: np.ndarray (n_find_peaks, m)
        The mask array defining which of the m measurements of the pixel corresponds to a peak.
    - amplitudes: np.ndarray (n_peaks, )
        The amplitudes of the peaks.
    - angles: np.ndarray (m, )
        The angles at which the intensities of the pixel are measured.
    - gof_weight: float
        The weight of the goodnes-of-fit ranging from 0 to 1.
    """

    sig_mask = (heights >= 1)
    peaks_gof = peaks_gof[sig_mask]
    mus = mus[sig_mask]
    scales = scales[sig_mask]
    heights = heights[sig_mask]
    
    for k, gof in enumerate(peaks_gof):
        gof_color = norm_default_colormap[min(k, len(default_colormap))]
        if gof < 0: gof = 0
        amplitude = heights[k] * distribution_pdf(0, 0, scales[k], distribution)
        ymax = gof**gof_weight * amplitude

        x_f = np.linspace(0, 2*np.pi, 1000, endpoint=False)
        peaks_x_f_1 = np.empty(0)
        peaks_x_f_2 = np.empty(0)
        wrapped = False
        for l in range(len(angles)):
            if l+1 < len(angles) and peaks_mask[k, l] and peaks_mask[k, l+1]:
                if not wrapped:
                    peaks_x_f_1 = np.append(peaks_x_f_1, np.linspace(angles[l], 
                                                            angles[l+1], 50, endpoint=False))
                else:
                    peaks_x_f_2 = np.append(peaks_x_f_2, np.linspace(angles[l], 
                                                            angles[l+1], 50, endpoint=False))
            elif l+1 == len(angles) and peaks_mask[k, l] and peaks_mask[k, 0]:
                if not wrapped:
                    peaks_x_f_1 = np.append(peaks_x_f_1, np.linspace(angles[l], 
                                                            2*np.pi, 50, endpoint=False))
                else:
                    peaks_x_f_2 = np.append(peaks_x_f_2, np.linspace(angles[l], 
                                                            2*np.pi, 50, endpoint=False))
            elif l > 0 and l+1 < len(angles) and not peaks_mask[k, l] \
                    and peaks_mask[k, l+1] and len(peaks_x_f_1) > 0:
                wrapped = True

        peak_y_f_1 = heights[k] * distribution_pdf(peaks_x_f_1, mus[k], scales[k], distribution)
        peak_y_f_2 = heights[k] * distribution_pdf(peaks_x_f_2, mus[k], scales[k], distribution)
        peak_y_f = heights[k] * distribution_pdf(x_f, mus[k], scales[k], distribution)
        plt.plot((peaks_x_f_1*180/np.pi) % 360, peak_y_f_1, marker='None', linestyle="-", color=gof_color)
        plt.plot((peaks_x_f_2*180/np.pi) % 360, peak_y_f_2, marker='None', linestyle="-", color=gof_color)
        plt.plot((x_f*180/np.pi) % 360, peak_y_f, marker='None', linestyle="--", color=gof_color)
        plt.vlines((mus[k]*180/np.pi) % 360, 0, ymax , color = gof_color)

def plot_directions(peak_pairs, mus, distribution, heights = None, scales = None, 
                        exclude_lone_peaks = False):
    """
    Plots the found directions (peak pairs) in the current plot.

    Parameters:
    - peaks_pairs: np.ndarray (3, 2)
        The array containing the number of peaks that are paired (max 3 pairings and 2 peaks per pair).
    - mus: np.ndarray (n_peaks, )
        The mus (centers) of the peaks.
    - heights: np.ndarray (n_peaks, )
        The heights of the peaks.
    - scales: np.ndarray (n_peaks, )
        The scales of the peaks.
    - distribution: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
        The name of the distribution.
    - exclude_lone_peaks: bool
        Whether to exclude lone peak directions in the plot.
    """
    for k, pair in enumerate(peak_pairs):
        if pair[0] == -1 and pair[1] == -1: continue
        if exclude_lone_peaks and (pair[0] == -1 or pair[1] == -1): continue
        mixed_colors = []
        for index in pair:
            if index == -1: continue
            mixed_colors.append(norm_default_colormap[min(index, len(default_colormap))])

        if len(mixed_colors) == 1:
            ax = plt.gca()
            limit_min, limit_max = ax.get_ylim()
            direction = mus[pair[0]] % (np.pi)
            if isinstance(heights, np.ndarray) and isinstance(scales , np.ndarray):
                if direction == mus[pair[0]]:
                    ymin = heights[pair[0]] * distribution_pdf(0, 0, scales[pair[0]], distribution)
                    ymin2 = limit_min
                else:
                    ymin = limit_min
                    ymin2 = heights[pair[0]] * distribution_pdf(0, 0, scales[pair[0]], distribution)
            else:
                ymin, ymin2 = limit_min, limit_min
            direction = direction * 180/np.pi
            
            plt.vlines(direction, ymin, limit_max, color = mixed_colors[0])
            plt.vlines(direction + 180, ymin2, limit_max, color = mixed_colors[0])
        else:
            distance = angle_distance(mus[pair[0]], mus[pair[1]])
            direction = (mus[pair[0]] + distance / 2) % (np.pi)
            direction = direction * 180/np.pi
            alternating_vline(direction, colors=mixed_colors, num_segments=10)
            alternating_vline(direction + 180, colors=mixed_colors, num_segments=10)

def show_pixel(intensities, intensities_err, best_parameters, peaks_mask, distribution):
    """
    Plots and shows the intensity profile of a pixel.

    Parameters:
    - intensities: np.ndarray (n, )
        The measured intensities of the pixel.
    - intensities_err: np.ndarray (n, )
        The errors (standard deviation) of the measured intensities.
    - best_parameters: np.ndarray (m, )
        The found best parameters of the full fitfunction from the fitting process.
    - peaks_mask: np.ndarray (max_find_peaks, n)
        The mask array defining which intensities corresponds to which peak.
    - distribution: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
        The name of the distribution.
    """
    angles = np.linspace(0, 2*np.pi, num=len(intensities), endpoint=False)

    max_fit_peaks = int((len(best_parameters) - 1) / 3)
    model_y = full_fitfunction(angles, best_parameters, distribution)
    peaks_gof = calculate_peaks_gof(intensities, model_y, peaks_mask[:max_fit_peaks], method = "r2")

    plt.errorbar(angles*180/np.pi, intensities, yerr=intensities_err, marker = "o", linestyle="", capsize=5)
    plt.xlabel("Angle / $^\\circ$", fontsize = 12)
    plt.ylabel("Intensity", fontsize = 12)

    x_f = np.linspace(0, 2*np.pi, 2000, endpoint=False)
    y_f = full_fitfunction(x_f, best_parameters, distribution)
    heights = best_parameters[0:-1:3]
    scales = best_parameters[2::3]
    mus = best_parameters[1::3]

    FitLine, = plt.plot(x_f*180/np.pi, y_f, marker='None', linestyle="-", color="black")

    plot_peaks_gof(peaks_gof, heights, mus, scales, 
                            distribution, peaks_mask, angles, gof_weight = 1)

    plt.show()


def plot_data_pixels(data, image_params, image_peaks_mask, image_peak_pairs = None, only_mus = False,
                distribution = "wrapped_cauchy", indices = None, data_err = "sqrt(data)",
                directory = "plots"):
    """
    Plots all the intensity profiles of the pixels of given data.

    Parameters:
    - data: np.ndarray (n, m, p)
        The image stack containing the measured intensities.
        n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
    - image_params: np.ndarray (n, m, q)
        The output of fitting the image stack, which stores the parameters of the full fitfunction.
        q = 3 * n_peaks + 1, is the number of parameters (max 19 for 6 peaks).
    - image_peaks_mask: np.ndarray (n, m, n_peaks, p)
        The mask defining which of the p-measurements corresponds to one of the peaks.
        The first two dimensions are the image dimensions.
    - image_peak_pairs: np.ndarray (n, m, 3, 2)
        The peak pairs for every pixel, where the fourth dimension contains both peak numbers of
        a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the third dimension
        is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
        The first two dimensions are the image dimensions.
    - only_mus: bool
        Defines if only the mus (for every pixel) are given in the image_params.
    - distribution: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
        The name of the distribution.
    - indices: np.ndarray (n, m, 2)
        The array storing the x- and y-coordinate of one pixel (if plotted data != full data).
    - data_err: np.ndarray (n, m, p)
        The standard deviation (error) of the measured intensities in the image stack.
        Default options is the sqrt of the intensities.
    - directory: string
        The directory path where the plots should be written to.
    """

    angles = np.linspace(0, 2*np.pi, num=data.shape[2], endpoint=False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):

            intensities = data[i, j]
            if isinstance(data_err, np.ndarray):
                intensities_err = data_err[i, j]
            elif isinstance(data_err, (int, float)):
                intensities_err = data_err
            else:
                intensities_err = np.ceil(np.sqrt(intensities)).astype(intensities.dtype)
                intensities_err[intensities_err == 0] = 1
            peaks_mask = image_peaks_mask[i, j]
            num_found_peaks = np.count_nonzero(np.any(peaks_mask, axis = -1), axis = -1)
            #if num_peaks == 0: continue

            intensities_err = np.array(intensities_err, dtype = np.int32)
            plt.errorbar(angles*180/np.pi, intensities, yerr=intensities_err, marker = "o", 
                            linestyle="", capsize=5, color="black")

            if not only_mus:
                params = image_params[i, j]
                heights = params[0:-1:3]
                scales = params[2::3]
                mus = params[1::3]

                model_y = full_fitfunction(angles, params, distribution)
                peaks_gof = calculate_peaks_gof(intensities, model_y, peaks_mask[:len(mus)], method = "r2")
                x_f = np.linspace(0, 2*np.pi, 2000, endpoint=False)
                y_f = full_fitfunction(x_f, params, distribution)
                FitLine, = plt.plot(x_f*180/np.pi, y_f, marker='None', linestyle="-", color="black")

                plot_peaks_gof(peaks_gof, heights, mus, scales, 
                                distribution, peaks_mask, angles)
            else:
                mus = image_params[i, j]   
                heights = None
                scales = None   
                for k, mask in enumerate(peaks_mask):
                    if not np.any(mask): break
                    color = norm_default_colormap[min(k, len(default_colormap))]
                    plt.errorbar(angles[mask]*180/np.pi, intensities[mask], 
                            yerr=intensities_err[mask], marker = "o", linestyle="", capsize=5, color=color)

            plt.xlabel("Winkel")
            plt.ylabel("Intensität")

            if isinstance(image_peak_pairs, np.ndarray):
                plot_directions(image_peak_pairs[i, j], mus, distribution, 
                                    heights = heights, scales = scales)

            max_rows = len(str(data.shape[0] - 1)) 
            max_cols = len(str(data.shape[1] - 1))
            if indices is None:
                x_str = f"{i:0{max_rows}d}"
                y_str = f"{j:0{max_cols}d}"
            else:
                x_str = f"{indices[i, j, 0]:0{max_rows}d}"
                y_str = f"{indices[i, j, 1]:0{max_cols}d}"
                
            if not os.path.exists(directory):
                os.makedirs(directory)

            plt.savefig(f"{directory}/x{x_str}y{y_str}.png")
            plt.clf()
            
def map_number_of_peaks(image_stack = None, image_params = None, 
                            image_peaks_mask = None, distribution = "wrapped_cauchy", 
                            amplitude_threshold = 3000, rel_amplitude_threshold = 0.1,
                            gof_threshold = 0.5, only_mus = False,
                            directory = "maps", colormap = None,
                            image_sig_peaks_mask = None, image_num_peaks = None):
    """
    Maps the number of peaks for every pixel.

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
    - directory: string
        The directory path defining where the resulting image should be writen to.
        If None, no image will be writen.
    - colormap: np.ndarray (6, 3)
        Colormap used for the map generation.
    - image_sig_peaks_mask: np.ndarray (n, m, max_find_peaks)
        When significant peaks mask is already calculated, 
        it can be provided here to speed up the process.
    - image_num_peaks: np.ndarray (n, m)
        If the number of peaks already have been calculated, they can be inserted here.

    Returns:
    - image_num_peaks: np.ndarray (n, m)
        The number of peaks for every pixel.
    """

    if not isinstance(image_num_peaks, np.ndarray):
        image_num_peaks = get_number_of_peaks(image_stack = image_stack, 
                                image_params = image_params, image_peaks_mask = image_peaks_mask, 
                                distribution = distribution, 
                                amplitude_threshold = amplitude_threshold, 
                                rel_amplitude_threshold = rel_amplitude_threshold, 
                                gof_threshold = gof_threshold,
                                only_mus = only_mus, image_sig_peaks_mask = image_sig_peaks_mask)

    if directory != None:
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not isinstance(colormap, np.ndarray):
            colormap = np.insert(default_colormap, 0, (0, 0, 0), axis = 0).astype(np.uint8)
        
        image = np.swapaxes(image_num_peaks, 0, 1)
        image = np.clip(image, 0, 7)
        image = colormap[image]

        imageio.imwrite(f'{directory}/n_peaks_map.tiff', image, format = 'tiff')

    return image_num_peaks
    
def map_peak_distances(image_stack = None, image_params = None, image_peaks_mask = None, 
                            distribution = "wrapped_cauchy", 
                            amplitude_threshold = 3000, rel_amplitude_threshold = 0.1,
                            gof_threshold = 0.5, only_mus = False, deviation = False,
                            image_peak_pairs = None, only_peaks_count = -1, 
                            normalize = False, normalize_to = [None, None],
                            directory = "maps", num_processes = 2,
                            image_num_peaks = None, image_sig_peaks_mask = None,
                            image_distances = None):
    """
    Maps the distance between two paired peaks for every pixel.

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
    - deviation: boolean
        If true, the distance deviation to 180 degrees will be mapped, so that values of 0
        represent peak distances of 180 degrees.
    - only_peaks_count: int
        Only use pixels where the number of peaks equals this number. -1 for use of every number of peaks.
    - directory: string
        The directory path defining where the resulting image should be writen to.
        If None, no image will be writen.
    - normalize: bool
        Whether the created image should be normalized (amd displayed with colors)
    - normalize_to: list
        List of min and max value that defines the range the image is normalized to.
        If min (or max) is None, the minimum (or maximum) of the image will be used.
    - num_processes: int
        Defines the number of processes to split the task into.
    - image_sig_peaks_mask: np.ndarray (n, m, max_find_peaks)
        If significant peaks mask is already calculated, 
        it can be provided here to speed up the process.
    - image_num_peaks: np.ndarray (n, m)
        If number of peaks are already calculated,
        they can be provided here to speed up the process.
    - image_distances: np.ndarray (n, m)
        If peak distances already have been calculated, they can be inserted here.

    Returns:
    - image_distances: np.ndarray (n, m)
        The distance between paired peaks for every pixel.
    """
    
    if not isinstance(image_distances, np.ndarray):
        image_distances = get_peak_distances(image_stack = image_stack, 
                                image_params = image_params, 
                                image_peaks_mask = image_peaks_mask, 
                                distribution = distribution,
                                amplitude_threshold = amplitude_threshold, 
                                rel_amplitude_threshold = rel_amplitude_threshold, 
                                gof_threshold = gof_threshold, 
                                only_mus = only_mus,
                                image_peak_pairs = image_peak_pairs,
                                only_peaks_count = only_peaks_count,
                                num_processes = num_processes,
                                image_sig_peaks_mask = image_sig_peaks_mask,
                                image_num_peaks = image_num_peaks)

    image_distances[image_distances != -1] = image_distances[image_distances != -1] * 180 / np.pi
    file_name = "peak_distances_map"
    if deviation:
        image_distances[image_distances != -1] = 180 - image_distances[image_distances != -1]
        file_name = "peak_distances_deviation_map"

    if len(image_distances.shape) == 3:
        for dir_n in range(image_distances.shape[-1]):    
            image = np.swapaxes(image_distances[:, :, dir_n], 0, 1)
            if normalize:
                image = normalize_to_rgb(image, normalize_to)
            filepath = f'{directory}/{file_name}_{dir_n + 1}.tiff'
            imageio.imwrite(filepath, image, format = 'tiff')
    else:
        image = np.swapaxes(image_distances, 0, 1)
        if normalize:
            image = normalize_to_rgb(image, normalize_to)
        filepath = f'{directory}/{file_name}.tiff'
        imageio.imwrite(filepath, image, format = 'tiff')


    return image_distances

def map_mean_peak_amplitudes(image_stack = None, image_params = None, image_peaks_mask = None, 
                            distribution = "wrapped_cauchy", 
                            amplitude_threshold = 3000, rel_amplitude_threshold = 0.1,
                            gof_threshold = 0.5, only_mus = False, directory = "maps", 
                            normalize = False, normalize_to = [None, None],
                            image_sig_peaks_mask = None, image_mean_amplitudes = None):
    """
    Maps the mean peak amplitude for every pixel.

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
    - directory: string
        The directory path defining where the resulting image should be writen to.
        If None, no image will be writen.
    - normalize: bool
        Whether the created image should be normalized (amd displayed with colors)
    - normalize_to: list
        List of min and max value that defines the range the image is normalized to.
        If min (or max) is None, the minimum (or maximum) of the image will be used.
    - image_sig_peaks_mask: np.ndarray (n, m)
        If significant peaks mask is already calculated, 
        it can be provided here to speed up the process.
    - image_mean_ampplitudes: np.ndarray (n, m)
        If the mean amplitudes have already been calculated, they can be inserted here.

    Returns:
    - image_mean_amplitudes: np.ndarray (n, m)
        The mean amplitude for every pixel.
    """

    if not isinstance(image_mean_amplitudes, np.ndarray):
        image_mean_amplitudes = get_mean_peak_amplitudes(image_stack = image_stack, 
                            image_params = image_params, 
                            image_peaks_mask = image_peaks_mask, 
                            distribution = distribution,
                            amplitude_threshold = amplitude_threshold, 
                            rel_amplitude_threshold = rel_amplitude_threshold, 
                            gof_threshold = gof_threshold,
                            only_mus = only_mus, 
                            image_sig_peaks_mask = image_sig_peaks_mask)

    image = np.swapaxes(image_mean_amplitudes, 0, 1)
    if normalize:
        image = normalize_to_rgb(image, normalize_to)

    imageio.imwrite(f'{directory}/peak_amplitudes_map.tiff', image, format = 'tiff')

    return image_mean_amplitudes

def map_mean_peak_widths(image_stack = None, image_params = None, 
                            image_peaks_mask = None, distribution = "wrapped_cauchy", 
                            amplitude_threshold = 3000, rel_amplitude_threshold = 0.1,
                            gof_threshold = 0.5, directory = "maps",
                            normalize = False, normalize_to = [None, None],
                            image_sig_peaks_mask = None, image_mean_widths = None):
    """
    Maps the mean peak width for every pixel.

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
    - directory: string
        The directory path defining where the resulting image should be writen to.
        If None, no image will be writen.
    - normalize: bool
        Whether the created image should be normalized (amd displayed with colors)
    - normalize_to: list
        List of min and max value that defines the range the image is normalized to.
        If min (or max) is None, the minimum (or maximum) of the image will be used.
    - image_sig_peaks_mask: np.ndarray (n, m, max_find_peaks)
        If the significant peaks mask is already calculated,
        it can be provided here to speed up the process.
    - image_mean_widths: np.ndarray (n, m)
        If the mean widths have already been calculated, they can be inserted here.

    Returns:
    - image_mean_widths: np.ndarray (n, m)
        The mean amplitude for every pixel.
    """

    if not isinstance(image_mean_widths, np.ndarray):
        image_mean_widths = get_mean_peak_widths(image_stack = image_stack, 
                                image_params = image_params, 
                                image_peaks_mask = image_peaks_mask, 
                                distribution = distribution,
                                amplitude_threshold = amplitude_threshold, 
                                rel_amplitude_threshold = rel_amplitude_threshold, 
                                gof_threshold = gof_threshold,
                                image_sig_peaks_mask = image_sig_peaks_mask)

    # scale = hwhm for wrapped cauchy distribution
    image_mean_widths = image_mean_widths * 180 / np.pi * 2

    image = np.swapaxes(image_mean_widths, 0, 1)
    if normalize:
        image = normalize_to_rgb(image, normalize_to)

    imageio.imwrite(f'{directory}/peak_widths_map.tiff', image, format = 'tiff')

    return image_mean_widths

def map_directions(image_peak_pairs = None, image_mus = None, only_peaks_count = -1, 
                    exclude_lone_peaks = True,
                    image_direction_sig = None, significance_threshold = 0,
                    directory = "maps", normalize = False, normalize_to = [0, 180],
                    image_directions = None):
    """
    Maps the directions for every pixel.

    Parameters:
     - image_peak_pairs: np.ndarray (n, m, 3, 2)
        The peak pairs for every pixel, where the fourth dimension contains both peak numbers of
        a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the third dimension
        is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
        The first two dimensions are the image dimensions.
    - image_mus: np.ndarray (n, m, q)
        The mus (peak centers) for every pixel.
    - only_peaks_count: int or list of ints
        Only use pixels where the number of peaks equals this number. -1 uses every number of peaks.
    - exclude_lone_peaks: bool
        Whether to exclude the directions for lone peaks 
        (for peak pairs with only one number unequal -1 e.g. [2, -1]).
    - image_direction_sig: np.ndarray (n, m, 3)
        Image containing the significance of every direction for every pixel.
        Can be created with "map_direction_significances" or "get_image_direction_significances".
        Used for threshold filtering with significance_threshold.
    - significance_threshold: float
        Value between 0 and 1.
        Directions with a significance below this threshold will not be mapped.
    - directory: string
        The directory path defining where the resulting image should be writen to.
        If None, no image will be writen.
    - normalize: bool
        Whether the created image should be normalized (amd displayed with colors)
    - normalize_to: list
        List of min and max value that defines the range the image is normalized to.
        If min (or max) is None, the minimum (or maximum) of the image will be used.
    - image_directions: np.ndarray (n, m, 3)
        If the directions have already been calculated, they can be inserted here.

    Returns:
    - image_directions: np.ndarray (n, m, 3)
        The directions for every pixel.
    """

    if not isinstance(image_directions, np.ndarray):
        print("Calculating directions...")
        image_directions = calculate_directions(image_peak_pairs, image_mus, 
                            only_peaks_count = only_peaks_count, exclude_lone_peaks = exclude_lone_peaks)
        print("Done")

    # Apply significance threshold filter if given
    if isinstance(image_direction_sig, np.ndarray) and significance_threshold > 0:
        image_directions[image_direction_sig < significance_threshold] = -1

    if directory != None:
        if not os.path.exists(directory):
            os.makedirs(directory)

        image_directions[image_directions != -1] = image_directions[image_directions != -1] * 180 / np.pi
        for dir_n in range(image_directions.shape[-1]):
            image = np.swapaxes(image_directions[:, :, dir_n], 0, 1)
            if normalize:
                image = normalize_to_rgb(image, normalize_to)
            filepath = f'{directory}/dir_{dir_n + 1}.tiff'
            imageio.imwrite(filepath, image, format = 'tiff')

    return image_directions

def map_direction_significances(image_stack = None, image_peak_pairs = None, image_params = None, 
                                image_peaks_mask = None, distribution = "wrapped_cauchy", 
                                amplitude_threshold = 0, rel_amplitude_threshold = 0,
                                gof_threshold = 0, weights = [1, 1], sens = [1, 1],
                                only_mus = False, directory = "maps",
                                normalize = False, normalize_to = [None, None],
                                num_processes = 2, image_direction_sig = None):
    """
    Maps the significance of the directions for every pixel.
    -Function that could be updated without need for multi processing and in similar manner
    as the other map functions.-

    Parameters:
    - image_stack: np.ndarray (n, m, p)
        The image stack containing the measured intensities.
        n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
    - image_peak_pairs: np.ndarray (n, m, 3, 2)
        The peak pairs for every pixel, where the fourth dimension contains both peak numbers of
        a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the third dimension
        is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
        The first two dimensions are the image dimensions.
    - image_params: np.ndarray (n, m, q)
        The output of fitting the image stack, which stores the parameters of the full fitfunction.
    - image_peaks_mask: np.ndarray (n, m, max_find_peaks, p)
        The mask defining which of the p-measurements corresponds to one of the peaks.
        The first two dimensions are the image dimensions.
    - distribution: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
        The name of the distribution.
    - amplitude_threshold: float
        Peak-Pairs with a amplitude below this threshold will not be evaluated.
    - rel_amplitude_threshold: float
        Value between 0 and 1.
        Peak-Pairs  with a relative amplitude (to maximum - minimum intensity of the pixel) below
        this threshold will not be evaluated.
    - gof_threshold: float
        Value between 0 and 1.
        Peak-Pairs with a goodness-of-fit value below this threshold will not be evaluated.
    - weights: list (2, )
        The weights for the amplitude and for the goodnes-of-fit, when calculating the significance
    - sens: list (2, )
        The sensitivity values for the amplitude (first value) and for the goodness-of-fit (second value),
        when calculating the significance.
    - only_mus: boolean
        Whether only the mus are provided in image_params. If so, only amplitude_threshold is used.
    - directory: string
        The directory path defining where the resulting image should be writen to.
        If None, no image will be writen.
    - normalize: bool
        Whether the created image should be normalized (amd displayed with colors)
    - normalize_to: list
        List of min and max value that defines the range the image is normalized to.
        If min (or max) is None, the minimum (or maximum) of the image will be used.
    - num_processes: int
        Defines the number of processes to split the task into.
    - image_direction_sig: np.ndarray (n, m, 3)
        If the direction significances have already been calculated, they can be inserted here.

    Returns:
    - image_direction_sig: np.ndarray (n, m, 3)
        The significances of the directions for every pixel.
    """

    if not isinstance(image_direction_sig, np.ndarray):
        image_direction_sig = get_image_direction_significances(image_stack = image_stack, 
                                    image_peak_pairs = image_peak_pairs, 
                                    image_params = image_params, 
                                    image_peaks_mask = image_peaks_mask, distribution = distribution, 
                                    amplitude_threshold = amplitude_threshold,
                                    rel_amplitude_threshold = rel_amplitude_threshold, 
                                    gof_threshold = gof_threshold,
                                    weights = weights, sens = sens, only_mus = only_mus, 
                                    num_processes = num_processes)
    
    if directory != None:
        if not os.path.exists(directory):
            os.makedirs(directory)

        for dir_n in range(image_direction_sig.shape[-1]):
            image = np.swapaxes(image_direction_sig[:, :, dir_n], 0, 1)
            if normalize:
                image = normalize_to_rgb(image, normalize_to)
            filepath = f'{directory}/dir_{dir_n + 1}_sig.tiff'
            imageio.imwrite(filepath, image, format = 'tiff')

    return image_direction_sig