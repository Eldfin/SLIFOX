import numpy as np
import matplotlib.pyplot as plt
import os
from .SLIF import full_fitfunction
from .peaks_evaluator import calculate_peaks_gof, get_number_of_peaks, get_direction_significances, \
                                get_peak_distances, get_peak_amplitudes, get_peak_widths, \
                                    calculate_directions
from .wrapped_distributions import distribution_pdf
from .utils import angle_distance
import imageio

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
        The goodness of fit values for both peaks.
    - peaks_mask: np.ndarray (n_peaks, m)
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
        if k == 0: color = "blue"
        elif k == 1: color = "red"
        elif k == 2: color = "green"
        elif k == 3: color = "brown"
        elif k == 4: color = "purple"
        elif k == 5: color = "orange"
        if gof < 0: gof = 0
        amplitude = heights[k] * distribution_pdf(0, 0, scales[k], distribution)
        ymax = gof**gof_weight * amplitude

        x_f = np.linspace(0, 2*np.pi, 1000, endpoint=False)
        peaks_x_f_1 = np.empty(0)
        peaks_x_f_2 = np.empty(0)
        wrapped = False
        for l in range(len(angles)):
            if l+1 < len(angles) and peaks_mask[k][l] and peaks_mask[k][l+1]:
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
        plt.plot((peaks_x_f_1*180/np.pi) % 360, peak_y_f_1, marker='None', linestyle="-", color=color)
        plt.plot((peaks_x_f_2*180/np.pi) % 360, peak_y_f_2, marker='None', linestyle="-", color=color)
        plt.plot((x_f*180/np.pi) % 360, peak_y_f, marker='None', linestyle="--", color=color)
        plt.vlines((mus[k]*180/np.pi) % 360, 0, ymax , color = color)

def plot_directions(peak_pairs, mus, distribution, heights = None, scales = None):
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
    """
    for k, pair in enumerate(peak_pairs):
        if pair[0] == -1 and pair[1] == -1: continue
        colors = []
        for index in pair:
            if index == 0: colors.append("blue")
            elif index == 1: colors.append("red")
            elif index == 2: colors.append("green")
            elif index == 3: colors.append("brown")
            elif index == 4: colors.append("purple")
            elif index == 5: colors.append("orange")

        if len(colors) == 1:
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
            
            plt.vlines(direction, ymin, limit_max, color = colors[0])
            plt.vlines(direction + 180, ymin2, limit_max, color = colors[0])
        else:
            distance = angle_distance(mus[pair[0]], mus[pair[1]])
            direction = (mus[pair[0]] + distance / 2) % (np.pi)
            direction = direction * 180/np.pi
            alternating_vline(direction, colors=colors, num_segments=10)
            alternating_vline(direction + 180, colors=colors, num_segments=10)

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
    - peaks_mask: np.ndarray (n_peaks, n)
        The mask array defining which intensities corresponds to which peak.
    - distribution: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
        The name of the distribution.
    """
    angles = np.linspace(0, 2*np.pi, num=len(intensities), endpoint=False)

    model_y = full_fitfunction(angles, best_parameters, distribution)
    peaks_gof = calculate_peaks_gof(intensities, model_y, peaks_mask, method = "r2")

    plt.errorbar(angles*180/np.pi, intensities, yerr=intensities_err, marker = "o", linestyle="", capsize=5)
    plt.xlabel("Angle")
    plt.ylabel("Intensity")

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
                distribution = "wrapped_cauchy", indices = None,
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
    - directory: string
        The directory path where the plots should be written to.
    """

    angles = np.linspace(0, 2*np.pi, num=data.shape[2], endpoint=False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):

            intensities = data[i, j]
            intensities_err = np.sqrt(intensities)
            peaks_mask = image_peaks_mask[i, j]
            num_peaks = np.count_nonzero(np.any(peaks_mask, axis = -1), axis = -1)
            if num_peaks == 0: continue

            plt.errorbar(angles*180/np.pi, intensities, yerr=intensities_err, marker = "o", 
                            linestyle="", capsize=5, color="black")

            if not only_mus:
                params = image_params[i, j]
                heights = params[0:-1:3]
                scales = params[2::3]
                mus = params[1::3]
                offset = params[-1]
                params = params[:(3 * num_peaks)]
                params = np.append(params, offset)

                model_y = full_fitfunction(angles, params, distribution)
                peaks_gof = calculate_peaks_gof(intensities, model_y, peaks_mask, method = "r2")
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
                    if k == 0: color = "blue"
                    elif k == 1: color = "red"
                    elif k == 2: color = "green"
                    elif k == 3: color = "brown"
                    elif k == 4: color = "purple"
                    elif k == 5: color = "orange"
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
            
def map_number_of_peaks(image_stack, image_params, image_peaks_mask, distribution = "wrapped_cauchy", 
                            gof_threshold = 0.5, amplitude_threshold = 0.1, only_mus = False,
                            colors = ["black", "green", "red", "yellow", "blue", "magenta", "cyan"],
                            directory = "maps"):
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
    - gof_threshold: float
        Value between 0 and 1.
        Peaks with a goodness-of-fit value below this threshold will not be counted.
    - amplitude_threshold: float
        Value between 0 and 1.
        Peaks with a relative amplitude (to maximum - minimum intensity of the pixel) below
        this threshold will not be counted.
    - colors: list
        List of the color names that should be used for the colormap in the image.
        First color will be used for zero peaks, second for 1 peaks, third for 2 peaks, ...
    - directory: string
        The directory path defining where the resulting image should be writen to.
        If None, no image will be writen.

    Returns:
    - image_num_peaks: np.ndarray (n, m)
        The number of peaks for every pixel.
    - image_used_peaks_mask: np.ndarray (n, m, max_peaks)
        Mask that stores the information, which peaks are used in the counting process.
    """

    image_num_peaks, image_valid_peaks_mask = get_number_of_peaks(image_stack, 
                            image_params, image_peaks_mask, distribution = distribution, 
                            gof_threshold = gof_threshold, amplitude_threshold = amplitude_threshold, 
                            only_mus = only_mus)

    if directory != None:
        if not os.path.exists(directory):
                os.makedirs(directory)

        colormap = np.empty((len(colors), 3), dtype = np.uint8)
        for i, color in enumerate(colors):
            if color == "black":
                colormap[i] = [0, 0, 0]
            elif color == "green":
                colormap[i] = [0, 255, 0]
            elif color == "red":
                colormap[i] = [255, 0, 0]
            elif color == "yellow":
                colormap[i] = [255, 255, 0]
            elif color == "blue":
                colormap[i] = [0, 0, 255]
            elif color == "magenta":
                colormap[i] = [255, 0, 255]
            elif color == "cyan":
                colormap[i] = [0, 255, 255]

        image = colormap[image_num_peaks]

        imageio.imwrite(f'{directory}/n_peaks_map.tiff', np.swapaxes(image, 0, 1), format = 'tiff')

    return image_num_peaks
    
def map_peak_distances(image_stack, image_params, image_peaks_mask, distribution = "wrapped_cauchy", 
                            gof_threshold = 0.5, amplitude_threshold = 0.1, only_mus = False,
                            only_peaks_count = 2, directory = "maps"):
    
    image_distances = get_peak_distances(image_stack, image_params, image_peaks_mask, 
                            distribution = distribution, gof_threshold = gof_threshold, 
                            amplitude_threshold = amplitude_threshold, only_mus = only_mus)

    imageio.imwrite(f'{directory}/peak_distances.tiff', np.swapaxes(image_distances, 0, 1), format = 'tiff')

    return image_distances

def map_peak_amplitudes(image_stack, image_params, image_peaks_mask, distribution = "wrapped_cauchy", 
                            gof_threshold = 0.5, amplitude_threshold = 0.1, only_mus = False,
                            directory = "maps"):

    image_amplitudes = get_peak_amplitudes(image_stack, image_params, image_peaks_mask, 
                        distribution = distribution, gof_threshold = gof_threshold, 
                        amplitude_threshold = amplitude_threshold, only_mus = only_mus)

    imageio.imwrite(f'{directory}/peak_amplitudes.tiff', np.swapaxes(image_amplitudes, 0, 1), format = 'tiff')

    return image_amplitudes

def map_peak_widths(image_stack, image_params, image_peaks_mask, distribution = "wrapped_cauchy", 
                            gof_threshold = 0.5, amplitude_threshold = 0.1, directory = "maps"):

    image_scales = get_peak_widths(image_stack, image_params, image_peaks_mask, distribution = distribution, 
                            gof_threshold = gof_threshold, amplitude_threshold = amplitude_threshold)

    imageio.imwrite(f'{directory}/peak_widths_map.tiff', np.swapaxes(image_scales, 0, 1), format = 'tiff')

    return image_scales

def map_directions(image_peak_pairs, image_mus, only_peaks_count = -1, directory = "maps"):

    image_directions = calculate_directions(image_peak_pairs, image_mus, only_peaks_count = only_peaks_count)

    if directory != None:
        if not os.path.exists(directory):
                os.makedirs(directory)

        for dir_n in range(max_directions):
            write_directions = np.swapaxes(image_directions[:, :, dir_n], 0, 1)
            write_directions[write_directions != -1] = write_directions[write_directions != -1] * 180 / np.pi
            imageio.imwrite(f'{directory}/dir_{dir_n + 1}.tiff', write_directions)

def map_direction_significances(image_stack, image_peak_pairs, image_params, 
                                image_peaks_mask, distribution = "wrapped_cauchy", weights = [1, 1], 
                                num_processes = 2, directory = "maps"):

    image_significances = get_direction_significances(image_stack, image_peak_pairs, image_params, 
                                image_peaks_mask, distribution = distribution, weights = weights, 
                                num_processes = num_processes)
    
    if directory != None:
        if not os.path.exists(directory):
                os.makedirs(directory)

        for dir_n in range(max_significances):
            imageio.imwrite(f'{directory}/dir_{dir_n + 1}_sig.tiff', 
                                np.swapaxes(image_significances[:, :, dir_n], 0, 1))