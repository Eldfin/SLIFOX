import numpy as np
import matplotlib.pyplot as plt
import os
from .SLIF import full_fitfunction
from .peaks_evaluator import calculate_peaks_gof
from .wrapped_distributions import distribution_pdf
from .utils import angle_distance

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


def plot_data_pixels(data, image_params, image_peaks_mask, peak_pairs = None, only_mus = False,
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
    - peak_pairs: np.ndarray (n, m, 3, 2)
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

            if not only_mus:
                params = image_params[i, j]
                heights = params[0:-1:3]
                scales = params[2::3]
                mus = params[1::3]
                mus = mus[heights >= 1]
                num_peaks = len(mus)
            else:
                mus = image_params[i, j]
                num_peaks = len(mus)
                if not np.any(peaks_mask[0]):
                    num_peaks = 0
            
            if num_peaks == 0: continue
            if not only_mus:
                offset = params[-1]
                params = params[:(3 * len(mus))]
                params = np.append(params, offset)

                model_y = full_fitfunction(angles, params, distribution)
                peaks_gof = calculate_peaks_gof(intensities, model_y, peaks_mask, method = "r2")

                scales = scales[heights >= 1]
                peaks_gof = peaks_gof[heights >= 1]
                heights = heights[heights >= 1]
            
            plt.errorbar(angles*180/np.pi, intensities, yerr=intensities_err, marker = "o", 
                            linestyle="", capsize=5, color="black")
            if only_mus:
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

            if not only_mus:
                x_f = np.linspace(0, 2*np.pi, 2000, endpoint=False)
                y_f = full_fitfunction(x_f, params, distribution)
                FitLine, = plt.plot(x_f*180/np.pi, y_f, marker='None', linestyle="-", color="black")

                plot_peaks_gof(peaks_gof, heights, mus, scales, 
                                distribution, peaks_mask, angles)

            if isinstance(peak_pairs, np.ndarray):
                if not only_mus:
                    plot_directions(peak_pairs[i, j], mus, distribution, 
                                        heights = heights, scales = scales)
                else:
                    plot_directions(peak_pairs[i, j], mus, distribution)

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
            