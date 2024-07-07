import numpy as np
import matplotlib.pyplot as plt
import os
from .SLIF import full_fitfunction
from .peaks_evaluator import calculate_peaks_gof
from .wrapped_distributions import distribution_pdf
from .utils import angle_distance

def alternating_vline(x, ax=None, colors=['blue', 'red'], num_segments=100, **kwargs):
    if ax is None:
        ax = plt.gca()
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)
    y_vals = np.linspace(ymin, ymax, num_segments + 1)
    for i in range(num_segments):
        ax.vlines(x, y_vals[i], y_vals[i+1], color=colors[i % len(colors)], **kwargs)

def plot_peaks_gof(peaks_gof, heights, mus, scales, 
                        distribution, peaks_mask, angles, gof_weight = 1):
    for k, gof in enumerate(peaks_gof):
        if k == 0: color = "blue"
        elif k == 1: color = "red"
        elif k == 2: color = "green"
        elif k == 3: color = "brown"
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
            elif l+1 == len(angles) and peaks_mask[k][l] and peaks_mask[k][0]:
                if not wrapped:
                    peaks_x_f_1 = np.append(peaks_x_f_1, np.linspace(angles[l], 
                                                            2*np.pi, 50, endpoint=False))
                else:
                    peaks_x_f_2 = np.append(peaks_x_f_2, np.linspace(angles[l], 
                                                            2*np.pi, 50, endpoint=False))
            elif l > 0 and l+1 < len(angles) and not peaks_mask[k][l] \
                    and peaks_mask[k][l+1] and len(peaks_x_f_1) > 0:
                wrapped = True

        peak_y_f_1 = heights[k] * distribution_pdf(peaks_x_f_1, mus[k], scales[k], distribution)
        peak_y_f_2 = heights[k] * distribution_pdf(peaks_x_f_2, mus[k], scales[k], distribution)
        peak_y_f = heights[k] * distribution_pdf(x_f, mus[k], scales[k], distribution)
        plt.plot((peaks_x_f_1*180/np.pi) % 360, peak_y_f_1, marker='None', linestyle="-", color=color)
        plt.plot((peaks_x_f_2*180/np.pi) % 360, peak_y_f_2, marker='None', linestyle="-", color=color)
        plt.plot((x_f*180/np.pi) % 360, peak_y_f, marker='None', linestyle="--", color=color)
        plt.vlines((mus[k]*180/np.pi) % 360, 0, ymax , color = color)

def plot_directions(peak_pairs, heights, mus, scales, distribution):
    for k, pair in enumerate(peak_pairs):
        if pair[0] == -1 and pair[1] == -1: continue
        colors = []
        for index in pair:
            if index == 0: colors.append("blue")
            elif index == 1: colors.append("red")
            elif index == 2: colors.append("green")
            elif index == 3: colors.append("brown")

        if len(colors) == 1:
            ax = plt.gca()
            limit_min, limit_max = ax.get_ylim()
            direction = mus[pair[0]] % (np.pi)
            if direction == mus[pair[0]]:
                ymin = heights[pair[0]] * distribution_pdf(0, 0, scales[pair[0]], distribution)
                ymin2 = limit_min
            else:
                ymin = limit_min
                ymin2 = heights[pair[0]] * distribution_pdf(0, 0, scales[pair[0]], distribution)
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


def plot_data_pixels(data, output_params, output_peaks_mask, peak_pairs, distribution, indices = None,
                directory = "plots"):
    angles = np.linspace(0, 2*np.pi, num=data.shape[2], endpoint=False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):

            intensities = data[i][j]
            intensities_err = np.sqrt(intensities)

            params = output_params[i][j]
            heights = params[0:-1:3]
            scales = params[2::3]
            mus = params[1::3]
            mus = mus[heights >= 1]
            if len(mus) == 0: continue
            offset = params[-1]
            params = params[:(3 * len(mus))]
            params = np.append(params, offset)

            model_y = full_fitfunction(angles, params, distribution)
            peaks_mask = output_peaks_mask[i][j]
            peaks_gof = calculate_peaks_gof(intensities, model_y, peaks_mask, method = "r2")

            scales = scales[heights >= 1]
            peaks_gof = peaks_gof[heights >= 1]
            heights = heights[heights >= 1]
            
            plt.errorbar(angles*180/np.pi, intensities, yerr=intensities_err, marker = "o", 
                        linestyle="", capsize=5, color="black")
            plt.xlabel("Winkel")
            plt.ylabel("Intensität")

            x_f = np.linspace(0, 2*np.pi, 2000, endpoint=False)
            y_f = full_fitfunction(x_f, params, distribution)
            FitLine, = plt.plot(x_f*180/np.pi, y_f, marker='None', linestyle="-", color="black")

            plot_peaks_gof(peaks_gof, heights, mus, scales, 
                            distribution, peaks_mask, angles)

            plot_directions(peak_pairs[i, j], heights, mus, scales, distribution)

            max_rows = len(str(data.shape[0] - 1)) 
            max_cols = len(str(data.shape[1] - 1))
            if indices is None:
                x_str = f"{str(i):0{max_rows}d}"
                y_str = f"{str(j):0{max_cols}d}"
            else:
                x_str = f"{indices[i][j][0]:0{max_rows}d}"
                y_str = f"{indices[i][j][1]:0{max_cols}d}"
                
            if not os.path.exists(directory):
                os.makedirs(directory)

            plt.savefig(f"{directory}/x{x_str}y{y_str}.png")
            plt.clf()
            