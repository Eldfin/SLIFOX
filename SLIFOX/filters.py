import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, gaussian_filter1d, median_filter
from scipy.signal import savgol_filter

@njit(cache=True, fastmath=True)
def _fftfreq(n, d=1.0):
    val = 1.0 / (n * d)
    results = np.empty(n)
    N = (n-1)//2 + 1
    p1 = np.arange(0, N)
    results[:N] = p1
    p2 = np.arange(-(n//2), 0)
    results[N:] = p2
    return results * val

@njit(cache=True, fastmath=True)
def _fourier_transform(signal):
    N = len(signal)
    result = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        for n in range(N):
            result[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
    return result

@njit(cache=True, fastmath=True)
def _inverse_fourier_transform(fft_result):
    N = len(fft_result)
    result = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        for k in range(N):
            result[n] += fft_result[k] * np.exp(2j * np.pi * k * n / N)
    return result / N

@njit(cache=True, fastmath=True)
def fourier_smoothing(signal, threshold, window):
    """
    Finds the closest true pixel for a given 2d-mask and a start_pixel.

    Parameters:
    - signal: np.ndarray (n, )
        Array of values (e.g. intensities) that should be filtered.
    - threshold: float
        Threshold value between 0 and 1. Lower threshold leads to stronger smoothing.
    - window: float
        Value between 0 and 1 that defines the transition region around the threshold.
        Should be smaller than the threshold value. Increasing window leads to smoother transition
        between peaks/edges, smoothing occurs more gradually.

    Returns:
    - result: np.ndarray (n, )
        The filtered signal.
    """
    fft_result = _fourier_transform(signal)
    frequencies = _fftfreq(len(fft_result))
    frequencies = frequencies / frequencies.max()

    multiplier = 1 - (0.5 + 0.5 * np.tanh((np.abs(frequencies) - threshold) / window))
    fft_result = fft_result * multiplier

    return np.real(_inverse_fourier_transform(fft_result)).astype(signal.dtype)

@njit(cache = True, fastmath = True)
def circular_moving_average_filter(data, window_size):
    n = len(data)
    smoothed_arr = np.zeros_like(data)

    for i in range(n):
        window_indices = [(i + j) % n for j in range(-window_size//2, window_size//2 + 1)]
        window = data[window_indices]
        smoothed_arr[i] = np.mean(window)

    return smoothed_arr

# To-Do: Add numba for all filters
def apply_filter(data, filter_params):
    """
    Applies a filter to data.

    Parameters:
    - data: np.ndarray (n, ) or (p, q, n)
        Array of values (e.g. intensities) that should be filtered.
    - filter_params: list (m, )
        List that defines which filter to use. First value of list is a string with
        "fourier", "gauss", "uniform", "median", "moving_average", or "savgol".
        The following one to two values are the params for this filter.

    Returns:
    - result: np.ndarray (n, ) or (p, q, n)
        The filtered data.
    """

    def apply_filter_1d(data_1d):
        if filter_params[0] == "fourier":
            return fourier_smoothing(data_1d, filter_params[1], filter_params[2])
        elif filter_params[0] == "gauss":
            order = filter_params[2] if len(filter_params) == 3 else 0
            return gaussian_filter1d(data_1d, filter_params[1], order=order, mode="wrap")
        elif filter_params[0] == "uniform":
            return uniform_filter1d(data_1d, filter_params[1], mode="wrap")
        elif filter_params[0] == "median":
            return median_filter(data_1d, size=filter_params[1], mode="wrap")
        elif filter_params[0] == "moving_average":
            return circular_moving_average_filter(data_1d, filter_params[1])
        elif filter_params[0] == "savgol":
            order = filter_params[2] if len(filter_params) == 3 else 0
            return savgol_filter(data_1d, filter_params[1], order, mode="wrap")
        return data_1d

    return np.apply_along_axis(apply_filter_1d, axis=-1, arr=data)