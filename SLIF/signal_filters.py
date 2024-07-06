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
    # signal is 1d-array
    # threshold between 0 and 1
    # lower threshold leads to stronger smoothing
    # window defines the transition region around the threshold
    # increasing window leads to smoother transition between peaks/edges, smmothing occurs more gradually
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
def apply_filter(data, init_fit_filter):
    if init_fit_filter[0] == "fourier":
        data = fourier_smoothing(data, init_fit_filter[1], init_fit_filter[2])
    elif init_fit_filter[0] == "gauss":
        if len(init_fit_filter) == 3:
            order = init_fit_filter[2]
        else:
            order = 0
        data = gaussian_filter1d(data, init_fit_filter[1], order = order, mode="wrap")
    elif init_fit_filter[0] == "uniform":
        data = uniform_filter1d(data, init_fit_filter[1], mode="wrap")
    elif init_fit_filter[0] == "median":
        data = median_filter(data, size=init_fit_filter[1], mode="wrap")
    elif init_fit_filter[0] == "moving_average":
        data = circular_moving_average_filter(data, init_fit_filter[1])
    elif init_fit_filter[0] == "savgol":
        if len(init_fit_filter) == 3:
            order = init_fit_filter[2]
        else:
            order = 0
        data = savgol_filter(data, init_fit_filter[1], order, mode="wrap")

    return data
