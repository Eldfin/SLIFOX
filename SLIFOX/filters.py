import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, gaussian_filter1d, median_filter
from scipy.signal import savgol_filter
import pymp
from tqdm import tqdm

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

def fourier_smoothing(signal, threshold, window):
    """
    Filters a one dimensional signal with the fourier filter.

    Parameters:
    - signal: np.ndarray (n, )
        Array of values (e.g. intensities) that should be filtered.
        Could also be multidimensional (image), with last axis as signals.
    - threshold: float
        Threshold value between 0 and 1. Frequencies (normalized) above the threshold will be cut off.
        Lower threshold leads to more filtering.
    - window: float
        Window value between 0 and 1.
        Controls the sharpness of the cutoff. 
        A larger value allows a smoother transition, blending frequencies more gradually.

    Returns:
    - result: np.ndarray (n, )
        The filtered signal.
    """
    num_values = signal.shape[-1]
    fft = np.fft.fft(signal, axis=-1)
    frequencies = np.fft.fftfreq(num_values)
    frequencies = frequencies / np.max(frequencies)

    if window == 0:
        multiplier = np.ones(num_values)
        mask = (np.abs(frequencies) >= threshold) 
        multiplier[mask] = 0
    else:
        multiplier = 1 - (0.5 + 0.5 * np.tanh((np.abs(frequencies) - threshold) / window))

    fft = np.multiply(fft, multiplier)
    result = np.real(np.fft.ifft(fft)).astype(signal.dtype)
    result[result < 0] = 0

    return result

#@njit(cache=True, fastmath=True)
def fourier_gauss_smoothing(signal, sigma):
    """
    Filters a one dimensional signal in the fourier domain using a gaussian window.

    Parameters:
    - signal: np.ndarray (n, )
        Array of values (e.g. intensities) that should be filtered.
        Could also be multidimensional (image), with last axis as signals.
    - sigma: float
        Standard deviation of the gaussian filter used to smooth frequencies in the frequency domain.

    Returns:
    - result: np.ndarray (n, )
        The filtered signal.
    """
    num_values = signal.shape[-1]
    fft = np.fft.fft(signal, axis=-1)
    frequencies = np.fft.fftfreq(num_values)
    frequencies = frequencies / np.max(frequencies)
    multiplier = np.exp(-0.5 * (frequencies / sigma) ** 2)
    fft = np.multiply(fft, multiplier)
    result = np.real(np.fft.ifft(fft)).astype(signal.dtype)
    result[result < 0] = 0

    return result

@njit(cache = True, fastmath = True)
def circular_moving_average_filter(data, window_size):
    n = len(data)
    smoothed_arr = np.zeros_like(data)

    for i in range(n):
        window_indices = [(i + j) % n for j in range(-window_size//2, window_size//2 + 1)]
        window = data[window_indices]
        smoothed_arr[i] = np.mean(window)

    return smoothed_arr

@njit(cache = True, fastmath = True)
def gaussian_kernel(sigma):
    radius = round(4 * sigma)
    kernel_size = (2 * radius) + 1
    kernel = np.zeros(kernel_size, dtype=np.float64)
    
    for x in range(-radius, radius + 1):
        kernel[x + radius] = np.exp(-0.5 * (x / sigma) ** 2)
    
    # Normalize the kernel
    return kernel / np.sum(kernel)

@njit(cache = True, fastmath = True)
def apply_gaussian_filter1d(arr, sigma):
    if sigma <= 0:
        raise ValueError("Sigma must be greater than 0.")
        
    kernel = gaussian_kernel(sigma)
    radius = len(kernel) // 2
    
    # Prepare the output array
    output = np.zeros_like(arr)
    padded_arr = np.empty(len(arr) + 2 * radius)
    
    # Wrap padding
    padded_arr[:radius] = arr[-radius:]  # Left wrap
    padded_arr[radius:radius + len(arr)] = arr  # Original values
    padded_arr[radius + len(arr):] = arr[:radius]  # Right wrap

    # Apply the Gaussian filter
    for i in range(len(arr)):
        output[i] = np.sum(kernel * padded_arr[i:i + kernel.size])
    
    return output

# To-Do: Add numba for all filters
def apply_filter(data, filter_params, num_processes = 2):
    """
    Applies a filter to data.

    Parameters:
    - data: np.ndarray (n, ) or (p, q, n)
        Array of values (e.g. intensities) that should be filtered.
    - filter_params: list (m, )
        List that defines which filter to use. First value of list is a string with
        "fourier", "gauss", "uniform", "median", "moving_average", or "savgol".
        The following one to two values are the params for this filter.
    - num_processes: int
        The number of processes to use.

    Returns:
    - result: np.ndarray (n, ) or (p, q, n)
        The filtered data.
    """

    def apply_filter_1d(data_1d):
        if filter_params[0] == "fourier":
            return fourier_smoothing(data_1d, filter_params[1], filter_params[2])
        elif filter_params[0] == "fourier_gauss":
            return fourier_gauss_smoothing(data_1d, filter_params[1])    
        elif filter_params[0] == "gauss":
            order = filter_params[2] if len(filter_params) == 3 else 0
            return gaussian_filter1d(data_1d, filter_params[1], order=order, mode="wrap")
            #return apply_gaussian_filter1d(data_1d, filter_params[1])
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

    if data.ndim == 1:
        return apply_filter_1d(data)

    n_rows, n_cols = data.shape[0], data.shape[1]
    total_pixels = n_rows * n_cols
    flat_data = data.reshape((total_pixels, data.shape[-1]))
    data_dtype = data.dtype
    result_data = pymp.shared.array((total_pixels, data.shape[-1]), dtype=data_dtype)
    # Initialize the progress bar
    pbar = tqdm(total = total_pixels, 
                desc = f'Applying filter to data',
                smoothing = 0)
    shared_counter = pymp.shared.array((num_processes, ), dtype = int)

    with pymp.Parallel(num_processes) as p:
        # Process data in square chunks
        for i in p.range(total_pixels):
            data_1d = flat_data[i].astype(np.float32)
            result_1d = apply_filter_1d(data_1d).astype(data_dtype)
            result_data[i] = result_1d

        # Update progress bar
            shared_counter[p.thread_num] += 1
            status = np.sum(shared_counter)
            pbar.update(status - pbar.n)
        
    # Set the progress bar to 100%
    pbar.update(pbar.total - pbar.n)

    result_data = result_data.reshape((n_rows, n_cols, result_data.shape[-1]))

    return result_data