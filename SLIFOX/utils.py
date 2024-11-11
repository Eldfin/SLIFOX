import numpy as np
import h5py
from numba import njit
import nibabel as nib
import tifffile
from collections import deque
from tqdm import tqdm
import pymp
import os
import contextlib

@njit(cache = True, fastmath = True)
def angle_distance(angle1, angle2, wrap = 2*np.pi):
    """
    Calculates the circular distance between two angles.

    Parameters:
    - angle1: float
        First angle in radians.
    - angle2: float
        Second angle in radians.

    Returns:
    - distance: float
        Shortest distance between both angles considering the cyclic nature.
    """
    # Calculate the difference between the angles
    diff = angle2 - angle1
    
    # Calculate the shortest distance considering the cyclic nature
    distance = (diff + wrap / 2) % wrap - wrap / 2
    
    return distance
    
# Define function like numpy.sum with parameter axis = -1 for use with numba
@njit(cache = True, fastmath = True)
def numba_sum_last_axis(arr):
    # Ensure input is a numpy array
    arr = np.asarray(arr)
    
    # Get the shape of the array
    shape = arr.shape
    
    # Initialize the output array for the result
    result_shape = shape[:-1]
    result = np.empty(result_shape, dtype = arr.dtype)
    
    # Iterate over all indices except the last axis
    for index in np.ndindex(result_shape):
        # Extract the sub-array for the current index
        sub_array = arr[index]
        
        # Compute the sum along the last axis
        sum_value = 0
        for value in sub_array:
            if value == value:
                sum_value += value
        
        # Store the result
        result[index] = sum_value
    
    return result

@njit(cache=True, fastmath=True)
def numba_sum_second_last_axis(arr):
    # Ensure input is a numpy array
    arr = np.asarray(arr)
    
    # Get the shape of the array
    shape = arr.shape

    # Get the size of the second-to-last axis
    second_last_axis_size = shape[-2]
    
    # Determine the shape of the output array
    result_shape = shape[:-2] + (shape[-1],)
    result = np.empty(result_shape, dtype = arr.dtype)
    
    # Iterate over all indices except the second-to-last axis
    for index in np.ndindex(result_shape):
        # Initialize the sum to zero
        sum_value = 0
        
        # Iterate over the second-to-last axis to compute the sum
        for i in range(second_last_axis_size):
            # Construct the full index manually
            full_index = (*index[:-1], i, index[-1])
            value = arr[full_index]
            
            # Only add the value if it is not NaN
            if value == value:
                sum_value += value
        
        # Store the result
        result[index] = sum_value
    
    return result

# Define function like numpy.repeat with parameter axis = -1 for numba use
@njit(cache = True, fastmath = True)
def numba_repeat_last_axis(arr, repeats):
    # Ensure input is a numpy array
    arr = np.asarray(arr)
    
    # Get the shape of the array
    shape = arr.shape
    
    # Get the size of the last axis
    last_axis_size = shape[-1]
    
    # Determine the shape of the output array
    result_shape = shape[:-1] + (last_axis_size * repeats,)
    result = np.empty(result_shape, dtype = arr.dtype)
    
    # Iterate over all indices except the last axis
    for index in np.ndindex(shape[:-1]):
        # Extract the sub-array for the current index
        sub_array = arr[index]
        
        # Repeat the elements along the last axis
        start = 0
        for value in sub_array:
            for _ in range(repeats):
                result[index + (start,)] = value
                start += 1
    
    return result

# Define function line numpy.max with parameter axis = -1 for use with numba
@njit(cache = True, fastmath = True)
def numba_max_last_axis(arr):
    # Ensure input is a numpy array
    arr = np.asarray(arr)
    
    # Get the shape of the array
    shape = arr.shape
    
    # Initialize the output array for the result
    result_shape = shape[:-1]
    result = np.empty(result_shape, dtype = arr.dtype)
    
    # Iterate over all indices except the last axis
    for index in np.ndindex(result_shape):
        # Extract the sub-array for the current index
        sub_array = arr[index]
        
        # Initialize the max value to the first element of the sub-array
        max_value = sub_array[0]
        
        # Iterate over the last axis to find the maximum value
        for value in sub_array[1:]:
            if value > max_value:
                max_value = value
        
        # Store the result
        result[index] = max_value
    
    return result

@njit(cache = True, fastmath = True)
def numba_min_last_axis(arr):
    # Ensure input is a numpy array
    arr = np.asarray(arr)
    
    # Get the shape of the array
    shape = arr.shape
    
    # Initialize the output array for the result
    result_shape = shape[:-1]
    result = np.empty(result_shape, dtype = arr.dtype)
    
    # Iterate over all indices except the last axis
    for index in np.ndindex(result_shape):
        # Extract the sub-array for the current index
        sub_array = arr[index]
        
        # Initialize the min value to the first element of the sub-array
        min_value = sub_array[0]
        
        # Iterate over the last axis to find the minimum value
        for value in sub_array[1:]:
            if value < min_value:
                min_value = value
        
        # Store the result
        result[index] = min_value
    
    return result

# Define function like numpy.nanmin with parameter arg = -1 for use with numba
@njit(cache = True)
def numba_nanmin_last_axis(arr):
    # Ensure input is a numpy array
    arr = np.asarray(arr)
    
    # Get the shape of the array
    shape = arr.shape
    
    # Initialize the output array for the result
    result_shape = shape[:-1]
    result = np.empty(result_shape, dtype = arr.dtype)
    
    # Iterate over all indices except the last axis
    for index in np.ndindex(result_shape):
        # Extract the sub-array for the current index
        sub_array = arr[index]
        
        # Initialize the min value to a very large number
        min_value = np.nan
        
        # Iterate over the last axis
        for value in sub_array:
            if not np.isnan(value):
                if np.isnan(min_value) or value < min_value:
                    min_value = value
        
        # Store the result
        result[index] = min_value
    
    return result

# Define function like numpy.nanmax with parameter arg = -1 for use with numba
@njit(cache = True)
def numba_nanmax_last_axis(arr):
    # Ensure input is a numpy array
    arr = np.asarray(arr)
    
    # Get the shape of the array
    shape = arr.shape
    
    # Initialize the output array for the result
    result_shape = shape[:-1]
    result = np.empty(result_shape, dtype = arr.dtype)
    
    # Iterate over all indices except the last axis
    for index in np.ndindex(result_shape):
        # Extract the sub-array for the current index
        sub_array = arr[index]
        
        # Initialize the max value to a very small number
        max_value = np.nan
        
        # Iterate over the last axis
        for value in sub_array:
            if not np.isnan(value):
                if np.isnan(max_value) or value > max_value:
                    max_value = value
        
        # Store the result
        result[index] = max_value
    
    return result

# Define function like numpy.nansum with axis = -1 for numba use
@njit(cache = True)
def numba_nansum_last_axis(arr):
    # Ensure input is a numpy array
    arr = np.asarray(arr)
    
    # Get the shape of the array
    shape = arr.shape
    
    # Initialize the output array for the result
    result_shape = shape[:-1]
    result = np.empty(result_shape, dtype = arr.dtype)
    
    # Iterate over all axes except the last one
    for index in np.ndindex(result_shape):
        # Extract the sub-array for the current index
        sub_array = arr[index]
        
        # Compute the sum of non-NaN values along the last axis
        sum_values = 0
        for value in sub_array:
            if not np.isnan(value):
                sum_values += value
        
        # Store the result
        result[index] = sum_values
    
    return result

# Define function like numpy.nanmean with axis = -1 for numba
@njit(cache = True)
def numba_nanmean_last_axis(arr):
    # Ensure input is a numpy array
    arr = np.asarray(arr)
    
    # Get the shape of the array
    shape = arr.shape
    
    # Initialize the output array for the mean
    mean_result = np.empty(shape[:-1], dtype = np.float64)
    
    # Iterate over all axes except the last one
    for index in np.ndindex(shape[:-1]):
        # Extract the sub-array for the current index
        sub_array = arr[index]
        
        # Initialize sum and count
        sum_values = 0.0
        count = 0
        
        # Iterate over the last axis values
        for value in sub_array:
            if not np.isnan(value):
                sum_values += value
                count += 1
        
        # Compute the mean for this sub-array
        if count > 0:
            mean_result[index] = sum_values / count
        else:
            mean_result[index] = np.nan
    
    return mean_result

# Define function like numpy.any with axis = -1 for numba
@njit(cache = True)
def numba_any_last_axis(arr):
    # Ensure input is a numpy array
    arr = np.asarray(arr)
    
    # Get the shape of the array
    shape = arr.shape
    
    # Initialize the output array for the result
    result_shape = shape[:-1]
    result = np.empty(result_shape, dtype=np.bool_)
    
    # Iterate over all axes except the last one
    for index in np.ndindex(result_shape):
        # Extract the sub-array for the current index
        sub_array = arr[index]
        
        # Check if any value in the last axis is True
        result[index] = np.any(sub_array)
    
    return result

# define function like numpy.insert for use with numba
@njit(cache = True, fastmath = True)
def numba_insert(array, index, value):
    new_array = np.empty(len(array) + 1, dtype=array.dtype)
    new_array[:index] = array[:index]
    new_array[index] = value
    new_array[index + 1:] = array[index:]

    return new_array

# define function like numpy.unique for use with numba
@njit(cache = True, fastmath = True)
def numba_unique(arr):
    # Sort the array
    sorted_arr = np.sort(arr)
    unique_values = []
    counts = []
    
    # Initialize counters
    prev_value = sorted_arr[0]
    count = 1
    
    # Iterate through sorted array
    for i in range(1, len(sorted_arr)):
        if sorted_arr[i] == prev_value:
            count += 1
        else:
            unique_values.append(prev_value)
            counts.append(count)
            prev_value = sorted_arr[i]
            count = 1
    
    # Append the last value and count
    unique_values.append(prev_value)
    counts.append(count)
    
    return np.array(unique_values), np.array(counts)

# define function like numpy.setdiff1d for numba
@njit(cache = True, fastmath = True)
def set_diff(arr1, arr2):
    # Create a boolean array to mark elements of arr1 that are not in arr2
    mask = np.ones(arr1.shape, dtype=np.bool_)

    for i in range(arr1.size):
        for j in range(arr2.size):
            if arr1[i] == arr2[j]:
                mask[i] = False
                break
    
    # Use the boolean mask to filter out the elements
    result = arr1[mask]
    return result

@njit(cache = True, fastmath = True)
def merge_similar_rows(array, threshold):
    """
    Merged the rows of a two dimensional array if their deviations are below threshold.
    - Function not used yet, but probably could be used to merge similar init parameters - 

    Parameters:
    - array: np.ndarray (n, m)
        The array containing the rows that should be merged
    - threshold: float (between 0 and 1)
        Threshold value defining the relative deviation at which rows should be merged.

    Returns:
    - merged_array: np.ndarray (q, m)
        The array with the merged rows.
    """
    n_rows, n_cols = array.shape
    merged = np.zeros(n_rows, dtype=np.bool_)
    merged_array = np.zeros((n_rows, n_cols), dtype=np.float64)
    count = 0

    for i in range(n_rows):
        if merged[i]:
            continue
        
        # Initialize the row to be merged
        current_group = [array[i]]
        
        for j in range(i + 1, n_rows):
            if merged[j]:
                continue
            
            # Check if all deviations are below threshold
            is_below_threshold = True
            for k in range(n_cols):
                deviation = abs(array[i, k] - array[j, k]) / array[i, k]
                if deviation >= threshold:
                    is_below_threshold = False
                    break
            
            if is_below_threshold:
                current_group.append(array[j])
                merged[j] = True
        
        # Compute the mean of the group
        mean_row = np.zeros(n_cols)
        for row in current_group:
            for col in range(n_cols):
                mean_row[col] += row[col]
        mean_row /= len(current_group)
        
        merged_array[count] = mean_row
        count += 1
    
    return merged_array[:count]


@njit(cache = True, fastmath = True)
def cartesian_product(arrays):
    # Determine the total number of combinations (like itertools.product but with numba)
    n = 1
    for arr in arrays:
        n *= len(arr)
    result = np.empty((n, len(arrays)), dtype = arrays.dtype)
    
    # Generate the Cartesian product
    repeats = 1
    for i, arr in enumerate(arrays):
        n_elements = len(arr)
        block_size = n // (n_elements * repeats)
        
        for j in range(repeats):
            for k in range(n_elements):
                start = j * block_size * n_elements + k * block_size
                result[start:start + block_size, i] = arr[k]
        repeats *= n_elements
    return result

@njit(cache = True, fastmath = True)
def calculate_chi2(model_y, ydata, ydata_err, num_params = 0):
    """
    Calculates the (reduces) chi2 of given data.

    Parameters:
    - model_y: np.ndarray (n, )
        The model values.
    - ydata: np.ndarray (n, )
        The y-data values.
    - ydata_err: np.ndarray (n, )
        The error (standard deviation) of the y-data values.
    - num_params: int
        The number of parameters for calculating the reduced chi2.
        If 0, normal chi2 will be calculated.

    Returns:
    - chi2: float
    """
    
    residuals = np.abs(model_y - ydata)
    n_res = residuals/ydata_err # normalized Residuals
    chi2 = np.sum(n_res**2) # Chi-squared
    if num_params > 0:
        dof = ydata.shape[-1] - num_params # Degrees of Freedom = amount of data - amount of parameters
        chi2 = chi2 / dof

    return chi2

@njit(cache = True, fastmath = True)
def mean_angle(angles):
    """
    Calculates the cyclic mean of given angles.

    Parameters:
    - angles: np.ndarray (n, )
        The angle values in radians.

    Returns:
    - mean_angle: float
    """
    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)
    mean_sin = np.mean(sin_angles)
    mean_cos = np.mean(cos_angles)
    mean_angle = np.arctan2(mean_sin, mean_cos)
    mean_angle = mean_angle % (2 * np.pi)
    
    return mean_angle

def pick_data(filepath, dataset_path = "", area = None, randoms = 0, indices = None, 
                dtype = None):
    """
    Picks data from a HDF5 or nii file.

    Parameters:
    - filepath: string
        The path to the file.
    - dataset_path: string
        The name (path) to the dataset in the HDF5 file.
    - area: list
        Should have the layout: [x_left, x_right, y_top, y_bot]
        Where x_left and x_right are the x_borders and y_top and y_bot are the y_borders.
    - randoms: int
        The number of randoms to pick from the data. 0 equals picking the full data.
    - indices: np.ndarray (n, m, 2)
        Array storing the both indices in the last dimension, which are used to pick from the data.
    - dtype: The dtype the returned data should have. Default (None) is same as in file.

    Returns:
    - data: np.ndarray (n, m) or (n, m, p)
        Returns the chosen data.
    - indices: np.ndarray (n, m, 2)
        Stores the picked indices from the data. 
        For every data point (n, m) the indices array has two values,
        which are the picked indices from the data.
    """

    if filepath.endswith(".nii"):
        nii_file = nib.load(filepath)
        data_shape = nii_file.shape
        data_dtype = nii_file.get_data_dtype()
        data_proxy = nii_file.dataobj

        if data_shape[-1] == 1:
            data_shape = data_shape[:-1]
            data_proxy = data_proxy[..., 0]
    else:
        with h5py.File(filepath, "r") as h5f:
            data_shape = h5f[dataset_path].shape
            data_dtype = h5f[dataset_path].dtype

    if area == None:
        x_indices, y_indices = np.indices((data_shape[0], data_shape[1]), dtype = np.uint16)

    elif not isinstance(indices, np.ndarray):
        x_indices, y_indices = np.indices((area[1] - area[0], area[3] - area[2]), dtype = np.uint16)
        x_indices += area[0]
        y_indices += area[2]

    if randoms > 0 and not isinstance(indices, np.ndarray):
        random_x_indices = np.random.choice(x_indices[:, 0], randoms)
        random_y_indices = np.random.choice(y_indices[0, :], randoms)

        data = np.empty((randoms, 1) + (data_shape[2:]), dtype = data_dtype)
        indices = np.empty((randoms, 1, 2), dtype = np.uint16)
        for i in range(randoms):
            if filepath.endswith(".nii"):
                data[i, 0, ...] = data_proxy[random_x_indices[i], random_y_indices[i], ...]
            else:
                with h5py.File(filepath, "r") as h5f:
                    data[i, 0, ...] = h5f[dataset_path][random_x_indices[i], random_y_indices[i], ...]
            indices[i, 0, 0] = random_x_indices[i]
            indices[i, 0, 1] = random_y_indices[i]

    else:
        if isinstance(indices, np.ndarray):
            flat_indices = indices.reshape(-1, indices.shape[-1])
            num_pixels = len(flat_indices)
            data = np.empty((num_pixels, 1) + (data_shape[2:]), dtype = data_dtype)
            for i in range(num_pixels):
                if filepath.endswith(".nii"):
                    data[i, 0, ...] = data_proxy[flat_indices[i, 0], flat_indices[i, 1], ...]
                else:
                    with h5py.File(filepath, "r") as h5f:
                        data[i, 0, ...] = h5f[dataset_path][flat_indices[i, 0], flat_indices[i, 1], ...]
            return data, indices
        elif area == None:
            if filepath.endswith(".nii"):
                data = data_proxy[:]
            else:
                with h5py.File(filepath, "r") as h5f:
                    data = h5f[dataset_path][:]
        else:
            if filepath.endswith(".nii"):
                data = data_proxy[area[0]:area[1], area[2]:area[3], ...]
            else:
                with h5py.File(filepath, "r") as h5f:
                    data = h5f[dataset_path][area[0]:area[1], area[2]:area[3], ...]

        indices = np.stack((x_indices, y_indices), axis = -1, dtype = np.uint16)
    
    if not dtype is None:
        data = data.astype(dtype)

    return data, indices

def process_image_in_chunks(filepaths, func, square_size = None, dataset_paths = None, 
                            num_processes_main = 2, suppress_prints = True, *args, **kwargs):
    """
    Processes image data in square chunks and applies a given function `func` to each chunk.
    This is usefull e.g. for map creation of very large datasets that does not fit into memory.

    Parameters:
    - filepaths: list of strings
        List of file paths (HDF5 or NII).
    - func: function
        The function to apply to each square chunk of data.
        The function `func` has to be an image processing function in that way that it returns
        an numpy array which first two dimensions are the image dimensions.
        The first arguments of the function must be the datas of the filepaths
    - square_size: int
        The size of the square chunks (the length of one edge in pixels).
        If None, it defaults to 1/10th of the total image size.
    - dataset_paths: list of strings
        List of dataset paths within the corresponding HDF5 files.
    - num_processes_main: int
        The number of processes to use for chunk calculation. Every process will calculate one chunk.
        Should be lower than the number of square chunks created.
    - surpress_prints: bool
        If True, prints within the function `func` will be suppressed.
    - *args: tuple
        Additional positional arguments for the function `func`.
    - **kwargs: dict
        Additional keyword arguments for the function `func`.

    Returns:
    - full_result: np.ndarray
        A numpy array where the first two dimensions match the input data and are filled with the processed results.
    """

    if isinstance(dataset_paths, list):
        if len(filepaths) != len(dataset_paths):
            raise ValueError("filepaths and dataset_paths must have the same length.")
    else:
        dataset_paths = [None for i in range(len(filepaths))]

    # Get the shape of the dataset
    data_shape, data_dtype = get_data_shape_and_dtype(filepaths[0], dataset_paths[0])
    total_rows, total_cols = data_shape[0], data_shape[1]

    if square_size is None:
        square_size = min(total_rows, total_cols) // 10
    
    total_chunks = (((total_rows + square_size - 1) // square_size) 
                    * ((total_cols + square_size - 1) // square_size))

    data_arguments = []
    for idx in range(len(filepaths)):
        initial_chunk_data, _ = pick_data(filepaths[idx], dataset_paths[idx], 
                                    area = [0, min(square_size, total_rows), 0, 
                                            min(square_size, total_cols)])
        data_arguments.append(initial_chunk_data)

    if suppress_prints:
        # Suppress stdout and stderr for the execution of func
        with open(os.devnull, 'w') as devnull, \
                contextlib.redirect_stdout(devnull), \
                contextlib.redirect_stderr(devnull):
            initial_result = func(*tuple(data_arguments), *args, **kwargs)
    else:
        initial_result = func(*tuple(data_arguments), *args, **kwargs)
    
    multi_dim_result = False
    if isinstance(initial_result, list) or isinstance(initial_result, tuple):
        multi_dim_result = True
        initial_result = initial_result[0]
    
    # Determine the full result shape based on the initial function's output
    result_shape = (total_rows, total_cols) + initial_result.shape[2:]
    full_result = pymp.shared.array(result_shape, dtype=initial_result.dtype)

    full_result[0:initial_chunk_data.shape[0], 0:initial_chunk_data.shape[1], ...] = initial_result
    
    # Create a list of chunk coordinates to process
    chunk_coords = [(row_start, col_start)
                    for row_start in range(0, total_rows, square_size)
                    for col_start in range(0, total_cols, square_size)]

    # Initialize the progress bar
    pbar = tqdm(total = len(chunk_coords), 
                desc = f'Processing chunks',
                smoothing = 0)
    shared_counter = pymp.shared.array((num_processes_main, ), dtype = int)

    # Activate nested looping in case func also uses pymp
    pymp.config.nested = True

    with pymp.Parallel(num_processes_main) as p:
        # Process data in square chunks
        for i in p.range(len(chunk_coords)):
            row_start, col_start = chunk_coords[i]

            row_end = min(row_start + square_size, total_rows)
            col_end = min(col_start + square_size, total_cols)
            if row_start == 0 and col_start == 0:
                result_chunk = initial_result
            else: 
                full_result
                area = [row_start, row_end, col_start, col_end]

                data_arguments = []
                for idx in range(len(filepaths)):
                    initial_chunk_data, _ = pick_data(filepaths[idx], dataset_paths[idx], area = area)
                    data_arguments.append(initial_chunk_data)
                
                if suppress_prints:
                    # Suppress stdout and stderr for the execution of func
                    with open(os.devnull, 'w') as devnull, \
                            contextlib.redirect_stdout(devnull), \
                            contextlib.redirect_stderr(devnull):
                        result_chunk = func(*tuple(data_arguments), *args, **kwargs)
                else:
                    result_chunk = func(*tuple(data_arguments), *args, **kwargs)

                if multi_dim_result:
                    result_chunk = result_chunk[0]
            
            full_result[row_start:row_end, col_start:col_end, ...] = result_chunk

            # Update progress bar
            shared_counter[p.thread_num] += 1
            status = np.sum(shared_counter)
            pbar.update(status - pbar.n)
        
    # Set the progress bar to 100%
    pbar.update(pbar.total - pbar.n)

    return full_result

def get_data_shape_and_dtype(filepath, dataset_path = ""):
    if filepath.endswith(".nii") or filepath.endswith(".nii-gz"):
        nii_file = nib.load(filepath)
        data_shape = nii_file.shape
        data_dtype = nii_file.get_data_dtype()
    else:
        with h5py.File(filepath, "r") as h5f:
            data_shape = h5f[dataset_path].shape
            data_dtype = h5f[dataset_path].dtype

    return data_shape, data_dtype


# Following function is copied from SLIX software:
# https://github.com/3d-pli/SLIX
def imread(filepath, dataset="/Image"):
    # Load NIfTI dataset
    if filepath.endswith('.nii') or filepath.endswith('.nii.gz'):
        data = nib.load(filepath).get_fdata()
        data = np.squeeze(np.swapaxes(data, 0, 1))
    elif filepath.endswith('.tiff') or filepath.endswith('.tif'):
        data = tifffile.imread(filepath)
        #if len(data.shape) == 3:
        #    data = np.squeeze(np.moveaxis(data, 0, -1))
    elif filepath.endswith('.h5'):
        with h5py.File(filepath, "r") as h5f:
            data = h5f[dataset][:]
        #if len(data.shape) == 3:
        #    data = np.squeeze(np.moveaxis(data, 0, -1))
    
    return data

@njit(cache = True, fastmath = True)
def add_birefringence_orig(
    dir_1,
    ret_1,
    dir_2,
    ret_2,
    symmetric=False,
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
def add_birefringence(
    dirs,
    rets,
    symmetric=False,
):
    """Add multiple birefringencies"""
    dirs = np.asarray(dirs)
    rets = np.asarray(rets)

    total_real, total_im = 0.0, 0.0

    # Process each (dir, ret) pair
    for i in range(len(dirs)):
        delta_i = np.arcsin(rets[i])
        for j in range(i + 1, len(dirs)):
            delta_j = np.arcsin(rets[j])

            # Calculate main term contributions
            if symmetric:
                # Symmetric case interactions
                real_ij, im_ij = mod2cplx(dirs[i], np.sin(delta_i / 2) * np.cos(delta_j / 2))
                real_ji, im_ji = mod2cplx(dirs[j], np.sin(delta_j / 2) * np.cos(delta_i / 2))
            else:
                # Non-symmetric main terms
                real_ij, im_ij = mod2cplx(dirs[i], np.sin(delta_i) * np.cos(delta_j))
                real_ji, im_ji = mod2cplx(dirs[j], np.cos(delta_i / 2) ** 2 * np.sin(delta_j))

            # Sum main terms for this pair
            total_real += real_ij + real_ji
            total_im += im_ij + im_ji

            # Add cross term for the non-symmetric case
            if not symmetric:
                real_cross, im_cross = mod2cplx(
                    2 * dirs[i] - dirs[j],
                    -np.sin(delta_i / 2) ** 2 * np.sin(delta_j)
                )
                total_real += real_cross
                total_im += im_cross

    # Convert accumulated real and imaginary parts to new direction and retardation
    dir_new, ret_new = cplx2mod(total_real, total_im)

    if symmetric:
        ret_new = np.sin(np.arcsin(ret_new) * 2)

    # wrap around 180 degree
    dir_new = dir_new % np.pi

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

def find_closest_true_pixel(mask, start_pixel, radius, queue=None, visited=None):
    """
    Finds the closest true pixel for a given 2d-mask and a start_pixel within a given radius.
    
    Parameters:
    - mask: np.ndarray (n, m)
        The boolean mask defining which pixels are False or True.
    - start_pixel: tuple
        The x- and y-coordinates of the start_pixel.
    - radius: int
        The radius within which to search for the closest true pixel.
    - queue: deque (optional)
        The current state of the search queue for resuming.
    - visited: np.ndarray (optional)
        The visited array tracking the search state.
        
    Returns:
    - closest_true_pixel: tuple
        The x- and y-coordinates of the closest true pixel or (-1, -1) if no true pixel is found.
    - queue: deque
        The queue at the current state for resuming the search.
    - visited: np.ndarray
        The visited array at the current state for resuming the search.
    """
    rows, cols = mask.shape
    sr, sc = start_pixel

    # Define the cropping boundaries within the mask limits
    left = max(0, sr - radius)
    right = min(rows, sr + radius + 1)
    top = max(0, sc - radius)
    bottom = min(cols, sc + radius + 1)

    cropped_mask = mask[left:right, top:bottom]

    if not np.any(cropped_mask):
        return (-1, -1), None, None

    # Adjust the starting pixel's coordinates for the cropped mask
    cropped_start = (sr - left, sc - top)
    cropped_rows, cropped_cols = cropped_mask.shape

    # Initialize queue and visited array if not resuming
    if queue is None:
        queue = deque([cropped_start])
    if visited is None:
        visited = np.zeros_like(cropped_mask, dtype = np.bool_)
        visited[cropped_start] = True

    while queue:
        r, c = queue.popleft()

        # Check neighbors within the cropped region bounds
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < cropped_rows and 0 <= nc < cropped_cols and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))

        if cropped_mask[r, c]:
            # Return the original coordinates by offsetting with the crop boundaries
            return (r + left, c + top), queue, visited
    
    # If no true pixel is found within the radius, return (-1, -1)
    return (-1, -1), queue, visited


@njit(cache = True, fastmath = True)
def calculate_inclination(retardation, birefringence, thickness , wavelength,):
    inclination = np.arccos(np.sqrt(np.arcsin(retardation) * wavelength 
                                / (2 * np.pi * thickness * birefringence)))

    return inclination

@njit(cache = True, fastmath = True)
def calculate_retardation(inclination, birefringence, thickness, wavelength):
    retardation = np.abs(np.sin(2 * np.pi * thickness * birefringence * np.cos(inclination)**2 
                                    / wavelength))

    return retardation

@njit(cache = True, fastmath = True)
def calculate_birefringence(retardation, inclination, thickness, wavelength):
    birefringence = np.arcsin(retardation) * wavelength / (2 * np.pi * thickness * np.cos(inclination)**2)

    return birefringence