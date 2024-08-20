import numpy as np
import h5py
from numba import njit

@njit(cache = True, fastmath = True)
def angle_distance(angle1, angle2):
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
    distance = (diff + np.pi) % (2*np.pi) - np.pi
    
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
            sum_value += arr[full_index]
        
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

@njit(cache=True, fastmath = True)
def calculate_chi2(model_y, ydata, xdata, ydata_err, num_params = 0):
    """
    Calculates the (reduces) chi2 of given data.

    Parameters:
    - model_y: np.ndarray (n, )
        The model values.
    - ydata: np.ndarray (n, )
        The y-data values.
    - xdata: np.ndarray (n, )
        The x-data values (only the number of them is used).
    - ydata_err: np.ndarray (n, )
        The error (standard deviation) of the y-data values.
    - num_params: int
        The number of parameters for calculating the reduced chi2.

    Returns:
    - chi2: float
    """
    
    residuals = np.abs(model_y - ydata)
    n_res = residuals/ydata_err # normalized Residuals
    chi2 = np.sum(n_res**2) # Chi-squared
    if num_params > 0:
        dof = len(xdata) - num_params # Degrees of Freedom = amount of data - amount of parameters
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

def pick_data(filepath, dataset_path, area = None, randoms = 0, indices = None):
    """
    Picks data from a HDF5 file.

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

    Returns:
    - data: np.ndarray (n, m) or (n, m, p)
        Returns the chosen data.
    - indices: np.ndarray (n, m, 2)
        Stores the picked indices from the data. 
        For every data point (n, m) the indices array has two values,
        which are the picked indices from the data.
    """

    with h5py.File(filepath, "r") as h5f:

        data_shape = h5f[dataset_path].shape

        if area == None:
            x_indices, y_indices = np.indices((data_shape[0], data_shape[1]))

        elif not isinstance(indices, np.ndarray):
            x_indices, y_indices = np.indices((area[1] - area[0], area[3] - area[2]))
            x_indices += area[0]
            y_indices += area[2]

        if randoms > 0 and not isinstance(indices, np.ndarray):
            random_x_indices = np.random.choice(x_indices[:, 0], randoms)
            random_y_indices = np.random.choice(y_indices[0, :], randoms)

            data = np.empty((randoms, 1) + (data_shape[2:]), dtype = h5f[dataset_path].dtype)
            indices = np.empty((randoms, 1, 2), dtype = int)
            for i in range(randoms):
                data[i, 0, ...] = h5f[dataset_path][random_x_indices[i], random_y_indices[i], ...]
                indices[i, 0, 0] = random_x_indices[i]
                indices[i, 0, 1] = random_y_indices[i]

        else:
            if isinstance(indices, np.ndarray):
                flat_indices = indices.reshape(-1, indices.shape[-1])
                num_pixels = len(flat_indices)
                data = np.empty((num_pixels, 1) + (data_shape[2:]), dtype = h5f[dataset_path].dtype)
                for i in range(num_pixels):
                    data[i, 0, ...] = h5f[dataset_path][flat_indices[i, 0], flat_indices[i, 1], ...]
                return data, indices
            elif area == None:
                data = h5f[dataset_path][:]
            else:
                data = h5f[dataset_path][area[0]:area[1], area[2]:area[3], ...]

            indices = np.stack((x_indices, y_indices), axis = -1, dtype = np.int64)

    return data, indices