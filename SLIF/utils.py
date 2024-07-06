import numpy as np
import h5py
from numba import njit

@njit(cache = True, fastmath = True)
def angle_distance(angle1, angle2):
    # Calculate the difference between the angles
    diff = angle2 - angle1
    
    # Calculate the shortest distance considering the cyclic nature
    distance = (diff + np.pi) % (2*np.pi) - np.pi
    
    return distance

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

@njit(cache=True, fastmath = True)
def calculate_chi2(model_y, ydata, xdata, ydata_err, num_params):
    
    residuals = np.abs(model_y - ydata)
    n_res = residuals/ydata_err # normalized Residuals
    chi2 = np.sum(n_res**2) # Chi-squared
    if num_params > 0:
        dof = len(xdata) - num_params # Degrees of Freedom = amount of data - amount of parameters
        chi2 = chi2 / dof

    return chi2

@njit(cache = True, fastmath = True)
def mean_angle(angles):
    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)
    mean_sin = np.mean(sin_angles)
    mean_cos = np.mean(cos_angles)
    mean_angle = np.arctan2(mean_sin, mean_cos)
    mean_angle = mean_angle % (2 * np.pi)
    
    return mean_angle

def pick_data(filename, dataset_path, area = None, randoms = 0):

    # area = [x_left, x_right, y_top, y_bot]
    # where x_left and x_right are the x-borders
    # and y_top and y_bot are the y_borders

    # randoms = number of random pixels to pick from area

    with h5py.File(filename, "r") as h5f:

        data_shape = h5f[dataset_path].shape

        if area == None:
            x_indices, y_indices = np.indices((data_shape[0], data_shape[1]))

        else:
            x_indices, y_indices = np.indices((area[1] - area[0], area[3] - area[2]))
            x_indices += area[0]
            y_indices += area[2]

        if randoms > 0:
            x_indices = x_indices.flatten()
            y_indices = y_indices.flatten()
            random_indices = np.random.choice(len(x_indices), randoms, replace=False)
            x_indices = x_indices[random_indices]
            y_indices = y_indices[random_indices]

            data = np.empty((randoms, 1, data_shape[2]), dtype=h5f[dataset_path].dtype)
            indices = np.empty((randoms, 1, 2), dtype=int)
            for i in range(randoms):
                data[i, 0] = h5f[dataset_path][x_indices[i], y_indices[i], :]
                indices[i, 0, 0] = x_indices[i]
                indices[i, 0, 1] = y_indices[i]

        else:
            if area == None:
                data = h5f[dataset_path][:]
            else:
                data = h5f[dataset_path][area[0]:area[1], area[2]:area[3], :]

            indices = np.stack((x_indices, y_indices), axis=-1, dtype=np.int64)

    return data, indices


    if fit_full_pyramid:
        reduced_data = h5f[dataset_path][:]
    else:
        reduced_data = np.empty((image_size, image_size, data_shape[2]), dtype=h5f[dataset_path].dtype)
        
        sample_size = image_size**2
        row_indices = np.random.randint(0, data_shape[0], sample_size)
        col_indices = np.random.randint(0, data_shape[1], sample_size)
        
        indices = np.empty((image_size, image_size, 2), dtype=int)
        
        for k in range(sample_size):
            i = k // image_size
            j = k % image_size
            reduced_data[i, j] = h5f[dataset_path][row_indices[k], col_indices[k]]
            indices[i, j, 0] = row_indices[k]
            indices[i, j, 1] = col_indices[k]

