import numpy as np
from numba import njit
from .utils import angle_distance, numba_insert, numba_unique, mean_angle, set_diff
from collections import namedtuple

PeakFinderParams = namedtuple('PeakFinderParams', [
    'local_maxima', 'local_minima', 'turning_points',
    'turning_points_directions', 'local_maxima_angles', 'local_minima_angles', 'is_merged', 
    'is_hidden_peak', 'max_A', 'max_peak_hwhm', 'min_peak_hwhm', 'mu_range', 'min_int', 
    'global_amplitude', 'extrema_tolerance', 
    'turning_point_tolerance', 'turning_point_tolerance_2', 'turning_point_tolerance_3',
    'only_peaks_count', 'max_peaks'
])

@njit(cache = True, fastmath = True)
def _find_extremas(intensities, first_diff, second_diff, max_A, extrema_tolerance,
                    turning_point_tolerance, turning_point_tolerance_2, turning_point_tolerance_3,
                    reverse = False):
    """
    Finds extremas from array of intensities

    Parameters:
    - intensities: np.ndarray (n, )
        Array of intensities.
    - first_diff: np.ndarray (n, )
        Differences between neighbouring intensities with wrapping at last value.
    - second_diff: np.ndarray (n, )
        Differences between neighbouring first_diff with wrapping at last value.
    - max_A: float
        Estimated maximum (border) of the underground of the intensity profile.
    - extrema_tolerance: float
        Tolerance to detect minima and maxima
    - turning_point_tolerance: float
        Tolerance to detect turning points

    Returns:
    - local_maxima: np.ndarray (m, )
        Array that stores the indices of the measurements that are local maxima.
    - local_minima: np.ndarray (p, )
        Array that stores the indices of the measurements that are local minima.
    - turning_points: np.ndarray (q, )
        Array that stores the indices of the measurements that are turning_points.
    - turning_points_directions: np.ndarray (n, )
        Array that stores the information if a measurement is a turning point and
        the direction of the turning point. 
        -1 for left, +1 for right and 0 for no turning point.
    """

    num_points = len(intensities)

    local_maxima = np.zeros(num_points, dtype = np.bool_)
    local_minima = np.zeros(num_points, dtype = np.bool_)
    turning_points = np.zeros(num_points, dtype = np.bool_)
    turning_points_directions = np.zeros(num_points, dtype = np.int64)

    # Iterate in reverse order
    indices = np.arange(num_points)
    if reverse:
        indices = np.flip(indices)

    for i in indices:

        if intensities[i] <= max_A:
            # Handle underground values as local minima
            local_minima[i] = True
        # Check for maxima
        elif first_diff[i - 1] >= 0 and first_diff[i] <= 0 and second_diff[i - 1] <= 0:

            left_condition = False
            right_condition = False
            for k in [-1, 1]:
                for j in range(1, 5):
                    index = (i + k * j) % num_points
                    if intensities[i] - intensities[index] >= extrema_tolerance:
                        if k == -1: left_condition = True
                        elif k == 1: right_condition = True
                        break
                    # For indices that are exremas break
                    if local_minima[index] or local_maxima[index] or turning_points[index]:
                        break
            if left_condition and right_condition:
                local_maxima[i] = True

        # Check for minima
        elif first_diff[i - 1] <= 0 and first_diff[i] >= 0 and second_diff[i - 1] >= 0:

            left_condition = False
            right_condition = False
            for k in [-1, 1]:
                for j in range(1, 5):
                    index = (i + k * j) % num_points
                    if intensities[i] - intensities[index] <= -extrema_tolerance:
                        if k == -1: left_condition = True
                        elif k == 1: right_condition = True
                        break
                    if local_minima[index] or local_maxima[index] or turning_points[index]:
                        break
            if left_condition and right_condition:
                local_minima[i] = True

    for i in indices:
        # Check for (right/left) turning points
        if (second_diff[i-2] <= turning_point_tolerance_3 \
                and second_diff[i-1] >= turning_point_tolerance \
                and np.abs(first_diff[i-1]) < turning_point_tolerance_2 \
                and first_diff[i] > extrema_tolerance \
                and not local_maxima[i] and not local_minima[i]):

            # if neighbours are extremas do not consider it as turning point
            extrema_found = False
            next_index = (i + 1) % num_points
            if local_maxima[i-1] or local_minima[i-1] or turning_points[i-1] \
                or local_maxima[next_index] or local_minima[next_index] or turning_points[next_index]:
                extrema_found = True
            
            if not extrema_found:
                turning_points[i] = True
                # -1 for tp on left side
                turning_points_directions[i] = -1

        # Check for (left/right) turning_point
        elif (second_diff[i-1] >= turning_point_tolerance \
                and second_diff[i] <= turning_point_tolerance_3 \
                and first_diff[i-1] < -extrema_tolerance \
                and np.abs(first_diff[i]) < turning_point_tolerance_2 \
                and not local_maxima[i] and not local_minima[i]):
            
            extrema_found = False
            next_index = (i + 1) % num_points
            if local_maxima[i-1] or local_minima[i-1] or turning_points[i-1] \
                or local_maxima[next_index] or local_minima[next_index] or turning_points[next_index]:
                extrema_found = True
                
            if not extrema_found:
                turning_points[i] = True
                # +1 for tp on right side
                turning_points_directions[i] = 1


    local_maxima = local_maxima.nonzero()[0]
    local_minima = local_minima.nonzero()[0]

    turning_points = turning_points.nonzero()[0]
    #turning_points_directions = turning_points_directions[turning_points_directions != 0]

    return local_maxima, local_minima, turning_points, turning_points_directions

@njit(cache = True, fastmath = True)
def _merge_maxima(local_maxima, local_minima, intensities, angles, local_maxima_angles, 
                    max_peak_hwhm, global_amplitude):
    """
    Merges local maxima if no local minima is between them

    Parameters:
    - local_maxima: np.ndarray (m, )
        Array that stores the indices of the measurements that are local maxima.
    - local_minima: np.ndarray (p, )
        Array that stores the indices of the measurements that are local minima.
    - intensities: np.ndarray (n, )
        The measured intensities of the pixel
    - angles: np.ndarray (n, )
        The angles at which the intensities are measured.
    - local_maxima_angles: np.ndarray (m, )
        Angles of the local maxima.
    - max_peak_hwhm: float
        Estimated maximum peak half width at half maximum.
    - global_amplitude: float
        The difference between the maximum and minimum intensity of the pixel.

    Returns:
    - local_maxima: np.ndarray (m, )
        Array that stores the indices of the measurements that are local maxima.
    - local_minima: np.ndarray (m, )
        Array that stores the indices of the measurements that are local minima.
    - local_maxima_angles: np.ndarray (m, )
        Angles of the local maxima.
    """

    # If two or more local maxima are between two local minima
    # merge them to the mean of them
    if len(local_maxima) > 0:
        insert_indices = np.searchsorted(local_minima, local_maxima)
        # insertion index after end equals insertion at start (because of circular data)
        insert_indices = np.where(insert_indices == len(local_minima), 0, insert_indices)
        # find the sorted insertion indices that are equal (duplicates)
        unique_insert_indices, counts = numba_unique(insert_indices)

        duplicate_insert_indices = unique_insert_indices[counts > 1]
        mask_maxima = np.ones(len(local_maxima), dtype = np.bool_)
        mean_maxima = np.empty(0, dtype = np.int64)
        mean_maxima_angles = np.empty(0)
        for i, duplicate_insert_index in enumerate(duplicate_insert_indices):
            # Get the maxima indices from the insertion indices that equals a duplicate insert index 
            duplicate_maxima_indices = np.asarray(insert_indices == duplicate_insert_index).nonzero()[0]
            duplicate_maxima = local_maxima[duplicate_maxima_indices]
            duplicate_maxima_angles = local_maxima_angles[duplicate_maxima_indices]

            # If max difference between intensity values of the maxima is high or
            # of max distance between duplicates is more than max peak hwhm, dont merge them
            # but append a local minimum at the lowest intensity between them
            max_distance = 0
            for j in range(len(duplicate_maxima_angles)):
                for k in range(i + 1, len(duplicate_maxima_angles)):
                    dist = np.abs(angle_distance(duplicate_maxima_angles[i], duplicate_maxima_angles[j]))
                    if dist > max_distance:
                        max_distance = dist
                        first_maximum = duplicate_maxima_angles[i]
                        second_maximum = duplicate_maxima_angles[j]

            max_difference = np.max(intensities[duplicate_maxima]) - np.min(intensities[duplicate_maxima])
            if max_distance > max_peak_hwhm or max_difference > 0.1 * global_amplitude:
                if first_maximum < second_maximum:
                    condition_between = (angles > first_maximum) & (angles < second_maximum)
                else:
                    condition_between = (angles > first_maximum) | (angles < second_maximum)
                
                between_indices = condition_between.nonzero()[0]
                minimum_between = between_indices[np.argmin(intensities[condition_between])]
                local_minima = np.append(local_minima, minimum_between)
                continue

            #duplicate_maxima += len(angles)
            #mean_maximum = np.int64(np.rint(np.mean(duplicate_maxima)) - len(angles))
            mean_maximum_angle = mean_angle(duplicate_maxima_angles)
            mean_maxima_angles = np.append(mean_maxima_angles, mean_maximum_angle)
            mean_distances = np.abs(angle_distance(angles, mean_maximum_angle))
            mean_maxima = np.append(mean_maxima, np.argmin(mean_distances))
            mask_maxima[duplicate_maxima_indices] = 0

        # remove duplicate maxima and append mean (sorting later)
        local_maxima = local_maxima[mask_maxima]
        local_maxima_angles = local_maxima_angles[mask_maxima]
        local_maxima = np.concatenate((local_maxima, mean_maxima))
        local_maxima_angles = np.concatenate((local_maxima_angles, mean_maxima_angles))

    return local_maxima, local_maxima_angles, local_minima

@njit(cache = True, fastmath = True)
def _append_similar_minima(local_minima, local_maxima, turning_points, intensities, 
                            intensities_err, global_amplitude):
    """
    Append neighbouring intensities to local minima, if they have a similar intensity

    Parameters:
    - local_minima: np.ndarray (m, )
        Array that stores the indices of the measurements that are local minima.
    - local_maxima: np.ndarray (p, )
        Array that stores the indices of the measurements that are local maxima.
    - turning_points: np.ndarray (q, )
        Array that stores the indices of the measurements that are turning points.
    - intensities: np.ndarray (n, )
        The measured intensities of the pixel.
    - intensities_err: np.ndarray (n, )
        The error (standard deviation) of the corresponding measured intensities of the pixel.
    - global_amplitude: float
        The difference between the maximum and minimum intensity of the pixel.

    Returns:
    - local_minima: np.ndarray (m, )
        Array that stores the indices of the measurements that are local minima.
    """
    for index_minimum in local_minima:
        # Get closest maxima regarding height
        # diff to closest maximum has to be high enough
        closest_maximum_index = np.argmin(np.abs(intensities[index_minimum] - intensities[local_maxima]))
        closest_maximum = local_maxima[closest_maximum_index]
        for k in [-1, 1]:
            half_len_angles = np.int64(np.floor(len(intensities) / 2))
            for i in range(1, half_len_angles):
                neighbour_index = (index_minimum + k * i) % len(intensities)
                if neighbour_index in local_maxima or neighbour_index in local_minima \
                        or neighbour_index in turning_points:
                    break

                diff = np.abs(intensities[neighbour_index] - intensities[index_minimum])
                diff_to_max = np.abs(intensities[closest_maximum] - intensities[index_minimum])
                if (diff < 0.02 * global_amplitude or diff < intensities_err[index_minimum] \
                        or diff < intensities_err[neighbour_index]) \
                        and diff_to_max > max(0.08 * global_amplitude, 1.5 * intensities_err[closest_maximum]):
                    local_minima = np.append(local_minima, neighbour_index)
                else:
                    break

    return local_minima

@njit(cache = True, fastmath = True)
def _handle_turning_points(turning_points, turning_points_directions,
                            local_maxima, local_minima, local_maxima_angles, angles, 
                            intensities, first_diff, min_peak_hwhm, global_amplitude, mu_range):
    """
    Check if the turning points are hidden peaks.

    Parameters:
    - turning_points: np.ndarray (n, )
    - ...

    Returns:
    - turning_points: np.ndarray (n, )
    ...
    """

    is_merged = np.zeros(len(local_maxima), dtype = np.bool_)

    # Append turning point as minimum and set maximum accordingly
    real_tp = np.ones(len(turning_points), dtype = np.bool_)
    for tp_index, tp in enumerate(turning_points):
        if tp in local_maxima: continue
        # Append local maximum in mu range away from merged turning point
        # and set merged turning point as local minimum

        # Check if next 3 values after turning point have a similar height
        # if not, it is not a real turning point (not a merged peak)
        # but only if the first diff before the turning point is enough different
        # from the first diff after the turning point (check for max 3 points)
        mean_left = 0
        for i in range(1, 4):
            index = (tp - i) % len(angles)
            if index in local_minima or index in local_maxima or index in turning_points:
                if i > 1: mean_left = mean_left / (i-1)
                break
            mean_left += first_diff[index]
            if i == 4:
                mean_left = mean_left / i

        mean_right = 0
        for i in range(3):
            index = (tp + i) % len(angles)
            if i != 0 and (index in local_minima or index in local_maxima or index in turning_points):
                mean_right = mean_right / i
                break
            mean_right += first_diff[index]
            if i == 3:
                mean_right = mean_right / (i + 1)

        tp_direction = turning_points_directions[tp_index]

        if len(local_minima) > 0 and len(local_maxima) > 0:
            # Get the closest minimum in the direction of the turning point
            distances_min = angle_distance(angles[tp], angles[local_minima])
            distances_max = angle_distance(angles[tp], angles[local_maxima])
            if tp_direction < 0:
                directed_distances_min = distances_min[distances_min < 0]
                directed_distances_max = distances_max[distances_max < 0]
            else:
                directed_distances_min = distances_min[distances_min > 0]
                directed_distances_max = distances_max[distances_max > 0]
            if len(directed_distances_min) != 0:
                closest_directed_distance_min = directed_distances_min[np.argmin(np.abs(directed_distances_min))]    
                closest_minimum_index = (distances_min == closest_directed_distance_min).nonzero()[0][0]
                closest_minimum = local_minima[closest_minimum_index]

                # Only consider turning points where the closest minimum is more than 5 * min_peak_hwhm away
                if np.abs(closest_directed_distance_min) < 5 * min_peak_hwhm:
                    real_tp[tp_index] = False
                    continue

                # Only consider turning points where the height to the closest minimum is high enough
                if (intensities[tp] - intensities[closest_minimum]) < 0.15 * global_amplitude:
                    real_tp[tp_index] = False
                    continue
            
            if len(directed_distances_max) != 0:
                closest_directed_distance_max = directed_distances_max[np.argmin(np.abs(directed_distances_max))]
                closest_maximum_index = (distances_max == closest_directed_distance_max).nonzero()[0][0]
                closest_maximum = local_maxima[closest_maximum_index]

                if np.abs(closest_directed_distance_max) < 4 * min_peak_hwhm:
                    real_tp[tp_index] = False
                    continue

        # Check if left side and right side of tp are similar in terms of difference
        # and if intensities after the tp differ much in intensity (more than 0.1 of global amplitude)
        # if they do, its not a real turning point (only noise)
        tp_maximum_index = tp
        for i in range(1, 3):
            index = (tp + tp_direction * i) % len(angles)
            if index in local_minima or index in local_maxima or index in turning_points:
                break
            if tp_maximum_index == tp or intensities[index] > intensities[tp_maximum_index]:
                tp_maximum_index = index
            if np.abs(mean_left - mean_right) < 0.1 * global_amplitude \
                    and intensities[tp] - intensities[index] > 0.15 * global_amplitude:
                real_tp[tp_index] = False

        if not real_tp[tp_index]:
            continue 

        tp_maximum = angles[tp_maximum_index]
        # if maximum index equals tp
        # set maximum in half mu range away from the turning point
        #if distances_max[closest_maximum_index] > 0:
        # in the direction of lowest diff

        if tp_maximum_index == tp:
            tp_maximum = (angles[tp] + tp_direction * mu_range / 2) % (2*np.pi)

            # Append the angle index which is the closest to the merged point maximum angle
            tp_maximum_index = np.searchsorted(angles, tp_maximum)
            if tp_maximum_index == len(angles): 
                tp_maximum_index = 0
            if np.abs(angle_distance(tp_maximum, angles[tp_maximum_index])) \
                > np.abs(angle_distance(tp_maximum, angles[tp_maximum_index - 1])):
                tp_maximum_index = (tp_maximum_index - 1) % len(angles)

        if not tp_maximum_index in local_maxima:
            local_maxima = np.append(local_maxima, tp_maximum_index)
            local_maxima_angles = np.append(local_maxima_angles, tp_maximum)
            local_minima = np.append(local_minima, tp)
            is_merged = np.append(is_merged, True)

    # Delete all false turning points
    turning_points = turning_points[real_tp]

    sort_indices = np.argsort(local_minima)
    local_minima = local_minima[sort_indices]

    return local_minima, local_maxima, local_maxima_angles, is_merged, turning_points

@njit(cache = True, fastmath = True)
def _find_hidden_peaks(local_maxima, local_minima, local_minima_angles, local_maxima_angles,
                        angles, intensities, first_diff, is_merged, 
                        max_peak_hwhm, max_A, global_amplitude):
    # If the distance between two extrema is more than 2 * max_peak_hwhm
    # and no local maximum is in between them and the mean intensity value between them
    # is larger than max_A + 0.15 * global_amplitude
    # append a "hidden peak" in the middle of this space

    is_hidden_peak = np.zeros(len(local_maxima), dtype = np.bool_)

    if len(local_maxima) > 0:

        extremas = np.concatenate((local_minima, local_maxima))
        extremas = np.sort(extremas)
        extremas_angles = angles[extremas]

        hidden_minima = np.empty(0, dtype = np.int64)
        hidden_maxima = np.empty(0, dtype = np.int64)
        hidden_maxima_angles = np.empty(0)
        for i in range(len(extremas)):
            next_index = (i + 1) % len(extremas)
            left_angle = extremas_angles[i]
            right_angle = extremas_angles[next_index]

            if left_angle < right_angle:
                condition = (angles >= left_angle) & (angles <= right_angle)
            else:
                condition = (angles >= left_angle) | (angles <= right_angle)
            hidden_intensities = intensities[condition]
            distance = angle_distance(left_angle, right_angle)
            # if peak is so wide that closest distance from left to right is negative
            # correct the distance
            if distance < 0:
                    distance = 2 * np.pi - np.abs(distance)

            if np.abs(distance) < 2 * max_peak_hwhm \
                    or np.mean(hidden_intensities) < max_A + 0.15 * global_amplitude \
                    or np.min(hidden_intensities) < max_A + 0.05 * global_amplitude:
                continue

            # If one of the pair is a maximum, add distance to closest minimum
            one_is_maximum = False
            if extremas[i] in local_maxima and extremas[next_index] in local_minima:
                one_is_maximum = True

                # Add a local minimum at index where first diff has minimum
                min_diff = np.inf
                for k in range(len(hidden_intensities)):
                    if k >= len(hidden_intensities) / 2:
                        break
                    index = (extremas[i] + k) % len(angles)
                    if np.abs(first_diff[index]) < min_diff:
                        min_diff = np.abs(first_diff[index])
                        new_minimum = index
                
            elif extremas[next_index] in local_maxima and extremas[i] in local_minima:
                one_is_maximum = True

                min_diff = np.inf
                for k in range(len(hidden_intensities)):
                    if k >= len(hidden_intensities) / 2:
                        break
                    index = (extremas[next_index] - k) % len(angles)
                    if np.abs(first_diff[index]) < min_diff:
                        min_diff = np.abs(first_diff[index])
                        new_minimum = index
                        
            if one_is_maximum:
                left_angle = angles[new_minimum]

            if left_angle < right_angle:
                condition = (angles >= left_angle) & (angles <= right_angle)
            else:
                condition = (angles >= left_angle) | (angles <= right_angle)
            hidden_intensities = intensities[condition]
            distance = angle_distance(left_angle, right_angle)
            # if peak is so wide that closest distance from left to right is negative
            # correct the distance
            if distance < 0:
                    distance = 2 * np.pi - np.abs(distance)

            if (np.abs(distance) < 2 * max_peak_hwhm and not one_is_maximum)\
                    or (one_is_maximum and np.abs(distance) < max_peak_hwhm)\
                    or np.mean(hidden_intensities) < max_A + 0.15 * global_amplitude \
                    or np.min(hidden_intensities) < max_A + 0.05 * global_amplitude:
                continue
            else:

                peak_angle = (left_angle + (distance / 2)) % (2 * np.pi)

                index = np.searchsorted(angles, peak_angle)
                if index == len(angles): 
                    index = 0
                if np.abs(angle_distance(peak_angle, angles[index])) \
                    > np.abs(angle_distance(peak_angle, angles[index - 1])):
                    index = (index - 1) % len(angles)

                if not index in local_maxima:
                    if one_is_maximum:
                        if not new_minimum in local_minima:
                            hidden_minima = np.append(hidden_minima, new_minimum)
                    hidden_maxima = np.append(hidden_maxima, index)
                    hidden_maxima_angles = np.append(hidden_maxima_angles, peak_angle)
                    is_merged = np.append(is_merged, True)
                    is_hidden_peak = np.append(is_hidden_peak, True)

    local_minima = np.concatenate((local_minima, hidden_minima))
    local_minima_angles = np.concatenate((local_minima_angles, angles[hidden_minima]))
    local_maxima = np.concatenate((local_maxima, hidden_maxima))
    local_maxima_angles = np.concatenate((local_maxima_angles, hidden_maxima_angles))

    return local_maxima, local_maxima_angles, local_minima, local_minima_angles, is_merged, is_hidden_peak

@njit(cache = True, fastmath = True)
def _find_best_center(angles, current_angles, intensities, current_intensities,
                        peak_angles, peak_intensities, angles_left, index_maximum,
                        angles_right, intensities_left, intensities_right, angle_spacing, mu_maximum, 
                        max_peak_hwhm, local_minima, local_minima_angles, closest_left_border,
                        closest_right_border):
    shortest_len = min(len(intensities_left), len(intensities_right))
    if shortest_len >= 2:
        # Find left/right values with lowest diff (most symmetric) and caculate best center
        # check diff also for neighbours (k)
        right_index = (index_maximum + 1) % len(angles)
        lowest_diff = np.abs(intensities[index_maximum] - intensities[right_index])
        best_left, best_right = angles[index_maximum], angles[right_index]
        for i in range(1, shortest_len + 1):
            for k in range(4):
                if (i + k - 1) > shortest_len: break
                left_index = (index_maximum - i) % len(angles)
                right_index = (index_maximum + i + k - 1) % len(angles)
                if not (angles[right_index] in peak_angles): break
                #if left_index == right_index: continue
                diff_mirrow = np.abs(intensities[left_index] - intensities[right_index])
                if diff_mirrow < lowest_diff:
                    lowest_diff = diff_mirrow
                    best_left, best_right = angles[left_index], angles[right_index]

        # if peak is over 180 degrees wide and best angles are on bottom
        # angle_distance is negative, and the right distance the rest of the angle
        best_distance = angle_distance(best_left, best_right)
        if best_distance < 0:
            best_distance = 2 * np.pi - np.abs(best_distance)

        best_center = (best_left + best_distance / 2) % (2 * np.pi)

        if np.abs(angle_distance(best_center, mu_maximum)) > 0.25 * angle_spacing:
            # Rearrange peak and correct peak values

            mu_maximum = best_center

            left_border = (mu_maximum - 1.5 * max_peak_hwhm) % (2 * np.pi)
            right_border = (mu_maximum + 1.5 * max_peak_hwhm) % (2 * np.pi)

            left_border, right_border, index_left_min, index_right_min, \
                left_minima_distances, right_minima_distances = _adjust_borders(left_border, right_border, 
                                mu_maximum, local_minima_angles, angles, local_minima)

            if left_border < right_border:
                condition = (current_angles < left_border) | (current_angles > right_border)
            else:
                condition = (current_angles < left_border) & (current_angles > right_border)

            peak_angles = current_angles[~ condition]
            peak_intensities = current_intensities[~ condition]

            if len(local_minima) > 0:
                left_min = angles[index_left_min]
                right_min = angles[index_right_min]
                if len(left_minima_distances) > 0 and left_min not in peak_angles:
                    if np.abs(angle_distance(left_min, left_border)) <= angle_spacing:
                        insert_index = np.searchsorted(peak_angles, left_min)
                        peak_angles = numba_insert(peak_angles, insert_index, left_min)
                        peak_intensities = numba_insert(peak_intensities, 
                                        insert_index, intensities[index_left_min])
                if len(right_minima_distances) > 0 and right_min not in peak_angles:
                    if np.abs(angle_distance(right_min, right_border)) <= angle_spacing:
                        insert_index = np.searchsorted(peak_angles, right_min)
                        peak_angles = numba_insert(peak_angles, insert_index, right_min)
                        peak_intensities = numba_insert(peak_intensities, insert_index, 
                                        intensities[index_right_min])

            closest_left_border = left_border
            closest_right_border = right_border
            relative_angles = angle_distance(mu_maximum, peak_angles)
            angles_left = peak_angles[relative_angles < 0]
            angles_right = peak_angles[relative_angles > 0]
            intensities_left = peak_intensities[relative_angles < 0]
            intensities_right = peak_intensities[relative_angles > 0]

    return mu_maximum, peak_angles, peak_intensities, angles_left, \
            angles_right, intensities_left, intensities_right, closest_left_border, closest_right_border

@njit(cache = True, fastmath = True)
def _is_real_peak(angles, intensities, peak_angles, peak_intensities, intensities_left,
                intensities_right, angles_left, angles_right, global_amplitude, 
                    local_max_int, intensities_err, extrema_tolerance, index_maximum,
                    local_maxima, local_minima, turning_points, is_merged, full_peak):

    #Check if a peak is real or not.
    
    is_peak = True
    if len(intensities_left) == 0 or len(intensities_right) == 0 \
                or (len(intensities_left) == 1 and len(intensities_right) == 1):
        is_peak = False
    # if one side of (not merged) peak is small and prominence is also small, its a fake peak
    elif not is_merged:
        # If all peak intensities except maximum have a similar intensitiey
        # its not a real peak
        peak_lr_intensities = np.concatenate((intensities_left, intensities_right))
        mean_peak_intensity = np.mean(peak_lr_intensities)
        peak_mean_diffs = np.abs(mean_peak_intensity - peak_lr_intensities)
        if np.all(peak_mean_diffs < max(0.06 * global_amplitude, 1.5 * intensities_err[index_maximum])):
            is_peak = False
        elif len(intensities_left) < 2 or len(intensities_right) < 2:
            left_height = local_max_int - np.min(intensities_left)
            right_height = local_max_int - np.min(intensities_right)
            prominence = min(left_height, right_height)
            if prominence < 0.05 * global_amplitude \
                or prominence < 1.5 * intensities_err[index_maximum]:
                is_peak = False
            # Check if the intensities after the side with lower height (and lower length)
            # are going already up again after the peak
            # and if thats the case the peak is considered noise and not a real peak 
            
            elif prominence < extrema_tolerance:
                if right_height < left_height and left_height > extrema_tolerance \
                    and len(angles_right) < len(angles_left):
                    mean_diff = 0
                    for i in range(1, 4):
                        last_index = index_maximum + len(angles_left)
                        next_index = index_maximum + len(angles_left) + i
                        if angles[index_maximum] in angles_right:
                            last_index -= 1
                            next_index -= 1
                        last_index = last_index % len(angles)
                        next_index = next_index % len(angles)
                        mean_diff += intensities[next_index] - intensities[last_index]

                        if i == 3 or next_index in local_maxima or next_index in local_minima:
                            mean_diff = mean_diff / i
                            if mean_diff > extrema_tolerance:
                                is_peak = False
                            break
                elif left_height < right_height and right_height > extrema_tolerance \
                    and len(angles_left) < len(angles_right):
                    mean_diff = 0
                    for i in range(1, 4):
                        last_index = index_maximum - len(angles_left)
                        next_index = index_maximum - len(angles_left) - i
                        if angles[index_maximum] in angles_left:
                            last_index += 1
                            next_index += 1
                        last_index = last_index % len(angles)
                        next_index = next_index % len(angles)
                        mean_diff += intensities[next_index] - intensities[last_index]
                        if i == 3 or next_index in local_maxima or next_index in local_minima:
                            mean_diff = mean_diff / i
                            if mean_diff > extrema_tolerance:
                                is_peak = False
                            break
        elif full_peak:
            # If symmetry of peak is not good (and it does not contain fake peaks or tp)
            # mark it as false peak
            fake_in_left = False
            fake_in_right = False
            for index in local_maxima:
                if angles[index] in angles_right: 
                    fake_in_right = True
                elif angles[index] in angles_left:
                    fake_in_left = True
            tp_in_left = False
            tp_in_right = False
            distances = np.abs(angle_distance(angles[index_maximum], angles_left))
            left_border = angles_left[np.argmax(distances)]
            distances = np.abs(angle_distance(angles[index_maximum], angles_right))
            right_border = angles_right[np.argmax(distances)]
            for tp in turning_points:
                if angles[tp] == left_border:
                    tp_in_left = True
                elif angles[tp] == right_border:
                    tp_in_right = True

            if np.min(intensities_left) > (np.max(intensities_right) \
                    + 0.1 * global_amplitude) and not fake_in_left and not tp_in_left:
                is_peak = False

            elif np.min(intensities_right) > (np.max(intensities_left) \
                        + 0.1 * global_amplitude) and not fake_in_right and not tp_in_right:
                is_peak = False

    return is_peak
    
@njit(cache = True, fastmath = True)
def _adjust_borders(left_border, right_border, mu_maximum, local_minima_angles, angles, local_minima):
    # adjust peak borders around the found maximum

    if len(local_minima) == 0:
        closest_left_border = left_border
        closest_right_border = right_border
    else:
        minima_distances = angle_distance(mu_maximum, local_minima_angles)
        left_minima_distances = minima_distances[minima_distances < 0]
        right_minima_distances = minima_distances[minima_distances > 0]
        if len(left_minima_distances) == 0: 
            closest_left_border = left_border
        else:
            index_left_min = local_minima[minima_distances == np.max(left_minima_distances)][0]
            left_min = angles[index_left_min]
            if np.abs(angle_distance(mu_maximum, left_border)) > np.abs(angle_distance(mu_maximum, left_min)):
                closest_left_border = left_min
            else:
                closest_left_border = left_border
        if len(right_minima_distances) == 0: 
            closest_right_border = right_border
        else: 
            index_right_min = local_minima[minima_distances == np.min(right_minima_distances)][0]
            right_min = angles[index_right_min]
            if np.abs(angle_distance(mu_maximum, right_border)) \
                > np.abs(angle_distance(mu_maximum, right_min)):
                closest_right_border = right_min
            else:
                closest_right_border = right_border

    return closest_left_border, closest_right_border, index_left_min, index_right_min, \
                left_minima_distances, right_minima_distances

@njit(cache = True, fastmath = True)
def _handle_extrema(angles, intensities, intensities_err, first_diff, params):
    # Do stuff with found extrema

    local_maxima = params.local_maxima
    local_minima = params.local_minima
    turning_points = params.turning_points
    turning_points_directions = params.turning_points_directions
    min_peak_hwhm = params.min_peak_hwhm
    max_peak_hwhm = params.max_peak_hwhm
    max_A = params.max_A
    global_amplitude = params.global_amplitude
    mu_range = params.mu_range

    local_maxima_angles = angles[local_maxima]

    local_maxima, local_maxima_angles, local_minima = _merge_maxima(local_maxima, local_minima, 
                intensities, angles, local_maxima_angles, max_peak_hwhm, global_amplitude)

    # Append all neighbours of minima to minima when they have similar height
    # and lower than neighbouring maxima

    local_minima = _append_similar_minima(local_minima, local_maxima, turning_points, intensities,
                                            intensities_err, global_amplitude)

    local_minima, local_maxima, local_maxima_angles, is_merged, turning_points = \
                            _handle_turning_points(turning_points, turning_points_directions,
                            local_maxima, local_minima, local_maxima_angles, angles, 
                            intensities, first_diff, min_peak_hwhm, global_amplitude, mu_range)

    local_minima_angles = angles[local_minima]

    local_maxima, local_maxima_angles, local_minima, local_minima_angles, is_merged, is_hidden_peak = \
                        _find_hidden_peaks(local_maxima, local_minima, local_minima_angles,
                        local_maxima_angles, angles, intensities, first_diff, is_merged, 
                        max_peak_hwhm, max_A, global_amplitude)

    # Sort maxima from lowest intensity to highest
    sort_indices = np.argsort(intensities[local_maxima])
    local_maxima = local_maxima[sort_indices]
    local_maxima_angles = local_maxima_angles[sort_indices]
    is_merged = is_merged[sort_indices]
    is_hidden_peak = is_hidden_peak[sort_indices]

    return local_maxima, local_minima, turning_points, local_maxima_angles, \
            local_minima_angles, is_merged, is_hidden_peak

@njit(cache = True, fastmath = True)
def _equalize_difference(angles, intensities, indices, indices_reverse, extrema_tolerance):

    different_indices = set_diff(indices, indices_reverse)
    different_indices_reverse = set_diff(indices_reverse, indices)

    for index in different_indices:
        if len(different_indices_reverse) == 0: break
        # Find closest reverse index
        distances = np.abs(angle_distance(angles[index], angles[different_indices_reverse]))
        closest_reverse_index = different_indices_reverse[np.argmin(distances)]

        # If difference of intensities between both indices are all below tolerance
        # append both indices (and they will be merged later)
        direction = 1
        if angle_distance(angles[index], angles[closest_reverse_index]) < 0:
            direction = -1
        all_below = True
        for i in range(1, len(angles)):
            next_index = (index + i * direction) % len(angles)
            if np.abs(intensities[next_index] - intensities[index]) > extrema_tolerance:
                all_below = False
                break
            if next_index == closest_reverse_index: 
                break

        if all_below:
            insert_index = np.searchsorted(indices, closest_reverse_index)
            indices = numba_insert(indices, insert_index, closest_reverse_index)
            insert_index = np.searchsorted(indices_reverse, index)
            indices_reverse = numba_insert(indices_reverse, insert_index, index)

    return indices, indices_reverse

@njit(cache = True, fastmath = True)
def _find_extremas_full(angles, intensities, first_diff, second_diff, params):

    max_A = params.max_A
    extrema_tolerance = params.extrema_tolerance
    turning_point_tolerance = params.turning_point_tolerance
    turning_point_tolerance_2 = params.turning_point_tolerance_2
    turning_point_tolerance_3 = params.turning_point_tolerance_3

    # Find extremas starting from left
    local_maxima, local_minima, turning_points, turning_points_directions = _find_extremas(intensities,
                    first_diff, second_diff, max_A, extrema_tolerance,
                    turning_point_tolerance, turning_point_tolerance_2, turning_point_tolerance_3,
                    reverse = False)

    # Find extremas starting from right
    local_maxima_reverse, local_minima_reverse, turning_points_reverse, \
    turning_points_directions_reverse = _find_extremas(intensities, first_diff, 
                    second_diff, max_A, extrema_tolerance,
                    turning_point_tolerance, turning_point_tolerance_2, turning_point_tolerance_3,
                    reverse = True)

    # If difference of intensities between different indices are all below tolerance
    # append both indices (and they will be merged later)
    local_maxima, local_maxima_reverse = _equalize_difference(angles, intensities, local_maxima, 
                                            local_maxima_reverse, extrema_tolerance)
    local_minima, local_minima_reverse = _equalize_difference(angles, intensities, local_minima, 
                                            local_minima_reverse, extrema_tolerance)

    # Pick extremas found independent from search direction
    local_maxima = np.intersect1d(local_maxima, local_maxima_reverse)
    local_minima = np.intersect1d(local_minima, local_minima_reverse)

    #turning_points = np.concatenate((turning_points, turning_points_reverse))
    #unique_turning_points = numba_unique(turning_points)[0]

    tps = np.zeros(len(angles), dtype=np.bool_)
    tp_directions = np.zeros(len(angles), dtype=np.int64)
    for i in range(len(angles)):
        if turning_points_directions[i] != 0 and turning_points_directions_reverse[i] == 0:
            tps[i] = True
            tp_directions[i] = turning_points_directions[i]
        elif turning_points_directions[i] == 0 and turning_points_directions_reverse[i] != 0:
            tps[i] = True
            tp_directions[i] = turning_points_directions_reverse[i]
        elif turning_points_directions[i] != 0 and turning_points_directions_reverse[i] != 0 \
                and turning_points_directions[i] == turning_points_directions_reverse[i]:
            tps[i] = True
            tp_directions[i] = turning_points_directions[i]   

    turning_points = tps.nonzero()[0]
    turning_points_directions = tp_directions[tp_directions != 0]

    return local_maxima, local_minima, turning_points, turning_points_directions

@njit(cache = True, fastmath = True)
def _find_peaks_from_extrema(angles, intensities, intensities_err, params):
    peaks_found = 0
    angle_spacing = 2 * np.pi / len(angles)

    current_intensities = np.copy(intensities)
    current_angles = np.copy(angles)

    local_maxima = params.local_maxima
    local_minima = params.local_minima
    turning_points = params.turning_points
    local_maxima_angles = params.local_maxima_angles
    local_minima_angles = params.local_minima_angles
    is_merged = params.is_merged
    is_hidden_peak = params.is_hidden_peak
    max_A = params.max_A
    max_peak_hwhm = params.max_peak_hwhm
    min_peak_hwhm = params.min_peak_hwhm
    min_int = params.min_int
    global_amplitude = params.global_amplitude
    extrema_tolerance = params.extrema_tolerance
    only_peaks_count = params.only_peaks_count
    max_peaks = params.max_peaks

    # Array to store the indices (mask) that (mainly) relates to a peak
    # first dimension is the peak number
    peaks_mask = np.zeros((len(local_maxima), len(angles)))
    
    # Array to store the angle of maximum (mu) for every peak
    peaks_mus = np.empty(len(local_maxima))

    is_peak = np.ones(len(local_maxima), dtype = np.bool_)
    for n, mu_maximum in enumerate(local_maxima_angles):
        # iterate from lowest to highest maximum
        index_maximum = local_maxima[n]
        local_max_int =  intensities[index_maximum]
        #mu_maximum = angles[index_maximum]
        amplitude = local_max_int - min_int

        # Define left and right border for the the peak angles
        # Pick all angles for the peak which are in the range of double hwhm per side
        left_border = (mu_maximum - 1.5 * max_peak_hwhm) % (2 * np.pi)
        right_border = (mu_maximum + 1.5 * max_peak_hwhm) % (2 * np.pi)

        # Adjust borders to the closest local minima around the peak

        closest_left_border, closest_right_border, index_left_min, index_right_min, \
            left_minima_distances, right_minima_distances = _adjust_borders(left_border, right_border, 
                                                    mu_maximum, local_minima_angles, angles, local_minima)

        # Define condition for the angles exluding the peak angles
        # edit: probably could be done easier with: index_maximum +- 2 * max_peak_hwhm / (angle_spacing)
        if closest_left_border < closest_right_border:
            condition = (current_angles < closest_left_border) | (current_angles > closest_right_border)
        else:
            condition = (current_angles < closest_left_border) & (current_angles > closest_right_border)

        if left_border < right_border:
            full_condition = (angles < left_border) | (angles > right_border)
        else:
            full_condition = (angles < left_border) & (angles > right_border)

        peak_angles = current_angles[~ condition]
        peak_intensities = current_intensities[~ condition]

        full_peak_angles = angles[~ full_condition]
        full_peak = True
        if len(peak_angles) != len(full_peak_angles): full_peak = False

        is_peak[n] = False
        # full peak width of angles must be at least 1.5 * min hwhm
        # and peak height from peak lowest minimum to peak maximum must be 5% of global amplitude
        # (except for hidden peaks)
        if ((len(peak_angles) * angle_spacing > 1.5 * min_peak_hwhm \
                and (local_max_int - np.min(peak_intensities) > 0.05 * global_amplitude)) \
                or is_hidden_peak[n]) and amplitude > (max_A - min_int) + 0.05 * global_amplitude:

            is_peak[n] = True
            
            # If a local minimum is neighbour of peak, append it to peak_angles
            # so that the local minima can be used by both neighbouring peaks
            if len(local_minima) > 0:
                left_min = angles[index_left_min]
                right_min = angles[index_right_min]
                if len(left_minima_distances) > 0 and left_min not in peak_angles:
                    if np.abs(angle_distance(left_min, closest_left_border)) <= angle_spacing:
                        insert_index = np.searchsorted(peak_angles, left_min)
                        peak_angles = numba_insert(peak_angles, insert_index, left_min)
                        peak_intensities = numba_insert(peak_intensities, insert_index, intensities[index_left_min])
                if len(right_minima_distances) > 0 and right_min not in peak_angles:
                    if np.abs(angle_distance(right_min, closest_right_border)) <= angle_spacing:
                        insert_index = np.searchsorted(peak_angles, right_min)
                        peak_angles = numba_insert(peak_angles, insert_index, right_min)
                        peak_intensities = numba_insert(peak_intensities, insert_index, intensities[index_right_min])
        
            relative_angles = angle_distance(mu_maximum, peak_angles)
            angles_left = peak_angles[relative_angles < 0]
            angles_right = peak_angles[relative_angles > 0]
            intensities_left = peak_intensities[relative_angles < 0]
            intensities_right = peak_intensities[relative_angles > 0]

            # If peak is so strongly merged that only half of the peak visible
            # so that local_maximum = local_minimum, then make the empty side contain this point
            if not full_peak:
                if len(intensities_left) == 0:
                    angles_left = peak_angles[relative_angles <= 0]
                    intensities_left = peak_intensities[relative_angles <= 0]
                elif len(intensities_right) == 0:
                    intensities_right = peak_intensities[relative_angles >= 0]
                    angles_right = peak_angles[relative_angles >= 0]

            if not is_merged[n]:
                mu_maximum, peak_angles, peak_intensities, angles_left, angles_right, \
                    intensities_left, intensities_right, closest_left_border, \
                        closest_right_border = _find_best_center(angles, current_angles, intensities,
                        current_intensities, peak_angles, peak_intensities, angles_left, index_maximum,
                        angles_right, intensities_left, intensities_right, angle_spacing, mu_maximum, 
                        max_peak_hwhm, local_minima, local_minima_angles, closest_left_border,
                        closest_right_border)

            is_peak[n] = _is_real_peak(angles, intensities, peak_angles, peak_intensities, intensities_left,
                intensities_right, angles_left, angles_right, global_amplitude, 
                    local_max_int, intensities_err, extrema_tolerance, index_maximum,
                    local_maxima, local_minima, turning_points, is_merged[n], full_peak)

        if not is_peak[n] and (is_merged[n] or not full_peak):
            # if fake merged remove the local minimum in peak_angles 
            # which is closest to another local maximum (without fake maxima)
            current_closest_distance = np.inf
            closest_minimum = -1
            real_maxima = local_maxima[is_peak]
            if len(real_maxima) > 0:
                for i, index_minimum in enumerate(local_minima):
                    if angles[index_minimum] not in peak_angles: continue
                    distances_to_max = angle_distance(angles[index_minimum], angles[real_maxima])
                    closest_distance = np.min(np.abs(distances_to_max))
                    if closest_distance < current_closest_distance:
                        current_closest_distance = closest_distance
                        closest_minimum = i

                if closest_minimum != -1 and intensities[local_minima[closest_minimum]] > max_A:
                    local_minima = np.delete(local_minima, closest_minimum)
                    local_minima_angles = np.delete(local_minima_angles, closest_minimum)

        elif is_peak[n]:
            peaks_found += 1

            # Store peak indices into peaks mask in reverse
            # so that the lowest peaks will be at the end
            for i, angle in enumerate(angles):
                if angle in peak_angles:
                    peaks_mask[-peaks_found][i] = 1

            # Store peak maximum angle (mu) into array
            peaks_mus[-peaks_found] = mu_maximum

            # Remove peak angles from current angles array
            current_angles = current_angles[condition]
            current_intensities = current_intensities[condition]

    # When no peaks found or not as many as wished return zeros
    if peaks_found == 0 or (only_peaks_count != -1 and peaks_found != only_peaks_count):
        peaks_mask = np.zeros(len(angles))
        peaks_mus = np.zeros(1)
        return peaks_mask, peaks_mus

    num_peaks = min(max_peaks, peaks_found)

    # Cut off peaks mask array for the peaks_found highest peaks
    start_index = len(local_maxima) - peaks_found
    cut_off_index = start_index + num_peaks
    peaks_mask = peaks_mask[start_index:cut_off_index, :]
    
    # Flattend peaks mask array for numba handling (same return type)
    peaks_mask = peaks_mask.ravel()

    # Cut off peaks mus array
    peaks_mus = peaks_mus[start_index:cut_off_index]

    return peaks_mask, peaks_mus

def find_peaks(angles, intensities, intensities_err, only_peaks_count = -1, max_peaks = 4,
                    max_peak_hwhm = 50 * np.pi/180, min_peak_hwhm = 10 * np.pi/180, 
                    mu_range = 40 * np.pi/180, scale_range = 0.4):
    """
    Finds peaks from given array of measured intensities of a pixel.

    Parameters:
    - angles: np.ndarray (n, )
        Array that stores the angles at which the intensities are measured.
    - intensities: np.ndarray (n, )
        The measured intensities of the pixel.
    - intensities_err: np.ndarray (n, )
        The error (standard deviation) of the corresponding measured intensities of the pixel.
    - only_peaks_count: int
        Defines a filter for found peaks, so that if the count of found peaks is not equal that number,
        the function return the same as if no peaks are found.
    - max_peaks: int
        Defines the maximum number of peaks that should be returned from the 
        total found peaks (starting from highest peak).
    - max_peak_hwhm: float
        Estimated maximum peak half width at half maximum.
    - min_peak_hwhm: float
        Estimated minimum peak half width at half maximum.
    - mu_range: float
        Range of mu (regarding estimated maximum and minimum bounds around true mu).
    - scale_range: float
        Range of scale (regarding estimated maximum and minimum bounds around true scale).

    Returns:
    - peaks_mask: np.ndarray (n_peaks, n)
        Array that stores the indices of the measurements that corresponds (mainly) to a peak.
    - peaks_mus: np.ndarray (n_peaks, )
        Array that stores the angles of the centers (mus) of the peaks.
    """

    # Ensure no overflow in (subtract) operations happen:
    intensities = intensities.astype(np.int32)

    min_int = np.min(intensities)
    max_int = np.max(intensities)
    global_amplitude = max_int - min_int
    max_A = min_int + 0.05 * global_amplitude

    # Define tolerance that differences must exceed tolerance to be classified as extrema
    extrema_tolerance = max(0.05 * global_amplitude, 1.5 * np.max(intensities_err))
    turning_point_tolerance = max(0.06 * global_amplitude, 3 * np.max(intensities_err))
    turning_point_tolerance_2 = 1.5 * extrema_tolerance
    turning_point_tolerance_3 = 2 * extrema_tolerance

    first_diff = np.diff(intensities)
    first_diff = np.append(first_diff, intensities[0] - intensities[-1])
    second_diff = np.diff(first_diff)
    second_diff = np.append(second_diff, first_diff[0] - first_diff[-1])

    params = PeakFinderParams(
        None, None, None, None, None, None, None, None, 
        max_A, max_peak_hwhm, min_peak_hwhm, mu_range, 
        min_int, global_amplitude, extrema_tolerance, 
        turning_point_tolerance, turning_point_tolerance_2, turning_point_tolerance_3,
        only_peaks_count, max_peaks
    )

    local_maxima, local_minima, turning_points, turning_points_directions = \
                    _find_extremas_full(angles, intensities, first_diff, second_diff, params)

    params = params._replace(
        local_maxima = local_maxima,
        local_minima = local_minima,
        turning_points = turning_points,
        turning_points_directions = turning_points_directions
    )

    local_maxima, local_minima, turning_points, local_maxima_angles, local_minima_angles, \
        is_merged, is_hidden_peak = _handle_extrema(angles, intensities, intensities_err, 
                    first_diff, params)

    params = params._replace(
        local_maxima = local_maxima,
        local_minima = local_minima,
        turning_points = turning_points,
        local_maxima_angles = local_maxima_angles,
        local_minima_angles = local_minima_angles,
        is_merged = is_merged,
        is_hidden_peak = is_hidden_peak
    )

    peaks_mask, peaks_mus = _find_peaks_from_extrema(angles, intensities, intensities_err, 
                params)

    # unflatten peaks_mask array
    peaks_mask = np.reshape(peaks_mask, (len(peaks_mus), len(angles))).astype(np.bool_)

    return peaks_mask, peaks_mus
