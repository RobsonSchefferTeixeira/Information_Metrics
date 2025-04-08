import numpy as np
import os
import sys
from scipy import interpolate
import numpy as np
from scipy import signal as sig
import math
import warnings

def correct_coordinates(x_coordinates, y_coordinates, environment_edges):
    # Assign NaN to out-of-bound x and y coordinates
    x_coordinates[(x_coordinates < environment_edges[0][0]) | (x_coordinates > environment_edges[0][1])] = np.nan
    y_coordinates[(y_coordinates < environment_edges[1][0]) | (y_coordinates > environment_edges[1][1])] = np.nan

    # If an index in y_coordinates is NaN, set the corresponding x_coordinates to NaN, and vice versa
    nan_mask = np.isnan(x_coordinates) | np.isnan(y_coordinates)
    x_coordinates[nan_mask] = np.nan
    y_coordinates[nan_mask] = np.nan

    return x_coordinates, y_coordinates

def get_sparsity(activity_map, position_occupancy):
    """
    Calculate the sparsity of a place field with respect to position occupancy.

    Parameters:
    - activity_map (numpy.ndarray): A place field map representing spatial preferences.
    - position_occupancy (numpy.ndarray): Positional occupancy map, typically representing time spent in each spatial bin.

    Returns:
    - sparsity (float): The sparsity measure indicating how selective the place field is with respect to position occupancy.

    """
    
    position_occupancy_norm = np.nansum(position_occupancy / np.nansum(position_occupancy))
    sparsity = np.nanmean(position_occupancy_norm * activity_map) ** 2 / np.nanmean(
        position_occupancy_norm * activity_map ** 2)

    return sparsity

def get_2D_activity_map(signal, x_coordinates, y_coordinates, x_grid, y_grid, smoothing_sigma):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # calculate mean calcium per pixel
    # Notice that this is the calcium trace normalized by occupancy.

    activity_map = np.nan * np.zeros((y_grid.shape[0] - 1, x_grid.shape[0] - 1))
    for xx in range(0, x_grid.shape[0] - 1):
        for yy in range(0, y_grid.shape[0] - 1):
            check_x_occupancy = np.logical_and(x_coordinates >= x_grid[xx], x_coordinates < (x_grid[xx + 1]))
            check_y_occupancy = np.logical_and(y_coordinates >= y_grid[yy], y_coordinates < (y_grid[yy + 1]))

            activity_map[yy, xx] = np.nanmean(signal[np.logical_and(check_x_occupancy, check_y_occupancy)])

    # Taking the mean is equivalent to:
    # activity_map[yy, xx] = np.nansum(input_signal[np.logical_and(check_x_occupancy, check_y_occupancy)])
    # and then normalize by the number of points inside each bin:
    # activity_map_normalized = activity_map/(position_occupancy*sampling_rate)
    # one must multiply by sampling_rate because position_occupancy is the time spent inside each bin


    activity_map_to_smooth = np.copy(activity_map)
    activity_map_to_smooth[np.isnan(activity_map_to_smooth)] = 0
    activity_map_smoothed = gaussian_smooth_2d(activity_map_to_smooth, smoothing_sigma)

    return activity_map, activity_map_smoothed



def identify_islands_1D(input_array):
    
    """
    Identifies islands (continuous sequences of 1s) in a 1D input array.

    Parameters:
    - input_array (ndarray): Input array containing 1s and 0s.

    Returns:
    - input_array2 (ndarray): Modified array with islands labeled.
    """
        
    length = len(input_array)
    count = 0
    input_array2 = np.copy(input_array)

    for i in range(length):
        if input_array[i] == 1:
            count += 1
            dfs_1D(input_array, input_array2, count, length, i)

    return input_array2


def dfs_1D(input_array, input_array2, count, length, i):

    """
    Depth-First Search (DFS) algorithm to identify islands in a 1D array.

    Parameters:
    - input_array (ndarray): Input array containing 1s and 0s.
    - input_array2 (ndarray): Modified array with islands labeled.
    - count (int): Counter for island labeling.
    - length (int): Length of the input array.
    - i (int): Current index being evaluated in the DFS algorithm.

    Returns:
    - None
    """
     
    if input_array[i] == 0:
        return
    input_array[i] = 0
    input_array2[i] = count

    if i != 0:
        dfs_1D(input_array, input_array2, count, length, i - 1)

    if i != length - 1:
        dfs_1D(input_array, input_array2, count, length, i + 1)


def field_coordinates_using_shifted_1D(activity_map, activity_map_shifted, visits_map, percentile_threshold=95,min_num_of_bins=4):

    """
    Identifies and characterizes place fields using shifted 1D data.

    Parameters:
    - activity_map (ndarray): Original place field data.
    - activity_map_shifted (ndarray): Shifted place field data.
    - visits_map (ndarray): Map of visited locations.
    - percentile_threshold (int): Percentile threshold for identifying place fields.
    - min_num_of_bins (int): Minimum number of pixels to consider as a place field.

    Returns:
    - num_of_islands (int): Number of identified place field islands.
    - islands_x_max (ndarray): X coordinates of identified islands.
    - pixels_place_cell_absolute (ndarray): Relative pixels of place cells in absolute terms.
    - pixels_place_cell_relative (ndarray): Relative pixels of place cells in relation to visited locations.
    - activity_map_identity (ndarray): Identified place field islands labeled.
    """


    activity_map_threshold = np.percentile(activity_map_shifted, percentile_threshold, 0)

    field_above_threshold_binary = activity_map.copy()
    field_above_threshold_binary[field_above_threshold_binary <= activity_map_threshold] = 0
    field_above_threshold_binary[field_above_threshold_binary > activity_map_threshold] = 1

    if np.any(field_above_threshold_binary > 0):
        sys.setrecursionlimit(10000000)
        activity_map_identity = identify_islands_1D(np.copy(field_above_threshold_binary))
        num_of_islands_pre = np.unique(activity_map_identity)[1:].shape[0]

        num_of_islands = 0
        for ii in range(1, num_of_islands_pre + 1):
            if np.where(activity_map_identity == ii)[0].shape[0] > min_num_of_bins:
                num_of_islands += 1
            else:
                activity_map_identity[np.where(activity_map_identity == ii)] = 0


        islands_id = np.unique(activity_map_identity[~np.isnan(activity_map_identity)])[1:]
        islands_x_max = []
        pixels_above = []
        for ii in islands_id:
            max_val = np.nanmax(activity_map[(activity_map_identity == ii)])
            I_x_max = np.where(activity_map == max_val)
            islands_x_max.append(I_x_max[0])

            pixels_above.append(np.nansum(activity_map_identity == ii))

        islands_x_max = np.squeeze(islands_x_max)
        pixels_above = np.array(pixels_above)

        total_visited_pixels = np.nansum(visits_map != 0)
        pixels_total = activity_map.shape[0]

        pixels_place_cell_relative = pixels_above / total_visited_pixels
        pixels_place_cell_absolute = pixels_above / pixels_total

    else:
        num_of_islands = 0
        islands_x_max = np.nan
        pixels_place_cell_relative = np.nan
        pixels_place_cell_absolute = np.nan
        activity_map_identity = np.nan


    return num_of_islands, islands_x_max,pixels_place_cell_absolute,pixels_place_cell_relative,correct_island_identifiers(activity_map_identity)





def get_visits_1D( x_coordinates,position_binned, x_center_bins):
    I_x_coord = []

    for xx in range(0, x_coordinates.shape[0]):
        if np.isnan(x_coordinates[xx]):
            I_x_coord.append(np.nan)
        else:
            I_x_coord.append(np.nanargmin(np.abs(x_coordinates[xx] - x_center_bins)))

    I_x_coord = np.array(I_x_coord)

    dx = np.diff(np.hstack([I_x_coord[0] - 1, I_x_coord]))

    new_visits_times = ((dx != 0) & (~np.isnan(dx)))

    visits_id, visits_counts = np.unique(position_binned[new_visits_times], return_counts=True)

    visits_bins = np.zeros(position_binned.shape) * np.nan
    for ids in range(visits_id.shape[0]):
        if ~np.isnan(visits_id[ids]):
            I_pos = position_binned == visits_id[ids]
            visits_bins[I_pos] = visits_counts[ids]

    return visits_bins, new_visits_times * 1

def get_visits( x_coordinates, y_coordinates, position_binned, x_center_bins, y_center_bins):
    I_x_coord = []
    I_y_coord = []

    for x_cord,y_cord in zip(x_coordinates,y_coordinates):
        if np.isnan(x_cord) | np.isnan(y_cord):
            I_x_coord.append(np.nan)
            I_y_coord.append(np.nan)
        else:
            I_x_coord.append(np.nanargmin(np.abs(x_cord - x_center_bins)))
            I_y_coord.append(np.nanargmin(np.abs(y_cord - y_center_bins)))

    I_x_coord = np.array(I_x_coord)
    I_y_coord = np.array(I_y_coord)

    dx = np.diff(np.hstack([I_x_coord[0] - 1, I_x_coord]))
    dy = np.diff(np.hstack([I_y_coord[0] - 1, I_y_coord]))

    new_visits_times = (np.logical_or(((dy != 0) & (~np.isnan(dy))), ((dx != 0) & (~np.isnan(dx)))))

    visits_id, visits_counts = np.unique(position_binned[new_visits_times], return_counts=True)

    visits_bins = np.zeros(position_binned.shape) * np.nan
    for ids in range(visits_id.shape[0]):
        if ~np.isnan(visits_id[ids]):
            I_pos = position_binned == visits_id[ids]
            visits_bins[I_pos] = visits_counts[ids]

    return visits_bins, new_visits_times * 1

def get_visits_occupancy_1D( x_coordinates, new_visits_times, x_grid):
    I_visit = np.where(new_visits_times > 0)[0]

    x_coordinate_visit = x_coordinates[I_visit]

    visits_occupancy = np.zeros(x_grid.shape[0] - 1) * np.nan
    for xx in range(0, x_grid.shape[0] - 1):
        check_x_occupancy = np.logical_and(x_coordinate_visit >= x_grid[xx],
                                               x_coordinate_visit < (x_grid[xx + 1]))

        visits_occupancy[xx] = np.sum(check_x_occupancy)

    # visits_occupancy[visits_occupancy < min_visits] = np.nan

    return visits_occupancy


def get_visits_occupancy(x_coordinates, y_coordinates, new_visits_times, x_grid, y_grid):
    I_visit = np.where(new_visits_times > 0)[0]

    x_coordinate_visit = x_coordinates[I_visit]
    y_coordinate_visit = y_coordinates[I_visit]

    visits_occupancy = np.zeros((y_grid.shape[0] - 1, x_grid.shape[0] - 1)) * np.nan
    for xx in range(0, x_grid.shape[0] - 1):
        for yy in range(0, y_grid.shape[0] - 1):
            check_x_occupancy = np.logical_and(x_coordinate_visit >= x_grid[xx],
                                               x_coordinate_visit < (x_grid[xx + 1]))
            check_y_occupancy = np.logical_and(y_coordinate_visit >= y_grid[yy],
                                               y_coordinate_visit < (y_grid[yy + 1]))

            visits_occupancy[yy, xx] = np.sum(np.logical_and(check_x_occupancy, check_y_occupancy))

    # visits_occupancy[visits_occupancy < min_visits] = np.nan

    return visits_occupancy


def get_position_time_spent( position_binned, sampling_rate):
    positions_id, positions_counts = np.unique(position_binned, return_counts=True)

    time_spent_inside_bins = np.zeros(position_binned.shape) * np.nan
    for ids in range(positions_id.shape[0]):
        if ~np.isnan(positions_id[ids]):
            I_pos = position_binned == positions_id[ids]
            time_spent_inside_bins[I_pos] = positions_counts[ids] / sampling_rate

    return time_spent_inside_bins


def get_occupancy_1D(x_coordinates, x_grid, sampling_rate):
    # calculate position occupancy
    position_occupancy = np.zeros(x_grid.shape[0] - 1)
    for xx in range(0, x_grid.shape[0] - 1):
        check_x_occupancy = np.logical_and(x_coordinates >= x_grid[xx], x_coordinates < (x_grid[xx + 1]))
        position_occupancy[xx] = np.sum(check_x_occupancy)/sampling_rate

    # position_occupancy[position_occupancy == 0] = np.nan
    return position_occupancy

def get_speed_occupancy(speed,x_coordinates, y_coordinates, x_grid, y_grid):
    # calculate position occupancy

    speed_occupancy = np.zeros((y_grid.shape[0] - 1, x_grid.shape[0] - 1))
    for xx in range(0, x_grid.shape[0] - 1):
        for yy in range(0, y_grid.shape[0] - 1):
            check_x_occupancy = np.logical_and(x_coordinates >= x_grid[xx], x_coordinates < (x_grid[xx + 1]))
            check_y_occupancy = np.logical_and(y_coordinates >= y_grid[yy], y_coordinates < (y_grid[yy + 1]))

            speed_occupancy[yy, xx] = np.nanmean(speed[np.logical_and(check_x_occupancy, check_y_occupancy)])

    return speed_occupancy

def get_occupancy(x_coordinates, y_coordinates, x_grid, y_grid, sampling_rate):
    # calculate position occupancy, defined as time spent inside each bin

    position_occupancy = np.zeros((y_grid.shape[0] - 1, x_grid.shape[0] - 1))
    for xx in range(0, x_grid.shape[0] - 1):
        for yy in range(0, y_grid.shape[0] - 1):
            check_x_occupancy = np.logical_and(x_coordinates >= x_grid[xx], x_coordinates < (x_grid[xx + 1]))
            check_y_occupancy = np.logical_and(y_coordinates >= y_grid[yy], y_coordinates < (y_grid[yy + 1]))

            position_occupancy[yy, xx] = np.sum(
                np.logical_and(check_x_occupancy, check_y_occupancy)) / sampling_rate

    # position_occupancy[position_occupancy == 0] = np.nan
    return position_occupancy

def get_position_grid_1D(x_coordinates, x_bin_size=1, environment_edges=None):
    # x_bin_size and y_bin_size in cm
    # environment_edges = [[x1, x2], [y1, y2]]


    if environment_edges is None:
        x_min = np.nanmin(x_coordinates)
        x_max = np.nanmax(x_coordinates)    
        environment_edges = [[x_min, x_max]]
    
    x_grid = np.linspace(environment_edges[0][0], environment_edges[0][1], int((environment_edges[0][1] - environment_edges[0][0]) / x_bin_size) + 1)
    x_center_bins = x_grid[:-1] + x_bin_size / 2
    x_center_bins_repeated = np.repeat(x_center_bins, x_center_bins.shape[0])


    return x_grid, x_center_bins, x_center_bins_repeated


def get_position_grid(x_coordinates, y_coordinates=None, x_bin_size=1, y_bin_size=None, environment_edges=None):
    # x_bin_size and y_bin_size in cm
    # environment_edges = [[x1, x2], [y1, y2]]
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if y_coordinates is None:
        # 1D tracking
    
        if environment_edges is None:
            x_min = np.nanmin(x_coordinates)
            x_max = np.nanmax(x_coordinates)    
            environment_edges = [[x_min, x_max]]
        
        x_grid = np.linspace(environment_edges[0][0], environment_edges[0][1], int((environment_edges[0][1] - environment_edges[0][0]) / x_bin_size) + 1)
        x_center_bins = x_grid[:-1] + x_bin_size / 2
        x_center_bins_repeated = np.repeat(x_center_bins, x_center_bins.shape[0])

        y_grid = np.nan
        y_center_bins = np.nan
        y_center_bins_repeated = np.nan

    else:
        # 2D tracking
        if environment_edges is None:
            x_min = np.nanmin(x_coordinates)
            x_max = np.nanmax(x_coordinates)
            y_min = np.nanmin(y_coordinates)
            y_max = np.nanmax(y_coordinates)
    
            environment_edges = [[x_min, x_max], [y_min, y_max]]

            
        x_range = environment_edges[0][1] - environment_edges[0][0]
        y_range = environment_edges[1][1] - environment_edges[1][0]

        num_x_bins = np.nanmax([int(np.floor(x_range / x_bin_size)), 2])
        num_y_bins = np.nanmax([int(np.floor(y_range / y_bin_size)), 2])

        x_grid = np.linspace(environment_edges[0][0], environment_edges[0][1], num_x_bins + 1)
        y_grid = np.linspace(environment_edges[1][0], environment_edges[1][1], num_y_bins + 1)

        x_center_bins = x_grid[:-1] + np.diff(x_grid) / 2
        y_center_bins = y_grid[:-1] + np.diff(y_grid) / 2

        x_center_bins_repeated = np.repeat(x_center_bins, y_center_bins.shape[0])
        y_center_bins_repeated = np.tile(y_center_bins, x_center_bins.shape[0])

        # TODO: check if this way is better for spatial prediction
        # x_center_bins_repeated, y_center_bins_repeated = np.meshgrid(x_center_bins, y_center_bins)
        # x_center_bins_repeated = x_center_bins_repeated.flatten()
        # y_center_bins_repeated = y_center_bins_repeated.flatten()

    return x_grid, y_grid, x_center_bins, y_center_bins, x_center_bins_repeated, y_center_bins_repeated



def get_binned_position(x_coordinates, y_coordinates = None, x_grid = None, y_grid = None):
    # calculate position occupancy

    if y_coordinates is None:
        position_binned = np.zeros(x_coordinates.shape) * np.nan
        count = 0
        for xx in range(0, x_grid.shape[0] - 1):
            if xx == x_grid.shape[0] - 2:
                check_x_occupancy = np.logical_and(x_coordinates >= x_grid[xx], x_coordinates <= (x_grid[xx + 1]))
            else:
                check_x_occupancy = np.logical_and(x_coordinates >= x_grid[xx], x_coordinates < (x_grid[xx + 1]))

            position_binned[check_x_occupancy] = count
            count += 1


    else:
            
        position_binned = np.zeros(x_coordinates.shape) * np.nan
        count = 0
        for xx in range(0, x_grid.shape[0] - 1):
            for yy in range(0, y_grid.shape[0] - 1):
                if xx == x_grid.shape[0] - 2:
                    check_x_occupancy = np.logical_and(x_coordinates >= x_grid[xx], x_coordinates <= (x_grid[xx + 1]))
                else:
                    check_x_occupancy = np.logical_and(x_coordinates >= x_grid[xx], x_coordinates < (x_grid[xx + 1]))

                if yy == y_grid.shape[0] - 2:
                    check_y_occupancy = np.logical_and(y_coordinates >= y_grid[yy], y_coordinates <= (y_grid[yy + 1]))
                else:
                    check_y_occupancy = np.logical_and(y_coordinates >= y_grid[yy], y_coordinates < (y_grid[yy + 1]))

                position_binned[np.logical_and(check_x_occupancy, check_y_occupancy)] = count
                count += 1

    return position_binned


def get_speed(x_coords, y_coords, time_vector, speed_smoothing_sigma=1):
    """
    Calculate the instantaneous speed from position coordinates and time stamps,
    and apply Gaussian smoothing to the speed.

    Parameters:
    x_coords (array-like): Array of x-coordinate positions.
    y_coords (array-like): Array of y-coordinate positions.
    time_stamps (array-like): Array of time stamps corresponding to the positions.
    smoothing_sigma (float, optional): Standard deviation for Gaussian kernel used in smoothing. Default is 1.

    Returns:
    tuple:
        raw_speed (numpy.ndarray): Array of calculated speeds before smoothing.
        smoothed_speed (numpy.ndarray): Array of speeds after applying Gaussian smoothing.
    """
    sampling_rate = 1 / np.nanmean(np.diff(time_vector))
    smoothing_sigma = (speed_smoothing_sigma/1000)*sampling_rate

    delta_x = np.diff(x_coords)
    delta_y = np.diff(y_coords)

    distances = np.sqrt(delta_x**2 + delta_y**2)
    time_intervals = np.diff(time_vector)
    speed = distances / time_intervals
    speed = np.append(speed, 0)
    smoothed_speed = gaussian_smooth_1d(speed, sigma = smoothing_sigma)
    
    return speed, smoothed_speed


def correct_lost_tracking(x_coordinates, y_coordinates, track_timevector, sampling_rate, min_epoch_length=1):
    x_coordinates_interpolated = x_coordinates.copy()
    y_coordinates_interpolated = y_coordinates.copy()

    I_nan_events = np.where(np.isnan(x_coordinates * y_coordinates))[0]
    I_start = np.where(np.diff(I_nan_events) > 1)[0] + 1

    all_epoch_length = []
    for cc in range(I_start.shape[0] - 1):
        beh_index = np.arange(I_start[cc], I_start[cc + 1])

        epoch_length = (beh_index.shape[0] / sampling_rate)
        all_epoch_length.append(epoch_length)

        if epoch_length <= min_epoch_length:
            events_around_nan = I_nan_events[beh_index]
            events_to_replace = np.arange(events_around_nan[0] - 1, events_around_nan[-1] + 2)

            window_beg = events_to_replace[0]
            window_end = events_to_replace[-1]

            x_original = track_timevector[[window_beg, window_end]]
            y_original = x_coordinates[[window_beg, window_end]]
            x_new = track_timevector[window_beg:window_end + 1]
            interpol_func = interpolate.interp1d(x_original, y_original, kind='slinear')
            y_new = interpol_func(x_new)

            x_coordinates_interpolated[events_to_replace] = y_new

            x_original = track_timevector[[window_beg, window_end]]
            y_original = y_coordinates[[window_beg, window_end]]
            x_new = track_timevector[window_beg:window_end + 1]
            interpol_func = interpolate.interp1d(x_original, y_original, kind='slinear')
            y_new = interpol_func(x_new)

            y_coordinates_interpolated[events_to_replace] = y_new

    all_epoch_length = np.array(all_epoch_length)

    return x_coordinates_interpolated, y_coordinates_interpolated


def filename_constructor(saving_string, animal_id, dataset, day, neuron, trial):
    first_string = saving_string
    animal_id_string = '.' + animal_id
    dataset_string = '.Dataset.' + dataset
    day_string = '.Day.' + str(day)
    neuron_string = '.Neuron.' + str(neuron)
    trial_string = '.Trial.' + str(trial)

    filename_checklist = np.array([first_string, animal_id, dataset, day, neuron, trial])
    inlcude_this = np.where(filename_checklist != None)[0]

    filename_backbone = [first_string, animal_id_string, dataset_string, day_string, neuron_string, trial_string]

    filename = ''.join([filename_backbone[i] for i in inlcude_this])

    return filename


def caller_saving(inputdict, filename, saving_path):
    os.chdir(saving_path)
    output = open(filename, 'wb')
    np.save(output, inputdict)
    output.close()
    print('File saved.')


def identify_islands(input_array):
    """
    Identifies islands in a binary input array.

    Parameters
    ----------
    input_array : numpy.ndarray
        Binary input array containing islands marked as 1.

    Returns
    -------
    input_array2 : numpy.ndarray
        Array with islands labeled sequentially.
    """

    row = input_array.shape[0]
    col = input_array.shape[1]
    count = 0
    input_array2 = np.copy(input_array)

    for i in range(row):
        for j in range(col):
            if input_array[i, j] == 1:
                count += 1
                dfs(input_array, input_array2, count, row, col, i, j)

    return input_array2


def dfs(input_array, input_array2, count, row, col, i, j):
    """
    Performs Depth-First Search (DFS) to label connected regions in the input array.

    Parameters
    ----------
    input_array : numpy.ndarray
        Binary input array containing islands marked as 1.
    input_array2 : numpy.ndarray
        Copy of the input array.
    count : int
        Count of identified islands.
    row : int
        Number of rows in the array.
    col : int
        Number of columns in the array.
    i : int
        Current row index.
    j : int
        Current column index.
    """

    if input_array[i, j] == 0:
        return
    
    input_array[i, j] = 0
    input_array2[i, j] = count

    if i != 0:
        dfs(input_array, input_array2, count, row, col, i - 1, j)

    if i != row - 1:
        dfs(input_array, input_array2, count, row, col, i + 1, j)

    if j != 0:
        dfs(input_array, input_array2, count, row, col, i, j - 1)

    if j != col - 1:
        dfs(input_array, input_array2, count, row, col, i, j + 1)



def field_coordinates_using_shifted(activity_map, activity_map_shifted, visits_map, x_center_bins,y_center_bins,percentile_threshold=95, min_num_of_bins=4, detection_smoothing_sigma=2):
    """
    Identifies and characterizes regions in a spatial field based on shifted field criteria,
    with the center of mass used as the location for each place field.

    Parameters
    ----------
    activity_map : numpy.ndarray
        Original spatial field data.
    activity_map_shifted : numpy.ndarray
        Shifted spatial field data used for threshold calculation.
    visits_map : numpy.ndarray
        Map of visits to locations in the field.
    percentile_threshold : int, optional
        Percentile threshold for identifying regions. Default is 95.
    min_num_of_bins : int, optional
        Minimum number of pixels to constitute a region. Default is 4.

    Returns
    -------
    num_of_islands : int
        Number of identified regions.
    islands_x_com : numpy.ndarray
        X-coordinates of the center of mass in each identified region.
    islands_y_com : numpy.ndarray
        Y-coordinates of the center of mass in each identified region.
    pixels_place_cell_absolute : numpy.ndarray
        Absolute pixel count for each identified region.
    pixels_place_cell_relative : numpy.ndarray
        Relative pixel count for each identified region.
    activity_map_identity : numpy.ndarray
        Identification map for the regions.
    """

    activity_map_smoothed = gaussian_smooth_2d(activity_map, detection_smoothing_sigma)

    # Calculate the threshold using the shifted place field data
    activity_map_threshold = np.percentile(activity_map_shifted, percentile_threshold, 0)

    # Create a binary map of areas above the threshold
    field_above_threshold_binary = np.copy(activity_map)
    field_above_threshold_binary[field_above_threshold_binary <= activity_map_threshold] = 0
    field_above_threshold_binary[field_above_threshold_binary > activity_map_threshold] = 1

    if np.any(field_above_threshold_binary > 0):
        sys.setrecursionlimit(10000000)
        activity_map_identity = identify_islands(np.copy(field_above_threshold_binary))
        num_of_islands_pre = np.unique(activity_map_identity)[1:].shape[0]

        num_of_islands = 0
        for ii in range(1, num_of_islands_pre + 1):
            if np.where(activity_map_identity == ii)[0].shape[0] > min_num_of_bins:
                num_of_islands += 1
            else:
                activity_map_identity[np.where(activity_map_identity == ii)] = 0

        islands_id = np.unique(activity_map_identity[~np.isnan(activity_map_identity)])[1:]

        islands_y_com = []
        islands_x_com = []
        pixels_above = []
        
        for ii in islands_id:
            island_mask = (activity_map_identity == ii)

            x_com, y_com = center_of_mass(island_mask, activity_map_smoothed,x_center_bins,y_center_bins)

            islands_y_com.append(y_com)
            islands_x_com.append(x_com)

            pixels_above.append(np.nansum(island_mask))

        islands_x_com = np.array(islands_x_com)
        islands_y_com = np.array(islands_y_com)
        pixels_above = np.array(pixels_above)

        total_visited_pixels = np.nansum(visits_map != 0)
        pixels_total = activity_map.shape[0] * activity_map.shape[1]

        pixels_place_cell_relative = pixels_above / total_visited_pixels
        pixels_place_cell_absolute = pixels_above / pixels_total

    else:
        num_of_islands = 0
        islands_y_com = np.nan
        islands_x_com = np.nan
        pixels_place_cell_relative = np.nan
        pixels_place_cell_absolute = np.nan
        activity_map_identity = np.nan

    return num_of_islands, islands_x_com, islands_y_com, pixels_place_cell_absolute, pixels_place_cell_relative, correct_island_identifiers(activity_map_identity)


def correct_island_identifiers(island_ids):
    """
    Corrects island identifiers by renumbering them sequentially starting from 0.

    Parameters
    ----------
    island_ids : numpy.ndarray
        Array of island identifiers.

    Returns
    -------
    corrected_ids : numpy.ndarray
        Corrected island identifiers.

    Notes
    -----
    If all island identifiers are NaN, no correction is performed, and the original array is returned.
    Otherwise, unique island identifiers are found, and a new sequential numbering starting from 0
    is applied to the identifiers, correcting them accordingly.
    """
        
    # Check if all island identifiers are NaN; no correction needed.
    if np.all(np.isnan(island_ids)):
        return island_ids

    # Find unique island identifiers.
    unique_ids = np.unique(island_ids)

    # Make a copy of the original island identifiers.
    corrected_ids = np.zeros(island_ids.shape)*np.nan

    # If there is more than one unique identifier, renumber them sequentially starting from 0.
    if len(unique_ids) > 1:
        for new_id, old_id in enumerate(unique_ids[1:]):
            # Update island identifiers based on the new sequential numbering.
            corrected_ids[island_ids == old_id] = new_id

    return corrected_ids


def center_of_mass(island_mask, activity_map_smoothed, x_center_bins, y_center_bins):
    """
    Calculate the center of mass for a given region (island) in the place field,
    taking into account the center bins for x and y coordinates.

    Parameters
    ----------
    island_mask : numpy.ndarray
        A binary mask indicating the pixels belonging to the region (island).
    activity_map_smoothed : numpy.ndarray
        The smoothed place field data.
    x_center_bins : numpy.ndarray
        The x-coordinate bin centers corresponding to the place field grid.
    y_center_bins : numpy.ndarray
        The y-coordinate bin centers corresponding to the place field grid.

    Returns
    -------
    x_com : float
        X-coordinate of the center of mass.
    y_com : float
        Y-coordinate of the center of mass.
    """
    # Get the pixel values of the place field corresponding to the island
    island_values = activity_map_smoothed[island_mask]

    # Get the coordinates of the pixels in the island
    y_coords, x_coords = np.where(island_mask)

    # Calculate the total weight (sum of island values)
    total_weight = np.nansum(island_values)

    # Compute the weighted center of mass using the provided center bins
    if total_weight > 0:
        x_com = np.nansum(x_center_bins[x_coords] * island_values) / total_weight
        y_com = np.nansum(y_center_bins[y_coords] * island_values) / total_weight
    else:
        # If no weight, use the mean of the coordinates
        x_com = np.nanmean(x_center_bins[x_coords])
        y_com = np.nanmean(y_center_bins[y_coords])

    return x_com, y_com


def field_coordinates_using_threshold(activity_map, visits_map, x_center_bins,y_center_bins,detection_smoothing_sigma=1, field_threshold=2, min_num_of_bins=4):
    """
    Identify and characterize spatial regions in a field based on threshold criteria,
    with the center of mass as the place field location.

    Parameters
    ----------
    activity_map : numpy.ndarray
        Input spatial field data.
    visits_map : numpy.ndarray
        Map of visits to locations in the field.
    detection_smoothing_sigma : int, optional
        Size of the smoothing window. Default is 1.
    field_threshold : float, optional
        Threshold value for identifying regions. Default is 2.
    min_num_of_bins : int, optional
        Minimum number of pixels to constitute a region. Default is 4.

    Returns
    -------
    num_of_islands : int
        Number of identified regions.
    islands_x_com : numpy.ndarray
        X-coordinates of the center of mass in each identified region.
    islands_y_com : numpy.ndarray
        Y-coordinates of the center of mass in each identified region.
    pixels_place_cell_absolute : numpy.ndarray
        Absolute pixel count for each identified region.
    pixels_place_cell_relative : numpy.ndarray
        Relative pixel count for each identified region.
    activity_map_identity : numpy.ndarray
        Identification map for the regions.
    """

    activity_map_to_smooth = np.copy(activity_map)
    I_nan = np.isnan(activity_map)
    
    activity_map_to_smooth[np.isnan(activity_map_to_smooth)] = 0
    activity_map_smoothed = gaussian_smooth_2d(activity_map_to_smooth, detection_smoothing_sigma)

    I_threshold = np.nanmean(activity_map_smoothed) + field_threshold * np.nanstd(activity_map_smoothed)

    field_above_threshold_binary = np.copy(activity_map_smoothed)
    field_above_threshold_binary[field_above_threshold_binary < I_threshold] = 0
    field_above_threshold_binary[field_above_threshold_binary >= I_threshold] = 1
    field_above_threshold_binary[I_nan] = 0
    
    if np.any(field_above_threshold_binary > 0):
        sys.setrecursionlimit(10000000)
        activity_map_identity = identify_islands(np.copy(field_above_threshold_binary))
        num_of_islands_pre = np.unique(activity_map_identity)[1:].shape[0]

        island_counter = 0
        for ii in range(1, num_of_islands_pre + 1):
            if np.where(activity_map_identity == ii)[0].shape[0] > min_num_of_bins:
                island_counter += 1
            else:
                activity_map_identity[np.where(activity_map_identity == ii)] = 0

        num_of_islands = island_counter

        islands_id = np.unique(activity_map_identity)[1:]
        islands_y_com = []
        islands_x_com = []
        pixels_above = []
        
        for ii in islands_id:
            # Create a mask for the current island
            island_mask = (activity_map_identity == ii)
            
            # Calculate the center of mass for this island
            x_com, y_com = center_of_mass(island_mask, activity_map_smoothed,x_center_bins,y_center_bins)
            
            islands_y_com.append(y_com)
            islands_x_com.append(x_com)

            pixels_above.append(np.nansum(island_mask))

        islands_x_com = np.array(islands_x_com)
        islands_y_com = np.array(islands_y_com)
        pixels_above = np.array(pixels_above)

        total_visited_pixels = np.nansum(visits_map != 0)
        pixels_total = activity_map_smoothed.shape[0] * activity_map_smoothed.shape[1]

        pixels_place_cell_relative = pixels_above / total_visited_pixels
        pixels_place_cell_absolute = pixels_above / pixels_total

    else:
        num_of_islands = 0
        islands_y_com = np.nan
        islands_x_com = np.nan
        pixels_place_cell_relative = np.nan
        pixels_place_cell_absolute = np.nan
        activity_map_identity = np.nan
        

    return num_of_islands, islands_x_com, islands_y_com, pixels_place_cell_absolute, pixels_place_cell_relative, correct_island_identifiers(activity_map_identity)



def preprocess_signal(input_signal, sampling_rate, signal_type, z_threshold=2):
    filtered_signal = eegfilt(input_signal, sampling_rate, 0, 2, order=2)
    diff_signal = np.hstack([np.diff(filtered_signal), 0])

    if signal_type == 'Raw':
        output_signal = input_signal

    elif signal_type == 'Filtered':
        output_signal = filtered_signal

    elif signal_type == 'Diff':
        diff_signal_truncated = np.copy(diff_signal)
        diff_signal_truncated[diff_signal < 0] = 0
        output_signal = diff_signal_truncated


    elif signal_type == 'Binarized':

        # filtered_signal = eegfilt(input_signal, sampling_rate, 0,2)
        diff_signal = np.hstack([0,np.diff(input_signal)])
        norm_signal = input_signal / np.nanstd(input_signal)
        binarized_idx = (norm_signal >= z_threshold) & (diff_signal > 0)
        binarized_signal = np.zeros(diff_signal.shape[0])
        binarized_signal[binarized_idx] = 1
        output_signal = binarized_signal

    else:
        raise NotImplementedError('Signal type unknown')

    return output_signal


def eegfilt(signal, fs, lowcut, highcut, order = 3):
    """
    Apply a bandpass or lowpass/highpass filter to the input signal.
    
    Parameters:
    - signal: np.ndarray
        Input signal
    - fs: float
        Sampling frequency of the signal
    - lowcut: float
        Low cutoff frequency of the filter (set to 0 for highpass)
    - highcut: float
        High cutoff frequency of the filter (set to 0 for lowpass)
    - order: int, optional
        Filter order (default is 3)
    
    Returns:
    - filtered: np.ndarray
        Filtered signal
    """
    
    # Create a non-NaN mask
    non_nan_mask = ~np.isnan(signal)
    signal_non_nan = signal[non_nan_mask]
    filtered = np.nan*np.zeros(signal.shape[0])

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if low == 0:
        b, a = sig.butter(order, high, btype='low')
        filtered_masked = sig.filtfilt(b, a, signal_non_nan)
    elif high == 0:
        b, a = sig.butter(order, low, btype='high')
        filtered_masked = sig.filtfilt(b, a, signal_non_nan)
    else:
        b, a = sig.butter(order, [low, high], btype='band')
        filtered_masked = sig.filtfilt(b, a, signal_non_nan)

    filtered[non_nan_mask] = filtered_masked


    return filtered





def find_matching_indexes(target_timestamps, reference_time_vector,error_threshold = 0.1):
    """
    Find the indexes in a reference time vector that match target timestamps with some maximum error.
    It will find the closest point in reference_time_vector such that
    reference_time_vector[matching_indexes] - target_timestamps < error_threshold

    Parameters:
        target_timestamps (array-like): Timestamps to match in the reference time vector.
        reference_time_vector (array-like): The reference time vector.

    Returns:
        matching_indexes (array): Indexes in the reference time vector that match the target timestamps.
        
        
    """
    if len(target_timestamps) > 0:
        # Find the indexes in the reference time vector that correspond to target timestamps.
        matching_indexes = search_sorted_indices(reference_time_vector, target_timestamps)
        
        # Filter the indexes to retain only those with a matching error less than 100 ms.
        
        # matching_indexes = [idx for idx in matching_indexes if abs(reference_time_vector[idx] - target_timestamps[idx]) < error_threshold]
        I_keep = np.abs((reference_time_vector[matching_indexes]-target_timestamps)) < error_threshold
        matching_indexes = matching_indexes[I_keep]

    else:
        matching_indexes = []
    
    return matching_indexes

def search_sorted_indices(known_array, test_array):
    """
    Search for the indexes in a sorted known array that correspond to values in a test array.

    Parameters:
        known_array (array-like): The sorted array containing known values.
        test_array (array-like): The array containing values to search for in the known array.

    Returns:
        indices (array): Indexes in the known array that correspond to values in the test array.
    """
    # Sort the known array and calculate the middle values between sorted elements.
    sorted_indices = np.argsort(known_array)
    sorted_known_array = known_array[sorted_indices]
    middle_values = sorted_known_array[1:] - np.diff(sorted_known_array.astype('f')) / 2
    
    # Search for the indexes in the middle values that correspond to values in the test array.
    search_results = np.searchsorted(middle_values, test_array)
    indices = sorted_indices[search_results]
    
    return indices


def searchsorted2(self,known_array, test_array):
    index_sorted = np.argsort(known_array)
    known_array_sorted = known_array[index_sorted]
    known_array_middles = known_array_sorted[1:] - np.diff(known_array_sorted.astype('f'))/2
    idx1 = np.searchsorted(known_array_middles, test_array)
    indices = index_sorted[idx1]
    return indices



def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """
    Detect peaks in data based on amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        Data to search for peaks.
    mph : {None, number}, optional (default=None)
        Minimum peak height. Peaks must exceed this value.
    mpd : int, optional (default=1)
        Minimum peak distance. Peaks must be separated by at least this many data points.
    threshold : float, optional (default=0)
        Minimum difference between a peak and its immediate neighbors.
    edge : {'rising', 'falling', 'both', None}, optional (default='rising')
        Type of edge to detect flat peaks: 'rising', 'falling', 'both', or None.
    kpsh : bool, optional (default=False)
        Keep peaks with the same height even if they are closer than `mpd`.
    valley : bool, optional (default=False)
        If True, detect valleys (local minima) instead of peaks.
    show : bool, optional (default=False)
        If True, plot the data and the detected peaks.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes on which to plot if `show` is True.

    Returns
    -------
    ind : 1D array_like
        Indices of the detected peaks in `x`.

    Notes
    -----
    To detect valleys instead of peaks, the input signal is inverted.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(100)
    >>> peaks = detect_peaks(x, show=True)
    """
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    
    if valley:
        x = -x
    
    dx = np.diff(x)
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf

    # Detect edges
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    if edge in ['rising', 'both']:
        ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
    if edge in ['falling', 'both']:
        ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]

    ind = np.unique(np.hstack((ine, ire, ife)))

    # Exclude indices near NaNs
    if ind.size and indnan.size:
        invalid = np.unique(np.hstack((indnan, indnan - 1, indnan + 1)))
        ind = ind[~np.in1d(ind, invalid)]

    # Filter by minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]

    # Apply threshold
    if ind.size and threshold > 0:
        ind = ind[(x[ind] - np.maximum(x[ind - 1], x[ind + 1])) > threshold]

    # Enforce minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])[::-1]]  # Sort by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                close = (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd)
                if not kpsh:
                    close &= x[ind[i]] > x[ind]
                idel |= close
                idel[i] = False  # Keep current peak
        ind = np.sort(ind[~idel])

    # Optional plotting
    if show:
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(x, label='Data')
        ax.plot(ind, x[ind], 'ro', label='Peaks')
        ax.legend()
        ax.set_title("Detected Peaks")
        plt.show()

    return ind