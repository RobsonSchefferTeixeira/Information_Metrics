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

def get_sparsity(place_field, position_occupancy):
    """
    Calculate the sparsity of a place field with respect to position occupancy.

    Parameters:
    - place_field (numpy.ndarray): A place field map representing spatial preferences.
    - position_occupancy (numpy.ndarray): Positional occupancy map, typically representing time spent in each spatial bin.

    Returns:
    - sparsity (float): The sparsity measure indicating how selective the place field is with respect to position occupancy.

    """
    
    position_occupancy_norm = np.nansum(position_occupancy / np.nansum(position_occupancy))
    sparsity = np.nanmean(position_occupancy_norm * place_field) ** 2 / np.nanmean(
        position_occupancy_norm * place_field ** 2)

    return sparsity

def get_2D_place_field(signal, x_coordinates, y_coordinates, x_grid, y_grid, smoothing_size):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # calculate mean calcium per pixel
    place_field = np.nan * np.zeros((y_grid.shape[0] - 1, x_grid.shape[0] - 1))
    for xx in range(0, x_grid.shape[0] - 1):
        for yy in range(0, y_grid.shape[0] - 1):
            check_x_occupancy = np.logical_and(x_coordinates >= x_grid[xx], x_coordinates < (x_grid[xx + 1]))
            check_y_occupancy = np.logical_and(y_coordinates >= y_grid[yy], y_coordinates < (y_grid[yy + 1]))

            place_field[yy, xx] = np.nanmean(signal[np.logical_and(check_x_occupancy, check_y_occupancy)])

    place_field_to_smooth = np.copy(place_field)
    place_field_to_smooth[np.isnan(place_field_to_smooth)] = 0
    place_field_smoothed = gaussian_smooth_2d(place_field_to_smooth, smoothing_size)

    return place_field, place_field_smoothed




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


def field_coordinates_using_shifted_1D(place_field, place_field_shifted, visits_map, percentile_threshold=95,min_num_of_bins=4):

    """
    Identifies and characterizes place fields using shifted 1D data.

    Parameters:
    - place_field (ndarray): Original place field data.
    - place_field_shifted (ndarray): Shifted place field data.
    - visits_map (ndarray): Map of visited locations.
    - percentile_threshold (int): Percentile threshold for identifying place fields.
    - min_num_of_bins (int): Minimum number of pixels to consider as a place field.

    Returns:
    - num_of_islands (int): Number of identified place field islands.
    - islands_x_max (ndarray): X coordinates of identified islands.
    - pixels_place_cell_absolute (ndarray): Relative pixels of place cells in absolute terms.
    - pixels_place_cell_relative (ndarray): Relative pixels of place cells in relation to visited locations.
    - place_field_identity (ndarray): Identified place field islands labeled.
    """


    place_field_threshold = np.percentile(place_field_shifted, percentile_threshold, 0)

    field_above_threshold_binary = place_field.copy()
    field_above_threshold_binary[field_above_threshold_binary <= place_field_threshold] = 0
    field_above_threshold_binary[field_above_threshold_binary > place_field_threshold] = 1

    if np.any(field_above_threshold_binary > 0):
        sys.setrecursionlimit(10000000)
        place_field_identity = identify_islands_1D(np.copy(field_above_threshold_binary))
        num_of_islands_pre = np.unique(place_field_identity)[1:].shape[0]

        num_of_islands = 0
        for ii in range(1, num_of_islands_pre + 1):
            if np.where(place_field_identity == ii)[0].shape[0] > min_num_of_bins:
                num_of_islands += 1
            else:
                place_field_identity[np.where(place_field_identity == ii)] = 0


        islands_id = np.unique(place_field_identity[~np.isnan(place_field_identity)])[1:]
        islands_x_max = []
        pixels_above = []
        for ii in islands_id:
            max_val = np.nanmax(place_field[(place_field_identity == ii)])
            I_x_max = np.where(place_field == max_val)
            islands_x_max.append(I_x_max[0])

            pixels_above.append(np.nansum(place_field_identity == ii))

        islands_x_max = np.squeeze(islands_x_max)
        pixels_above = np.array(pixels_above)

        total_visited_pixels = np.nansum(visits_map != 0)
        pixels_total = place_field.shape[0]

        pixels_place_cell_relative = pixels_above / total_visited_pixels
        pixels_place_cell_absolute = pixels_above / pixels_total

    else:
        num_of_islands = 0
        islands_x_max = np.nan
        pixels_place_cell_relative = np.nan
        pixels_place_cell_absolute = np.nan
        place_field_identity = np.nan


    return num_of_islands, islands_x_max,pixels_place_cell_absolute,pixels_place_cell_relative,correct_island_identifiers(place_field_identity)





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
    # calculate position occupancy
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

            
        x_grid = np.linspace(environment_edges[0][0], environment_edges[0][1], int((environment_edges[0][1] - environment_edges[0][0]) / x_bin_size) + 1)
        y_grid = np.linspace(environment_edges[1][0], environment_edges[1][1], int((environment_edges[1][1] - environment_edges[1][0]) / y_bin_size) + 1)
    
        x_center_bins = x_grid[:-1] + x_bin_size / 2
        y_center_bins = y_grid[:-1] + y_bin_size / 2
    
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


def get_speed(x_coordinates, y_coordinates, time_vector,sigma_points=1):
    
    if y_coordinates is None:
        distances = np.abs(np.diff(x_coordinates))
    else:
        distances = np.sqrt(np.diff(x_coordinates)**2 + np.diff(y_coordinates)**2)
 
    time_vector_diff = np.diff(time_vector)

    speed = np.divide(distances, time_vector_diff)
    speed = np.hstack([speed, 0])
    speed_smoothed = gaussian_smooth_1d(speed, sigma_points)

    return speed,speed_smoothed

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



def field_coordinates_using_shifted(place_field, place_field_shifted, visits_map, x_center_bins,y_center_bins,percentile_threshold=95, min_num_of_bins=4, detection_smoothing_size=2):
    """
    Identifies and characterizes regions in a spatial field based on shifted field criteria,
    with the center of mass used as the location for each place field.

    Parameters
    ----------
    place_field : numpy.ndarray
        Original spatial field data.
    place_field_shifted : numpy.ndarray
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
    place_field_identity : numpy.ndarray
        Identification map for the regions.
    """

    place_field_smoothed = gaussian_smooth_2d(place_field, detection_smoothing_size)

    # Calculate the threshold using the shifted place field data
    place_field_threshold = np.percentile(place_field_shifted, percentile_threshold, 0)

    # Create a binary map of areas above the threshold
    field_above_threshold_binary = np.copy(place_field)
    field_above_threshold_binary[field_above_threshold_binary <= place_field_threshold] = 0
    field_above_threshold_binary[field_above_threshold_binary > place_field_threshold] = 1

    if np.any(field_above_threshold_binary > 0):
        sys.setrecursionlimit(10000000)
        place_field_identity = identify_islands(np.copy(field_above_threshold_binary))
        num_of_islands_pre = np.unique(place_field_identity)[1:].shape[0]

        num_of_islands = 0
        for ii in range(1, num_of_islands_pre + 1):
            if np.where(place_field_identity == ii)[0].shape[0] > min_num_of_bins:
                num_of_islands += 1
            else:
                place_field_identity[np.where(place_field_identity == ii)] = 0

        islands_id = np.unique(place_field_identity[~np.isnan(place_field_identity)])[1:]

        islands_y_com = []
        islands_x_com = []
        pixels_above = []
        
        for ii in islands_id:
            island_mask = (place_field_identity == ii)

            x_com, y_com = center_of_mass(island_mask, place_field_smoothed,x_center_bins,y_center_bins)

            islands_y_com.append(y_com)
            islands_x_com.append(x_com)

            pixels_above.append(np.nansum(island_mask))

        islands_x_com = np.array(islands_x_com)
        islands_y_com = np.array(islands_y_com)
        pixels_above = np.array(pixels_above)

        total_visited_pixels = np.nansum(visits_map != 0)
        pixels_total = place_field.shape[0] * place_field.shape[1]

        pixels_place_cell_relative = pixels_above / total_visited_pixels
        pixels_place_cell_absolute = pixels_above / pixels_total

    else:
        num_of_islands = 0
        islands_y_com = np.nan
        islands_x_com = np.nan
        pixels_place_cell_relative = np.nan
        pixels_place_cell_absolute = np.nan
        place_field_identity = np.nan

    return num_of_islands, islands_x_com, islands_y_com, pixels_place_cell_absolute, pixels_place_cell_relative, correct_island_identifiers(place_field_identity)


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


def center_of_mass(island_mask, place_field_smoothed, x_center_bins, y_center_bins):
    """
    Calculate the center of mass for a given region (island) in the place field,
    taking into account the center bins for x and y coordinates.

    Parameters
    ----------
    island_mask : numpy.ndarray
        A binary mask indicating the pixels belonging to the region (island).
    place_field_smoothed : numpy.ndarray
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
    island_values = place_field_smoothed[island_mask]

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


def field_coordinates_using_threshold(place_field, visits_map, x_center_bins,y_center_bins,smoothing_size=1, field_threshold=2, min_num_of_bins=4):
    """
    Identify and characterize spatial regions in a field based on threshold criteria,
    with the center of mass as the place field location.

    Parameters
    ----------
    place_field : numpy.ndarray
        Input spatial field data.
    visits_map : numpy.ndarray
        Map of visits to locations in the field.
    smoothing_size : int, optional
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
    place_field_identity : numpy.ndarray
        Identification map for the regions.
    """
      
    place_field_to_smooth = np.copy(place_field)
    I_nan = np.isnan(place_field)
    
    place_field_to_smooth[np.isnan(place_field_to_smooth)] = 0
    place_field_smoothed = gaussian_smooth_2d(place_field_to_smooth, smoothing_size)

    I_threshold = np.nanmean(place_field_smoothed) + field_threshold * np.nanstd(place_field_smoothed)

    field_above_threshold_binary = np.copy(place_field_smoothed)
    field_above_threshold_binary[field_above_threshold_binary < I_threshold] = 0
    field_above_threshold_binary[field_above_threshold_binary >= I_threshold] = 1
    field_above_threshold_binary[I_nan] = 0
    
    if np.any(field_above_threshold_binary > 0):
        sys.setrecursionlimit(10000000)
        place_field_identity = identify_islands(np.copy(field_above_threshold_binary))
        num_of_islands_pre = np.unique(place_field_identity)[1:].shape[0]

        island_counter = 0
        for ii in range(1, num_of_islands_pre + 1):
            if np.where(place_field_identity == ii)[0].shape[0] > min_num_of_bins:
                island_counter += 1
            else:
                place_field_identity[np.where(place_field_identity == ii)] = 0

        num_of_islands = island_counter

        islands_id = np.unique(place_field_identity)[1:]
        islands_y_com = []
        islands_x_com = []
        pixels_above = []
        
        for ii in islands_id:
            # Create a mask for the current island
            island_mask = (place_field_identity == ii)
            
            # Calculate the center of mass for this island
            x_com, y_com = center_of_mass(island_mask, place_field_smoothed,x_center_bins,y_center_bins)
            
            islands_y_com.append(y_com)
            islands_x_com.append(x_com)

            pixels_above.append(np.nansum(island_mask))

        islands_x_com = np.array(islands_x_com)
        islands_y_com = np.array(islands_y_com)
        pixels_above = np.array(pixels_above)

        total_visited_pixels = np.nansum(visits_map != 0)
        pixels_total = place_field_smoothed.shape[0] * place_field_smoothed.shape[1]

        pixels_place_cell_relative = pixels_above / total_visited_pixels
        pixels_place_cell_absolute = pixels_above / pixels_total

    else:
        num_of_islands = 0
        islands_y_com = np.nan
        islands_x_com = np.nan
        pixels_place_cell_relative = np.nan
        pixels_place_cell_absolute = np.nan
        place_field_identity = np.nan
        

    return num_of_islands, islands_x_com, islands_y_com, pixels_place_cell_absolute, pixels_place_cell_relative, correct_island_identifiers(place_field_identity)



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

        filtered_signal = eegfilt(input_signal, sampling_rate, 0,2)
        diff_signal = np.hstack([0,np.diff(filtered_signal)])
        norm_signal = filtered_signal / np.nanstd(filtered_signal)
        binarized_idx = (norm_signal > z_threshold) & (diff_signal > 0)
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



def smooth(x, window_len=11, window='hanning'):
    """
    Smooth the data using a window with the requested size.
    
    This method involves the convolution of a scaled window with the input signal.
    The input signal is prepared by introducing reflected copies at both ends,
    reducing transient effects at the beginning and end of the output signal.
    
    Parameters:
    - x: numpy.ndarray
        The input signal

    - window_len: int
        The size of the smoothing window; should be an odd integer

    - window: str
        The type of window used for smoothing; options: 'flat', 'hanning', 'hamming',
        'bartlett', 'blackman'. A flat window produces a moving average smoothing.
    
    Returns:
    - smoothed_signal: numpy.ndarray
        Tthe smoothed signal
    
    Example:
    t = np.linspace(-2, 2, 0.1)
    x = np.sin(t) + np.random.randn(len(t)) * 0.1
    y = smooth(x)
        
    Note:
    - The length of output is not equal to the input length. To correct this, use:
      return y[(window_len//2-1):-(window_len//2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1-dimensional arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be larger than window size.")
    if window_len < 3:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window should be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]

    if window == 'flat':  # Moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)  # Using getattr to retrieve window function by name

    y = np.convolve(w / w.sum(), s, mode='valid')

    return y[int(window_len / 2 - 1):-int(window_len / 2)]



def gaussian_smooth_1d(input_data, sigma_points):
    """
    Perform 1D Gaussian smoothing on input data.
    Notice that when using it for time series, sigma_points are set in poins, not time.
    In order to set the correct amount of points that correspond to ms, for instance,
    one should use it like this: sigma_points = (s/1000)*sampling_rate
    where s in the standard deviation in ms.

    Parameters:
        input_data (numpy.ndarray): The 1D input data to be smoothed.
        sigma_points (float): The standard deviation of the Gaussian kernel in data points.

    Returns:
        smoothed_data (numpy.ndarray): The smoothed 1D data.
    """
    # Generate a 1D Gaussian kernel.
    gaussian_kernel_1d = generate_1d_gaussian_kernel(sigma_points)

    # Convolve the input data with the Gaussian kernel.
    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data,0)
    # smoothed_data = sig.convolve(input_data, gaussian_kernel_1d, mode='same')
    smoothed_data = np.apply_along_axis(lambda x: sig.convolve(x, gaussian_kernel_1d, mode='same'), axis=1, arr=input_data)

    return np.squeeze(smoothed_data)


def generate_1d_gaussian_kernel(sigma):
    """
    Generate a 1D Gaussian kernel with a specified standard deviation.

    Parameters:
        sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
        gaussian_kernel (numpy.ndarray): The 1D Gaussian kernel.
    """
    x_values = np.arange(-3.0 * sigma, 3.0 * sigma + 1.0)
    constant = 1 / (np.sqrt(2 * math.pi) * sigma)
    gaussian_kernel = constant * np.exp(-((x_values**2) / (2 * (sigma**2))))

    return gaussian_kernel


def gaussian_smooth_2d(input_matrix, sigma_points):
    """
    Perform 2D Gaussian smoothing on input data.

    Parameters:
        input_matrix (numpy.ndarray): The 2D input matrix to be smoothed.
        sigma_points (float): The standard deviation of the 2D Gaussian kernel in data points.

    Returns:
        smoothed_matrix (numpy.ndarray): The smoothed 2D data.
    """
    # Generate a 2D Gaussian kernel.
    gaussian_kernel_2d = generate_2d_gaussian_kernel(sigma_points)

    # Convolve the input matrix with the 2D Gaussian kernel.
    smoothed_matrix = sig.convolve2d(input_matrix, gaussian_kernel_2d, mode='same')

    return smoothed_matrix

def generate_2d_gaussian_kernel(sigma):
    """
    Generate a 2D Gaussian kernel with a specified standard deviation.

    Parameters:
        sigma (float): The standard deviation of the 2D Gaussian kernel.

    Returns:
        gaussian_kernel (numpy.ndarray): The 2D Gaussian kernel.
    """
    x_values = np.arange(-3.0 * sigma, 3.0 * sigma + 1.0)
    y_values = np.arange(-3.0 * sigma, 3.0 * sigma + 1.0)
    
    gaussian_kernel = np.zeros([y_values.shape[0], x_values.shape[0]])
    
    for x_count, x_val in enumerate(x_values):
        for y_count, y_val in enumerate(y_values):
            gaussian_kernel[y_count, x_count] = np.exp(-((x_val**2 + y_val**2) / (2 * (sigma**2))))

    return gaussian_kernel


def min_max_norm(input_signal, axis=0, custom_min=0, custom_max=1):
    """
    Perform min-max normalization on an input signal with the option to set custom minimum and maximum values.

    Parameters:
        input_signal (numpy.ndarray): The input signal to be normalized.
        axis (int, optional): The axis along which normalization is performed (default is 0).
        custom_min (float, optional): Custom minimum value for normalization (default is 0).
        custom_max (float, optional): Custom maximum value for normalization (default is 1).

    Returns:
        scaled_signal (numpy.ndarray): The min-max normalized signal within the custom range.
    """
    # Calculate the minimum and maximum values of the input signal along the specified axis,
    # handling NaN values.
    min_value = np.nanmin(input_signal, axis=axis, keepdims=True)
    max_value = np.nanmax(input_signal, axis=axis, keepdims=True)

    # Perform min-max normalization with custom minimum and maximum values.
    scaled_signal = (((custom_max - custom_min) * (input_signal - min_value)) / 
                     (max_value - min_value)) + custom_min

    return scaled_signal


def z_score_norm(input_matrix, axis=0):
    """
    Perform z-score normalization on an input matrix.

    Parameters:
        input_matrix (numpy.ndarray): The input matrix to be normalized.
        axis (int, optional): The axis along which normalization is performed (default is 0).

    Returns:
        z_scored_matrix (numpy.ndarray): The z-score normalized matrix.
    """
    # Calculate the mean and standard deviation along the specified axis, handling NaN values.
    mean = np.nanmean(input_matrix, axis=axis, keepdims=True)
    std = np.nanstd(input_matrix, axis=axis, keepdims=True)

    # Check for cases where all values in std are zero to avoid division by zero.
    all_same_values = np.all(std == 0, axis=axis, keepdims=True)

    # If all values in std are zero, set std to 1 to prevent division by zero.
    std[all_same_values] = 1

    # Calculate the z-scored matrix using the formula (input_matrix - mean) / std.
    z_scored_matrix = (input_matrix - mean) / std

    # Return the z-scored matrix.
    return z_scored_matrix


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