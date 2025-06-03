import numpy as np
import os
import sys
from scipy import interpolate
import numpy as np
from scipy import signal as sig
import math
import warnings
import src.utils.smoothing_functions as smooth











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


def correct_coordinates(x_coordinates, y_coordinates, environment_edges):
    """
    Set coordinates outside of defined environment edges to NaN.

    Parameters:
    - x_coordinates: 1D array of x values
    - y_coordinates: 1D array of y values (or None for 1D use)
    - environment_edges: tuple of ((x_min, x_max), (y_min, y_max))

    Returns:
    - x_coordinates: corrected x values (with NaNs where needed)
    - y_coordinates: corrected y values (or None)
    """

    # Copy to avoid modifying input in-place
    x_coordinates = x_coordinates.copy()

    if y_coordinates is None:
        # 1D case: only check x bounds
        x_min, x_max = environment_edges[0]
        out_of_bounds = (x_coordinates < x_min) | (x_coordinates > x_max)
        x_coordinates[out_of_bounds] = np.nan
        return x_coordinates, None

    # 2D case: check both x and y bounds
    x_coordinates = x_coordinates.copy()
    y_coordinates = y_coordinates.copy()

    x_min, x_max = environment_edges[0]
    y_min, y_max = environment_edges[1]

    x_oob = (x_coordinates < x_min) | (x_coordinates > x_max)
    y_oob = (y_coordinates < y_min) | (y_coordinates > y_max)

    # Set NaN where either coordinate is out of bounds
    nan_mask = x_oob | y_oob | np.isnan(x_coordinates) | np.isnan(y_coordinates)

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



def get_activity_map(signal, x_coordinates, x_grid, sigma_x=1.0, y_coordinates=None, y_grid=None, sigma_y=None):
    """
    Computes a spatial activity map (mean signal per spatial bin) and applies Gaussian smoothing.
    Supports both 1D (x only) and 2D (x and y) coordinates.

    Parameters
    ----------
    signal : ndarray
        1D array of signal values (e.g., dF/F, spikes, etc.), same length as coordinate arrays.
    x_coordinates : ndarray
        1D array of x-coordinates of the animal's position.
    x_grid : ndarray
        Bin edges along the x-axis.
    sigma_x : float
        Smoothing standard deviation along x-axis, in spatial units (same as x_grid).
    y_coordinates : ndarray, optional
        1D array of y-coordinates of the animal's position. If None, function runs in 1D mode.
    y_grid : ndarray, optional
        Bin edges along the y-axis. Required if y_coordinates is provided.
    sigma_y : float, optional
        Smoothing standard deviation along y-axis, in spatial units (same as y_grid). If None and y_coordinates is provided, uses sigma_x.

    Returns
    -------
    activity_map : ndarray
        Raw (unsmoothed) activity map: 1D if y_coordinates is None, 2D otherwise.
    activity_map_smoothed : ndarray
        Smoothed activity map using Gaussian kernel.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if y_coordinates is None:
        # -------- 1D MODE --------
        activity_map = np.full(len(x_grid) - 1, np.nan)

        for xx in range(len(x_grid) - 1):
            in_x_bin = np.logical_and(x_coordinates >= x_grid[xx], x_coordinates < x_grid[xx + 1])
            activity_map[xx] = np.nanmean(signal[in_x_bin])

        # Smoothing
        sigma_x_points = smooth.get_sigma_points(sigma_x, x_grid)
        kernel, x_mesh = smooth.generate_1d_gaussian_kernel(sigma_x_points, truncate=4.0)
        activity_map_smoothed = smooth.gaussian_smooth_1d(activity_map, kernel, handle_nans=False)

    else:
        # -------- 2D MODE --------
        if sigma_y is None:
            sigma_y = sigma_x

        activity_map = np.full((len(y_grid) - 1, len(x_grid) - 1), np.nan)

        for xx in range(len(x_grid) - 1):
            for yy in range(len(y_grid) - 1):
                in_x_bin = np.logical_and(x_coordinates >= x_grid[xx], x_coordinates < x_grid[xx + 1])
                in_y_bin = np.logical_and(y_coordinates >= y_grid[yy], y_coordinates < y_grid[yy + 1])
                in_bin = np.logical_and(in_x_bin, in_y_bin)

                activity_map[yy, xx] = np.nanmean(signal[in_bin])

        # Smoothing
        sigma_x_points = smooth.get_sigma_points(sigma_x, x_grid)
        sigma_y_points = smooth.get_sigma_points(sigma_y, y_grid)

        kernel, (x_mesh, y_mesh) = smooth.generate_2d_gaussian_kernel(
            sigma_x_points, sigma_y_points, radius_x=None, radius_y=None, truncate=4.0
        )
        activity_map_smoothed = smooth.gaussian_smooth_2d(activity_map, kernel, handle_nans=False)

    # Taking the mean is equivalent to:
    # activity_map[yy, xx] = np.nansum(input_signal[np.logical_and(check_x_occupancy, check_y_occupancy)])
    # and then normalize by the number of points inside each bin:
    # activity_map_normalized = activity_map/(position_occupancy*sampling_rate)
    # one must multiply by sampling_rate because position_occupancy is the time spent inside each bin

    return activity_map, activity_map_smoothed


def get_visits(x_coordinates, position_binned, x_center_bins, y_coordinates=None, y_center_bins=None):
    """
    Identify new visits to spatial bins in either 1D or 2D space.

    Parameters:
    ----------
    x_coordinates : array_like
        X coordinates of the animal's trajectory (1D array).
    position_binned : array_like
        Pre-binned position data (e.g., bin ID per timepoint).
    x_center_bins : array_like
        Centers of spatial bins along the X-axis.
    y_coordinates : array_like, optional
        Y coordinates of the trajectory (for 2D case). Default is None.
    y_center_bins : array_like, optional
        Centers of spatial bins along the Y-axis (for 2D case). Default is None.

    Returns:
    -------
    visits_bins : np.ndarray
        Array of the same shape as `position_binned`, with visit counts per bin.
    new_visits_times : np.ndarray
        Binary array (0 or 1) indicating where new visits start.
    """
    I_x_coord, I_y_coord = [], []
    is_2d = y_coordinates is not None and y_center_bins is not None

    for i in range(len(x_coordinates)):
        x = x_coordinates[i]
        y = y_coordinates[i] if is_2d else None

        if np.isnan(x) or (is_2d and np.isnan(y)):
            I_x_coord.append(np.nan)
            if is_2d:
                I_y_coord.append(np.nan)
        else:
            I_x_coord.append(np.nanargmin(np.abs(x - x_center_bins)))
            if is_2d:
                I_y_coord.append(np.nanargmin(np.abs(y - y_center_bins)))

    I_x_coord = np.array(I_x_coord)
    dx = np.diff(np.hstack([I_x_coord[0] - 1, I_x_coord]))

    if is_2d:
        I_y_coord = np.array(I_y_coord)
        dy = np.diff(np.hstack([I_y_coord[0] - 1, I_y_coord]))
        new_visits_times = ((dx != 0) | (dy != 0)) & (~np.isnan(dx) & ~np.isnan(dy))
    else:
        new_visits_times = (dx != 0) & (~np.isnan(dx))

    visits_id, visits_counts = np.unique(position_binned[new_visits_times], return_counts=True)

    visits_bins = np.full(position_binned.shape, np.nan)
    for i in range(len(visits_id)):
        if not np.isnan(visits_id[i]):
            mask = position_binned == visits_id[i]
            visits_bins[mask] = visits_counts[i]

    return visits_bins, new_visits_times.astype(int)


def get_visits_occupancy(x_coordinates, new_visits_times, x_grid, y_coordinates=None, y_grid=None):
    """
    Count number of visits per spatial bin based on visit timepoints.

    Parameters:
    ----------
    x_coordinates : array_like
        X coordinates of the animal's trajectory (1D array).
    new_visits_times : array_like
        Binary array indicating timepoints where new visits occur.
    x_grid : array_like
        Bin edges along the X-axis.
    y_coordinates : array_like, optional
        Y coordinates of the trajectory (for 2D case). Default is None.
    y_grid : array_like, optional
        Bin edges along the Y-axis (for 2D case). Default is None.

    Returns:
    -------
    visits_occupancy : np.ndarray
        Array with shape (n_bins_x,) for 1D or (n_bins_y, n_bins_x) for 2D,
        containing visit counts per bin.
    """
    I_visit = np.where(new_visits_times > 0)[0]
    x_coordinate_visit = x_coordinates[I_visit]
    is_2d = y_coordinates is not None and y_grid is not None

    if is_2d:
        y_coordinate_visit = y_coordinates[I_visit]
        visits_occupancy = np.full((y_grid.shape[0] - 1, x_grid.shape[0] - 1), np.nan)

        for xx in range(x_grid.shape[0] - 1):
            for yy in range(y_grid.shape[0] - 1):
                in_x = (x_coordinate_visit >= x_grid[xx]) & (x_coordinate_visit < x_grid[xx + 1])
                in_y = (y_coordinate_visit >= y_grid[yy]) & (y_coordinate_visit < y_grid[yy + 1])
                visits_occupancy[yy, xx] = np.sum(in_x & in_y)
    else:
        visits_occupancy = np.full(x_grid.shape[0] - 1, np.nan)
        for xx in range(x_grid.shape[0] - 1):
            in_bin = (x_coordinate_visit >= x_grid[xx]) & (x_coordinate_visit < x_grid[xx + 1])
            visits_occupancy[xx] = np.sum(in_bin)

    return visits_occupancy


def get_position_time_spent(position_binned, sampling_rate):
    """
    Compute the time spent at each spatial bin over time.

    Parameters:
    ----------
    position_binned : array_like
        1D array indicating the bin index of the animal's position at each timepoint.
        NaNs are expected where tracking failed.
    sampling_rate : float
        Sampling rate of the position signal, in Hz (samples per second).

    Returns:
    -------
    time_spent_inside_bins : np.ndarray
        1D array (same shape as `position_binned`) where each timepoint is
        assigned the total time (in seconds) spent in the corresponding bin.
        NaN for timepoints with undefined bin (e.g. NaNs in input).
    """

    positions_id, positions_counts = np.unique(position_binned, return_counts=True)
    time_spent_inside_bins = np.zeros(position_binned.shape) * np.nan

    for ids in range(positions_id.shape[0]):
        if ~np.isnan(positions_id[ids]):
            I_pos = position_binned == positions_id[ids]
            # Time spent in this bin = count of samples / sampling rate
            time_spent_inside_bins[I_pos] = positions_counts[ids] / sampling_rate

    return time_spent_inside_bins

def get_occupancy(x_coordinates, x_grid, sampling_rate, y_coordinates=None, y_grid=None):
    """
    Calculate position occupancy — the time spent in each spatial bin.

    Parameters
    ----------
    x_coordinates : ndarray
        Array of x-coordinates (1D or 2D tracking).
    x_grid : ndarray
        Array defining the x-bin edges.
    sampling_rate : float
        Sampling rate of the tracking signal, in Hz.
    y_coordinates : ndarray, optional
        Array of y-coordinates for 2D tracking. If None, assumes 1D.
    y_grid : ndarray, optional
        Array defining the y-bin edges for 2D tracking. Required if y_coordinates is given.

    Returns
    -------
    position_occupancy : ndarray
        Array of occupancy values. Shape is:
        - (n_bins_x,) for 1D
        - (n_bins_y, n_bins_x) for 2D
        Each value is in seconds.
    """

    if y_coordinates is None or y_grid is None:
        # Handle 1D case
        position_occupancy = np.zeros(len(x_grid) - 1)
        for xx in range(len(x_grid) - 1):
            check_x = np.logical_and(x_coordinates >= x_grid[xx], x_coordinates < x_grid[xx + 1])
            position_occupancy[xx] = np.nansum(check_x) / sampling_rate
    else:
        # Handle 2D case
        position_occupancy = np.zeros((len(y_grid) - 1, len(x_grid) - 1))
        for xx in range(len(x_grid) - 1):
            for yy in range(len(y_grid) - 1):
                check_x = np.logical_and(x_coordinates >= x_grid[xx], x_coordinates < x_grid[xx + 1])
                check_y = np.logical_and(y_coordinates >= y_grid[yy], y_coordinates < y_grid[yy + 1])
                position_occupancy[yy, xx] = np.nansum(np.logical_and(check_x, check_y)) / sampling_rate

    # Optionally uncomment below to ignore empty bins:
    # position_occupancy[position_occupancy == 0] = np.nan

    return position_occupancy


def get_speed_occupancy(speed, x_coordinates, x_grid, y_coordinates=None, y_grid=None):
    """
    Compute the average speed within each spatial bin (1D or 2D).

    Parameters
    ----------
    speed : ndarray
        Array of instantaneous speed values (same length as x/y_coordinates).
    x_coordinates : ndarray
        Array of x-coordinates.
    x_grid : ndarray
        Array defining the x-bin edges.
    y_coordinates : ndarray, optional
        Array of y-coordinates for 2D tracking. If None, assumes 1D.
    y_grid : ndarray, optional
        Array defining the y-bin edges for 2D tracking. Required if y_coordinates is given.

    Returns
    -------
    speed_occupancy : ndarray
        Array of average speed per bin:
        - (n_bins_x,) for 1D
        - (n_bins_y, n_bins_x) for 2D
        Each value is the mean speed within that spatial bin.
    """

    if y_coordinates is None or y_grid is None:
        # Handle 1D case
        speed_occupancy = np.zeros(len(x_grid) - 1) * np.nan
        for xx in range(len(x_grid) - 1):
            check_x = np.logical_and(x_coordinates >= x_grid[xx], x_coordinates < x_grid[xx + 1])
            speed_occupancy[xx] = np.nanmean(speed[check_x])
    else:
        # Handle 2D case
        speed_occupancy = np.zeros((len(y_grid) - 1, len(x_grid) - 1)) * np.nan
        for xx in range(len(x_grid) - 1):
            for yy in range(len(y_grid) - 1):
                check_x = np.logical_and(x_coordinates >= x_grid[xx], x_coordinates < x_grid[xx + 1])
                check_y = np.logical_and(y_coordinates >= y_grid[yy], y_coordinates < y_grid[yy + 1])
                mask = np.logical_and(check_x, check_y)
                speed_occupancy[yy, xx] = np.nanmean(speed[mask])

    return speed_occupancy


def get_position_grid(x_coordinates, y_coordinates=None, x_bin_size=1, y_bin_size=None, environment_edges=None):
    """
    Generate bin edges and centers for position data in 1D or 2D environments.

    Parameters:
    - x_coordinates (array-like): X position data.
    - y_coordinates (array-like or None): Y position data. If None, assumes 1D.
    - x_bin_size (float): Bin size in cm for x-axis.
    - y_bin_size (float or None): Bin size in cm for y-axis. Required for 2D.
    - environment_edges (list or None): [[x_min, x_max], [y_min, y_max]]. If None, estimated from data.

    Returns:
    - x_grid (np.ndarray): Edges of x bins.
    - y_grid (np.ndarray or np.nan): Edges of y bins (or nan if 1D).
    - x_center_bins (np.ndarray): Centers of x bins.
    - y_center_bins (np.ndarray or np.nan): Centers of y bins (or nan if 1D).
    - x_center_bins_repeated (np.ndarray): Flattened meshgrid x bin centers (for 2D decoding).
    - y_center_bins_repeated (np.ndarray or np.nan): Flattened meshgrid y bin centers (or nan if 1D).
    """
    x_coordinates = np.asarray(x_coordinates)

    if y_coordinates is None:
        # --- 1D case ---
        if environment_edges is None:
            x_min, x_max = np.nanmin(x_coordinates), np.nanmax(x_coordinates)
            environment_edges = [[x_min, x_max]]

        x_grid = np.linspace(environment_edges[0][0], environment_edges[0][1],
                             int((environment_edges[0][1] - environment_edges[0][0]) / x_bin_size) + 1)
        x_center_bins = x_grid[:-1] + x_bin_size / 2
        x_center_bins_repeated = np.repeat(x_center_bins, len(x_center_bins))

        y_grid = np.nan
        y_center_bins = np.nan
        y_center_bins_repeated = np.nan

    else:
        # --- 2D case ---
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        y_coordinates = np.asarray(y_coordinates)

        if environment_edges is None:
            x_min, x_max = np.nanmin(x_coordinates), np.nanmax(x_coordinates)
            y_min, y_max = np.nanmin(y_coordinates), np.nanmax(y_coordinates)
            environment_edges = [[x_min, x_max], [y_min, y_max]]

        if y_bin_size is None:
            warnings.warn("y_bin_size not specified, using x_bin_size for y-axis.")
            y_bin_size = x_bin_size

        x_min, x_max = environment_edges[0]
        y_min, y_max = environment_edges[1]

        x_range = x_max - x_min
        y_range = y_max - y_min

        num_x_bins = max(int(np.ceil(x_range / x_bin_size)), 2)
        num_y_bins = max(int(np.ceil(y_range / y_bin_size)), 2)

        x_grid = np.linspace(x_min, x_max, num_x_bins + 1)
        y_grid = np.linspace(y_min, y_max, num_y_bins + 1)

        x_center_bins = x_grid[:-1] + np.diff(x_grid) / 2
        y_center_bins = y_grid[:-1] + np.diff(y_grid) / 2

        x_center_bins_repeated = np.repeat(x_center_bins, len(y_center_bins))
        y_center_bins_repeated = np.tile(y_center_bins, len(x_center_bins))

    return (
        x_grid,
        y_grid,
        x_center_bins,
        y_center_bins,
        x_center_bins_repeated,
        y_center_bins_repeated,
    )


def get_binned_position(x_coordinates, x_grid, y_coordinates=None, y_grid=None):
    """
    Bin 1D or 2D position data and return bin IDs and center coordinates.

    Parameters:
    - x_coordinates: 1D array of x position values
    - x_grid: 1D array of x bin edges
    - y_coordinates: 1D array of y position values (optional, for 2D)
    - y_grid: 1D array of y bin edges (optional, for 2D)

    Returns:
    - position_binned: 1D array with unique bin ID per position
    - bin_coordinates: array with bin centers per position
        - shape (n,) with x_center values in 1D
        - shape (n, 2) with [x_center, y_center] in 2D
    """
    position_binned = np.full(x_coordinates.shape, np.nan)

    if y_coordinates is None:
        # 1D binning
        bin_coordinates = np.full(x_coordinates.shape, np.nan)
        count = 0

        for xx in range(x_grid.shape[0] - 1):
            x_start = x_grid[xx]
            x_end = x_grid[xx + 1]
            x_center = (x_start + x_end) / 2

            if xx == x_grid.shape[0] - 2:
                check_x = (x_coordinates >= x_start) & (x_coordinates <= x_end)
            else:
                check_x = (x_coordinates >= x_start) & (x_coordinates < x_end)

            position_binned[check_x] = count
            bin_coordinates[check_x] = x_center
            count += 1

    else:
        # 2D binning
        bin_coordinates = np.full((x_coordinates.shape[0], 2), np.nan)
        count = 0

        for xx in range(x_grid.shape[0] - 1):
            x_start = x_grid[xx]
            x_end = x_grid[xx + 1]
            x_center = (x_start + x_end) / 2

            check_x = x_coordinates >= x_start
            if xx == x_grid.shape[0] - 2:
                check_x &= x_coordinates <= x_end
            else:
                check_x &= x_coordinates < x_end

            for yy in range(y_grid.shape[0] - 1):
                y_start = y_grid[yy]
                y_end = y_grid[yy + 1]
                y_center = (y_start + y_end) / 2

                check_y = y_coordinates >= y_start
                if yy == y_grid.shape[0] - 2:
                    check_y &= y_coordinates <= y_end
                else:
                    check_y &= y_coordinates < y_end

                in_bin = check_x & check_y
                position_binned[in_bin] = count
                bin_coordinates[in_bin, 0] = x_center
                bin_coordinates[in_bin, 1] = y_center
                count += 1

    return position_binned, bin_coordinates

def get_speed(x_coords, y_coords, time_vector, speed_smoothing_sigma=1):
    """
    Calculate instantaneous speed from position coordinates and time vector,
    handling both 1D and 2D positions. Applies Gaussian smoothing.

    Parameters:
    - x_coords (array-like): X-coordinate positions (or 1D positions if y_coords is None).
    - time_vector (array-like): Time stamps corresponding to the positions.
    - y_coords (array-like or None): Y-coordinate positions. If None, assumes 1D data.
    - speed_smoothing_sigma (float): Std deviation for Gaussian smoothing. Default is 1.

    Returns:
    - speed (np.ndarray): Instantaneous speed (not smoothed).
    - smoothed_speed (np.ndarray): Smoothed speed.
    """
    x_coords = np.asarray(x_coords)
    time_vector = np.asarray(time_vector)

    if y_coords is None:
        distances = np.abs(np.diff(x_coords))
    else:
        y_coords = np.asarray(y_coords)
        delta_x = np.diff(x_coords)
        delta_y = np.diff(y_coords)
        distances = np.sqrt(delta_x**2 + delta_y**2)

    time_intervals = np.diff(time_vector)
    if np.any(time_intervals <= 0):
        raise ValueError("Time vector must be strictly increasing.")

    speed = distances / time_intervals
    speed = np.append(speed, 0)  # Pad to match original array length

    sigma_points = smooth.get_sigma_points(speed_smoothing_sigma, time_vector)
    kernel, _ = smooth.generate_1d_gaussian_kernel(sigma_points, radius=None, truncate=4.0)
    smoothed_speed = smooth.gaussian_smooth_1d(speed, kernel, handle_nans=False)

    return speed, smoothed_speed


def correct_lost_tracking(x_coordinates, y_coordinates, track_timevector, sampling_rate, min_epoch_length=0.5):
    """
    Interpolates short lost-tracking (NaN) periods in position data.
    Works for both 1D (x only) and 2D (x and y) data.

    Parameters:
    - x_coordinates: array of x positions
    - y_coordinates: array of y positions or None (for 1D)
    - track_timevector: array of time points
    - sampling_rate: data sampling rate in Hz
    - min_epoch_length: max duration (in seconds) of NaN epoch to interpolate

    Returns:
    - x_coordinates_interpolated
    - y_coordinates_interpolated (or None)
    """

    x_coordinates_interpolated = x_coordinates.copy()
    y_coordinates_interpolated = y_coordinates.copy() if y_coordinates is not None else None

    # Define NaNs for interpolation
    if y_coordinates is None:
        nan_mask = np.isnan(x_coordinates)
    else:
        nan_mask = np.isnan(x_coordinates) | np.isnan(y_coordinates)

    # Indices of NaNs
    I_nan_events = np.where(nan_mask)[0]

    if len(I_nan_events) == 0:
        return x_coordinates_interpolated, y_coordinates_interpolated

    # Find starts of contiguous NaN groups
    split_points = np.where(np.diff(I_nan_events) > 1)[0] + 1
    groups = np.split(I_nan_events, split_points)

    for group in groups:
        if len(group) == 0:
            continue

        epoch_length = len(group) / sampling_rate
        if epoch_length <= min_epoch_length:
            window_beg = max(group[0] - 1, 0)
            window_end = min(group[-1] + 1, len(track_timevector) - 1)

            if window_end <= window_beg:
                continue  # skip if interpolation window is invalid

            time_window = track_timevector[window_beg:window_end + 1]

            # Interpolate x
            x_orig = track_timevector[[window_beg, window_end]]
            y_orig = x_coordinates[[window_beg, window_end]]
            interp_func = interpolate.interp1d(x_orig, y_orig, kind='slinear')
            x_coordinates_interpolated[window_beg:window_end + 1] = interp_func(time_window)

            # Interpolate y if available
            if y_coordinates is not None:
                y_orig = y_coordinates[[window_beg, window_end]]
                interp_func = interpolate.interp1d(x_orig, y_orig, kind='slinear')
                y_coordinates_interpolated[window_beg:window_end + 1] = interp_func(time_window)

    return x_coordinates_interpolated, y_coordinates_interpolated


def filename_constructor(saving_string, animal_id=None, dataset=None, day=None, neuron=None, trial=None):
    """
    Construct a filename by concatenating provided components with specific prefixes.

    Parameters:
        saving_string (str): Base string for the filename.
        animal_id (str, optional): Identifier for the animal.
        dataset (str, optional): Dataset name.
        day (int or str, optional): Day identifier.
        neuron (int or str, optional): Neuron identifier.
        trial (int or str, optional): Trial identifier.

    Returns:
        str: Constructed filename.
    """
    parts = [saving_string]

    if animal_id is not None:
        parts.append(f".{animal_id}")
    if dataset is not None:
        parts.append(f".Dataset.{dataset}")
    if day is not None:
        parts.append(f".Day.{day}")
    if neuron is not None:
        parts.append(f".Neuron.{neuron}")
    if trial is not None:
        parts.append(f".Trial.{trial}")

    return ''.join(parts)

'''
def caller_saving(inputdict, filename, saving_path):
    os.chdir(saving_path)
    output = open(filename, 'wb')
    np.save(output, inputdict)
    output.close()
    print('File saved.')
'''

def caller_saving(inputdict, filename, saving_path, overwrite=False):
    """
    Save a dictionary to a .npy file, with an option to overwrite existing files.

    Parameters:
        inputdict (dict): The dictionary to save.
        filename (str): The name of the file to save.
        saving_path (str): The directory path where the file will be saved.
        overwrite (bool): If True, overwrite the file if it exists. Default is False.

    Returns:
        None
    """
    # Ensure the saving path exists
    os.makedirs(saving_path, exist_ok=True)

    # Construct the full file path
    full_path = os.path.join(saving_path, filename)

    # Check if the file exists and handle based on the overwrite flag
    if os.path.exists(full_path):
        if overwrite:
            print(f"Overwriting existing file: {full_path}")
        else:
            print(f"File already exists and overwrite is set to False: {full_path}")
            return

    # Save the dictionary to the .npy file
    with open(full_path, 'wb') as output:
        np.save(output, inputdict)

    print(f"File saved at: {full_path}")


def identify_islands(input_array):
    """
    Identifies islands (connected components of 1s) in a binary array, 
    supporting both 1D and 2D input. NaNs are preserved and ignored during labeling.

    Parameters
    ----------
    input_array : numpy.ndarray
        Input array containing 1s for islands, 0s for background, NaNs ignored.

    Returns
    -------
    labeled_array : numpy.ndarray
        Array of the same shape with islands labeled sequentially (NaNs preserved).
    """
    if input_array.ndim == 1:
        return _identify_islands_1d(input_array)
    elif input_array.ndim == 2:
        return _identify_islands_2d(input_array)
    else:
        raise ValueError("Input array must be 1D or 2D.")

# ==============================
#        2D Island Finder
# ==============================

def _identify_islands_2d(input_array):
    """
    Identifies connected components (4-neighbor islands) in a 2D binary array.
    """
    working_array = np.where(np.isnan(input_array), 0, input_array.copy())
    labeled_array = np.full_like(input_array, 0, dtype=float)
    count = 0

    rows, cols = input_array.shape

    for i in range(rows):
        for j in range(cols):
            if not np.isnan(input_array[i, j]) and working_array[i, j] == 1:
                count += 1
                _dfs_2d(working_array, labeled_array, count, rows, cols, i, j)

    # Restore NaNs from the original input
    labeled_array = np.where(np.isnan(input_array), np.nan, labeled_array)
    return labeled_array

def _dfs_2d(input_array, output_array, count, rows, cols, i, j):
    """
    Recursive DFS for 2D island labeling, 4-directional connectivity.
    """
    if (i < 0 or i >= rows or j < 0 or j >= cols or 
        np.isnan(input_array[i, j]) or input_array[i, j] == 0):
        return

    input_array[i, j] = 0
    output_array[i, j] = count

    # Recurse to 4 neighbors
    _dfs_2d(input_array, output_array, count, rows, cols, i + 1, j)
    _dfs_2d(input_array, output_array, count, rows, cols, i - 1, j)
    _dfs_2d(input_array, output_array, count, rows, cols, i, j + 1)
    _dfs_2d(input_array, output_array, count, rows, cols, i, j - 1)

# ==============================
#        1D Island Finder
# ==============================

def _identify_islands_1d(input_array):
    """
    Identifies contiguous segments (islands) of 1s in a 1D array, ignoring NaNs.
    """
    arr = np.copy(input_array)
    labeled = np.copy(input_array)
    count = 0
    length = len(arr)

    for i in range(length):
        if arr[i] == 1:
            count += 1
            _dfs_1d(arr, labeled, count, length, i)

    return labeled

def _dfs_1d(input_array, labeled_array, count, length, i):
    """
    DFS for 1D islands (forward and backward traversal).
    """
    if i < 0 or i >= length or input_array[i] == 0 or np.isnan(input_array[i]):
        return

    input_array[i] = 0
    labeled_array[i] = count

    _dfs_1d(input_array, labeled_array, count, length, i - 1)
    _dfs_1d(input_array, labeled_array, count, length, i + 1)





def correct_island_identifiers(island_ids):
    """
    Corrects island identifiers by renumbering them sequentially starting from 0.

    Parameters
    ----------
    island_ids : numpy.ndarray
        Array of island identifiers (1D or 2D).

    Returns
    -------
    corrected_ids : numpy.ndarray
        Corrected island identifiers.

    Notes
    -----
    If all island identifiers are NaN, no correction is performed.
    """
    if np.all(np.isnan(island_ids)):
        return island_ids

    unique_ids = np.unique(island_ids[~np.isnan(island_ids)])
    corrected_ids = np.full_like(island_ids, np.nan, dtype=float)

    for new_id, old_id in enumerate(unique_ids):
        corrected_ids[island_ids == old_id] = new_id

    return corrected_ids


def center_of_mass(island_mask, activity_map_smoothed, center_bins):
    """
    Calculate the center of mass for a given region (island), in 1D or 2D.

    Parameters
    ----------
    island_mask : numpy.ndarray
        Binary mask indicating pixels in the region.
    activity_map_smoothed : numpy.ndarray
        Smoothed activity map (1D or 2D).
    center_bins : numpy.ndarray or tuple of arrays
        Bin centers corresponding to the place field grid.

    Returns
    -------
    com : float or tuple
        Center of mass coordinates.
    """
    island_values = activity_map_smoothed[island_mask]

    if island_mask.ndim == 1:
        coords = np.where(island_mask)[0]
        total_weight = np.nansum(island_values)
        if total_weight > 0:
            com = np.nansum(center_bins[coords] * island_values) / total_weight
        else:
            com = np.nanmean(center_bins[coords])
        return com

    elif island_mask.ndim == 2:
        y_coords, x_coords = np.where(island_mask)
        x_centers, y_centers = center_bins
        total_weight = np.nansum(island_values)
        if total_weight > 0:
            x_com = np.nansum(x_centers[x_coords] * island_values) / total_weight
            y_com = np.nansum(y_centers[y_coords] * island_values) / total_weight
        else:
            x_com = np.nanmean(x_centers[x_coords])
            y_com = np.nanmean(y_centers[y_coords])
        return x_com, y_com


def detect_islands_and_com(binary_map, activity_map_smoothed, visits_map, center_bins, min_num_of_bins=4):
    """
    Detects contiguous active regions (islands) in a thresholded binary map and computes their centers of mass.

    Parameters
    ----------
    binary_map : numpy.ndarray
        Binary map (1D or 2D) where islands are defined by contiguous True values.
    activity_map_smoothed : numpy.ndarray
        Smoothed activity map corresponding to the binary map.
    visits_map : numpy.ndarray
        Map indicating where activity was observed.
    center_bins : numpy.ndarray or tuple of arrays
        Bin centers used to compute center of mass.
    min_num_of_bins : int, optional
        Minimum number of bins required for an island to be considered valid.

    Returns
    -------
    n_islands : int
        Number of detected islands.
    x_coms : np.ndarray
        Array of center of mass x-coordinates (or values for 1D).
    y_coms : np.ndarray or float
        Array of center of mass y-coordinates for 2D or np.nan for 1D.
    pixels_absolute : np.ndarray
        Fraction of the entire grid each island occupies.
    pixels_relative : np.ndarray
        Fraction of visited grid each island occupies.
    corrected_ids : np.ndarray
        Island map with corrected sequential identifiers.
    activity_map_identity : np.ndarray
        Original island ID map (NaNs for discarded islands).
    """
    import sys
    sys.setrecursionlimit(10000000)

    if binary_map.ndim == 1 or binary_map.ndim == 2:
        activity_map_identity = identify_islands(np.copy(binary_map))
    else:
        raise ValueError("Input array must be 1D or 2D.")

    unique_islands = np.unique(activity_map_identity[~np.isnan(activity_map_identity)])[1:]

    islands_id = []
    islands_com = []
    pixels_above = []

    for ii in unique_islands:
        island_mask = (activity_map_identity == ii)
        if np.nansum(island_mask) > min_num_of_bins:
            com = center_of_mass(island_mask, activity_map_smoothed, center_bins)
            islands_id.append(ii)
            islands_com.append(com)
            pixels_above.append(np.nansum(island_mask))
        else:
            activity_map_identity[island_mask] = np.nan

    if not islands_id:
        return 0, np.array([np.nan]), np.array([np.nan]), np.nan, np.nan, np.full_like(activity_map_identity, np.nan)

    total_visited_pixels = np.nansum(visits_map != 0)
    pixels_total = activity_map_smoothed.size

    pixels_relative = np.array(pixels_above) / total_visited_pixels
    pixels_absolute = np.array(pixels_above) / pixels_total

    if binary_map.ndim == 1:
        return len(islands_id), np.array(islands_com), np.array([np.nan] * len(islands_com)), pixels_absolute, pixels_relative, correct_island_identifiers(activity_map_identity)
    else:
        x_coms = [c[0] for c in islands_com]
        y_coms = [c[1] for c in islands_com]
        return len(islands_id), np.array(x_coms), np.array(y_coms), pixels_absolute, pixels_relative, correct_island_identifiers(activity_map_identity)


def smooth_field_detection_map(activity_map, center_bins, sigma_x=2, sigma_y=None, handle_nans=False):

    nan_mask = np.isnan(activity_map)
    if activity_map.ndim == 1:
        sigma_x_points = smooth.get_sigma_points(sigma_x, center_bins[0])
        kernel,_ = smooth.generate_1d_gaussian_kernel(sigma_x_points)
        activity_map_smoothed = smooth.gaussian_smooth_1d(activity_map, kernel, handle_nans=handle_nans)
        activity_map_smoothed[nan_mask] = np.nan
        return activity_map_smoothed
    
    elif activity_map.ndim == 2:
        if sigma_y is None:
            sigma_y = sigma_x
        sigma_x_points = smooth.get_sigma_points(sigma_x, center_bins[0])
        sigma_y_points = smooth.get_sigma_points(sigma_y, center_bins[1])
        kernel, _ = smooth.generate_2d_gaussian_kernel(sigma_x_points, sigma_y_points)
        activity_map_smoothed = smooth.gaussian_smooth_2d(activity_map, kernel, handle_nans=handle_nans)
        activity_map_smoothed[nan_mask] = np.nan
        return activity_map_smoothed
    else:
        raise ValueError("Input array must be 1D or 2D.")


def detect_place_fields(activity_map,
                        activity_map_shifted,
                        visits_occupancy,
                        center_bins,
                        field_detection_method,
                        threshold,
                        min_num_of_bins=4,
                        detection_smoothing_sigma_x=1.0,
                        detection_smoothing_sigma_y=1.0):
    
    """
    Wrapper to detect place fields based on the specified detection method and threshold.

    Parameters
    ----------
    activity_map : np.ndarray
        The main activity map.
    activity_map_shifted : np.ndarray or None
        Null distribution (shifted activity map), used for 'random_fields' method.
    visits_occupancy : np.ndarray
        Map of visited bins (occupancy).
    center_bins : tuple of np.ndarray
        (x_center_bins, y_center_bins)
    field_detection_method : str
        Method to use: 'random_fields' or 'std_from_field'.
    threshold : tuple
        A tuple such as ('percentile', 95) or ('mean_std', 2).
    min_num_of_bins : int, optional
        Minimum size of region to be considered a field.
    detection_smoothing_sigma_x : float, optional
        Smoothing sigma in X direction.
    detection_smoothing_sigma_y : float, optional
        Smoothing sigma in Y direction.

    Returns
    -------
    num_of_fields : int
    fields_x_max : np.ndarray
    fields_y_max : np.ndarray
    pixels_place_cell_absolute : np.ndarray
    pixels_place_cell_relative : np.ndarray
    activity_map_identity : np.ndarray
    """
    import warnings

    threshold_type, threshold_value = threshold

    if field_detection_method == 'random_fields':
        return field_coordinates(
            activity_map,
            activity_map_shifted,
            visits_occupancy,
            center_bins,
            threshold_type=threshold_type,
            threshold_value=threshold_value,
            use_shifted=True,
            min_num_of_bins=min_num_of_bins,
            sigma_x=detection_smoothing_sigma_x,
            sigma_y=detection_smoothing_sigma_y
        )

    elif field_detection_method == 'std_from_field':
        return field_coordinates(
            activity_map,
            None,
            visits_occupancy,
            center_bins,
            threshold_type=threshold_type,
            threshold_value=threshold_value,
            use_shifted=False,
            min_num_of_bins=min_num_of_bins,
            sigma_x=detection_smoothing_sigma_x,
            sigma_y=detection_smoothing_sigma_y
        )
    
    else:
        warnings.warn("No valid field detection method set", UserWarning)
        return 0, np.array([]), np.array([]), np.array([]), np.array([]), np.full_like(activity_map, np.nan)


def field_coordinates(activity_map, activity_map_shifted, visits_map, center_bins, threshold_type, threshold_value, use_shifted=False, min_num_of_bins=4, sigma_x=2, sigma_y=None):
    activity_map_smoothed = smooth_field_detection_map(activity_map, center_bins, sigma_x, sigma_y, handle_nans=False)

    shifted_smoothed = None
    if use_shifted and activity_map_shifted is not None:
        if activity_map.ndim == 1:
            shifted_smoothed = np.array([
                smooth_field_detection_map(m, center_bins, sigma_x, handle_nans=True) for m in activity_map_shifted
            ])
        else:
            shifted_smoothed = np.array([
                smooth_field_detection_map(m, center_bins, sigma_x, sigma_y, handle_nans=True) for m in activity_map_shifted
            ])

    binary_map = apply_threshold(
        activity_map_smoothed,
        threshold_type=threshold_type,
        threshold_value=threshold_value,
        shifted_maps=shifted_smoothed
    )

    binary_map[np.isnan(activity_map)] = np.nan
    return detect_islands_and_com(binary_map, activity_map_smoothed, visits_map, center_bins, min_num_of_bins)


def apply_threshold(activity_map_smoothed, threshold_type="mean_std", threshold_value=2, shifted_maps=None):
    
    if threshold_type == "mean_std" and shifted_maps is None:
        I_threshold = np.nanmean(activity_map_smoothed) + threshold_value * np.nanstd(activity_map_smoothed)

    elif threshold_type == "percentile" and shifted_maps is None:
        I_threshold = np.nanpercentile(activity_map_smoothed[np.isfinite(activity_map_smoothed)], threshold_value)

    elif threshold_type == "percentile" and shifted_maps is not None:
        I_threshold = np.nanpercentile(shifted_maps, threshold_value, axis=0)

    elif threshold_type == "mean_std" and shifted_maps is not None:
        I_threshold = np.nanmean(shifted_maps, axis=0) + threshold_value * np.nanstd(shifted_maps, axis=0)

    else:
        raise ValueError("Invalid threshold_type or missing shifted_maps")

    binary_map = np.where(activity_map_smoothed > I_threshold, 1, 0).astype(float)
    return binary_map
