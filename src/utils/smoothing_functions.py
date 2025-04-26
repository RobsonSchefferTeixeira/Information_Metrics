import numpy as np
import scipy.signal as sig
import math

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
        The smoothed signal
    
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


def get_sigma_points(sigma,units_vector):
    
    sampling_rate = 1/np.nanmean(np.diff(units_vector))
    sigma_points = sigma*sampling_rate
    return sigma_points

def generate_1d_gaussian_kernel(sigma, radius=None, truncate=4.0):
    """
    Generate a 1D Gaussian kernel with a specified standard deviation.

    Parameters:
        sigma (float): Standard deviation of the Gaussian kernel. 
        
            Notice that this is in data points and reflect the unit step.
            E.g., a if a time vector, given in seconds, is sampled at 20 Hz (one at each 5 ms),
            to get a sigma of 0.5 s, one need to convert it to points: 

            sampling_rate = np.nanmean(np.diff(time_vector))
            and then 
            sigma = 0.5*sampling_rate

            Since the kernel will convolute with the original signal, it will step through each point in a discrete way,
            and its wideness reflect the standard deviation: a higher std need more points than lower ones.
            
        radius (int, optional): Radius of the kernel. If None, computed as round(truncate * sigma).
        truncate (float, optional): Truncate the filter at this many standard deviations (default is 4.0).

    Returns:

        gaussian_kernel (numpy.ndarray): The normalized 1D Gaussian kernel (sums to 1).
        x_values (numpy.ndarray): The x-coordinates corresponding to the kernel values.
    """

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    # Calculate radius if not provided
    if radius is None:
        radius = math.ceil(truncate * sigma)


    x_values = np.arange(-radius, radius + 1)
    dt = 1 # just making this more explicit
    constant = dt / (np.sqrt(2 * math.pi) * sigma)
    # We need to multiply by dt to normalize to area equal to one. Since here we have integer numbers meant to use as indexes, dt = 1
    kernel = constant * np.exp(-((x_values**2) / (2 * (sigma**2))))

    # The kernel should be normalized to sum to 1 (especially important for convolution to preserve signal 
    # magnitude). Currently, it's normalized for a continuous Gaussian (area=1), but in discrete form 
    # the sum may not be exactly 1.
    kernel /= kernel.sum()

    return kernel,x_values


def gaussian_smooth_1d(input_data, kernel, handle_nans=False):
    """
    Perform 1D Gaussian smoothing with mirror padding and proper NaN handling.
    
    Parameters:
        input_data (ndarray): 1D input array (may contain NaNs)
        kernel (ndarray): 1D Gaussian kernel (must be normalized to sum=1)
        handle_nans (bool): Whether to handle NaN values (default: True)
    
    Returns:
        ndarray: Smoothed data with original NaNs preserved
    """

    input_data = np.copy(input_data)
    if input_data.ndim != 1:
        raise ValueError("Input must be 1D array")
    
    if not handle_nans:
        nan_mask = np.isnan(input_data)
        input_data[nan_mask] = 0
        result = sig.convolve(input_data, kernel, mode='same', method='direct')
        # result[nan_mask] = np.nan
        return 
    
    # Create mask of valid numbers
    mask = ~np.isnan(input_data)
    data_clean = np.where(mask, input_data, 0)
    
    # Convolve both data and mask
    smoothed = sig.convolve(data_clean, kernel, mode='same', method='auto')
    norm_factor = sig.convolve(mask.astype(float), kernel, mode='same', method='auto')
    
    # Normalize and restore NaNs
    result = np.divide(smoothed, norm_factor, 
                      out=np.full_like(smoothed, np.nan), 
                      where=norm_factor > 1e-8)
    
    return result


def generate_2d_gaussian_kernel(sigma_x, sigma_y=None, radius_x=None, radius_y=None, truncate=4.0):
    """
    Generate 2D Gaussian kernel with rectangular support.
    
    Parameters:
        sigma_x (float): Standard deviation in x-direction (in pixels)
        sigma_y (float): Standard deviation in y-direction (optional)
        radius_x (int): Explicit x radius in pixels (optional)
        radius_y (int): Explicit y radius in pixels (optional)
        truncate (float): Number of sigmas to include (default: 4.0)
    
    Returns:
        kernel (ndarray): Normalized 2D Gaussian kernel
        (x_mesh, y_mesh) (tuple): Coordinate grids
    """
    if sigma_y is None:
        sigma_y = sigma_x
        
    # Calculate radii if not provided
    if radius_x is None:
        radius_x = math.ceil(truncate * sigma_x)
    if radius_y is None:
        radius_y = math.ceil(truncate * sigma_y)
    
    # Create coordinate ranges
    x = np.arange(-radius_x, radius_x + 1)
    y = np.arange(-radius_y, radius_y + 1)
    x_mesh, y_mesh = np.meshgrid(x, y)
    
    # Compute 2D Gaussian
    # constant = (dt*dt) / (2 * np.pi * sigma_x * sigma_y) # since we are in 2D, we need to dt: dtx and dty

    constant = (1*1) / (2 * np.pi * sigma_x * sigma_y)
    kernel = constant * np.exp(-(x_mesh**2/(2*sigma_x**2) + y_mesh**2/(2*sigma_y**2)))
    kernel /= kernel.sum()  # Normalize
    
    return kernel, (x_mesh, y_mesh)
    
    
def gaussian_smooth_2d(matrix, kernel, handle_nans=False):
    """
    Convolve 2D matrix with mirror padding and NaN handling using convolve2d.
    
    Parameters:
        matrix (ndarray): Input 2D array (may contain NaNs)
        kernel (ndarray): 2D convolution kernel (must be square and normalized)
        handle_nans (bool): If True, properly handles NaN values (slower)
    
    Returns:
        ndarray: Convolved matrix (same shape as input) with NaNs preserved
    """
    matrix = np.copy(matrix)
    if handle_nans:
        # Create mask of valid numbers
        nan_mask = np.isnan(matrix)
        mask = ~nan_mask
        matrix_clean = np.nan_to_num(matrix)
        
        # Convolve both data and mask
        convolved = sig.convolve2d(matrix_clean, kernel, mode='same', boundary='fill')
        kernel_mask = sig.convolve2d(mask.astype(float), kernel, mode='same', boundary='fill')
        
        # Normalize and restore NaNs
        result = np.divide(convolved, kernel_mask, 
                         out=np.full_like(convolved, np.nan), 
                         where=kernel_mask > 1e-8)
        
        # result[nan_mask] = np.nan
        return result
    else:
        nan_mask = np.isnan(matrix)
        matrix[nan_mask] = 0
        result = sig.convolve2d(matrix, kernel, mode='same', boundary='fill')
        # result[nan_mask] = np.nan

        return result


'''

def gaussian_smooth_2d(input_matrix, sigma):
    """
    Perform 2D Gaussian smoothing on input data.

    Parameters:
        input_matrix (numpy.ndarray): The 2D input matrix to be smoothed.
        sigma (float): The standard deviation of the 2D Gaussian kernel in data points.

    Returns:
        smoothed_matrix (numpy.ndarray): The smoothed 2D data.
    """
    # Generate a 2D Gaussian kernel.
    gaussian_kernel_2d = generate_2d_gaussian_kernel(sigma)

    # Convolve the input matrix with the 2D Gaussian kernel.
    smoothed_matrix = sig.convolve2d(input_matrix, gaussian_kernel_2d, mode='same')

    return smoothed_matrix
'''

'''
def generate_2d_gaussian_kernel(sigma_x, sigma_y=None, radius=None, truncate=4.0):
    """
    Generate a 2D Gaussian kernel with sigma values dependent on sampling rate.
    
    Parameters:
        sigma_x (float): Standard deviation in x-direction (in seconds/units)
        sigma_y (float): Standard deviation in y-direction (optional, uses sigma_x if None)
        radius (int, optional): Explicit kernel radius in pixels. If None, uses ceil(truncate * max_sigma_pixels)
        truncate (float): Number of standard deviations to include (default: 4.0)
    
    Returns:
        kernel (ndarray): Normalized 2D Gaussian kernel
        (x_mesh, y_mesh) (tuple): Coordinate grids
    """
    if sigma_y is None:
        sigma_y = sigma_x
        
    
    # Calculate radius if not provided
    if radius is None:
        max_sigma = max(sigma_x, sigma_y)
        radius = math.ceil(truncate * max_sigma)
    
    # Create coordinate ranges
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    x_mesh, y_mesh = np.meshgrid(x, y)
    
    constant = 1 / (2 * np.pi * sigma_x * sigma_y)
    kernel = constant * np.exp(-(x_mesh**2/(2*sigma_x**2) + y_mesh**2/(2*sigma_y**2)))
    kernel /= kernel.sum()  # Additional normalization for discrete case

    return kernel, (x_mesh, y_mesh)
    '''



'''
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
'''



# This is a 2d deconvolution from scratch. I'm not using it.

def apply_mirror_padding(matrix, pad_width):
    """
    Apply mirror/symmetric padding to a 2D matrix.
    """
    padded = np.pad(matrix, pad_width, mode='symmetric')
    return padded

def convolve2d_mirror_nan(matrix, kernel, handle_nans=True):
    """
    Full 2D convolution with mirror padding and NaN handling implemented from scratch.
    
    Parameters:
        matrix: 2D input array (may contain NaNs)
        kernel: 2D convolution kernel (must be square, odd dimensions, and normalized)
        handle_nans: If True, properly handles NaN values
    
    Returns:
        Convolved matrix with same shape as input, preserving NaNs
    """
    # Validate kernel
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Kernel must be square")
    if kernel.shape[0] % 2 == 0:
        raise ValueError("Kernel must have odd dimensions")
    
    h, w = matrix.shape
    k_size = kernel.shape[0]
    pad = k_size // 2
    
    # Apply mirror padding
    padded = apply_mirror_padding(matrix, pad)
    
    # Initialize output
    output = np.zeros_like(matrix)
    
    if handle_nans:
        # Create masked array
        mask = ~np.isnan(padded)
        matrix_clean = np.where(mask, padded, 0)
        
        # Initialize normalization array
        norm = np.zeros_like(matrix)
        
        for i in range(h):
            for j in range(w):
                # Extract region of interest
                region = matrix_clean[i:i+k_size, j:j+k_size]
                k_region = kernel * mask[i:i+k_size, j:j+k_size]
                
                # Compute weighted sum
                output[i,j] = np.sum(region * kernel)
                norm[i,j] = np.sum(k_region)
        
        # Normalize and restore NaNs
        output = np.divide(output, norm, out=np.full_like(output, np.nan), where=norm>1e-8)
    else:
        # Simple convolution without NaN handling
        for i in range(h):
            for j in range(w):
                output[i,j] = np.sum(padded[i:i+k_size, j:j+k_size] * kernel)
    
    return output