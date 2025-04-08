import numpy as np
import scipy.signal as sig

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
    constant = 1 / (np.sqrt(2 * math.pi) * sigma)
    gaussian_kernel = constant * np.exp(-((x_values**2) / (2 * (sigma**2))))


    # The kernel should be normalized to sum to 1 (especially important for convolution to preserve signal 
    # magnitude). Currently, it's normalized for a continuous Gaussian (area=1), but in discrete form 
    # the sum may not be exactly 1.
    gaussian_kernel /= np.sum(gaussian_kernel)

    return gaussian_kernel,x_values


def gaussian_smooth_1d(input_data, sigma, radius = None, truncate = 4.0):
    """
    Perform 1D Gaussian smoothing on input data, ignoring NaNs and preserving their positions.

    Parameters:
        input_data (numpy.ndarray): The 1D input data to be smoothed.
        sigma_points (float): The standard deviation of the Gaussian kernel in data points.

    Returns:
        numpy.ndarray: The smoothed 1D data with NaNs preserved.
    """
    if input_data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")

    # Generate the Gaussian kernel
    gaussian_kernel_1d,_ = generate_1d_gaussian_kernel(sigma,radius, truncate)

    # Identify NaN positions
    nan_mask = np.isnan(input_data)

    # Replace NaNs with zero for convolution
    data_filled = np.where(nan_mask, 0, input_data)

    # Convolve the filled data with the Gaussian kernel
    smoothed_data = sig.convolve(data_filled, gaussian_kernel_1d, mode='same')
    # smoothed_data = np.apply_along_axis(lambda x: sig.convolve(x, gaussian_kernel_1d, mode='same'), axis=1, arr=input_data)

    # Create a normalization factor to account for missing data
    normalization_factor = sig.convolve(~nan_mask, gaussian_kernel_1d, mode='same')

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        smoothed_data /= normalization_factor

    # Reinsert NaNs into their original positions
    smoothed_data[nan_mask] = np.nan

    return smoothed_data



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
