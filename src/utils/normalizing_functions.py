
import numpy as np

class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X):
        self.mean_ = np.nanmean(X, axis=(0, 2))
        self.std_ = np.nanstd(X, axis=(0, 2))
        self.std_ = np.where(self.std_ == 0, 1, self.std_)  # Replace zero standard deviations with 1
    
    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        X_transformed = (X - self.mean_[:, np.newaxis]) / self.std_[:, np.newaxis]
        return X_transformed
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

def min_max_norm(input_signal, axis=None, custom_min=0, custom_max=1):
    """
    Perform min-max normalization on an input signal with the option to set custom minimum and maximum values.

    Parameters:
        input_signal (numpy.ndarray): The input signal to be normalized.
        axis (int or None, optional): The axis along which normalization is performed.
                                       If None, normalization is applied to the entire array (default).
        custom_min (float, optional): Custom minimum value for normalization (default is 0).
        custom_max (float, optional): Custom maximum value for normalization (default is 1).

    Returns:
        scaled_signal (numpy.ndarray): The min-max normalized signal within the custom range.
    """
    min_value = np.nanmin(input_signal, axis=axis, keepdims=True)
    max_value = np.nanmax(input_signal, axis=axis, keepdims=True)
    
    # Prevent division by zero by ensuring max_value != min_value
    range_value = max_value - min_value
    range_value[range_value == 0] = 1
    
    # Perform min-max normalization
    scaled_signal = (((custom_max - custom_min) * (input_signal - min_value)) /
                     range_value) + custom_min

    return scaled_signal



def robust_z_score_norm(input_matrix, axis=0):
    """
    Perform robust z-score normalization on an input matrix using MAD.

    Parameters:
        input_matrix (numpy.ndarray): The input matrix to be normalized.
        axis (int, optional): The axis along which normalization is performed (default is 0).

    Returns:
        robust_z_scored_matrix (numpy.ndarray): The robust z-score normalized matrix.
    """
    # Calculate the median along the specified axis, handling NaN values.
    median = np.nanmedian(input_matrix, axis=axis, keepdims=True)

    # Calculate the median absolute deviation (MAD) along the specified axis.
    mad = np.nanmedian(np.abs(input_matrix - median), axis=axis, keepdims=True)

    # Check for cases where all values in mad are zero to avoid division by zero.
    all_same_values = np.all(mad == 0, axis=axis, keepdims=True)

    # If all values in mad are zero, set mad to 1 to prevent division by zero.
    mad[all_same_values] = 1

    # Calculate the robust z-scored matrix using the formula (input_matrix - median) / (1.4826 * mad).
    robust_z_scored_matrix = (input_matrix - median) / (1.4826 * mad)

    # Return the robust z-scored matrix.
    return robust_z_scored_matrix


def robust_z_score_norm_old(input_matrix, axis=0):
    """
    Perform robust z-score normalization on an input matrix.

    Parameters:
        input_matrix (numpy.ndarray): The input matrix to be normalized.
        axis (int, optional): The axis along which normalization is performed (default is 0).

    Returns:
        robust_z_scored_matrix (numpy.ndarray): The robust z-score normalized matrix.
    """
    # Calculate the median and IQR along the specified axis, handling NaN values.
    median = np.nanmedian(input_matrix, axis=axis, keepdims=True)
    q75 = np.nanpercentile(input_matrix, 75, axis=axis, keepdims=True)
    q25 = np.nanpercentile(input_matrix, 25, axis=axis, keepdims=True)
    iqr = q75 - q25

    # Check for cases where all values in iqr are zero to avoid division by zero.
    all_same_values = np.all(iqr == 0, axis=axis, keepdims=True)

    # If all values in iqr are zero, set iqr to 1 to prevent division by zero.
    iqr[all_same_values] = 1

    # Calculate the robust z-scored matrix using the formula (input_matrix - median) / iqr.
    robust_z_scored_matrix = (input_matrix - median) / iqr

    # Return the robust z-scored matrix.
    return robust_z_scored_matrix


def z_score_norm(input_matrix, axis=None, ddof=1):
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
    std = np.nanstd(input_matrix, axis=axis,ddof = ddof, keepdims=True)

    # Check for cases where all values in std are zero to avoid division by zero.
    all_same_values = np.all(std == 0, axis=axis, keepdims=True)

    # If all values in std are zero, set std to 1 to prevent division by zero.
    std[all_same_values] = 1

    # Calculate the z-scored matrix using the formula (input_matrix - mean) / std.
    z_scored_matrix = (input_matrix - mean) / std

    # Return the z-scored matrix.
    return z_scored_matrix


