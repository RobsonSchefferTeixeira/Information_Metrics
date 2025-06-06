import numpy as np
import scipy.signal as sig
import warnings
import src.utils.normalizing_functions as nf

def preprocess_signal(input_signal, sampling_rate, signal_type, z_threshold=2):
    """
    Preprocess a signal based on the specified signal type.

    Parameters:
    - input_signal: np.ndarray
        The input signal to preprocess.
    - sampling_rate: float
        The sampling rate of the input signal.
    - signal_type: str
        The type of signal to output. Options are 'raw', 'filtered', 'diff', or 'binary'.
        -- 'raw' - Return the original, unprocessed signal.
        -- 'filtered' - Apply a low-pass filter (cutoff at 2 Hz) to extract the slow components of the signal.
        -- 'diff' - Compute the first derivative (rate of change) of the low-pass filtered signal.
        -- 'diff_truncated' - Compute the first derivative of the filtered signal and truncate all negative values (i.e., keep only rising edges).
        -- 'binary' - Apply low-pass filtering and z-score normalization, then binarize the signal (0`s and 1`s) by thresholding and 
        detecting positive-going changes. This version highlights periods of rising activity above a z-threshold.


     - z_threshold: float, optional
        Z-score threshold for binarization (default is 2).

    Returns:
    - output_signal: np.ndarray
        The preprocessed signal.

    Raises:
    - NotImplementedError: If the signal_type is unknown.
    """
    
    if signal_type == 'raw':
        # Return the raw input signal
        return input_signal

    elif signal_type == 'filtered':
        # Return the lowpass filtered signal
        # Filter the input signal with a lowpass filter (cutoff at 2 Hz)
        filtered_signal = eegfilt(input_signal, sampling_rate, 0, 2, order=3)
        return filtered_signal
    
    elif signal_type == 'diff':
        # Compute the first derivative of the filtered signal
        filtered_signal = eegfilt(input_signal, sampling_rate, 0, 2, order=3)
        diff_signal = np.hstack([np.diff(filtered_signal), 0])
        return diff_signal

    elif signal_type == 'diff_truncated':
        # Return the positive part of the first derivative of the filtered signal
        filtered_signal = eegfilt(input_signal, sampling_rate, 0, 2, order=3)
        diff_signal = np.hstack([np.diff(filtered_signal), 0])
        diff_signal_truncated = np.copy(diff_signal)
        diff_signal_truncated[diff_signal < 0] = 0
        return diff_signal_truncated

    elif signal_type == 'binary':
        # Binarize the signal based on z-score threshold and positive derivative
        filtered_signal = eegfilt(input_signal, sampling_rate, 0, 2, order=3)
        diff_signal = np.hstack([np.diff(filtered_signal),0])
        norm_signal = nf.z_score_norm(filtered_signal)
        # norm_signal = input_signal / np.nanstd(input_signal)
        binarized_idx = (norm_signal >= z_threshold) & (diff_signal > 0)
        binarized_signal = np.zeros(diff_signal.shape[0])
        binarized_signal[binarized_idx] = 1
        return binarized_signal

    else:
        # Raise an error for unsupported signal types
        raise NotImplementedError('Signal type unknown')




def eegfilt(signal, fs, lowcut, highcut, order=3, nan_policy='propagate'):
    """
    Apply a bandpass or lowpass/highpass filter to the input signal.

    Parameters:
    - signal: np.ndarray
        Input signal to be filtered.
    - fs: float
        Sampling frequency of the signal.
    - lowcut: float
        Low cutoff frequency of the filter (set to 0 for highpass).
    - highcut: float
        High cutoff frequency of the filter (set to 0 for lowpass).
    - order: int, optional
        Filter order (default is 3).
    - nan_policy: str, optional
        Policy for handling NaN values in the signal. Options are:
        'propagate': Remove NaNs temporarily, filter, then reintroduce NaNs (default)
        'omit': Remove NaNs and return only filtered non-NaN values
        'raise': Raise an error if NaN values are present.

    Returns:
    - filtered: np.ndarray
        Filtered signal with NaN values handled according to the specified policy.

    Raises:
    - ValueError: If nan_policy is 'raise' and NaN values are present in the signal.
    """
    if nan_policy not in ['propagate', 'omit', 'raise']:
        raise ValueError("nan_policy must be 'propagate', 'omit', or 'raise'")

    if nan_policy == 'raise' and np.isnan(signal).any():
        raise ValueError("Signal contains NaNs but nan_policy='raise'")

    is_constant = np.allclose(signal, signal[0],equal_nan=True)

    if is_constant:
        warnings.warn(f"All values in the array are approximately equal to {signal[0]}.")

    # Create mask of non-NaN values
    non_nan_mask = ~np.isnan(signal)
    signal_non_nan = signal[non_nan_mask].copy()

    # Handle case where all values are NaN
    if len(signal_non_nan) == 0:
        if nan_policy == 'omit':
            return np.array([])  # Return empty array for 'omit'
        else:
            return signal.copy()  # Return original for 'propagate'

    # Normalize cutoff frequencies by Nyquist frequency
    nyq = 0.5 * fs
    low = lowcut / nyq if lowcut > 0 else None
    high = highcut / nyq if highcut > 0 else None

    # Design filter
    if low and high:
        btype = 'band'
        Wn = [low, high]
    elif low:
        btype = 'high'
        Wn = low
    elif high:
        btype = 'low'
        Wn = high
    else:
        raise ValueError("At least one of lowcut or highcut must be greater than 0.")

    b, a = sig.butter(order, Wn, btype=btype)
    filtered_non_nan = sig.filtfilt(b, a, signal_non_nan)

    # Handle different nan policies
    if nan_policy == 'omit':
        return filtered_non_nan
    else:  # 'propagate'
        filtered = np.full_like(signal, np.nan)
        filtered[non_nan_mask] = filtered_non_nan
        return filtered