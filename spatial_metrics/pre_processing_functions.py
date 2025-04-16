import numpy as np
import scipy.signal as sig

def preprocess_signal(input_signal, sampling_rate, signal_type, z_threshold=2):
    filtered_signal = eegfilt(input_signal, sampling_rate, 0, 2, order=2)
    diff_signal = np.hstack([np.diff(filtered_signal), 0])

    if signal_type == 'raw':
        output_signal = input_signal

    elif signal_type == 'filtered':
        output_signal = filtered_signal

    elif signal_type == 'diff':
        diff_signal_truncated = np.copy(diff_signal)
        diff_signal_truncated[diff_signal < 0] = 0
        output_signal = diff_signal_truncated


    elif signal_type == 'binary':

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
