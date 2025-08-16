import numpy as np


def get_spikes_idx_surrogate(time_stamps_idx, time_vector, sampling_rate, shift_time_limit):
    """
    Generate a surrogate set of spike indices by circularly shifting the original spike indices.

    Parameters:
        time_stamps_idx (numpy.ndarray): The original spike indices (positions in time_vector).
        time_vector (numpy.ndarray): The time vector associated with the data.
        sampling_rate (float): Sampling rate of the recording.
        shift_time_limit (float): Maximum time range (in seconds) for circular shifting.

    Returns:
        numpy.ndarray: Surrogate spike indices after circular shifting.
    """
    
    # max number of samples to shift
    max_shift_points = int(np.round(shift_time_limit * sampling_rate))
    shift_points = np.random.randint(-max_shift_points, max_shift_points + 1)

    n_points = len(time_vector)

    # circular shift
    shifted_idx = (time_stamps_idx + shift_points) % n_points

    return np.sort(shifted_idx)

    

def get_spikes_surrogate(time_stamps, time_vector, shift_time_limit):
    """
    Generate a surrogate set of spike timestamps by circularly shifting the original timestamps within a time range.

    Parameters:
        time_stamps (numpy.ndarray): The original spike timestamps.
        time_vector (numpy.ndarray): The time vector associated with the spike timestamps.
        shift_time_limit (float): The maximum time range for circular shifting.

    Returns:
        time_stamps_shifted (numpy.ndarray): The surrogate spike timestamps after circular shifting.
    """
    # Generate a random shift within the specified time range
    shift_this_much = np.random.uniform(-shift_time_limit, shift_time_limit, size=1)
    # Apply the circular shift to the timestamps
    time_stamps_shifted = time_stamps + shift_this_much

    # Get the lower and upper time limits from the time vector
    lower_time_limit = time_vector[0]
    upper_time_limit = time_vector[-1]

    # Apply circular shifting for timestamps exceeding the time range
    time_stamps_shifted[time_stamps_shifted <= lower_time_limit] += (upper_time_limit - lower_time_limit)
    time_stamps_shifted[time_stamps_shifted >= upper_time_limit] -= (upper_time_limit - lower_time_limit)

    return np.sort(time_stamps_shifted)



def circular_random_shift(input_array, sampling_rate, shift_time, axis=0):
    """
    Apply a random circular shift to an input array.

    Parameters:
        input_array (np.ndarray): Array to shift (1D or ND).
        sampling_rate (float): Sampling rate in Hz.
        shift_time (float): Maximum absolute shift (in seconds).
        axis (int): Axis along which to shift (default=0).

    Returns:
        np.ndarray: Circularly shifted array.
    """
    # Convert seconds to samples
    max_shift_samples = int(np.round(sampling_rate * shift_time))

    # Draw a random shift between -max and +max
    shift_samples = np.random.randint(-max_shift_samples, max_shift_samples + 1)

    # Apply circular shift
    return np.roll(input_array, shift=shift_samples, axis=axis)



def get_signal_surrogate(input_array, sampling_rate, shift_time, axis=0):
    """
    Generate a surrogate signal by applying a time shift to the input array.

    This function creates a surrogate signal by shifting the input array in time
    along the specified axis while maintaining the same signal characteristics.

    Parameters:
        input_array (numpy.ndarray): The input signal to generate a surrogate for.
        sampling_rate (float): The sampling rate of the input signal (samples per second).
        shift_time (float): The desired time shift for the surrogate signal (seconds).
        axis (int): The axis along which to apply the time shift.

    Returns:
        numpy.ndarray: The surrogate signal obtained by applying the time shift.
    """
    # Calculate the maximum allowable shift in samples
    max_shift_samples = int(np.floor(input_array.shape[axis] / 2))
    desired_shift_samples = int(np.round(sampling_rate * shift_time))

    # Adjust the shift if it exceeds half the length of the signal along the specified axis
    if abs(desired_shift_samples) > max_shift_samples:
        desired_shift_samples = max_shift_samples * np.sign(desired_shift_samples)

    # Generate a random shift within the allowable range
    shift_samples = np.random.randint(-abs(desired_shift_samples), abs(desired_shift_samples) + 1)

    # Apply the time shift using np.roll
    input_array_shifted = np.roll(input_array, shift=shift_samples, axis=axis)

    return input_array_shifted



def get_signal_surrogate_old(input_vector, sampling_rate, shift_time):
    """
    Generate a surrogate signal by applying a time shift to the input vector.

    This function creates a surrogate signal by shifting the input vector in time 
    while maintaining the same signal characteristics.

    Parameters:
        input_vector (numpy.ndarray): The input signal to generate a surrogate for.
        sampling_rate (float): The sampling rate of the input signal (samples per second).
        shift_time (float): The desired time shift for the surrogate signal (seconds).

    Returns:
        input_vector_shifted (numpy.ndarray): The surrogate signal obtained by applying the time shift.
    """
    if len(input_vector) < np.abs(sampling_rate * shift_time):
        # Adjust the shift time if it exceeds the length of the input signal.
        shift_time = np.floor(len(input_vector) / sampling_rate)

    # Generate a random time shift in samples within the specified range.
    shift_samples = np.random.randint(-shift_time * sampling_rate, sampling_rate * shift_time + 1)

    # Apply the time shift to create the surrogate signal.
    input_vector_shifted = np.concatenate([input_vector[shift_samples:], input_vector[0:shift_samples]])
    # np.roll could be used instead
    return input_vector_shifted
    

def get_surrogate_old(self,spike_times_idx,time_vector,sampling_rate,shift_time):
    eps = np.finfo(np.float64).eps
    time_vector_hist = np.append(time_vector,time_vector[-1] + eps)
    spike_timevector = np.histogram(time_vector[spike_times_idx],time_vector_hist)[0]

    I_break = np.random.choice(np.linspace(-shift_time*sampling_rate,sampling_rate*shift_time),1)[0].astype(int)
    input_vector_shifted = np.concatenate([spike_timevector[I_break:], spike_timevector[0:I_break]])

    timestamps_shifted = np.repeat(time_vector, input_vector_shifted)

    spike_times_idx_shifted = self.searchsorted2(time_vector, timestamps_shifted)
    I_keep = np.abs(time_vector[spike_times_idx_shifted]-timestamps_shifted)<0.1
    spike_times_idx_shifted = spike_times_idx_shifted[I_keep]

    return spike_times_idx_shifted
