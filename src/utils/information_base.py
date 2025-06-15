import numpy as np
from src.utils import smoothing_functions as sf
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
import warnings


''' 
This part calculates the mutual information for binarized signals 
(i.e., calcium imaging data that has been binned into 0s and 1s)

'''

def get_binarized_spatial_info_nkinsky(binarized_signal, position_binned):
    """
    Adapted from https://github.com/nkinsky/ImageCamp/blob/e48c6fac407ef3997b67474a2333184bbc4915dc/General/CalculateSpatialInfo.m
    Also check this nice work: On information metrics for spatial coding by Bryan Souza et al

    Compute spatial information per spike and per second for a single neuron.

    Parameters:
    - binarized_signal: 1D numpy array of binary values (1 = active, 0 = inactive).
    - position_binned: 1D numpy array of positions (same length as binarized_signal).

    Returns:
    - info_per_spike: Float, spatial information per spike (bits/spike).
    - info_per_second: Float, spatial information per second (bits/second).
    - occupancy: 1D numpy array of occupancy probabilities per bin.
    - firing_rate: 1D numpy array of firing rates per bin.
    """
    # Ensure inputs are numpy arrays
    binarized_signal = np.asarray(binarized_signal)
    position_binned = np.asarray(position_binned)

    # Identify unique position bins
    bin_vector = np.unique(position_binned)
    num_bins = len(bin_vector)

    # Initialize arrays
    occupancy = np.zeros(num_bins)
    firing_rate = np.zeros(num_bins)

    # Total number of time points
    total_time = len(binarized_signal)

    # Compute occupancy and firing rate per bin
    for i, bin_val in enumerate(bin_vector):
        # Indices where the position matches the current bin
        position_idx = position_binned == bin_val
        occupancy[i] = np.nansum(position_idx) / total_time
        if np.nansum(position_idx) > 0:
            firing_rate[i] = np.nansum(binarized_signal[position_idx]) / np.nansum(position_idx)
        else:
            firing_rate[i] = 0.0

    # Overall mean firing rate
    mean_firing_rate = np.nansum(binarized_signal) / total_time

    # Compute information per spike and per second
    info_per_spike = 0.0
    info_per_second = 0.0
    for i in range(num_bins):
        if occupancy[i] > 0 and firing_rate[i] > 0 and mean_firing_rate > 0:
            ratio = firing_rate[i] / mean_firing_rate
            log_term = np.log2(ratio)
            info_per_spike += occupancy[i] * ratio * log_term
            info_per_second += occupancy[i] * firing_rate[i] * log_term

    return info_per_spike, info_per_second


def get_binarized_spatial_info_etter(binarized_signal, position_binned):
    """
    Adapted from https://github.com/etterguillaume/CaImDecoding/blob/master/extract_1D_information.m
    Compute mutual information between a binarized neural signal and binned positions.

    Parameters:
    - binarized_signal: 1D numpy array of binary values (1 = active, 0 = inactive).
    - position_binned: 1D numpy array of position bin indices (same length as binarized_signal).

    Returns:
    - mutual_info: Float, mutual information value in bits.
    """
    # Ensure inputs are numpy arrays
    binarized_signal = np.asarray(binarized_signal)
    position_binned = np.asarray(position_binned)

    # Identify unique position bins
    bin_vector = np.unique(position_binned)

    # Compute overall probabilities of being active and inactive
    prob_active = np.nansum(binarized_signal == 1) / len(binarized_signal)
    prob_inactive = np.nansum(binarized_signal == 0) / len(binarized_signal)

    # Initialize arrays to store information per position bin and occupancy probabilities
    mutual_info_bin = np.zeros(bin_vector.shape[0])
    prob_in_bin = np.zeros(bin_vector.shape[0])

    # Iterate over each position bin to compute mutual information contributions
    for i, bin_val in enumerate(bin_vector):
        # Indices where the position matches the current bin
        position_idx = position_binned == bin_val

        if np.any(position_idx):
            # Compute occupancy probability for the current bin
            prob_in_bin[i] = np.nansum(position_idx) / len(position_binned)

            # Compute joint probabilities for active and inactive states within the bin
            joint_active = np.nansum((binarized_signal == 1) & position_idx) / len(binarized_signal)
            joint_inactive = np.nansum((binarized_signal == 0) & position_idx) / len(binarized_signal)

            # Compute conditional probabilities
            cond_active = joint_active / prob_in_bin[i] if prob_in_bin[i] > 0 else 0
            cond_inactive = joint_inactive / prob_in_bin[i] if prob_in_bin[i] > 0 else 0

            # Calculate mutual information contributions, ensuring no division by zero or log of zero
            mutual_info_aux = 0
           
            if joint_active > 0 and prob_active > 0:
                mutual_info_aux += joint_active * np.log2(cond_active / prob_active)

            if joint_inactive > 0 and prob_inactive > 0:
                mutual_info_aux += joint_inactive * np.log2(cond_inactive / prob_inactive)

            mutual_info_bin[i] = mutual_info_aux

    # Sum contributions from all bins to get total mutual information
    # mutual_info = np.nansum(info_per_bin)
    mutual_info = np.nansum(mutual_info_bin)

    return mutual_info


def get_binarized_spatial_info(binarized_signal, position_binned):
    """
    Compute mutual information between a binarized neural signal and binned positions.

    Returns:
    - info_per_spike: mutual information in bits/spike
    - info_per_second: mutual information in bits/second
    """
    binarized_signal = np.asarray(binarized_signal)
    position_binned = np.asarray(position_binned)

    bin_vector = np.unique(position_binned)
    prob_active = np.nansum(binarized_signal == 1) / len(binarized_signal)
    prob_inactive = 1.0 - prob_active

    info_per_second_bin = np.zeros(bin_vector.shape[0])
    info_per_spike_bin = np.zeros(bin_vector.shape[0])
    prob_in_bin = np.zeros(bin_vector.shape[0])

    for i, bin_val in enumerate(bin_vector):
        position_idx = position_binned == bin_val

        if np.any(position_idx):
            prob_in_bin[i] = np.nansum(position_idx) / len(position_binned)

            joint_active = np.nansum((binarized_signal == 1) & position_idx) / len(binarized_signal)
            joint_inactive = np.nansum((binarized_signal == 0) & position_idx) / len(binarized_signal)

            cond_active = joint_active / prob_in_bin[i] if prob_in_bin[i] > 0 else 0
            cond_inactive = joint_inactive / prob_in_bin[i] if prob_in_bin[i] > 0 else 0

            info_second = 0
            info_spike = 0

            if joint_active > 0 and prob_active > 0:
                info_second += joint_active * np.log2(cond_active / prob_active)
                info_spike += prob_in_bin[i] * cond_active * np.log2(cond_active / prob_active)

            if joint_inactive > 0 and prob_inactive > 0:
                info_second += joint_inactive * np.log2(cond_inactive / prob_inactive)

            info_per_second_bin[i] = info_second
            info_per_spike_bin[i] = info_spike

    info_per_second = np.nansum(info_per_second_bin)
    info_per_spike = np.nansum(info_per_spike_bin) / prob_active if prob_active > 0 else np.nan

    return info_per_spike, info_per_second



def get_spatial_info_from_map(activity_map, occupancy):
    """
    Compute spatial information per spike and per second from activity and occupancy maps.

    Parameters:
    - activity_map: 2D numpy array of summed binary activity per spatial bin.
    - occupancy: 2D numpy array of time spent in each spatial bin.
    Returns:
    - info_per_spike: Spatial information per spike (bits/spike).
    - info_per_second: Spatial information per second (bits/second).
    """
    # Ensure inputs are numpy arrays
    activity_map = np.asarray(activity_map, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)

    # Validate input shapes
    if activity_map.shape != occupancy.shape:
        raise ValueError("activity_map and occupancy must have the same shape.")

    # Compute total occupancy time and total spikes
    total_time = np.nansum(occupancy)
    total_spikes = np.nansum(activity_map)

    if total_time == 0 or total_spikes == 0:
        return 0.0, 0.0

    # Compute occupancy probability per bin
    P_i = occupancy / total_time

    # Compute firing rate per bin (spikes per second)
    with np.errstate(divide='ignore', invalid='ignore'):
        firing_rate = np.divide(activity_map, occupancy)
        firing_rate[np.isnan(firing_rate)] = 0.0  # Replace NaNs with zeros

    # Compute mean firing rate across all bins
    mean_firing_rate = total_spikes / total_time

    # Compute spatial information per spike and per second
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(firing_rate, mean_firing_rate)
        log_term = np.log2(ratio, where=ratio > 0)
        log_term[np.isnan(log_term)] = 0.0  # Replace NaNs with zeros
        info_per_spike = np.nansum(P_i * ratio * log_term)
        info_per_second = np.nansum(P_i * firing_rate * log_term)

    return info_per_spike, info_per_second

''' 
This part calculates the mutual information for continuous signals (i.e., calcium imaging data that has been binned into several bins or [near] continuous)

'''

def get_joint_entropy(bin_vector1, bin_vector2, nbins_1, nbins_2):

    eps = np.finfo(float).eps

    bin_vector1 = np.copy(bin_vector1)
    bin_vector2 = np.copy(bin_vector2)

    jointprobs = np.zeros([nbins_1, nbins_2])

    for i1 in range(nbins_1):
        for i2 in range(nbins_2):
            jointprobs[i1, i2] = np.nansum((bin_vector1 == i1) & (bin_vector2 == i2))

    jointprobs = jointprobs / np.nansum(jointprobs)
    joint_entropy = -np.nansum(jointprobs * np.log2(jointprobs + eps))

    return joint_entropy

def get_entropy(binned_input, num_bins):

    """
    Calculate the entropy of binned data.

    Parameters:
        binned_data (numpy.ndarray): An array of data points sorted into bins.
        num_bins (int): The number of bins used to group the data.

    Returns:
        entropy (float): The calculated entropy of the binned data.
    """
    
    eps = np.finfo(float).eps

    hdat = np.histogram(binned_input, num_bins)[0]
    hdat = hdat / np.nansum(hdat)
    entropy = -np.nansum(hdat * np.log2(hdat + eps))

    return entropy

def get_mutual_information_binned(calcium_imag_valid_binned,nbins_cal, position_binned_valid,nbins_pos):
    """
    Calculate the mutual information between two random variables.

    Parameters:
        entropy1 (float): Entropy of the first random variable.
        entropy2 (float): Entropy of the second random variable.
        joint_entropy (float): Joint entropy of both random variables.

    Returns:
        mutual_info (float): The calculated mutual information between the random variables.
    """

    entropy1 = get_entropy(position_binned_valid, nbins_pos)
    entropy2 = get_entropy(calcium_imag_valid_binned, nbins_cal)
    joint_entropy = get_joint_entropy(position_binned_valid, calcium_imag_valid_binned, nbins_pos,nbins_cal)

    mutual_info = entropy1 + entropy2 - joint_entropy
    return mutual_info

def get_mutual_information_zscored(mutual_info_original, mutual_info_shifted):
    mutual_info_centered = mutual_info_original - np.nanmean(mutual_info_shifted)
    mutual_info_zscored = (mutual_info_original - np.nanmean(mutual_info_shifted)) / np.nanstd(
        mutual_info_shifted)

    return mutual_info_zscored, mutual_info_centered


def get_binned_signal(calcium_imag, nbins_cal):
    """
    Bin calcium signals across time for each cell independently.

    Parameters
    ----------
    calcium_imag : np.ndarray
        Array of shape (n_cells, n_time) or (n_time,) containing calcium signals.
    nbins_cal : int
        Number of bins to divide each signal into.

    Returns
    -------
    calcium_imag_binned : np.ndarray
        Binned signal(s) with same shape as input.
    """
    calcium_imag = np.atleast_2d(calcium_imag)
    n_cells, n_time = calcium_imag.shape
    calcium_imag_binned = np.zeros((n_cells, n_time), dtype=int)

    for i in range(n_cells):
        signal = calcium_imag[i]
        min_val, max_val = np.nanmin(signal), np.nanmax(signal)
        if min_val == max_val:
            calcium_imag_binned[i] = 0  # all values the same, assign to bin 0
        else:
            bins = np.linspace(min_val, max_val, nbins_cal + 1)
            for jj in range(nbins_cal):
                I_amp = (signal > bins[jj]) & (signal <= bins[jj + 1])
                calcium_imag_binned[i, I_amp] = jj

    # Return original shape
    if calcium_imag.shape[0] == 1:
        return calcium_imag_binned[0]
    return calcium_imag_binned




def get_mutual_information_classif(calcium_imag, position_binned):
    mutual_info_classif_original = \
    mutual_info_classif(calcium_imag.reshape(-1, 1), position_binned, discrete_features='auto')[0]

    return mutual_info_classif_original

def get_mutual_information_regression(calcium_imag, position_binned):
    mutual_info_regression_original = \
    mutual_info_regression(calcium_imag.reshape(-1, 1), position_binned, discrete_features='auto')[0]

    return mutual_info_regression_original




def get_mutual_information_binarized(binarized_signal, position_binned):

    

    bin_vector = np.unique(position_binned)

    # Create bin vectors
    prob_being_active = np.nansum(binarized_signal == 1) / binarized_signal.shape[0]  # Expressed in probability of firing (<1)
    # prob_being_inactive = 1-prob_being_active
    prob_being_inactive = np.nansum(binarized_signal == 0) / binarized_signal.shape[0]  # Expressed in probability of firing (<1)

    # Compute joint probabilities (of cell being active while being in a state bin)
    info_pos = np.zeros(bin_vector.shape[0])
    prob_in_bin = np.zeros(bin_vector.shape[0])
    for i in range(bin_vector.shape[0]):
        position_idx = position_binned == bin_vector[i]

        info_bin = 0
        if np.nansum(position_idx) > 0:

            active_in_bin_idx = np.nansum((binarized_signal == 1) & position_idx)
            joint_prob_active = active_in_bin_idx / np.nansum(position_idx)
            if joint_prob_active /prob_being_active > 0:
                aux = joint_prob_active * np.log2(joint_prob_active /prob_being_active)

                if not np.isnan(aux):
                    info_bin += aux
    

            # inactivity_in_bin_idx = 1-active_in_bin_idx
            inactivity_in_bin_idx = np.nansum((binarized_signal == 0) & position_idx)
            joint_prob_inactive = inactivity_in_bin_idx / np.nansum(position_idx)
            if joint_prob_inactive /prob_being_inactive > 0:
                aux = joint_prob_inactive * np.log2(joint_prob_inactive /prob_being_inactive)
                if not np.isnan(aux):
                    info_bin += aux
                        
            info_pos[i] = info_bin.copy()
            prob_in_bin[i] = np.nansum(position_idx) / position_binned.shape[0]

    mutual_info = np.nansum(info_pos*prob_in_bin)
    return mutual_info





def get_mutual_information_2d(calcium_imag_binned,position_binned,nbins_cal,nbins_pos,x_grid,y_grid,map_smoothing_sigma_x,map_smoothing_sigma_y):

    total_num_events = calcium_imag_binned.shape[0]

    I_pos_xi = []
    I_pos_xi_c = []
    P_xi = []
    for i in range(nbins_pos):

        x_i = np.where(position_binned == i)[0]
        num_x_i_events = x_i.shape[0]
        P_xi.append(num_x_i_events / total_num_events)

        x_i_c = np.where(position_binned != i)[0]
        num_x_i_c_events = x_i_c.shape[0]

        mutual_info_xi = 0
        mutual_info_xi_c = 0
        if num_x_i_events > 0:
            for k in range(nbins_cal):

                num_k_events = np.where(calcium_imag_binned == k)[0].shape[0]
                P_k = num_k_events / total_num_events

                num_k_events_given_x_i = np.where(calcium_imag_binned[x_i] == k)[0].shape[0]
                P_k_xi = num_k_events_given_x_i / num_x_i_events

                num_k_events_given_x_i_c = np.where(calcium_imag_binned[x_i_c] == k)[0].shape[0]
                P_k_xi_c = num_k_events_given_x_i_c / num_x_i_c_events

                if (P_k != 0) & (P_k_xi != 0):
                    mutual_info_xi += P_k_xi * np.log2(P_k_xi / P_k)

                if (P_k != 0) & (P_k_xi_c != 0):
                    mutual_info_xi_c += P_k_xi_c * np.log2(P_k_xi_c / P_k)

        I_pos_xi.append(mutual_info_xi)
        I_pos_xi_c.append(mutual_info_xi_c)

    I_pos_xi = np.array(I_pos_xi)
    I_pos_xi_c = np.array(I_pos_xi_c)

    P_xi_c = 1 - np.array(P_xi)
    P_xi = np.array(P_xi)

    I_bezzi = P_xi * I_pos_xi + P_xi_c * I_pos_xi_c
    mutual_info_distribution_bezzi = I_bezzi.reshape((x_grid.shape[0] - 1), (y_grid.shape[0] - 1)).T

    mutual_info_distribution = P_xi * I_pos_xi
    mutual_info_distribution = mutual_info_distribution.reshape((x_grid.shape[0] - 1), (y_grid.shape[0] - 1)).T
    
    sigma_x_points = sf.get_sigma_points(map_smoothing_sigma_x,x_grid)
    sigma_y_points = sf.get_sigma_points(map_smoothing_sigma_y,y_grid)
    kernel,(x_mesh,y_mesh) = sf.generate_2d_gaussian_kernel(sigma_x = sigma_x_points, sigma_y=sigma_y_points)
                                
    mutual_info_distribution_smoothed = sf.gaussian_smooth_2d(mutual_info_distribution,kernel)
    mutual_info_distribution_bezzi_smoothed = sf.gaussian_smooth_2d(mutual_info_distribution_bezzi,kernel)

    return mutual_info_distribution,mutual_info_distribution_bezzi,mutual_info_distribution_smoothed,mutual_info_distribution_bezzi_smoothed


def get_kullback_leibler_normalized(calcium_imag, position_binned):

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    position_bins = np.unique(position_binned)
    nbin = position_bins.shape[0]

    mean_calcium_activity = []
    for pos in position_bins:
        I_pos = np.where(pos == position_binned)[0]
        mean_calcium_activity.append(np.nanmean(calcium_imag[I_pos]))
    mean_calcium_activity = np.array(mean_calcium_activity)

    observed_distr = -np.nansum((mean_calcium_activity / np.nansum(mean_calcium_activity)) * np.log(
        (mean_calcium_activity / np.nansum(mean_calcium_activity))))
    test_distr = np.log(nbin)
    modulation_index = (test_distr - observed_distr) / test_distr
    return modulation_index

def get_mutual_info_skaggs(calcium_imag, position_binned):
    
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    overall_mean_amplitude = np.nanmean(calcium_imag)

    position_bins = np.unique(position_binned)
    nbin = position_bins.shape[0]

    bin_probability = []
    mean_calcium_activity = []
    for pos in position_bins:
        I_pos = np.where(pos == position_binned)[0]
        bin_probability.append(I_pos.shape[0] / position_binned.shape[0])
        mean_calcium_activity.append(np.nanmean(calcium_imag[I_pos]))
    mean_calcium_activity = np.array(mean_calcium_activity)
    bin_probability = np.array(bin_probability)

    mutual_info_skaggs = np.nansum((bin_probability * (mean_calcium_activity / overall_mean_amplitude)) * np.log2(
        mean_calcium_activity / overall_mean_amplitude))

    # spatial info in bits per deltaF/F s^-1

    return mutual_info_skaggs








def extract_2D_information(binarized_trace, interp_behav_vec, X_bin_vector, Y_bin_vector, inclusion_vector):
    """
    # I translated it from https://github.com/etterguillaume/CaImDecoding/blob/master/extract_1D_information.m
    Analyzes spatial information by computing mutual information between neuronal activity and position.

    Parameters:
    - binarized_trace: 1D numpy array of binary values indicating neuron activity.
    - interp_behav_vec: 2D numpy array of shape (n, 2), interpolated behavioral positions.
    - X_bin_vector: 1D numpy array defining the bin edges along the X-axis.
    - Y_bin_vector: 1D numpy array defining the bin edges along the Y-axis.
    - inclusion_vector: 1D boolean numpy array indicating which time points to include.

    Returns:
    - MI: Mutual Information value.
    - posterior: 2D numpy array representing the posterior probability map.
    - occupancy_map: 2D numpy array representing the occupancy probability map.
    - prob_being_active: Float, probability of the neuron being active.
    - likelihood: 2D numpy array representing the likelihood of neuron activity given position.
    """
    # Apply inclusion vector
    binarized_trace = binarized_trace[inclusion_vector]
    interp_behav_vec = interp_behav_vec[inclusion_vector, :]

    prob_being_active = np.sum(binarized_trace) / len(binarized_trace)

    num_y_bins = len(Y_bin_vector) - 1
    num_x_bins = len(X_bin_vector) - 1

    likelihood = np.zeros((num_y_bins, num_x_bins))
    occupancy_map = np.zeros((num_y_bins, num_x_bins))
    MI = 0.0

    for y in range(num_y_bins):
        for x in range(num_x_bins):
            # Identify indices where position falls within the current bin
            in_bin = (
                (interp_behav_vec[:, 0] >= X_bin_vector[x]) & (interp_behav_vec[:, 0] < X_bin_vector[x + 1]) &
                (interp_behav_vec[:, 1] >= Y_bin_vector[y]) & (interp_behav_vec[:, 1] < Y_bin_vector[y + 1])
            )

            position_idx = np.where(in_bin)[0]

            if position_idx.size > 0:
                occupancy_map[y, x] = position_idx.size / len(binarized_trace)

                activity_in_bin = binarized_trace[position_idx]
                likelihood[y, x] = np.sum(activity_in_bin) / position_idx.size

                joint_prob_active = np.sum(activity_in_bin) / len(binarized_trace)
                joint_prob_inactive = (position_idx.size - np.sum(activity_in_bin)) / len(binarized_trace)
                prob_in_bin = position_idx.size / len(binarized_trace)

                if joint_prob_active > 0:
                    MI += joint_prob_active * np.log2(joint_prob_active / (prob_in_bin * prob_being_active))
                if joint_prob_inactive > 0:
                    MI += joint_prob_inactive * np.log2(joint_prob_inactive / (prob_in_bin * (1 - prob_being_active)))

    # Compute posterior probability map
    with np.errstate(divide='ignore', invalid='ignore'):
        posterior = np.divide(likelihood * occupancy_map, prob_being_active)
        posterior = np.nan_to_num(posterior)

    return MI, posterior, occupancy_map, prob_being_active, likelihood



def calculate_spatial_info_nkinsky(FT, x, y, speed, cmperbin):
    """
    I translated it from https://github.com/nkinsky/ImageCamp/blob/e48c6fac407ef3997b67474a2333184bbc4915dc/General/CalculateSpatialInfo.m

    Calculate spatial information for each neuron based on spatial binning.

    Parameters:
    - FT: (NumNeurons, NumFrames) binary array of neural activity (1 = active)
    - x, y: arrays of position coordinates (length = NumFrames)
    - speed: array of speed values (length = NumFrames), unused but can be for movement filtering
    - cmperbin: spatial bin size in cm

    Returns:
    - INFO: (NumNeurons,) spatial information per neuron (bits/spike)
    - p_i: (num_bins,) occupancy probability per bin
    - lambda_: (NumNeurons,) mean firing rate per neuron
    - lambda_i: (NumNeurons, num_bins) mean firing rate per neuron per bin
    """

    NumNeurons = FT.shape[0]
    NumFrames = FT.shape[1]

    # Assume all frames are valid (e.g., no movement filtering)
    turn_ind = np.ones(NumFrames, dtype=bool)

    # Compute spatial range
    Xrange = np.nanmax(x) - np.nanmin(x)
    Yrange = np.nanmax(y) - np.nanmin(y)

    # Determine number of spatial bins
    NumXBins = int(np.ceil(Xrange / cmperbin))
    NumYBins = int(np.ceil(Yrange / cmperbin))

    # Bin edges
    Xedges = np.arange(0, NumXBins + 1) * cmperbin + np.nanmin(x)
    Yedges = np.arange(0, NumYBins + 1) * cmperbin + np.nanmin(y)

    # Digitize positions into bins
    Xbin = np.digitize(x, Xedges) - 1
    Ybin = np.digitize(y, Yedges) - 1

    # Clip to valid range
    Xbin[Xbin >= NumXBins] = NumXBins - 1
    Ybin[Ybin >= NumYBins] = NumYBins - 1
    Xbin[Xbin < 0] = 0
    Ybin[Ybin < 0] = 0

    # Convert 2D bins to linear index
    loc_index = np.ravel_multi_index((Xbin, Ybin), (NumXBins, NumYBins))

    # Identify unique bins and initialize
    bins = np.unique(loc_index)
    num_bins = len(bins)

    lambda_ = np.full(NumNeurons, np.nan)          # overall mean firing rate per neuron
    lambda_i = np.full((NumNeurons, num_bins), np.nan)  # bin-specific mean firing rates
    p_i = np.full(num_bins, np.nan)                # occupancy probabilities
    INFO = np.full(NumNeurons, np.nan)             # output spatial information per neuron
    in_bin = np.zeros((NumFrames, num_bins), dtype=bool)  # mask of frames in each bin
    dwell = np.full(num_bins, np.nan)              # number of frames spent in each bin
    LR_Frames = np.sum(turn_ind)                   # total number of valid frames

    # Compute dwell time and occupancy for each bin
    for i, bin_val in enumerate(bins):
        in_bin[:, i] = loc_index == bin_val
        dwell[i] = np.nansum(in_bin[turn_ind, i])
        p_i[i] = dwell[i] / LR_Frames

    # Exclude low-occupancy bins
    dwell[dwell <= 1] = np.nan
    p_i[np.isnan(dwell)] = np.nan

    good = ~np.isnan(p_i)

    for this_neuron in range(NumNeurons):
        if (this_neuron + 1) % 10 == 0:
            print(f'Calculating spatial information for neuron #{this_neuron + 1}...')

        # Mean firing rate over all valid frames
        lambda_[this_neuron] = np.nanmean(FT[this_neuron, turn_ind])

        # Mean firing rate in each spatial bin
        for i in range(num_bins):
            if good[i]:
                frames_in_bin = turn_ind & in_bin[:, i]
                if np.nansum(frames_in_bin) > 0:
                    lambda_i[this_neuron, i] = np.nansum(FT[this_neuron, frames_in_bin]) / dwell[i]

        # Calculate spatial information
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = lambda_i[this_neuron, :] / lambda_[this_neuron]
            info_terms = p_i * lambda_i[this_neuron, :] * np.log2(ratio)
            # info_terms = p_i * ratio * np.log2(ratio)
            INFO[this_neuron] = np.nansum(info_terms)

    return INFO, p_i, lambda_, lambda_i




import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors, KDTree, BallTree
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed

def mutual_info_regression_scratch(X, y, n_neighbors=3, random_state=None, n_jobs=1, tree_method='auto'):
    """
    Estimate mutual information between each feature in X and the continuous target y
    using the Kraskov kNN method.

    Parameters:
    - X: ndarray of shape (n_samples, n_features)
    - y: ndarray of shape (n_samples,)
    - n_neighbors: int, number of nearest neighbors (default 3)
    - random_state: int or None, for reproducible neighbor tie-breaking
    - n_jobs: int, number of parallel jobs (default 1)
    - tree_method: str, one of {'auto', 'kd_tree', 'ball_tree', 'brute'} (default 'auto')

    Returns:
    - mi: ndarray of shape (n_features,), estimated mutual information values
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n_samples, n_features = X.shape

    # MinMax scaling to [0, 1], like sklearn does internally
    scaler_x = MinMaxScaler()
    X = scaler_x.fit_transform(X)
    scaler_y = MinMaxScaler()
    y = scaler_y.fit_transform(y)

    # Shuffle to break ties similarly to sklearn
    if random_state is not None:
        rng = np.random.default_rng(seed=random_state)
        indices = rng.permutation(n_samples)
        X = X[indices]
        y = y[indices]

    eps_machine = np.finfo(float).eps

    def get_tree(X_data, method):
        if method == 'kd_tree':
            return KDTree(X_data, metric='chebyshev')
        elif method == 'ball_tree':
            return BallTree(X_data, metric='chebyshev')
        else:
            return NearestNeighbors(metric='chebyshev').fit(X_data)

    def compute_mi_feature(x):
        x = x.reshape(-1, 1)
        xy = np.hstack((x, y))

        if tree_method in ['kd_tree', 'ball_tree']:
            tree_xy = get_tree(xy, tree_method)
            eps = tree_xy.query(xy, k=n_neighbors + 1, return_distance=True)[0][:, -1]
        else:
            nn_xy = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='chebyshev')
            nn_xy.fit(xy)
            distances, _ = nn_xy.kneighbors(xy)
            eps = distances[:, -1]

        tree_x = get_tree(x, tree_method)
        tree_y = get_tree(y, tree_method)

        nx = np.empty(n_samples, dtype=int)
        ny = np.empty(n_samples, dtype=int)
        for i in range(n_samples):
            radius = eps[i] - eps_machine
            if tree_method in ['kd_tree', 'ball_tree']:
                nx[i] = len(tree_x.query_radius([x[i]], r=radius, count_only=False)[0]) - 1
                ny[i] = len(tree_y.query_radius([y[i]], r=radius, count_only=False)[0]) - 1
            else:
                nx[i] = len(tree_x.radius_neighbors([x[i]], radius=radius, return_distance=False)[0]) - 1
                ny[i] = len(tree_y.radius_neighbors([y[i]], radius=radius, return_distance=False)[0]) - 1

        return digamma(n_neighbors) + digamma(n_samples) - np.nanmean(digamma(nx + 1) + digamma(ny + 1))

    mi = Parallel(n_jobs=n_jobs)(delayed(compute_mi_feature)(X[:, j]) for j in range(n_features))

    return np.array(mi)


import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors, KDTree, BallTree
from sklearn.preprocessing import MinMaxScaler

def mutual_info_regression_multivariate(X, y, n_neighbors=3, random_state=None, tree_method='auto'):
    """
    Estimate mutual information between the multivariate X (all features together)
    and continuous target y using Kraskov kNN method.

    Parameters:
    - X: ndarray of shape (n_samples, n_features)
    - y: ndarray of shape (n_samples,)
    - n_neighbors: int, number of nearest neighbors
    - random_state: int or None, for reproducible tie-breaking
    - tree_method: one of {'auto', 'kd_tree', 'ball_tree'}

    Returns:
    - mi: scalar, estimated mutual information between X and y
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n_samples, n_features = X.shape

    # Scale X and y to [0,1]
    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)

    # Shuffle to break ties
    if random_state is not None:
        rng = np.random.default_rng(seed=random_state)
        indices = rng.permutation(n_samples)
        X_scaled = X_scaled[indices]
        y_scaled = y_scaled[indices]

    eps_machine = np.finfo(float).eps

    # Helper to get tree object
    def get_tree(data):
        if tree_method == 'kd_tree':
            return KDTree(data, metric='chebyshev')
        elif tree_method == 'ball_tree':
            return BallTree(data, metric='chebyshev')
        else:
            return NearestNeighbors(metric='chebyshev').fit(data)

    # Build joint space (X + y)
    xy = np.hstack((X_scaled, y_scaled))

    # Build neighbor structures
    if tree_method in ['kd_tree', 'ball_tree']:
        tree_xy = get_tree(xy)
        distances, _ = tree_xy.query(xy, k=n_neighbors + 1, return_distance=True)
        eps = distances[:, -1]
    else:
        nn_xy = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='chebyshev')
        nn_xy.fit(xy)
        distances, _ = nn_xy.kneighbors(xy)
        eps = distances[:, -1]

    tree_x = get_tree(X_scaled)
    tree_y = get_tree(y_scaled)

    nx = np.empty(n_samples, dtype=int)
    ny = np.empty(n_samples, dtype=int)
    for i in range(n_samples):
        radius = eps[i] - eps_machine
        if tree_method in ['kd_tree', 'ball_tree']:
            nx[i] = len(tree_x.query_radius([X_scaled[i]], r=radius, count_only=False)[0]) - 1
            ny[i] = len(tree_y.query_radius([y_scaled[i]], r=radius, count_only=False)[0]) - 1
        else:
            nx[i] = len(tree_x.radius_neighbors([X_scaled[i]], radius=radius, return_distance=False)[0]) - 1
            ny[i] = len(tree_y.radius_neighbors([y_scaled[i]], radius=radius, return_distance=False)[0]) - 1

    mi = digamma(n_neighbors) + digamma(n_samples) - np.nanmean(digamma(nx + 1) + digamma(ny + 1))
    return mi

def estimate_optimal_k(n_samples):
    """
    Estimate a reasonable number of neighbors (k) for kNN-based mutual information estimation.
    This is heuristic and based on Kraskov et al. (2004).

    Parameters:
    - n_samples: int, number of data samples

    Returns:
    - k: int, suggested number of neighbors
    """
    if n_samples < 20:
        return 2  # Very small dataset
    elif n_samples < 100:
        return 3
    elif n_samples < 500:
        return 5
    elif n_samples < 1000:
        return 10
    else:
        return min(20, int(np.log(n_samples)))

