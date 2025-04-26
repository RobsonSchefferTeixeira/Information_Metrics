import numpy as np
from src.utils import smoothing_functions as sf
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
import warnings

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

    calcium_imag_bins = np.linspace(np.nanmin(calcium_imag), np.nanmax(calcium_imag), nbins_cal + 1)
    calcium_imag_binned = np.zeros(calcium_imag.shape[0])
    for jj in range(calcium_imag_bins.shape[0] - 1):
        I_amp = (calcium_imag > calcium_imag_bins[jj]) & (calcium_imag <= calcium_imag_bins[jj + 1])
        calcium_imag_binned[I_amp] = jj

    return calcium_imag_binned


def get_mutual_information_NN(calcium_imag, position_binned):
    mutual_info_NN_original = \
    mutual_info_classif(calcium_imag.reshape(-1, 1), position_binned, discrete_features=False)[0]

    return mutual_info_NN_original

def get_mutual_information_regression(calcium_imag, position_binned):
    mutual_info_regression_original = \
    mutual_info_regression(calcium_imag.reshape(-1, 1), position_binned, discrete_features=False)[0]

    return mutual_info_regression_original


def get_mutual_information_bkp(calcium_imag, position_binned):

    # I've translated this code to Python. 
    # Originally I took it from https://github.com/etterguillaume/CaImDecoding/blob/master/extract_1D_information.m
    # https://github.com/nkinsky/ImageCamp/blob/e48c6fac407ef3997b67474a2333184bbc4915dc/General/CalculateSpatialInfo.m

    # I'm calling the input variable as calcium_imag just for the sake of class inheritance, but a better name
    # would be binarized_signal
    bin_vector = np.unique(position_binned)

    # Create bin vectors
    prob_being_active = np.nansum(calcium_imag) / calcium_imag.shape[0]  # Expressed in probability of firing (<1)

    # Compute joint probabilities (of cell being active while being in a state bin)
    likelihood = []
    occupancy_vector = []

    mutual_info = 0
    for i in range(bin_vector.shape[0]):
        position_idx = position_binned == bin_vector[i]

        if np.sum(position_idx) > 0:
            occupancy_vector.append(position_idx.shape[0] / position_binned.shape[0])

            activity_in_bin_idx = np.where((calcium_imag == 1) & position_idx)[0]
            inactivity_in_bin_idx = np.where((calcium_imag == 0) & position_idx)[0]
            likelihood.append(activity_in_bin_idx.shape[0] / np.sum(position_idx))

            joint_prob_active = activity_in_bin_idx.shape[0] / calcium_imag.shape[0]
            joint_prob_inactive = inactivity_in_bin_idx.shape[0] / calcium_imag.shape[0]
            prob_in_bin = np.sum(position_idx) / position_binned.shape[0]

            if joint_prob_active > 0:
                mutual_info = mutual_info + joint_prob_active * np.log2(
                    joint_prob_active / (prob_in_bin * prob_being_active))

            if joint_prob_inactive > 0:
                mutual_info = mutual_info + joint_prob_inactive * np.log2(
                    joint_prob_inactive / (prob_in_bin * (1 - prob_being_active)))
    occupancy_vector = np.array(occupancy_vector)
    likelihood = np.array(likelihood)

    posterior = likelihood * occupancy_vector / prob_being_active

    return mutual_info


def get_mutual_information_binarized(binarized_signal, position_binned):

    # https://github.com/etterguillaume/CaImDecoding/blob/master/extract_1D_information.m
    # https://github.com/nkinsky/ImageCamp/blob/e48c6fac407ef3997b67474a2333184bbc4915dc/General/CalculateSpatialInfo.m

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



def get_mutual_information_binarized_from_map(x_coordinates, y_coordinates, activity, bin_size=4.0):

    '''
    Compute spatial mutual information between position (x, y) and binary activity.

    Parameters:
    - x_coordinates: 1D array of x positions.
    - y_coordinates: 1D array of y positions.
    - activity: 1D array of binary activity (0 or 1).
    - bin_size: Size of spatial bins (default is 4.0 units).

    Returns:
    - mutual_info: Mutual information value in bits.
    '''
    # Ensure inputs are numpy arrays
    x = np.asarray(x_coordinates)
    y = np.asarray(y_coordinates)
    a = np.asarray(activity)

    # Validate input lengths
    if not (len(x) == len(y) == len(a)):
        raise ValueError("Input arrays x_coordinates, y_coordinates, and activity must have the same length.")

    # Define spatial bin edges
    x_edges = np.arange(np.nanmin(x), np.nanmax(x) + bin_size, bin_size)
    y_edges = np.arange(np.nanmin(y), np.nanmax(y) + bin_size, bin_size)

    # Digitize positions to get bin indices
    x_bin = np.digitize(x, x_edges) - 1
    y_bin = np.digitize(y, y_edges) - 1

    # Initialize occupancy and activity maps
    occupancy = np.zeros((len(x_edges), len(y_edges)))
    activity_map = np.zeros((len(x_edges), len(y_edges)))

    # Populate maps
    for xi, yi, act in zip(x_bin, y_bin, a):
        if 0 <= xi < occupancy.shape[0] and 0 <= yi < occupancy.shape[1]:
            occupancy[xi, yi] += 1
            activity_map[xi, yi] += act

    # Compute probabilities
    total_time = np.nansum(occupancy)
    P_x = occupancy / total_time  # P(x_i)
    P_k = np.array([np.nanmean(a == k) for k in [0, 1]])  # P(k)

    # Compute conditional probabilities P(k|x_i)
    P_k_given_x = np.zeros((2, occupancy.shape[0], occupancy.shape[1]))
    with np.errstate(divide='ignore', invalid='ignore'):
        P_k_given_x[1] = activity_map / occupancy
        P_k_given_x[0] = 1 - P_k_given_x[1]
        P_k_given_x = np.nan_to_num(P_k_given_x)


    # Compute spatial information
    info_pos = 0
    for k in [0, 1]:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = P_k_given_x[k] / P_k[k]
            log_term = np.log2(ratio, where=ratio > 0)
            log_term = np.nan_to_num(log_term)
            info_pos += P_k_given_x[k] * log_term
    spatial_information = np.nansum(info_pos*P_x)

    '''
    mutual_info = 0.0
    for k in [0, 1]:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = P_k_given_x[k] / P_k[k]
            log_term = np.log2(ratio, where=ratio > 0)
            log_term = np.nan_to_num(log_term)
            mutual_info += np.nansum(P_x * P_k_given_x[k] * log_term)
    '''
    return spatial_information


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