import numpy as np
import os
from spatial_metrics import helper_functions as hf
from spatial_metrics import detect_peaks as dp
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
import warnings


class PlaceCell:

    def __init__(self, **kwargs):

        kwargs.setdefault('animal_id', None)
        kwargs.setdefault('day', None)
        kwargs.setdefault('neuron', None)
        kwargs.setdefault('trial', None)
        kwargs.setdefault('dataset', None)
        kwargs.setdefault('mean_video_srate', 30.)
        kwargs.setdefault('min_time_spent', 0.1)
        kwargs.setdefault('min_visits', 1)
        kwargs.setdefault('min_speed_threshold', 2.5)
        kwargs.setdefault('x_bin_size', 1)
        kwargs.setdefault('y_bin_size', 1)
        kwargs.setdefault('environment_edges', None)
        kwargs.setdefault('smoothing_size', 2)
        kwargs.setdefault('shift_time', 10)
        kwargs.setdefault('num_cores', 1)
        kwargs.setdefault('num_surrogates', 200)
        kwargs.setdefault('saving_path', os.getcwd())
        kwargs.setdefault('saving', False)
        kwargs.setdefault('saving_string', 'SpatialMetrics')
        kwargs.setdefault('nbins_cal', 10)
        kwargs.setdefault('percentile_threshold', 95)
        kwargs.setdefault('min_num_of_pixels', 4)

        valid_kwargs = ['animal_id', 'day', 'neuron', 'dataset', 'trial', 'mean_video_srate',
                        'min_time_spent', 'min_visits', 'min_speed_threshold', 'smoothing_size',
                        'x_bin_size', 'y_bin_size', 'shift_time', 'num_cores', 'percentile_threshold','min_num_of_pixels',
                        'num_surrogates', 'saving_path', 'saving', 'saving_string', 'environment_edges', 'nbins_cal']

        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            setattr(self, k, v)

        self.__dict__['input_parameters'] = kwargs

    def main(self, calcium_imag, track_timevector, x_coordinates, y_coordinates):

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        
        if np.all(np.isnan(calcium_imag)):
            warnings.warn("Signal contains only NaN's")
            inputdict = np.nan
            filename = self.filename_constructor(self.saving_string, self.animal_id, self.dataset, self.day,
                                                 self.neuron, self.trial)
        else:
            
            if np.any(np.isnan(calcium_imag)):
                I_keep = ~np.isnan(calcium_imag)
                calcium_imag = calcium_imag[I_keep]
                track_timevector = track_timevector[I_keep]
                x_coordinates = x_coordinates[I_keep]
                y_coordinates = y_coordinates[I_keep]

    
            speed = hf.get_speed(x_coordinates, y_coordinates, track_timevector)

            x_grid, y_grid, x_center_bins, y_center_bins, x_center_bins_repeated, y_center_bins_repeated = \
                hf.get_position_grid(x_coordinates, y_coordinates, self.x_bin_size,
                                       self.y_bin_size, environment_edges=self.environment_edges)

            position_binned = hf.get_binned_2Dposition(x_coordinates, y_coordinates, x_grid, y_grid)

            visits_bins, new_visits_times = hf.get_visits(x_coordinates, y_coordinates, position_binned,
                                                            x_center_bins, y_center_bins)
            time_spent_inside_bins = hf.get_position_time_spent(position_binned, self.mean_video_srate)

            I_keep = self.get_valid_timepoints(calcium_imag, speed, visits_bins, time_spent_inside_bins,
                                               self.min_speed_threshold, self.min_visits, self.min_time_spent)

            calcium_imag_valid = calcium_imag[I_keep].copy()
            x_coordinates_valid = x_coordinates[I_keep].copy()
            y_coordinates_valid = y_coordinates[I_keep].copy()
            track_timevector_valid = track_timevector[I_keep].copy()
            visits_bins_valid = visits_bins[I_keep].copy()
            position_binned_valid = position_binned[I_keep].copy()

            position_occupancy = hf.get_occupancy(x_coordinates_valid, y_coordinates_valid, x_grid, y_grid,
                                                    self.mean_video_srate)
            visits_occupancy = hf.get_visits_occupancy(x_coordinates, y_coordinates, new_visits_times, x_grid, y_grid,
                                                         self.min_visits)

            place_field, place_field_smoothed = self.get_place_field(calcium_imag_valid, x_coordinates_valid,
                                                                     y_coordinates_valid, x_grid, y_grid,
                                                                     self.smoothing_size)

            calcium_imag_valid_binned = self.get_binned_signal(calcium_imag_valid, self.nbins_cal)
            nbins_pos = (x_grid.shape[0] - 1) * (y_grid.shape[0] - 1)
            entropy1 = self.get_entropy(position_binned_valid, nbins_pos)
            entropy2 = self.get_entropy(calcium_imag_valid_binned, self.nbins_cal)
            joint_entropy = self.get_joint_entropy(position_binned_valid, calcium_imag_valid_binned, nbins_pos,
                                                   self.nbins_cal)

            mutual_info_original = self.get_mutual_information(entropy1, entropy2, joint_entropy)
            mutual_info_kullback_leibler_original = self.get_kullback_leibler_normalized(calcium_imag_valid,
                                                                                         position_binned_valid)
            mutual_info_NN_original = self.get_mutual_information_NN(calcium_imag_valid, position_binned_valid)
            mutual_info_skaggs_original = self.get_mutual_info_skaggs(calcium_imag_valid, position_binned_valid)

            mutual_info_distribution, mutual_info_distribution_bezzi = self.get_mutual_information_2d(
                calcium_imag_valid_binned,position_binned_valid, y_grid, x_grid, self.nbins_cal, nbins_pos,self.smoothing_size)

            results = self.parallelize_surrogate(calcium_imag, I_keep, position_binned_valid, self.mean_video_srate,
                                                 self.shift_time, self.nbins_cal, nbins_pos, x_coordinates_valid,
                                                 y_coordinates_valid, x_grid, y_grid, self.smoothing_size,
                                                 self.num_cores, self.num_surrogates)

            place_field_shuffled = []
            place_field_smoothed_shuffled = []
            mutual_info_shuffled = []
            mutual_info_NN_shuffled = []
            mutual_info_kullback_leibler_shuffled = []
            mutual_info_skaggs_shuffled = []
            mutual_info_distribution_shuffled = []
            mutual_info_distribution_bezzi_shuffled = []

            for perm in range(self.num_surrogates):
                mutual_info_shuffled.append(results[perm][0])
                mutual_info_kullback_leibler_shuffled.append(results[perm][1])
                mutual_info_NN_shuffled.append(results[perm][2])
                mutual_info_skaggs_shuffled.append(results[perm][3])
                place_field_shuffled.append(results[perm][4])
                place_field_smoothed_shuffled.append(results[perm][5])
                mutual_info_distribution_shuffled.append(results[perm][6])
                mutual_info_distribution_bezzi_shuffled.append(results[perm][7])

            mutual_info_NN_shuffled = np.array(mutual_info_NN_shuffled)
            mutual_info_shuffled = np.array(mutual_info_shuffled)
            mutual_info_kullback_leibler_shuffled = np.array(mutual_info_kullback_leibler_shuffled)
            mutual_info_skaggs_shuffled = np.array(mutual_info_skaggs_shuffled)
            place_field_shuffled = np.array(place_field_shuffled)
            place_field_smoothed_shuffled = np.array(place_field_smoothed_shuffled)
            mutual_info_distribution_shuffled = np.array(mutual_info_distribution_shuffled)
            mutual_info_distribution_bezzi_shuffled = np.array(mutual_info_distribution_bezzi_shuffled)

            

            mutual_info_zscored, mutual_info_centered = self.get_mutual_information_zscored(mutual_info_original,
                                                                                            mutual_info_shuffled)
            mutual_info_kullback_leibler_zscored, mutual_info_kullback_leibler_centered = self.get_mutual_information_zscored(
                mutual_info_kullback_leibler_original, mutual_info_kullback_leibler_shuffled)

            mutual_info_NN_zscored, mutual_info_NN_centered = self.get_mutual_information_zscored(
                mutual_info_NN_original, mutual_info_NN_shuffled)

            mutual_info_skaggs_zscored, mutual_info_skaggs_centered = self.get_mutual_information_zscored(
                mutual_info_skaggs_original, mutual_info_skaggs_shuffled)



            num_of_islands, islands_x_max, islands_y_max,pixels_place_cell_absolute,pixels_place_cell_relative,place_field_identity = \
                hf.field_coordinates_using_shuffled(place_field_smoothed,place_field_smoothed_shuffled,visits_occupancy,
                                                    percentile_threshold=self.percentile_threshold,
                                                    min_num_of_pixels = self.min_num_of_pixels)

            I_peaks = dp.detect_peaks(calcium_imag_valid, mpd=0.5 * self.mean_video_srate,
                                      mph=1. * np.nanstd(calcium_imag_valid))
            peaks_amplitude = calcium_imag_valid[I_peaks]
            x_peaks_location = x_coordinates_valid[I_peaks]
            y_peaks_location = y_coordinates_valid[I_peaks]

            sparsity = hf.get_sparsity(place_field, position_occupancy)

            inputdict = dict()
            inputdict['place_field'] = place_field
            inputdict['place_field_smoothed'] = place_field_smoothed

            inputdict['place_field_shuffled'] = place_field_shuffled
            inputdict['place_field_smoothed_shuffled'] = place_field_smoothed_shuffled

            inputdict['mutual_info_distribution'] = mutual_info_distribution
            inputdict['mutual_info_distribution_bezzi'] = mutual_info_distribution_bezzi

            inputdict['mutual_info_distribution_shuffled'] = mutual_info_distribution_shuffled
            inputdict['mutual_info_distribution_bezzi_shuffled'] = mutual_info_distribution_bezzi_shuffled

            inputdict['occupancy_map'] = position_occupancy
            inputdict['visits_map'] = visits_occupancy
            inputdict['x_grid'] = x_grid
            inputdict['y_grid'] = y_grid
            inputdict['x_center_bins'] = x_center_bins
            inputdict['y_center_bins'] = y_center_bins
            inputdict['numb_events'] = I_peaks.shape[0]
            inputdict['x_peaks_location'] = x_peaks_location
            inputdict['y_peaks_location'] = y_peaks_location
            inputdict['events_amplitude'] = peaks_amplitude

            inputdict['place_field_identity'] = place_field_identity
            inputdict['num_of_islands'] = num_of_islands
            inputdict['islands_x_max'] = islands_x_max
            inputdict['islands_y_max'] = islands_y_max
            inputdict['sparsity'] = sparsity

            inputdict['place_cell_extension_absolute'] = pixels_place_cell_absolute
            inputdict['place_cell_extension_relative'] = pixels_place_cell_relative

            inputdict['mutual_info_original'] = mutual_info_original
            inputdict['mutual_info_shuffled'] = mutual_info_shuffled
            inputdict['mutual_info_zscored'] = mutual_info_zscored
            inputdict['mutual_info_centered'] = mutual_info_centered

            inputdict['mutual_info_kullback_leibler_original'] = mutual_info_kullback_leibler_original
            inputdict['mutual_info_kullback_leibler_shuffled'] = mutual_info_kullback_leibler_shuffled
            inputdict['mutual_info_kullback_leibler_zscored'] = mutual_info_kullback_leibler_zscored
            inputdict['mutual_info_kullback_leibler_centered'] = mutual_info_kullback_leibler_centered

            inputdict['mutual_info_NN_original'] = mutual_info_NN_original
            inputdict['mutual_info_NN_shuffled'] = mutual_info_NN_shuffled
            inputdict['mutual_info_NN_zscored'] = mutual_info_NN_zscored
            inputdict['mutual_info_NN_centered'] = mutual_info_NN_centered

            inputdict['mutual_info_skaggs_original'] = mutual_info_skaggs_original
            inputdict['mutual_info_skaggs_shuffled'] = mutual_info_skaggs_shuffled
            inputdict['mutual_info_skaggs_zscored'] = mutual_info_skaggs_zscored
            inputdict['mutual_info_skaggs_centered'] = mutual_info_skaggs_centered

            inputdict['input_parameters'] = self.__dict__['input_parameters']

            filename = hf.filename_constructor(self.saving_string, self.animal_id, self.dataset, self.day, self.neuron,self.trial)

        if self.saving == True:
            hf.caller_saving(inputdict, filename, self.saving_path)
            print(filename + ' saved')

        else:
            print(filename + ' not saved')

        return inputdict


    def get_sparsity(self, place_field, position_occupancy):
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


    def get_valid_timepoints(self, calcium_imag, speed, visits_bins, time_spent_inside_bins, min_speed_threshold,
                             min_visits, min_time_spent):

        # min speed
        I_speed_thres = speed >= min_speed_threshold

        # min visits
        I_visits_times_thres = visits_bins >= min_visits

        # min time spent
        I_time_spent_thres = time_spent_inside_bins >= min_time_spent

        # valid calcium points
        I_valid_calcium = ~np.isnan(calcium_imag)

        I_keep = I_speed_thres * I_visits_times_thres * I_time_spent_thres * I_valid_calcium
        return I_keep

    def get_place_field(self, calcium_imag, x_coordinates, y_coordinates, x_grid, y_grid, smoothing_size):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # calculate mean calcium per pixel
        place_field = np.nan * np.zeros((y_grid.shape[0] - 1, x_grid.shape[0] - 1))
        for xx in range(0, x_grid.shape[0] - 1):
            for yy in range(0, y_grid.shape[0] - 1):
                check_x_occupancy = np.logical_and(x_coordinates >= x_grid[xx], x_coordinates < (x_grid[xx + 1]))
                check_y_occupancy = np.logical_and(y_coordinates >= y_grid[yy], y_coordinates < (y_grid[yy + 1]))

                place_field[yy, xx] = np.nanmean(calcium_imag[np.logical_and(check_x_occupancy, check_y_occupancy)])

        place_field_to_smooth = np.copy(place_field)
        place_field_to_smooth[np.isnan(place_field_to_smooth)] = 0
        place_field_smoothed = hf.gaussian_smooth_2d(place_field_to_smooth, smoothing_size)

        return place_field, place_field_smoothed


    def get_mutual_information_NN(self, calcium_imag, position_binned):
        mutual_info_NN_original = \
        mutual_info_classif(calcium_imag.reshape(-1, 1), position_binned, discrete_features=False, n_neighbors=5)[0]

        return mutual_info_NN_original

    def get_binned_signal(self, calcium_imag, nbins_cal):

        calcium_imag_bins = np.linspace(np.nanmin(calcium_imag), np.nanmax(calcium_imag), nbins_cal + 1)
        calcium_imag_binned = np.zeros(calcium_imag.shape[0])
        for jj in range(calcium_imag_bins.shape[0] - 1):
            I_amp = (calcium_imag > calcium_imag_bins[jj]) & (calcium_imag <= calcium_imag_bins[jj + 1])
            calcium_imag_binned[I_amp] = jj

        return calcium_imag_binned

    def get_joint_entropy(self, bin_vector1, bin_vector2, nbins_1, nbins_2):

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

    def get_entropy(self, binned_input, num_bins):

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

    def get_mutual_information(self, entropy1, entropy2, joint_entropy):
        """
        Calculate the mutual information between two random variables.

        Parameters:
            entropy1 (float): Entropy of the first random variable.
            entropy2 (float): Entropy of the second random variable.
            joint_entropy (float): Joint entropy of both random variables.

        Returns:
            mutual_info (float): The calculated mutual information between the random variables.
        """
        mutual_info = entropy1 + entropy2 - joint_entropy
        return mutual_info

    def get_mutual_information_zscored(self, mutual_info_original, mutual_info_shuffled):
        mutual_info_centered = mutual_info_original - np.nanmean(mutual_info_shuffled)
        mutual_info_zscored = (mutual_info_original - np.nanmean(mutual_info_shuffled)) / np.nanstd(
            mutual_info_shuffled)

        return mutual_info_zscored, mutual_info_centered

    def parallelize_surrogate(self, calcium_imag, I_keep, position_binned_valid, mean_video_srate, shift_time,
                              nbins_cal, nbins_pos, x_coordinates_valid, y_coordinates_valid, x_grid, y_grid,
                              smoothing_size, num_cores, num_surrogates):
        results = Parallel(n_jobs=num_cores)(
            delayed(self.get_mutual_info_surrogate)(calcium_imag, I_keep, position_binned_valid, mean_video_srate,
                                                    shift_time, nbins_cal, nbins_pos, x_coordinates_valid,
                                                    y_coordinates_valid, x_grid, y_grid, smoothing_size)
            for _ in range(num_surrogates))

        return results

    def get_surrogate(self,input_vector, sampling_rate, shift_time):
        """
        Generate a surrogate signal by applying a time shift to the input vector.

        This function creates a surrogate signal by shifting the input vector in time 
        while maintaining the same signal characteristics.

        Parameters:
            input_vector (numpy.ndarray): The input signal to generate a surrogate for.
            sampling_rate (float): The sampling rate of the input signal (samples per second).
            shift_time (float): The desired time shift for the surrogate signal (seconds).

        Returns:
            input_vector_shuffled (numpy.ndarray): The surrogate signal obtained by applying the time shift.
        """
        if len(input_vector) < np.abs(sampling_rate * shift_time):
            # Adjust the shift time if it exceeds the length of the input signal.
            shift_time = np.floor(len(input_vector) / sampling_rate)

        # Generate a random time shift in samples within the specified range.
        shift_samples = np.random.randint(-shift_time * sampling_rate, sampling_rate * shift_time + 1)

        # Apply the time shift to create the surrogate signal.
        input_vector_shuffled = np.concatenate([input_vector[shift_samples:], input_vector[0:shift_samples]])
        # np.roll could be used instead

        return input_vector_shuffled


    def get_mutual_info_surrogate(self, calcium_imag, I_keep, position_binned_valid, mean_video_srate, shift_time,
                                  nbins_cal, nbins_pos, x_coordinates_valid, y_coordinates_valid, x_grid, y_grid,
                                  smoothing_size):

        calcium_imag_shuffled = self.get_surrogate(calcium_imag, mean_video_srate, shift_time)
        calcium_imag_shuffled_valid = calcium_imag_shuffled[I_keep].copy()

        calcium_imag_shuffled_binned = self.get_binned_signal(calcium_imag_shuffled_valid, nbins_cal)
        entropy1 = self.get_entropy(position_binned_valid, nbins_pos)
        entropy2 = self.get_entropy(calcium_imag_shuffled_binned, nbins_cal)
        joint_entropy = self.get_joint_entropy(position_binned_valid, calcium_imag_shuffled_binned, nbins_pos,
                                               nbins_cal)
        mutual_info_shuffled = self.get_mutual_information(entropy1, entropy2, joint_entropy)

        mutual_info_shuffled_NN = self.get_mutual_information_NN(calcium_imag_shuffled_valid, position_binned_valid)

        modulation_index_shuffled = self.get_kullback_leibler_normalized(calcium_imag_shuffled_valid,
                                                                         position_binned_valid)

        mutual_info_skaggs_shuffled = self.get_mutual_info_skaggs(calcium_imag_shuffled_valid, position_binned_valid)

        place_field_shuffled, place_field_smoothed_shuffled = self.get_place_field(calcium_imag_shuffled_valid,
                                                                                   x_coordinates_valid,
                                                                                   y_coordinates_valid, x_grid, y_grid,
                                                                                   smoothing_size)
        
        mutual_info_distribution,mutual_info_distribution_bezzi = self.get_mutual_information_2d(calcium_imag_shuffled_binned,
                                                                                                  position_binned_valid,y_grid,
                                                                                                  x_grid,nbins_cal,nbins_pos,
                                                                                                  smoothing_size)

        return mutual_info_shuffled, modulation_index_shuffled, mutual_info_shuffled_NN, mutual_info_skaggs_shuffled,\
               place_field_shuffled, place_field_smoothed_shuffled,mutual_info_distribution,mutual_info_distribution_bezzi

    def get_mutual_information_2d(self,calcium_imag_binned,position_binned,y_grid,x_grid,nbins_cal,nbins_pos,smoothing_size):

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

        mutual_info_distribution_smoothed = hf.gaussian_smooth_2d(mutual_info_distribution, smoothing_size)
        mutual_info_distribution_bezzi_smoothed = hf.gaussian_smooth_2d(mutual_info_distribution_bezzi, smoothing_size)


        return mutual_info_distribution,mutual_info_distribution_bezzi


    def get_kullback_leibler_normalized(self, calcium_imag, position_binned):

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

    def get_mutual_info_skaggs(self, calcium_imag, position_binned):
        
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

