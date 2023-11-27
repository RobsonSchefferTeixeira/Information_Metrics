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
        kwargs.setdefault('sampling_rate', 30.)
        kwargs.setdefault('min_time_spent', 0.1)
        kwargs.setdefault('min_visits', 1)
        kwargs.setdefault('min_speed_threshold', 2.5)
        kwargs.setdefault('x_bin_size', 1)
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
        kwargs.setdefault('speed_smoothing_points', 1)

        

        valid_kwargs = ['animal_id', 'day', 'neuron', 'dataset', 'trial', 'sampling_rate',
                        'min_time_spent', 'min_visits', 'min_speed_threshold', 'smoothing_size','speed_smoothing_points',
                        'x_bin_size', 'shift_time', 'num_cores', 'percentile_threshold','min_num_of_pixels',
                        'num_surrogates', 'saving_path', 'saving', 'saving_string', 'environment_edges', 'nbins_cal']

        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            setattr(self, k, v)

        self.__dict__['input_parameters'] = kwargs

    def main(self, calcium_imag, time_vector, x_coordinates):

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        
        if np.all(np.isnan(calcium_imag)):
            warnings.warn("Signal contains only NaN's")
            inputdict = np.nan
            filename = self.filename_constructor(self.saving_string, self.animal_id, self.dataset, self.day,self.neuron, self.trial)
        else:
            
     
            I_keep_valid = self.validate_input_data(calcium_imag, time_vector, x_coordinates)

            calcium_imag = calcium_imag[I_keep_valid]
            time_vector = time_vector[I_keep_valid]
            x_coordinates = x_coordinates[I_keep_valid]

    
            _,speed = hf.get_speed_1D(x_coordinates, time_vector,sigma_points=self.speed_smoothing_points)

            x_grid, x_center_bins, x_center_bins_repeated = hf.get_position_grid_1D(x_coordinates, self.x_bin_size,environment_edges=self.environment_edges)

            position_binned = hf.get_binned_position_1D(x_coordinates, x_grid)

            visits_bins, new_visits_times = hf.get_visits_1D(x_coordinates, position_binned,x_center_bins)
            
            time_spent_inside_bins = hf.get_position_time_spent(position_binned, self.sampling_rate)

            I_keep = self.get_valid_timepoints(speed, visits_bins, time_spent_inside_bins,self.min_speed_threshold, self.min_visits, self.min_time_spent)

            calcium_imag_valid = calcium_imag[I_keep].copy()
            x_coordinates_valid = x_coordinates[I_keep].copy()
            time_vector_valid = time_vector[I_keep].copy()
            visits_bins_valid = visits_bins[I_keep].copy()
            position_binned_valid = position_binned[I_keep].copy()

            position_occupancy = hf.get_occupancy_1D(x_coordinates_valid, x_grid,self.sampling_rate)
            
            visits_occupancy = hf.get_visits_occupancy_1D(x_coordinates, new_visits_times, x_grid,self.min_visits)

            place_field, place_field_smoothed = self.get_place_field_1D(calcium_imag_valid, x_coordinates_valid,x_grid,self.smoothing_size)

            calcium_imag_valid_binned = self.get_binned_signal(calcium_imag_valid, self.nbins_cal)
            nbins_pos = (x_grid.shape[0] - 1)
            entropy1 = self.get_entropy(position_binned_valid, nbins_pos)
            entropy2 = self.get_entropy(calcium_imag_valid_binned, self.nbins_cal)
            joint_entropy = self.get_joint_entropy(position_binned_valid, calcium_imag_valid_binned, nbins_pos,
                                                   self.nbins_cal)

            mutual_info_original = self.get_mutual_information(entropy1, entropy2, joint_entropy)
            mutual_info_NN_original = self.get_mutual_information_NN(calcium_imag_valid, position_binned_valid)


            results = self.parallelize_surrogate(calcium_imag, I_keep, position_binned_valid, self.sampling_rate,
                                                 self.shift_time, self.nbins_cal, nbins_pos, x_coordinates_valid,
                                                 x_grid, self.smoothing_size,
                                                 self.num_cores, self.num_surrogates)

            place_field_shifted = []
            place_field_smoothed_shifted = []
            mutual_info_shifted = []
            mutual_info_NN_shifted = []

            for perm in range(self.num_surrogates):
                mutual_info_shifted.append(results[perm][0])
                mutual_info_NN_shifted.append(results[perm][1])
                place_field_shifted.append(results[perm][2])
                place_field_smoothed_shifted.append(results[perm][3])

            mutual_info_NN_shifted = np.array(mutual_info_NN_shifted)
            mutual_info_shifted = np.array(mutual_info_shifted)
            place_field_shifted = np.array(place_field_shifted)
            place_field_smoothed_shifted = np.array(place_field_smoothed_shifted)

            

            mutual_info_zscored, mutual_info_centered = self.get_mutual_information_zscored(mutual_info_original,
                                                                                            mutual_info_shifted)

            mutual_info_NN_zscored, mutual_info_NN_centered = self.get_mutual_information_zscored(
                mutual_info_NN_original, mutual_info_NN_shifted)

          

            num_of_islands, islands_x_max,pixels_place_cell_absolute,pixels_place_cell_relative,place_field_identity = \
                hf.field_coordinates_using_shifted_1D(place_field_smoothed,place_field_smoothed_shifted,visits_occupancy,
                                                    percentile_threshold=self.percentile_threshold,
                                                    min_num_of_pixels = self.min_num_of_pixels)
            


            I_peaks = dp.detect_peaks(calcium_imag_valid, mpd=0.5 * self.sampling_rate,
                                      mph=1. * np.nanstd(calcium_imag_valid))
            peaks_amplitude = calcium_imag_valid[I_peaks]
            x_peaks_location = x_coordinates_valid[I_peaks]

            sparsity = hf.get_sparsity(place_field, position_occupancy)

            inputdict = dict()
            inputdict['place_field'] = place_field
            inputdict['place_field_smoothed'] = place_field_smoothed

            inputdict['place_field_shifted'] = place_field_shifted
            inputdict['place_field_smoothed_shifted'] = place_field_smoothed_shifted

            inputdict['occupancy_map'] = position_occupancy
            inputdict['visits_map'] = visits_occupancy
            inputdict['x_grid'] = x_grid
            inputdict['x_center_bins'] = x_center_bins
            inputdict['numb_events'] = I_peaks.shape[0]
            inputdict['x_peaks_location'] = x_peaks_location
            inputdict['events_amplitude'] = peaks_amplitude

            inputdict['place_field_identity'] = place_field_identity
            inputdict['num_of_islands'] = num_of_islands
            inputdict['islands_x_max'] = islands_x_max
            inputdict['sparsity'] = sparsity

            inputdict['place_cell_extension_absolute'] = pixels_place_cell_absolute
            inputdict['place_cell_extension_relative'] = pixels_place_cell_relative

            inputdict['mutual_info_original'] = mutual_info_original
            inputdict['mutual_info_shifted'] = mutual_info_shifted
            inputdict['mutual_info_zscored'] = mutual_info_zscored
            inputdict['mutual_info_centered'] = mutual_info_centered

            inputdict['mutual_info_NN_original'] = mutual_info_NN_original
            inputdict['mutual_info_NN_shifted'] = mutual_info_NN_shifted
            inputdict['mutual_info_NN_zscored'] = mutual_info_NN_zscored
            inputdict['mutual_info_NN_centered'] = mutual_info_NN_centered

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


    
    def validate_input_data(self,calcium_imag, time_vector, x_coordinates):

        # valid calcium points
        I_valid_calcium = ~np.isnan(calcium_imag)

        # valid x coordinates
        I_valid_x_coord = ~np.isnan(x_coordinates)

        # valid time vector
        I_valid_time_vector = ~np.isnan(time_vector)

        I_keep_valid = I_valid_calcium * I_valid_x_coord * I_valid_time_vector

        return I_keep_valid


    def get_valid_timepoints(self, speed, visits_bins, time_spent_inside_bins, min_speed_threshold,
                             min_visits, min_time_spent):

        # min speed
        I_speed_thres = speed >= min_speed_threshold

        # min visits
        I_visits_times_thres = visits_bins >= min_visits

        # min time spent
        I_time_spent_thres = time_spent_inside_bins >= min_time_spent

        I_keep = I_speed_thres * I_visits_times_thres * I_time_spent_thres

        return I_keep

    def get_place_field_1D(self, calcium_imag, x_coordinates, x_grid, smoothing_size):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # calculate mean calcium per pixel
        place_field = np.nan * np.zeros(( x_grid.shape[0] - 1))
        for xx in range(0, x_grid.shape[0] - 1):
            check_x_occupancy = np.logical_and(x_coordinates >= x_grid[xx], x_coordinates < x_grid[xx + 1])
            place_field[xx] = np.nanmean(calcium_imag[check_x_occupancy])

        place_field_to_smooth = np.copy(place_field)
        place_field_to_smooth[np.isnan(place_field_to_smooth)] = 0
        place_field_smoothed = hf.gaussian_smooth_1d(place_field_to_smooth, smoothing_size)

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

    def get_mutual_information_zscored(self, mutual_info_original, mutual_info_shifted):
        mutual_info_centered = mutual_info_original - np.nanmean(mutual_info_shifted)
        mutual_info_zscored = (mutual_info_original - np.nanmean(mutual_info_shifted)) / np.nanstd(
            mutual_info_shifted)

        return mutual_info_zscored, mutual_info_centered

    def parallelize_surrogate(self, calcium_imag, I_keep, position_binned_valid, sampling_rate, shift_time,
                              nbins_cal, nbins_pos, x_coordinates_valid, x_grid,
                              smoothing_size, num_cores, num_surrogates):
        results = Parallel(n_jobs=num_cores)(
            delayed(self.get_mutual_info_surrogate)(calcium_imag, I_keep, position_binned_valid, sampling_rate,
                                                    shift_time, nbins_cal, nbins_pos, x_coordinates_valid,
                                                    x_grid, smoothing_size)
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


    def get_mutual_info_surrogate(self, calcium_imag, I_keep, position_binned_valid, sampling_rate, shift_time,
                                  nbins_cal, nbins_pos, x_coordinates_valid, x_grid,
                                  smoothing_size):

        calcium_imag_shifted = self.get_surrogate(calcium_imag, sampling_rate, shift_time)
        calcium_imag_shifted_valid = calcium_imag_shifted[I_keep].copy()

        calcium_imag_shifted_binned = self.get_binned_signal(calcium_imag_shifted_valid, nbins_cal)
        entropy1 = self.get_entropy(position_binned_valid, nbins_pos)
        entropy2 = self.get_entropy(calcium_imag_shifted_binned, nbins_cal)
        joint_entropy = self.get_joint_entropy(position_binned_valid, calcium_imag_shifted_binned, nbins_pos,
                                               nbins_cal)
        mutual_info_shifted = self.get_mutual_information(entropy1, entropy2, joint_entropy)

        mutual_info_shifted_NN = self.get_mutual_information_NN(calcium_imag_shifted_valid, position_binned_valid)


        place_field_shifted, place_field_smoothed_shifted = self.get_place_field_1D(calcium_imag_shifted_valid,
                                                                                   x_coordinates_valid,x_grid,smoothing_size)
        

        return mutual_info_shifted, mutual_info_shifted_NN, place_field_shifted, place_field_smoothed_shifted



