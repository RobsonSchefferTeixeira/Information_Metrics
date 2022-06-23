import numpy as np
import os
import spatial_metrics.helper_functions as hf
from joblib import Parallel, delayed
import warnings
import sys


class PlaceCellBinarized:
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
        kwargs.setdefault('percentile_threshold', 95)
        kwargs.setdefault('min_num_of_pixels', 4)

        valid_kwargs = ['animal_id', 'day', 'neuron', 'dataset', 'trial', 'mean_video_srate',
                        'min_time_spent', 'min_visits', 'min_speed_threshold', 'smoothing_size',
                        'x_bin_size', 'y_bin_size', 'shift_time', 'num_cores', 'percentile_threshold','min_num_of_pixels',
                        'num_surrogates', 'saving_path', 'saving', 'saving_string', 'environment_edges']

        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            setattr(self, k, v)

        self.__dict__['input_parameters'] = kwargs

    def main(self, calcium_imag, track_timevector, x_coordinates, y_coordinates):

        if np.all(np.isnan(calcium_imag)):
            warnings.warn("Signal contains only NaN's")
            inputdict = np.nan

        else:
            speed = hf.get_speed(x_coordinates, y_coordinates, track_timevector)

            x_grid, y_grid, x_center_bins, y_center_bins, x_center_bins_repeated, y_center_bins_repeated = hf.get_position_grid(
                x_coordinates, y_coordinates, self.x_bin_size, self.y_bin_size,
                environment_edges=self.environment_edges)

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

            mutual_info_original = self.get_mutual_information(calcium_imag_valid, position_binned_valid)

            results = self.parallelize_surrogate(calcium_imag,I_keep, position_binned_valid, self.mean_video_srate,
                                                 self.shift_time, x_coordinates_valid, y_coordinates_valid,x_grid,y_grid,
                                                 self.smoothing_size,self.num_cores,self.num_surrogates)

            place_field_shuffled = []
            place_field_smoothed_shuffled = []
            mutual_info_shuffled = []

            for perm in range(self.num_surrogates):
                mutual_info_shuffled.append(results[perm][0])
                place_field_shuffled.append(results[perm][1])
                place_field_smoothed_shuffled.append(results[perm][2])

            mutual_info_shuffled = np.array(mutual_info_shuffled)
            place_field_shuffled = np.array(place_field_shuffled)
            place_field_smoothed_shuffled = np.array(place_field_smoothed_shuffled)

            mutual_info_zscored, mutual_info_centered = self.get_mutual_information_zscored(mutual_info_original,
                                                                                            mutual_info_shuffled)

            num_of_islands, islands_x_max, islands_y_max,pixels_place_cell_absolute,pixels_place_cell_relative = \
                hf.field_coordinates_using_shuffled(place_field_smoothed,place_field_smoothed_shuffled,visits_occupancy,
                                                    percentile_threshold=self.percentile_threshold,
                                                    min_num_of_pixels = self.min_num_of_pixels)

            sparsity = hf.get_sparsity(place_field, position_occupancy)


            I_peaks = np.where(calcium_imag_valid == 1)[0]
            peaks_amplitude = calcium_imag_valid[I_peaks]
            x_peaks_location = x_coordinates_valid[I_peaks]
            y_peaks_location = y_coordinates_valid[I_peaks]

            inputdict = dict()
            inputdict['place_field'] = place_field
            inputdict['place_field_smoothed'] = place_field_smoothed
            inputdict['place_field_shuffled'] = place_field_shuffled
            inputdict['place_field_smoothed_shuffled'] = place_field_smoothed_shuffled

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

            inputdict['input_parameters'] = self.__dict__['input_parameters']

            filename = hf.filename_constructor(self.saving_string, self.animal_id, self.dataset, self.day,
                                                 self.neuron, self.trial)

        if self.saving == True:
            hf.caller_saving(inputdict, filename, self.saving_path)
            print(filename + ' saved')

        else:
            print(filename + ' not saved')

        return inputdict


    def get_calcium_occupancy(self, calcium_imag, x_coordinates, y_coordinates, x_grid, y_grid):

        # calculate mean calcium per pixel
        calcium_mean_occupancy = np.nan * np.zeros((y_grid.shape[0] - 1, x_grid.shape[0] - 1))
        for xx in range(0, x_grid.shape[0] - 1):
            for yy in range(0, y_grid.shape[0] - 1):
                check_x_occupancy = np.logical_and(x_coordinates >= x_grid[xx], x_coordinates < (x_grid[xx + 1]))
                check_y_occupancy = np.logical_and(y_coordinates >= y_grid[yy], y_coordinates < (y_grid[yy + 1]))

                calcium_mean_occupancy[yy, xx] = np.nanmean(
                    calcium_imag[np.logical_and(check_x_occupancy, check_y_occupancy)])

        return calcium_mean_occupancy

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


    def get_mutual_information_zscored(self, mutual_info_original, mutual_info_shuffled):
        mutual_info_centered = mutual_info_original - np.nanmean(mutual_info_shuffled)
        mutual_info_zscored = (mutual_info_original - np.nanmean(mutual_info_shuffled)) / np.nanstd(
            mutual_info_shuffled)

        return mutual_info_zscored, mutual_info_centered

    def parallelize_surrogate(self, calcium_imag, I_keep, position_binned_valid, mean_video_srate, shift_time,
                              x_coordinates_valid, y_coordinates_valid, x_grid, y_grid, smoothing_size,
                              num_cores, num_surrogates):
        results = Parallel(n_jobs=num_cores)(delayed(self.get_mutual_info_surrogate)
                                                          (calcium_imag, I_keep, position_binned_valid,
                                                           mean_video_srate,
                                                           shift_time, x_coordinates_valid, y_coordinates_valid,
                                                           x_grid, y_grid, smoothing_size)
                                                          for _ in range(num_surrogates))

        return results

    def get_surrogate(self, input_vector, mean_video_srate, shift_time):
        # eps = np.finfo(float).eps
        I_break = np.random.choice(np.arange(-shift_time * mean_video_srate, mean_video_srate * shift_time), 1)[
            0].astype(int)
        input_vector_shuffled = np.concatenate([input_vector[I_break:], input_vector[0:I_break]])

        return input_vector_shuffled

    def get_mutual_info_surrogate(self, calcium_imag, I_keep, position_binned_valid, mean_video_srate, shift_time,
                                  x_coordinates_valid, y_coordinates_valid, x_grid, y_grid, smoothing_size):

        calcium_imag_shuffled = self.get_surrogate(calcium_imag, mean_video_srate, shift_time)
        calcium_imag_shuffled_valid = calcium_imag_shuffled[I_keep].copy()

        # calcium_imag_shuffled = self.get_surrogate(calcium_imag,mean_video_srate,shift_time)
        mutual_info_shuffled = self.get_mutual_information(calcium_imag_shuffled_valid, position_binned_valid)
        place_field_shuffled, place_field_smoothed_shuffled = self.get_place_field(calcium_imag_shuffled_valid,
                                                                                   x_coordinates_valid,
                                                                                   y_coordinates_valid, x_grid, y_grid,
                                                                                   smoothing_size)
        return mutual_info_shuffled, place_field_shuffled, place_field_smoothed_shuffled

    def get_mutual_information(self, calcium_imag, position_binned):

        # I've translated this code to Python. 
        # Originally I took it from https://github.com/etterguillaume/CaImDecoding/blob/master/extract_1D_information.m

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
                occupancy_vector.append(position_idx.shape[0] / calcium_imag.shape[0])

                activity_in_bin_idx = np.where((calcium_imag == 1) & position_idx)[0]
                inactivity_in_bin_idx = np.where((calcium_imag == 0) & position_idx)[0]
                likelihood.append(activity_in_bin_idx.shape[0] / np.sum(position_idx))

                joint_prob_active = activity_in_bin_idx.shape[0] / calcium_imag.shape[0]
                joint_prob_inactive = inactivity_in_bin_idx.shape[0] / calcium_imag.shape[0]
                prob_in_bin = np.sum(position_idx) / calcium_imag.shape[0]

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
