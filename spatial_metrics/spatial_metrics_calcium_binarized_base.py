import numpy as np
import os
from . import helper_functions as hf
from joblib import Parallel, delayed
import warnings
import sys
from spatial_metrics import surrogate_functions as surrogate
from spatial_metrics.validators import ParameterValidator

'''
Method used in
https://www.sciencedirect.com/science/article/pii/S0960982218312624#sec4
and
https://www.frontiersin.org/articles/10.3389/fncir.2020.00019/abstract
'''

class PlaceCellBinarized:
    
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
        kwargs.setdefault('y_bin_size', None)
        kwargs.setdefault('environment_edges', None)
        kwargs.setdefault('smoothing_size', 2)
        kwargs.setdefault('shift_time', 10)
        kwargs.setdefault('num_cores', 1)
        kwargs.setdefault('num_surrogates', 200)
        kwargs.setdefault('saving_path', os.getcwd())
        kwargs.setdefault('saving', False)
        kwargs.setdefault('saving_string', 'SpatialMetrics')
        kwargs.setdefault('percentile_threshold', 95)
        kwargs.setdefault('min_num_of_bins', 4)
        kwargs.setdefault('speed_smoothing_points', 1)
        kwargs.setdefault('detection_threshold', 2)
        kwargs.setdefault('detection_smoothing_size', 2)
        kwargs.setdefault('field_detection_method','std_from_field')


        valid_kwargs = ['animal_id', 'day', 'neuron', 'dataset', 'trial', 'sampling_rate',
                        'min_time_spent', 'min_visits', 'min_speed_threshold', 'smoothing_size',
                        'x_bin_size', 'y_bin_size', 'shift_time', 'num_cores', 'percentile_threshold','min_num_of_bins',
                        'num_surrogates', 'saving_path', 'saving', 'saving_string', 'environment_edges','speed_smoothing_points',
                        'detection_threshold','detection_smoothing_size','field_detection_method']

        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            setattr(self, k, v)

        ParameterValidator.validate_all(kwargs)

        self.__dict__['input_parameters'] = kwargs



    def main(self, calcium_imag, time_vector, x_coordinates, y_coordinates=None):

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        
        if np.all(np.isnan(calcium_imag)):
            warnings.warn("Signal contains only NaN's")
            inputdict = np.nan
            filename = self.filename_constructor(self.saving_string, self.animal_id, self.dataset, self.day,
                                                 self.neuron, self.trial)
        else:
            
            x_coordinates, y_coordinates = hf.correct_coordinates(x_coordinates, y_coordinates,environment_edges=self.environment_edges)

            self.validate_input_data(calcium_imag, x_coordinates, y_coordinates,time_vector)

            speed,speed_smoothed = hf.get_speed(x_coordinates, y_coordinates, time_vector, self.speed_smoothing_points)

            x_grid, y_grid, x_center_bins, y_center_bins, x_center_bins_repeated, y_center_bins_repeated = hf.get_position_grid(
                x_coordinates, y_coordinates, self.x_bin_size, self.y_bin_size,
                environment_edges=self.environment_edges)

            position_binned = hf.get_binned_position(x_coordinates, y_coordinates, x_grid, y_grid)

            visits_bins, new_visits_times = hf.get_visits(x_coordinates, y_coordinates, position_binned, x_center_bins, y_center_bins)

            time_spent_inside_bins = hf.get_position_time_spent(position_binned, self.sampling_rate)

            keep_these_frames = self.get_valid_timepoints(speed, visits_bins, time_spent_inside_bins,
                                               self.min_speed_threshold, self.min_visits, self.min_time_spent)

            speed_valid = speed[keep_these_frames].copy()
            calcium_imag_valid = calcium_imag[keep_these_frames].copy()
            x_coordinates_valid = x_coordinates[keep_these_frames].copy()
            y_coordinates_valid = y_coordinates[keep_these_frames].copy()
            visits_bins_valid = visits_bins[keep_these_frames].copy()
            position_binned_valid = position_binned[keep_these_frames].copy()
            new_visits_times_valid = new_visits_times[keep_these_frames].copy()
            time_vector_valid = np.linspace(0,keep_these_frames.shape[0]/self.sampling_rate,keep_these_frames.shape[0])
        
            position_occupancy = hf.get_occupancy(x_coordinates_valid, y_coordinates_valid, x_grid, y_grid, self.sampling_rate)
            
            speed_occupancy = hf.get_speed_occupancy(speed_valid,x_coordinates_valid, y_coordinates_valid,x_grid, y_grid)
            
            visits_occupancy = hf.get_visits_occupancy(x_coordinates_valid, y_coordinates_valid, new_visits_times_valid, x_grid, y_grid)


            activity_map, activity_map_smoothed = hf.get_2D_activity_map(calcium_imag_valid, x_coordinates_valid,
                                                                    y_coordinates_valid, x_grid, y_grid,
                                                                    self.smoothing_size)



            mutual_info_original = self.get_mutual_information(calcium_imag_valid, position_binned_valid)

            results = self.parallelize_surrogate(calcium_imag_valid, position_binned_valid, self.sampling_rate,
                                        self.shift_time, x_coordinates_valid,
                                        y_coordinates_valid, x_grid, y_grid, self.smoothing_size,
                                        self.num_cores, self.num_surrogates)
            
            activity_map_shifted = []
            activity_map_smoothed_shifted = []
            mutual_info_shifted = []

            for perm in range(self.num_surrogates):
                mutual_info_shifted.append(results[perm][0])
                activity_map_shifted.append(results[perm][1])
                activity_map_smoothed_shifted.append(results[perm][2])

            mutual_info_shifted = np.array(mutual_info_shifted)
            activity_map_shifted = np.array(activity_map_shifted)
            activity_map_smoothed_shifted = np.array(activity_map_smoothed_shifted)

            mutual_info_zscored, mutual_info_centered = self.get_mutual_information_zscored(mutual_info_original,
                                                                                            mutual_info_shifted)
            

            if self.field_detection_method == 'random_fields':
                # num_of_islands, islands_x_max, islands_y_max,pixels_place_cell_absolute,pixels_place_cell_relative,activity_map_identity = \
                # hf.field_coordinates_using_shifted(activity_map,activity_map_shifted,visits_occupancy,
                #                                    percentile_threshold=self.percentile_threshold,
                #                                   min_num_of_bins = self.min_num_of_bins)
                
                num_of_islands, islands_x_max, islands_y_max,pixels_place_cell_absolute,pixels_place_cell_relative,activity_map_identity = \
                hf.field_coordinates_using_shifted(activity_map_smoothed,activity_map_smoothed_shifted,visits_occupancy,x_center_bins, y_center_bins,
                                                    percentile_threshold=self.percentile_threshold,
                                                    min_num_of_bins = self.min_num_of_bins)
                

            elif self.field_detection_method == 'std_from_field':
                num_of_islands, islands_x_max, islands_y_max,pixels_place_cell_absolute,pixels_place_cell_relative,activity_map_identity = \
                hf.field_coordinates_using_threshold(activity_map, visits_occupancy,x_center_bins, y_center_bins,smoothing_size = self.detection_smoothing_size,    
                                                field_threshold=self.detection_threshold,
                                                min_num_of_bins=self.min_num_of_bins)
            else:
                warnings.warn("No field detection method set", UserWarning)
                num_of_islands, islands_x_max, islands_y_max, pixels_place_cell_absolute, pixels_place_cell_relative, activity_map_identity = [[] for _ in range(6)]

            


            sparsity = hf.get_sparsity(activity_map, position_occupancy)


            I_peaks = np.where(calcium_imag_valid == 1)[0]
            peaks_amplitude = calcium_imag_valid[I_peaks]
            x_peaks_location = x_coordinates_valid[I_peaks]
            y_peaks_location = y_coordinates_valid[I_peaks]

            inputdict = dict()

            inputdict = dict()
            inputdict['activity_map'] = activity_map
            inputdict['activity_map_smoothed'] = activity_map_smoothed

            inputdict['activity_map_shifted'] = activity_map_shifted
            inputdict['activity_map_smoothed_shifted'] = activity_map_smoothed_shifted
            
            inputdict['timespent_map'] = position_occupancy
            inputdict['visits_map'] = visits_occupancy
            inputdict['speed_map'] = speed_occupancy

            inputdict['x_grid'] = x_grid
            inputdict['y_grid'] = y_grid
            inputdict['x_center_bins'] = x_center_bins
            inputdict['y_center_bins'] = y_center_bins
            inputdict['numb_events'] = I_peaks.shape[0]
            inputdict['x_peaks_location'] = x_peaks_location
            inputdict['y_peaks_location'] = y_peaks_location

            inputdict['activity_map_identity'] = activity_map_identity
            inputdict['num_of_islands'] = num_of_islands
            inputdict['islands_x_max'] = islands_x_max
            inputdict['islands_y_max'] = islands_y_max
            inputdict['sparsity'] = sparsity

            inputdict['place_cell_extension_absolute'] = pixels_place_cell_absolute
            inputdict['place_cell_extension_relative'] = pixels_place_cell_relative

            inputdict['mutual_info_original'] = mutual_info_original
            inputdict['mutual_info_shifted'] = mutual_info_shifted
            inputdict['mutual_info_zscored'] = mutual_info_zscored
            inputdict['mutual_info_centered'] = mutual_info_centered

            inputdict['input_parameters'] = self.__dict__['input_parameters']

            filename = hf.filename_constructor(self.saving_string, self.animal_id, self.dataset, self.day, 
                                        self.neuron,self.trial)

        if self.saving == True:
            hf.caller_saving(inputdict, filename, self.saving_path)
            print(filename + ' saved')

        else:
            print(filename + ' not saved')

        return inputdict



    def validate_input_data(self,calcium_imag, x_coordinates, y_coordinates,time_vector):

        # valid calcium points
        I_valid_calcium = ~np.isnan(calcium_imag)

        # valid x coordinates
        I_valid_x_coord = ~np.isnan(x_coordinates)

        # valid y coordinates
        I_valid_y_coord = ~np.isnan(y_coordinates)

        # valid time vector
        I_valid_time_vector = ~np.isnan(time_vector)

        I_keep_valid = I_valid_calcium * I_valid_x_coord * I_valid_y_coord * I_valid_time_vector

        calcium_imag = calcium_imag[I_keep_valid]
        time_vector = time_vector[I_keep_valid]
        x_coordinates = x_coordinates[I_keep_valid]
        y_coordinates = y_coordinates[I_keep_valid]


    def get_valid_timepoints(self, speed, visits_bins, time_spent_inside_bins, min_speed_threshold, min_visits, min_time_spent):

        # min speed
        I_speed_thres = speed >= min_speed_threshold

        # min visits
        I_visits_times_thres = visits_bins >= min_visits

        # min time spent
        I_time_spent_thres = time_spent_inside_bins >= min_time_spent


        I_keep = I_speed_thres * I_visits_times_thres * I_time_spent_thres

        return I_keep
    



    def get_mutual_information_zscored(self, mutual_info_original, mutual_info_shuffled):
        mutual_info_centered = mutual_info_original - np.nanmean(mutual_info_shuffled)
        mutual_info_zscored = (mutual_info_original - np.nanmean(mutual_info_shuffled)) / np.nanstd(
            mutual_info_shuffled)

        return mutual_info_zscored, mutual_info_centered

    def parallelize_surrogate(self, calcium_imag_valid, position_binned_valid, sampling_rate, shift_time,
                              x_coordinates_valid, y_coordinates_valid, x_grid, y_grid,
                              smoothing_size, num_cores, num_surrogates):
        results = Parallel(n_jobs=num_cores)(delayed(self.get_mutual_info_surrogate)
                                                          (calcium_imag_valid, position_binned_valid,
                                                           sampling_rate,
                                                           shift_time, x_coordinates_valid, y_coordinates_valid,
                                                           x_grid, y_grid, smoothing_size)
                                                          for _ in range(num_surrogates))

        return results



    def get_mutual_info_surrogate(self, calcium_imag_valid, position_binned_valid, sampling_rate, shift_time,
                                  x_coordinates_valid, y_coordinates_valid, x_grid, y_grid, smoothing_size):

        calcium_imag_shuffled_valid = surrogate.get_signal_surrogate(calcium_imag_valid, sampling_rate, shift_time)

        mutual_info_shuffled = self.get_mutual_information(calcium_imag_shuffled_valid, position_binned_valid)
        activity_map_shuffled, activity_map_smoothed_shuffled = hf.get_2D_activity_map(calcium_imag_shuffled_valid,
                                                                                   x_coordinates_valid,
                                                                                   y_coordinates_valid, x_grid, y_grid,
                                                                                   smoothing_size)
        return mutual_info_shuffled, activity_map_shuffled, activity_map_smoothed_shuffled



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
