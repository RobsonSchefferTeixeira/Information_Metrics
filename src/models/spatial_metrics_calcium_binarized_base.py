import numpy as np
import os
import sys
import warnings

from src.utils import helper_functions as hf
from src.utils import surrogate_functions as surrogate
from src.utils.validators import ParameterValidator,DataValidator
from src.utils import information_base as info
import src.utils.bootstrapped_estimation as be

from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed


'''
Method used in
https://www.sciencedirect.com/science/article/pii/S0960982218312624#sec4
and
https://www.frontiersin.org/articles/10.3389/fncir.2020.00019/abstract
'''

class PlaceCellBinarized:
    
    def __init__(self, **kwargs):
        kwargs.setdefault('signal_type',None)
        kwargs.setdefault('animal_id', None)
        kwargs.setdefault('day', None)
        kwargs.setdefault('neuron', None)
        kwargs.setdefault('trial', None)
        kwargs.setdefault('dataset', None)
        kwargs.setdefault('saving_path', os.getcwd())
        kwargs.setdefault('saving', False)
        kwargs.setdefault('saving_string', 'SpatialMetrics')
        kwargs.setdefault('overwrite', False)
        kwargs.setdefault('min_time_spent', 0.1)
        kwargs.setdefault('min_visits', 1)
        kwargs.setdefault('min_speed_threshold', 2.5)
        kwargs.setdefault('speed_smoothing_sigma', 1)
        kwargs.setdefault('x_bin_size', 1)
        kwargs.setdefault('y_bin_size', None)
        kwargs.setdefault('map_smoothing_sigma_x', 2)
        kwargs.setdefault('map_smoothing_sigma_y', 2)
        kwargs.setdefault('x_bin_size_info', 2)
        kwargs.setdefault('y_bin_size_info', 2)
        kwargs.setdefault('shift_time', 10)
        kwargs.setdefault('num_cores', 1)
        kwargs.setdefault('num_surrogates', 200)
        kwargs.setdefault('percentile_threshold', 95)
        kwargs.setdefault('min_num_of_bins', 4)
        kwargs.setdefault('detection_threshold', 2)
        kwargs.setdefault('detection_smoothing_sigma_x', 2)
        kwargs.setdefault('detection_smoothing_sigma_y', 2)
        kwargs.setdefault('field_detection_method','std_from_field')
        kwargs.setdefault('alpha',0.05)


        valid_kwargs = ['signal_type','animal_id', 'day', 'neuron', 'dataset', 'trial',
                        'min_time_spent', 'min_visits', 'min_speed_threshold','speed_smoothing_sigma','map_smoothing_sigma_x','map_smoothing_sigma_y',
                        'x_bin_size', 'y_bin_size', 'shift_time', 'num_cores', 'percentile_threshold','min_num_of_bins',
                        'num_surrogates', 'saving_path', 'saving', 'saving_string','x_bin_size_info','y_bin_size_info',
                        'detection_threshold','detection_smoothing_sigma_x','detection_smoothing_sigma_y',
                        'field_detection_method','alpha','overwrite']

        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            setattr(self, k, v)

        ParameterValidator.validate_all(kwargs)

        self.__dict__['input_parameters'] = kwargs


    def main(self, signal_data):
        
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        filename = hf.filename_constructor(self.saving_string, self.animal_id, self.dataset, self.day, self.neuron, self.trial)
        full_path = f"{self.saving_path}/{filename}"
        # Check if the file exists and handle based on the overwrite flag
        if os.path.exists(full_path) and not self.overwrite:
            print(f"File already exists and overwrite is set to False: {full_path}")
            return
            

        if DataValidator.is_empty_or_all_nan(signal_data.input_signal) or DataValidator.is_empty_or_all_nan(signal_data.x_coordinates):
            warnings.warn("Signal contains only NaN's or is empty", UserWarning)
            inputdict = np.nan
            
        else:
            
            if signal_data.speed is None:
                signal_data.add_speed(self.speed_smoothing_sigma)

            x_grid, y_grid, x_center_bins, y_center_bins, _, _ = hf.get_position_grid(
                signal_data.x_coordinates, signal_data.y_coordinates, self.x_bin_size, self.y_bin_size,
                environment_edges=signal_data.environment_edges)


            x_grid_info, y_grid_info, _, _, _, _ = hf.get_position_grid(
                signal_data.x_coordinates, signal_data.y_coordinates, self.x_bin_size_info, self.y_bin_size_info,
                environment_edges=signal_data.environment_edges)

            signal_data.add_position_binned(x_grid_info, y_grid_info)

            signal_data.add_visits(x_center_bins, y_center_bins)

            signal_data.add_position_time_spent()

            DataValidator.get_valid_timepoints(signal_data, self.min_speed_threshold, self.min_visits, self.min_time_spent)

            position_occupancy = hf.get_occupancy(signal_data.x_coordinates, signal_data.y_coordinates, x_grid, y_grid, signal_data.sampling_rate)
            
            speed_occupancy = hf.get_speed_occupancy(signal_data.speed,signal_data.x_coordinates, signal_data.y_coordinates,x_grid, y_grid)
            
            visits_occupancy = hf.get_visits_occupancy(signal_data.x_coordinates, signal_data.y_coordinates, signal_data.new_visits_times, x_grid, y_grid)


            activity_map, activity_map_smoothed = hf.get_2D_activity_map(signal_data.input_signal, signal_data.x_coordinates,
                                                                    signal_data.y_coordinates, x_grid, y_grid,
                                                                    self.map_smoothing_sigma_x,self.map_smoothing_sigma_y)

            mutual_info_per_spike_original, mutual_info_per_second_original = info.get_binarized_spatial_info(signal_data.input_signal, signal_data.position_binned)
            info_per_spike_original, info_per_second_original = info.get_binarized_spatial_info_nkinsky(signal_data.input_signal, signal_data.position_binned)
            info_per_spike_map_original, info_per_second_map_original = info.get_spatial_info_from_map(activity_map_smoothed, position_occupancy)
            results = self.parallelize_surrogate(signal_data.input_signal, signal_data.position_binned, position_occupancy, signal_data.sampling_rate,
                                        self.shift_time, signal_data.x_coordinates,signal_data.y_coordinates,
                                        x_grid, y_grid, self.map_smoothing_sigma_x,self.map_smoothing_sigma_y,
                                        self.num_cores, self.num_surrogates)
            

            mutual_info_per_spike_shifted = []
            mutual_info_per_second_shifted = []
            info_per_spike_shifted = []
            info_per_second_shifted = []
            info_per_spike_map_shifted = []
            info_per_second_map_shifted = []

            activity_map_shifted = []
            activity_map_smoothed_shifted = []

            for perm in range(self.num_surrogates):
                mutual_info_per_spike_shifted.append(results[perm][0])
                mutual_info_per_second_shifted.append(results[perm][1])
                info_per_spike_shifted.append(results[perm][2])
                info_per_second_shifted.append(results[perm][3])
                info_per_spike_map_shifted.append(results[perm][4])
                info_per_second_map_shifted.append(results[perm][5])
                activity_map_shifted.append(results[perm][6])
                activity_map_smoothed_shifted.append(results[perm][7])


            mutual_info_per_spike_shifted = np.array(mutual_info_per_spike_shifted)
            mutual_info_per_second_shifted = np.array(mutual_info_per_second_shifted)
            info_per_spike_shifted = np.array(info_per_spike_shifted)
            info_per_second_shifted = np.array(info_per_second_shifted)
            info_per_spike_map_shifted = np.array(info_per_spike_map_shifted)
            info_per_second_map_shifted = np.array(info_per_second_map_shifted)
            activity_map_shifted = np.array(activity_map_shifted)
            activity_map_smoothed_shifted = np.array(activity_map_smoothed_shifted)

            mutual_info_per_spike_zscored, mutual_info_per_spike_centered = info.get_mutual_information_zscored(mutual_info_per_spike_original, mutual_info_per_spike_shifted)
            mutual_info_per_second_zscored, mutual_info_per_second_centered = info.get_mutual_information_zscored(mutual_info_per_second_original, mutual_info_per_second_shifted)
            info_per_spike_zscored, info_per_spike_centered = info.get_mutual_information_zscored(info_per_spike_original, info_per_spike_shifted)
            info_per_second_zscored, info_per_second_centered = info.get_mutual_information_zscored(info_per_second_original, info_per_second_shifted)
            info_per_spike_map_zscored, info_per_spike_map_centered = info.get_mutual_information_zscored(info_per_spike_map_original, info_per_spike_map_shifted)
            info_per_second_map_zscored, info_per_second_map_centered = info.get_mutual_information_zscored(info_per_second_map_original, info_per_second_map_shifted)
         
            if self.field_detection_method == 'random_fields':

                num_of_fields, fields_x_max, fields_y_max,pixels_place_cell_absolute,pixels_place_cell_relative,activity_map_identity = \
                hf.field_coordinates_using_shifted(activity_map,activity_map_shifted,visits_occupancy,x_center_bins, y_center_bins,
                                                    percentile_threshold=self.percentile_threshold,
                                                    min_num_of_bins = self.min_num_of_bins,
                                                    sigma_x=self.detection_smoothing_sigma_x, 
                                                    sigma_y=self.detection_smoothing_sigma_y)


            elif self.field_detection_method == 'std_from_field':
                num_of_fields, fields_x_max, fields_y_max,pixels_place_cell_absolute,pixels_place_cell_relative,activity_map_identity = \
                hf.field_coordinates_using_threshold(activity_map, visits_occupancy,x_center_bins, y_center_bins,                                                        
                                                    field_threshold=self.detection_threshold, min_num_of_bins=self.min_num_of_bins,
                                                    sigma_x = self.detection_smoothing_sigma_x, 
                                                    sigma_y=self.detection_smoothing_sigma_y)
                
            else:
                warnings.warn("No field detection method set", UserWarning)
                num_of_fields, fields_x_max, fields_y_max, pixels_place_cell_absolute, pixels_place_cell_relative, activity_map_identity = [[] for _ in range(6)]

        

            sparsity = hf.get_sparsity(activity_map, position_occupancy)

            signal_data.add_peaks_detection(self.signal_type)

            mutual_info_per_spike_statistic = be.calculate_p_value(mutual_info_per_spike_original, mutual_info_per_spike_shifted, alternative='greater')
            mutual_info_per_spike_pvalue = mutual_info_per_spike_statistic.p_value

            mutual_info_per_second_statistic = be.calculate_p_value(mutual_info_per_second_original, mutual_info_per_second_shifted, alternative='greater')
            mutual_info_per_second_pvalue = mutual_info_per_second_statistic.p_value

            info_per_spike_statistic = be.calculate_p_value(info_per_spike_original, info_per_spike_shifted, alternative='greater')
            info_per_spike_pvalue = info_per_spike_statistic.p_value
            
            info_per_second_statistic = be.calculate_p_value(info_per_second_original, info_per_second_shifted, alternative='greater')
            info_per_second_pvalue = info_per_second_statistic.p_value

            info_per_spike_map_statistic = be.calculate_p_value(info_per_spike_map_original, info_per_spike_map_shifted, alternative='greater')
            info_per_spike_map_pvalue = info_per_spike_map_statistic.p_value
            
            info_per_second_map_statistic = be.calculate_p_value(info_per_second_map_original, info_per_second_map_shifted, alternative='greater')
            info_per_second_map_pvalue = info_per_second_map_statistic.p_value


            if mutual_info_per_second_pvalue > self.alpha:
                activity_map_identity = np.zeros(activity_map.shape)*np.nan
                num_of_fields = 0
                pixels_place_cell_absolute = np.nan
                fields_x_max = np.nan
                fields_y_max = np.nan
                pixels_place_cell_absolute = np.nan
                pixels_place_cell_relative = np.nan


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

            inputdict['numb_events'] = signal_data.peaks_idx[0].shape[0]
            inputdict['peaks_x_location'] = signal_data.peaks_x_location
            inputdict['peaks_y_location'] = signal_data.peaks_y_location

            inputdict['activity_map_identity'] = activity_map_identity
            inputdict['num_of_fields'] = num_of_fields
            inputdict['fields_x_max'] = fields_x_max
            inputdict['fields_y_max'] = fields_y_max

            inputdict['place_cell_extension_absolute'] = pixels_place_cell_absolute
            inputdict['place_cell_extension_relative'] = pixels_place_cell_relative

            inputdict['sparsity'] = sparsity

            inputdict['mutual_info_per_spike_original'] = mutual_info_per_spike_original
            inputdict['mutual_info_per_spike_shifted'] = mutual_info_per_spike_shifted
            inputdict['mutual_info_per_spike_zscored'] = mutual_info_per_spike_zscored
            inputdict['mutual_info_per_spike_centered'] = mutual_info_per_spike_centered
            inputdict['mutual_info_per_spike_pvalue'] = mutual_info_per_spike_pvalue

            inputdict['mutual_info_per_second_original'] = mutual_info_per_second_original
            inputdict['mutual_info_per_second_shifted'] = mutual_info_per_second_shifted
            inputdict['mutual_info_per_second_zscored'] = mutual_info_per_second_zscored
            inputdict['mutual_info_per_second_centered'] = mutual_info_per_second_centered
            inputdict['mutual_info_per_second_pvalue'] = mutual_info_per_second_pvalue

            inputdict['info_per_spike_original'] = info_per_spike_original
            inputdict['info_per_spike_shifted'] = info_per_spike_shifted
            inputdict['info_per_spike_zscored'] = info_per_spike_zscored
            inputdict['info_per_spike_centered'] = info_per_spike_centered
            inputdict['info_per_spike_pvalue'] = info_per_spike_pvalue

            inputdict['info_per_second_original'] = info_per_second_original
            inputdict['info_per_second_shifted'] = info_per_second_shifted
            inputdict['info_per_second_zscored'] = info_per_second_zscored
            inputdict['info_per_second_centered'] = info_per_second_centered
            inputdict['info_per_second_pvalue'] = info_per_second_pvalue

            inputdict['info_per_spike_map_original'] = info_per_spike_map_original
            inputdict['info_per_spike_map_shifted'] = info_per_spike_map_shifted
            inputdict['info_per_spike_map_zscored'] = info_per_spike_map_zscored
            inputdict['info_per_spike_map_centered'] = info_per_spike_map_centered
            inputdict['info_per_spike_map_pvalue'] = info_per_spike_map_pvalue

            inputdict['info_per_second_map_original'] = info_per_second_map_original
            inputdict['info_per_second_map_shifted'] = info_per_second_map_shifted
            inputdict['info_per_second_map_zscored'] = info_per_second_map_zscored
            inputdict['info_per_second_map_centered'] = info_per_second_map_centered
            inputdict['info_per_second_map_pvalue'] = info_per_second_map_pvalue

            inputdict['input_parameters'] = self.__dict__['input_parameters']

            filename = hf.filename_constructor(self.saving_string, self.animal_id, self.dataset, self.day, 
                                        self.neuron,self.trial)

        if self.saving == True:
            hf.caller_saving(inputdict, filename, self.saving_path, self.overwrite)
            print(filename + ' saved')

        else:
            print(filename + ' not saved')

        return inputdict


    def parallelize_surrogate(self, input_signal, position_binned, position_occupancy, sampling_rate, shift_time,
                            x_coordinates, y_coordinates, x_grid, y_grid,
                            map_smoothing_sigma_x,map_smoothing_sigma_y, num_cores, num_surrogates):
        with tqdm_joblib(tqdm(desc="Processing Surrogates", total=num_surrogates)) as progress_bar:
            results = Parallel(n_jobs=num_cores)(
                delayed(self.get_mutual_info_surrogate)
                (
                    input_signal, position_binned, position_occupancy, sampling_rate,
                    shift_time, x_coordinates, y_coordinates,
                    x_grid, y_grid, map_smoothing_sigma_x, map_smoothing_sigma_y
                ) 
                for _ in range(num_surrogates)
            )
        return results

   
    def get_mutual_info_surrogate(self, input_signal, position_binned, position_occupancy, sampling_rate, shift_time,
                                  x_coordinates, y_coordinates, x_grid, y_grid, map_smoothing_sigma_x,map_smoothing_sigma_y):

        input_signal_shifted = surrogate.get_signal_surrogate(input_signal, sampling_rate, shift_time)

        info_per_spike_shifted, info_per_second_shifted = info.get_binarized_spatial_info_nkinsky(input_signal_shifted, position_binned)
        mutual_info_per_spike_shifted, mutual_info_per_second_shifted = info.get_binarized_spatial_info_etter(input_signal_shifted, position_binned)

        activity_map_shifted, activity_map_smoothed_shifted = hf.get_2D_activity_map(input_signal_shifted,
                                                                                   x_coordinates,
                                                                                   y_coordinates, x_grid, y_grid,
                                                                                   map_smoothing_sigma_x,map_smoothing_sigma_y)
        
        info_per_spike_map_shifted, info_per_second_map_shifted = info.get_spatial_info_from_map(activity_map_smoothed_shifted, position_occupancy)

        return mutual_info_per_spike_shifted, mutual_info_per_second_shifted, info_per_spike_shifted, info_per_second_shifted, info_per_spike_map_shifted, info_per_second_map_shifted, activity_map_shifted, activity_map_smoothed_shifted

    
