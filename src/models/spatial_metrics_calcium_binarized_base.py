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
        kwargs.setdefault('min_time_spent', 0.1)
        kwargs.setdefault('min_visits', 1)
        kwargs.setdefault('min_speed_threshold', 2.5)
        kwargs.setdefault('speed_smoothing_sigma', 1)
        kwargs.setdefault('x_bin_size', 1)
        kwargs.setdefault('y_bin_size', None)
        kwargs.setdefault('map_smoothing_sigma_x', 2)
        kwargs.setdefault('map_smoothing_sigma_y', 2)
        kwargs.setdefault('shift_time', 10)
        kwargs.setdefault('num_cores', 1)
        kwargs.setdefault('num_surrogates', 200)
        kwargs.setdefault('saving_path', os.getcwd())
        kwargs.setdefault('saving', False)
        kwargs.setdefault('saving_string', 'SpatialMetrics')
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
                        'num_surrogates', 'saving_path', 'saving', 'saving_string',
                        'detection_threshold','detection_smoothing_sigma_x','detection_smoothing_sigma_y',
                        'field_detection_method','alpha']

        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            setattr(self, k, v)

        ParameterValidator.validate_all(kwargs)

        self.__dict__['input_parameters'] = kwargs



    # def main(self, input_signal, time_vector, x_coordinates, y_coordinates=None, speed=None):
    def main(self, signal_data):



        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        
        if np.all(np.isnan(signal_data.input_signal)):
            warnings.warn("Signal contains only NaN's")
            inputdict = np.nan
            filename = self.filename_constructor(self.saving_string, self.animal_id, self.dataset, self.day,
                                                 self.neuron, self.trial)
        else:
            
            signal_data.x_coordinates, signal_data.y_coordinates = hf.correct_coordinates(signal_data.x_coordinates,signal_data.y_coordinates,environment_edges=signal_data.environment_edges)

            if signal_data.speed is None:
                signal_data.add_speed(self.speed_smoothing_sigma)

            x_grid, y_grid, x_center_bins, y_center_bins, x_center_bins_repeated, y_center_bins_repeated = hf.get_position_grid(
                signal_data.x_coordinates, signal_data.y_coordinates, self.x_bin_size, self.y_bin_size,
                environment_edges=signal_data.environment_edges)

            signal_data.add_position_binned(x_grid, y_grid)

            signal_data.add_visits(x_center_bins, y_center_bins)

            signal_data.add_position_time_spent()

            DataValidator.get_valid_timepoints(signal_data, self.min_speed_threshold, self.min_visits, self.min_time_spent)


            position_occupancy = hf.get_occupancy(signal_data.x_coordinates, signal_data.y_coordinates, x_grid, y_grid, signal_data.sampling_rate)
            
            speed_occupancy = hf.get_speed_occupancy(signal_data.speed,signal_data.x_coordinates, signal_data.y_coordinates,x_grid, y_grid)
            
            visits_occupancy = hf.get_visits_occupancy(signal_data.x_coordinates, signal_data.y_coordinates, signal_data.new_visits_times, x_grid, y_grid)


            activity_map, activity_map_smoothed = hf.get_2D_activity_map(signal_data.input_signal, signal_data.x_coordinates,
                                                                    signal_data.y_coordinates, x_grid, y_grid,
                                                                    self.map_smoothing_sigma_x,self.map_smoothing_sigma_y)


            mutual_info_original = info.get_mutual_information_binarized(signal_data.input_signal, signal_data.position_binned)
            
            results = self.parallelize_surrogate(signal_data.input_signal, signal_data.position_binned, signal_data.sampling_rate,
                                        self.shift_time, signal_data.x_coordinates,signal_data.y_coordinates,
                                        x_grid, y_grid, self.map_smoothing_sigma_x,self.map_smoothing_sigma_y,
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

            mutual_info_zscored, mutual_info_centered = info.get_mutual_information_zscored(mutual_info_original,
                                                                                            mutual_info_shifted)
            

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

            statistic = be.calculate_p_value(mutual_info_original, mutual_info_shifted, alternative='greater')
            
            if statistic.p_value > self.alpha:

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

            inputdict['numb_events'] = signal_data.peaks_idx.shape[0]
            inputdict['peaks_x_location'] = signal_data.peaks_x_location
            inputdict['peaks_y_location'] = signal_data.peaks_y_location

            inputdict['mutual_information_pvalue'] = statistic.p_value

            inputdict['activity_map_identity'] = activity_map_identity
            inputdict['num_of_fields'] = num_of_fields
            inputdict['fields_x_max'] = fields_x_max
            inputdict['fields_y_max'] = fields_y_max

            inputdict['place_cell_extension_absolute'] = pixels_place_cell_absolute
            inputdict['place_cell_extension_relative'] = pixels_place_cell_relative

            inputdict['sparsity'] = sparsity

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


    def parallelize_surrogate(self, input_signal, position_binned, sampling_rate, shift_time,
                            x_coordinates, y_coordinates, x_grid, y_grid,
                            map_smoothing_sigma_x,map_smoothing_sigma_y, num_cores, num_surrogates):
        with tqdm_joblib(tqdm(desc="Processing Surrogates", total=num_surrogates)) as progress_bar:
            results = Parallel(n_jobs=num_cores)(
                delayed(self.get_mutual_info_surrogate)
                (
                    input_signal, position_binned, sampling_rate,
                    shift_time, x_coordinates, y_coordinates,
                    x_grid, y_grid, map_smoothing_sigma_x, map_smoothing_sigma_y
                ) 
                for _ in range(num_surrogates)
            )
        return results

   
    def get_mutual_info_surrogate(self, input_signal, position_binned, sampling_rate, shift_time,
                                  x_coordinates, y_coordinates, x_grid, y_grid, map_smoothing_sigma_x,map_smoothing_sigma_y):

        input_signal_shuffled = surrogate.get_signal_surrogate(input_signal, sampling_rate, shift_time)

        mutual_info_shuffled = info.get_mutual_information_binarized(input_signal_shuffled, position_binned)
        activity_map_shuffled, activity_map_smoothed_shuffled = hf.get_2D_activity_map(input_signal_shuffled,
                                                                                   x_coordinates,
                                                                                   y_coordinates, x_grid, y_grid,
                                                                                   map_smoothing_sigma_x,map_smoothing_sigma_y)
        return mutual_info_shuffled, activity_map_shuffled, activity_map_smoothed_shuffled

    
