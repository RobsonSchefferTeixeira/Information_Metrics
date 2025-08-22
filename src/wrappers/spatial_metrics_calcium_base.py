import numpy as np
import os
import warnings
from pathlib import Path

from src.utils import helper_functions as hf
from src.utils import surrogate_functions as surrogate
from src.utils import information_base as info
from src.utils.validators import ParameterValidator,DataValidator
import src.utils.bootstrapped_estimation as be

from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

class PlaceCell:

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
        kwargs.setdefault('environment_edges', None)
        kwargs.setdefault('map_smoothing_sigma_x', 2)
        kwargs.setdefault('map_smoothing_sigma_y', 2)
        kwargs.setdefault('x_bin_size_info', 2)
        kwargs.setdefault('y_bin_size_info', 2)
        kwargs.setdefault('shift_time', 10)
        kwargs.setdefault('num_cores', 1)
        kwargs.setdefault('num_surrogates', 200)
        kwargs.setdefault('saving_path', os.getcwd())
        kwargs.setdefault('saving', False)
        kwargs.setdefault('overwrite', False)
        kwargs.setdefault('saving_string', 'SpatialMetrics')
        kwargs.setdefault('nbins_cal', 10)
        kwargs.setdefault('min_num_of_bins', 4)        
        kwargs.setdefault('threshold',('mean_std',2))
        kwargs.setdefault('threshold_fraction',0.5)
        kwargs.setdefault('alpha',0.05)


        valid_kwargs = ['signal_type','animal_id', 'day', 'neuron', 'dataset', 'trial','x_bin_size_info','y_bin_size_info',
                        'min_time_spent', 'min_visits', 'min_speed_threshold', 'speed_smoothing_sigma','overwrite',
                        'x_bin_size', 'y_bin_size', 'shift_time', 'map_smoothing_sigma_x','map_smoothing_sigma_y','num_cores', 'percentile_threshold','min_num_of_bins',
                        'num_surrogates', 'saving_path', 'saving', 'saving_string', 'environment_edges', 'nbins_cal',
                        'threshold','threshold_fraction','alpha']

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
            

        if  DataValidator.is_empty_or_all_nan(signal_data.input_signal) or DataValidator.is_empty_or_all_nan(signal_data.x_coordinates):
            warnings.warn("Signal is constant, contains only NaN's or is empty", UserWarning)
            inputdict = np.nan
        
        elif np.allclose(signal_data.input_signal, signal_data.input_signal[0],equal_nan=True):
            warnings.warn("Signal is constant", UserWarning)
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

            position_occupancy = hf.get_occupancy(signal_data.x_coordinates,x_grid, signal_data.sampling_rate,signal_data.y_coordinates, y_grid)
            
            speed_occupancy = hf.get_speed_occupancy(signal_data.speed,signal_data.x_coordinates,x_grid, signal_data.y_coordinates, y_grid)
            
            visits_occupancy = hf.get_visits_occupancy(signal_data.x_coordinates, signal_data.new_visits_times, x_grid, signal_data.y_coordinates, y_grid)

            activity_map, activity_map_smoothed = hf.get_activity_map(signal_data.input_signal,
                                                                        signal_data.x_coordinates, x_grid, self.map_smoothing_sigma_x,
                                                                        signal_data.y_coordinates, y_grid, self.map_smoothing_sigma_y)

            signal_data.add_peaks_detection()
            
            signal_data.add_binned_input_signal(self.nbins_cal)

            if np.all(np.isnan(y_grid_info)):
                nbins_pos = (x_grid_info.shape[0] - 1)
            else:
                nbins_pos = (x_grid_info.shape[0] - 1) * (y_grid_info.shape[0] - 1)

            mutual_info_original = info.get_mutual_information_binned(signal_data.input_signal_binned,self.nbins_cal,signal_data.position_binned,nbins_pos)

            mutual_info_classif_original = info.get_mutual_information_classif(signal_data.input_signal,signal_data.position_binned)
            
            mutual_info_regression_original = info.get_mutual_information_regression(signal_data.input_signal,signal_data.position_binned)

            mutual_info_skaggs_original = info.get_mutual_info_skaggs(signal_data.input_signal,signal_data.position_binned)
            
            mutual_info_kullback_leibler_original = info.get_kullback_leibler_normalized(signal_data.input_signal,signal_data.position_binned)

            results = self.parallelize_surrogate(signal_data.input_signal,signal_data.position_binned, signal_data.sampling_rate,
                                                 self.shift_time, self.nbins_cal, nbins_pos, 
                                                 signal_data.x_coordinates, signal_data.y_coordinates, x_grid, y_grid, 
                                                 self.map_smoothing_sigma_x, self.map_smoothing_sigma_y, self.num_cores, self.num_surrogates)

            mutual_info_shifted = []
            mutual_info_classif_shifted = []
            mutual_info_kullback_leibler_shifted = []
            mutual_info_skaggs_shifted = []
            mutual_info_regression_shifted = []
            activity_map_shifted = []
            activity_map_smoothed_shifted = []

            for perm in range(self.num_surrogates):
                mutual_info_shifted.append(results[perm][0])
                mutual_info_kullback_leibler_shifted.append(results[perm][1])
                mutual_info_classif_shifted.append(results[perm][2])
                mutual_info_skaggs_shifted.append(results[perm][3])
                mutual_info_regression_shifted.append(results[perm][4])
                activity_map_shifted.append(results[perm][5])
                activity_map_smoothed_shifted.append(results[perm][6])
                
            mutual_info_classif_shifted = np.array(mutual_info_classif_shifted)
            mutual_info_shifted = np.array(mutual_info_shifted)
            mutual_info_kullback_leibler_shifted = np.array(mutual_info_kullback_leibler_shifted)
            mutual_info_skaggs_shifted = np.array(mutual_info_skaggs_shifted)
            mutual_info_regression_shifted = np.array(mutual_info_regression_shifted)
            activity_map_shifted = np.array(activity_map_shifted)
            activity_map_smoothed_shifted = np.array(activity_map_smoothed_shifted)
            

            mutual_info_zscored, mutual_info_centered = info.get_mutual_information_zscored(mutual_info_original, mutual_info_shifted)

            mutual_info_classif_zscored, mutual_info_classif_centered = info.get_mutual_information_zscored(
                mutual_info_classif_original, mutual_info_classif_shifted)

            mutual_info_regression_zscored, mutual_info_regression_centered = info.get_mutual_information_zscored(
                mutual_info_regression_original, mutual_info_regression_shifted)

            mutual_info_skaggs_zscored, mutual_info_skaggs_centered = info.get_mutual_information_zscored(
                mutual_info_skaggs_original, mutual_info_skaggs_shifted)

            mutual_info_kullback_leibler_zscored, mutual_info_kullback_leibler_centered = info.get_mutual_information_zscored(
                mutual_info_kullback_leibler_original, mutual_info_kullback_leibler_shifted)

            num_of_fields, fields_x_max, fields_y_max, field_ids, pixels_place_cell_absolute, pixels_place_cell_relative, activity_map_identity \
                = hf.detect_place_fields(activity_map_smoothed, activity_map_smoothed_shifted,
                                        visits_occupancy,
                                        (x_center_bins, y_center_bins),
                                        threshold=self.threshold,
                                        min_num_of_bins=self.min_num_of_bins,
                                        threshold_fraction = self.threshold_fraction
                                        )

            sparsity = hf.get_sparsity(activity_map, position_occupancy)
            
            mutual_info_statistic = be.calculate_p_value(mutual_info_original, mutual_info_shifted, alternative='greater')
            mutual_info_pvalue = mutual_info_statistic.p_value

            mutual_info_kullback_leibler_statistic = be.calculate_p_value(mutual_info_kullback_leibler_original, mutual_info_kullback_leibler_shifted, alternative='greater')
            mutual_info_kullback_leibler_pvalue = mutual_info_kullback_leibler_statistic.p_value
            
            mutual_info_classif_statistic = be.calculate_p_value(mutual_info_classif_original, mutual_info_classif_shifted, alternative='greater')
            mutual_info_classif_pvalue = mutual_info_classif_statistic.p_value

            mutual_info_regression_statistic = be.calculate_p_value(mutual_info_regression_original, mutual_info_regression_shifted, alternative='greater')
            mutual_info_regression_pvalue = mutual_info_regression_statistic.p_value
            
            mutual_info_skaggs_statistic = be.calculate_p_value(mutual_info_skaggs_original, mutual_info_skaggs_shifted, alternative='greater')
            mutual_info_skaggs_pvalue = mutual_info_skaggs_statistic.p_value
            
            if (mutual_info_pvalue > self.alpha) & (mutual_info_classif_pvalue > self.alpha) & (mutual_info_regression_pvalue > self.alpha):

                activity_map_identity = np.zeros(activity_map.shape)*np.nan
                num_of_fields = 0
                pixels_place_cell_absolute = np.nan
                fields_x_max = np.nan
                fields_y_max = np.nan
                field_ids = np.nan
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

            inputdict['numb_events'] = signal_data.numb_events[0]
            inputdict['peaks_x_location'] = signal_data.peaks_x_location[0]
            inputdict['peaks_y_location'] = signal_data.peaks_y_location[0]
            inputdict['events_amplitude'] = signal_data.input_signal[signal_data.peaks_idx[0]]

            inputdict['activity_map_identity'] = activity_map_identity
            inputdict['num_of_fields'] = num_of_fields
            inputdict['fields_x_max'] = fields_x_max
            inputdict['fields_y_max'] = fields_y_max
            inputdict['field_ids'] = field_ids

            inputdict['place_cell_extension_absolute'] = pixels_place_cell_absolute
            inputdict['place_cell_extension_relative'] = pixels_place_cell_relative

            inputdict['sparsity'] = sparsity

            inputdict['mutual_info_original'] = mutual_info_original
            inputdict['mutual_info_shifted'] = mutual_info_shifted
            inputdict['mutual_info_zscored'] = mutual_info_zscored
            inputdict['mutual_info_centered'] = mutual_info_centered
            inputdict['mutual_info_pvalue'] = mutual_info_pvalue

            inputdict['mutual_info_classif_original'] = mutual_info_classif_original
            inputdict['mutual_info_classif_shifted'] = mutual_info_classif_shifted
            inputdict['mutual_info_classif_zscored'] = mutual_info_classif_zscored
            inputdict['mutual_info_classif_centered'] = mutual_info_classif_centered
            inputdict['mutual_info_classif_pvalue'] = mutual_info_classif_pvalue

            inputdict['mutual_info_regression_original'] = mutual_info_regression_original
            inputdict['mutual_info_regression_shifted'] = mutual_info_regression_shifted
            inputdict['mutual_info_regression_zscored'] = mutual_info_regression_zscored
            inputdict['mutual_info_regression_centered'] = mutual_info_regression_centered
            inputdict['mutual_info_regression_pvalue'] = mutual_info_regression_pvalue

            # inputdict['mutual_info_kullback_leibler_original'] = mutual_info_kullback_leibler_original
            # inputdict['mutual_info_kullback_leibler_shifted'] = mutual_info_kullback_leibler_shifted
            # inputdict['mutual_info_kullback_leibler_zscored'] = mutual_info_kullback_leibler_zscored
            # inputdict['mutual_info_kullback_leibler_centered'] = mutual_info_kullback_leibler_centered
            # inputdict['mutual_info_kullback_leibler_pvalue'] = mutual_info_kullback_leibler_pvalue

            # inputdict['mutual_info_skaggs_original'] = mutual_info_skaggs_original
            # inputdict['mutual_info_skaggs_shifted'] = mutual_info_skaggs_shifted
            # inputdict['mutual_info_skaggs_zscored'] = mutual_info_skaggs_zscored
            # inputdict['mutual_info_skaggs_centered'] = mutual_info_skaggs_centered
            # inputdict['mutual_info_skaggs_pvalue'] = mutual_info_skaggs_pvalue

            inputdict['input_parameters'] = self.__dict__['input_parameters']


        if self.saving == True:
            hf.caller_saving(inputdict, filename, self.saving_path, self.overwrite)
        else:
            print(filename + ' not saved')

        return inputdict

    
 
    def parallelize_surrogate(self, input_signal, position_binned, sampling_rate, shift_time,
                              nbins_cal, nbins_pos, x_coordinates, y_coordinates, x_grid, y_grid,
                              map_smoothing_sigma_x,map_smoothing_sigma_y, num_cores, num_surrogates):
        with tqdm_joblib(tqdm(desc="Processing Surrogates", total=num_surrogates)) as progress_bar:
            results = Parallel(n_jobs=num_cores)(
                delayed(self.get_mutual_info_surrogate)
                (
                    input_signal, position_binned, sampling_rate,
                    shift_time, nbins_cal, nbins_pos,
                    x_coordinates, y_coordinates, x_grid, y_grid, map_smoothing_sigma_x,
                    map_smoothing_sigma_y
                )
                for _ in range(num_surrogates)
            )
        return results





    def get_mutual_info_surrogate(self, input_signal, position_binned, sampling_rate, shift_time,
                                  nbins_cal, nbins_pos, x_coordinates, y_coordinates, x_grid, y_grid,
                                  map_smoothing_sigma_x,map_smoothing_sigma_y):

        input_signal_shifted = surrogate.circular_random_shift(input_signal, sampling_rate, shift_time, axis = 0)

        input_signal_shifted_binned = info.get_binned_signal(input_signal_shifted, nbins_cal)

        mutual_info_shifted = info.get_mutual_information_binned(input_signal_shifted_binned,nbins_cal, 
                                                                 position_binned,nbins_pos)

        mutual_info_shifted_classif = info.get_mutual_information_classif(input_signal_shifted, position_binned)

        mutual_info_shifted_regression = info.get_mutual_information_regression(input_signal_shifted, position_binned)

        modulation_index_shifted = info.get_kullback_leibler_normalized(input_signal_shifted,position_binned)

        mutual_info_skaggs_shifted = info.get_mutual_info_skaggs(input_signal_shifted, position_binned)

        activity_map_shifted, activity_map_smoothed_shifted = hf.get_activity_map(input_signal_shifted,
                                                                                   x_coordinates, x_grid, map_smoothing_sigma_x,
                                                                                   y_coordinates, y_grid, map_smoothing_sigma_y)
    

        return mutual_info_shifted, modulation_index_shifted, mutual_info_shifted_classif, mutual_info_skaggs_shifted,\
               mutual_info_shifted_regression, activity_map_shifted, activity_map_smoothed_shifted


