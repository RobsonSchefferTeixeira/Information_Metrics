import numpy as np
import os
from spatial_metrics import helper_functions as hf
from joblib import Parallel, delayed
from spatial_metrics import surrogate_functions as surrogate
from spatial_metrics import information_base as info
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
        kwargs.setdefault('speed_smoothing_sigma', 1)

        kwargs.setdefault('x_bin_size', 1)
        kwargs.setdefault('y_bin_size', None)
        kwargs.setdefault('environment_edges', None)
        kwargs.setdefault('map_smoothing_sigma_x', 2)
        kwargs.setdefault('map_smoothing_sigma_y', 2)
        kwargs.setdefault('shift_time', 10)
        kwargs.setdefault('num_cores', 1)
        kwargs.setdefault('num_surrogates', 200)
        kwargs.setdefault('saving_path', os.getcwd())
        kwargs.setdefault('saving', False)
        kwargs.setdefault('saving_string', 'SpatialMetrics')
        kwargs.setdefault('nbins_cal', 10)
        kwargs.setdefault('percentile_threshold', 95)
        kwargs.setdefault('min_num_of_bins', 4)
        kwargs.setdefault('detection_threshold', 2)
        kwargs.setdefault('detection_smoothing_sigma_x', 2)
        kwargs.setdefault('detection_smoothing_sigma_y', 2)
        kwargs.setdefault('field_detection_method','std_from_field')


        valid_kwargs = ['animal_id', 'day', 'neuron', 'dataset', 'trial', 'sampling_rate',
                        'min_time_spent', 'min_visits', 'min_speed_threshold', 'speed_smoothing_sigma',
                        'x_bin_size', 'y_bin_size', 'shift_time', 'map_smoothing_sigma_x','map_smoothing_sigma_y','num_cores', 'percentile_threshold','min_num_of_bins',
                        'num_surrogates', 'saving_path', 'saving', 'saving_string', 'environment_edges', 'nbins_cal',
                        'detection_threshold','detection_smoothing_sigma_x','detection_smoothing_sigma_y','field_detection_method']

        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            setattr(self, k, v)

        self.__dict__['input_parameters'] = kwargs

    def main(self, calcium_imag, time_vector, x_coordinates, y_coordinates=None, speed = None):

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        
        if np.all(np.isnan(calcium_imag)):
            warnings.warn("Signal contains only NaN's")
            inputdict = np.nan
            filename = self.filename_constructor(self.saving_string, self.animal_id, self.dataset, self.day,
                                                 self.neuron, self.trial)
        else:
        
            x_coordinates, y_coordinates = hf.correct_coordinates(x_coordinates, y_coordinates,environment_edges=self.environment_edges)

            self.validate_input_data(calcium_imag, x_coordinates, y_coordinates,time_vector)

            if speed is None:
                speed,speed_smoothed = hf.get_speed(x_coordinates, y_coordinates, time_vector, self.speed_smoothing_sigma)

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
                                                                    self.map_smoothing_sigma_x,self.map_smoothing_sigma_y)

            calcium_imag_valid_binned = info.get_binned_signal(calcium_imag_valid, self.nbins_cal)

            nbins_pos = (x_grid.shape[0] - 1) * (y_grid.shape[0] - 1)

            mutual_info_original = info.get_mutual_information_binned(calcium_imag_valid_binned,self.nbins_cal,position_binned_valid,nbins_pos)

            mutual_info_kullback_leibler_original = info.get_kullback_leibler_normalized(calcium_imag_valid,position_binned_valid)

            mutual_info_NN_original = info.get_mutual_information_NN(calcium_imag_valid, position_binned_valid)
            
            mutual_info_regression_original = info.get_mutual_information_regression(calcium_imag_valid, position_binned_valid)

            mutual_info_skaggs_original = info.get_mutual_info_skaggs(calcium_imag_valid, position_binned_valid)

            mutual_info_distribution, mutual_info_distribution_bezzi = info.get_mutual_information_2d(
                calcium_imag_valid_binned,position_binned_valid, y_grid, x_grid, self.nbins_cal, nbins_pos,self.smoothing_size)

            results = self.parallelize_surrogate(calcium_imag_valid, position_binned_valid, self.sampling_rate,
                                                 self.shift_time, self.nbins_cal, nbins_pos, x_coordinates_valid,
                                                 y_coordinates_valid, x_grid, y_grid, self.smoothing_size,
                                                 self.num_cores, self.num_surrogates)

            place_field_shifted = []
            place_field_smoothed_shifted = []
            mutual_info_shifted = []
            mutual_info_NN_shifted = []
            mutual_info_kullback_leibler_shifted = []
            mutual_info_skaggs_shifted = []
            mutual_info_distribution_shifted = []
            mutual_info_distribution_bezzi_shifted = []
            mutual_info_regression_shifted = []

            for perm in range(self.num_surrogates):
                mutual_info_shifted.append(results[perm][0])
                mutual_info_kullback_leibler_shifted.append(results[perm][1])
                mutual_info_NN_shifted.append(results[perm][2])
                mutual_info_skaggs_shifted.append(results[perm][3])
                place_field_shifted.append(results[perm][4])
                place_field_smoothed_shifted.append(results[perm][5])
                mutual_info_distribution_shifted.append(results[perm][6])
                mutual_info_distribution_bezzi_shifted.append(results[perm][7])
                mutual_info_regression_shifted.append(results[perm][8])

            mutual_info_NN_shifted = np.array(mutual_info_NN_shifted)
            mutual_info_shifted = np.array(mutual_info_shifted)
            mutual_info_kullback_leibler_shifted = np.array(mutual_info_kullback_leibler_shifted)
            mutual_info_skaggs_shifted = np.array(mutual_info_skaggs_shifted)
            place_field_shifted = np.array(place_field_shifted)
            place_field_smoothed_shifted = np.array(place_field_smoothed_shifted)
            mutual_info_distribution_shifted = np.array(mutual_info_distribution_shifted)
            mutual_info_distribution_bezzi_shifted = np.array(mutual_info_distribution_bezzi_shifted)
            mutual_info_regression_shifted = np.array(mutual_info_regression_shifted)
            

            mutual_info_zscored, mutual_info_centered = info.get_mutual_information_zscored(mutual_info_original,
                                                                                            mutual_info_shifted)
            mutual_info_kullback_leibler_zscored, mutual_info_kullback_leibler_centered = info.get_mutual_information_zscored(
                mutual_info_kullback_leibler_original, mutual_info_kullback_leibler_shifted)

            mutual_info_NN_zscored, mutual_info_NN_centered = info.get_mutual_information_zscored(
                mutual_info_NN_original, mutual_info_NN_shifted)

            mutual_info_skaggs_zscored, mutual_info_skaggs_centered = info.get_mutual_information_zscored(
                mutual_info_skaggs_original, mutual_info_skaggs_shifted)

            mutual_info_regression_zscored, mutual_info_regression_centered = info.get_mutual_information_zscored(
                mutual_info_regression_original, mutual_info_regression_shifted)

            if self.field_detection_method == 'random_fields':
                # num_of_islands, islands_x_max, islands_y_max,pixels_place_cell_absolute,pixels_place_cell_relative,place_field_identity = \
                # hf.field_coordinates_using_shifted(place_field,place_field_shifted,visits_occupancy,
                #                                    percentile_threshold=self.percentile_threshold,
                #                                   min_num_of_bins = self.min_num_of_bins)
                
                num_of_islands, islands_x_max, islands_y_max,pixels_place_cell_absolute,pixels_place_cell_relative,place_field_identity = \
                hf.field_coordinates_using_shifted(place_field_smoothed,place_field_smoothed_shifted,visits_occupancy,x_center_bins, y_center_bins,
                                                    percentile_threshold=self.percentile_threshold,
                                                    min_num_of_bins = self.min_num_of_bins)
                

            elif self.field_detection_method == 'std_from_field':
                num_of_islands, islands_x_max, islands_y_max,pixels_place_cell_absolute,pixels_place_cell_relative,place_field_identity = \
                hf.field_coordinates_using_threshold(place_field, visits_occupancy,x_center_bins, y_center_bins,smoothing_size = self.detection_smoothing_size,    
                                                   field_threshold=self.detection_threshold,
                                                   min_num_of_bins=self.min_num_of_bins)
            else:
                 warnings.warn("No field detection method set", UserWarning)
                 num_of_islands, islands_x_max, islands_y_max, pixels_place_cell_absolute, pixels_place_cell_relative, place_field_identity = [[] for _ in range(6)]

            

            I_peaks = hf.detect_peaks(calcium_imag_valid, mpd=0.5 * self.sampling_rate,
                                      mph=1. * np.nanstd(calcium_imag_valid))
            peaks_amplitude = calcium_imag_valid[I_peaks]
            x_peaks_location = x_coordinates_valid[I_peaks]
            y_peaks_location = y_coordinates_valid[I_peaks]

            sparsity = hf.get_sparsity(place_field, position_occupancy)

            inputdict = dict()
            inputdict['place_field'] = place_field
            inputdict['place_field_smoothed'] = place_field_smoothed

            inputdict['place_field_shifted'] = place_field_shifted
            inputdict['place_field_smoothed_shifted'] = place_field_smoothed_shifted

            inputdict['mutual_info_distribution'] = mutual_info_distribution
            inputdict['mutual_info_distribution_bezzi'] = mutual_info_distribution_bezzi

            inputdict['mutual_info_distribution_shifted'] = mutual_info_distribution_shifted
            inputdict['mutual_info_distribution_bezzi_shifted'] = mutual_info_distribution_bezzi_shifted

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
            inputdict['events_amplitude'] = peaks_amplitude

            inputdict['place_field_identity'] = place_field_identity
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

            inputdict['mutual_info_kullback_leibler_original'] = mutual_info_kullback_leibler_original
            inputdict['mutual_info_kullback_leibler_shifted'] = mutual_info_kullback_leibler_shifted
            inputdict['mutual_info_kullback_leibler_zscored'] = mutual_info_kullback_leibler_zscored
            inputdict['mutual_info_kullback_leibler_centered'] = mutual_info_kullback_leibler_centered

            inputdict['mutual_info_NN_original'] = mutual_info_NN_original
            inputdict['mutual_info_NN_shifted'] = mutual_info_NN_shifted
            inputdict['mutual_info_NN_zscored'] = mutual_info_NN_zscored
            inputdict['mutual_info_NN_centered'] = mutual_info_NN_centered

            inputdict['mutual_info_regression_original'] = mutual_info_regression_original
            inputdict['mutual_info_regression_shifted'] = mutual_info_regression_shifted
            inputdict['mutual_info_regression_zscored'] = mutual_info_regression_zscored
            inputdict['mutual_info_regression_centered'] = mutual_info_regression_centered

            inputdict['mutual_info_skaggs_original'] = mutual_info_skaggs_original
            inputdict['mutual_info_skaggs_shifted'] = mutual_info_skaggs_shifted
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

 
    def parallelize_surrogate(self, calcium_imag_valid, position_binned_valid, sampling_rate, shift_time,
                              nbins_cal, nbins_pos, x_coordinates_valid, y_coordinates_valid, x_grid, y_grid,
                              smoothing_size, num_cores, num_surrogates):
        results = Parallel(n_jobs=num_cores)(
            delayed(self.get_mutual_info_surrogate)(calcium_imag_valid, position_binned_valid, sampling_rate,
                                                    shift_time, nbins_cal, nbins_pos, x_coordinates_valid,
                                                    y_coordinates_valid, x_grid, y_grid, smoothing_size)
            for _ in range(num_surrogates))

        return results



    def get_mutual_info_surrogate(self, calcium_imag_valid, position_binned_valid, sampling_rate, shift_time,
                                  nbins_cal, nbins_pos, x_coordinates_valid, y_coordinates_valid, x_grid, y_grid,
                                  smoothing_size):

        calcium_imag_shifted_valid = surrogate.get_signal_surrogate(calcium_imag_valid, sampling_rate, shift_time)
        calcium_imag_shifted_binned = info.get_binned_signal(calcium_imag_shifted_valid, nbins_cal)

        mutual_info_shifted = info.get_mutual_information_binned(calcium_imag_shifted_binned,nbins_cal, position_binned_valid,nbins_pos)

        mutual_info_shifted_NN = info.get_mutual_information_NN(calcium_imag_shifted_valid, position_binned_valid)

        mutual_info_shifted_regression = info.get_mutual_information_regression(calcium_imag_shifted_valid, position_binned_valid)

        modulation_index_shifted = info.get_kullback_leibler_normalized(calcium_imag_shifted_valid,
                                                                         position_binned_valid)

        mutual_info_skaggs_shifted = info.get_mutual_info_skaggs(calcium_imag_shifted_valid, position_binned_valid)

        place_field_shifted, place_field_smoothed_shifted = self.get_place_field(calcium_imag_shifted_valid,
                                                                                   x_coordinates_valid,
                                                                                   y_coordinates_valid, x_grid, y_grid,
                                                                                   smoothing_size)
        
        mutual_info_distribution,mutual_info_distribution_bezzi = info.get_mutual_information_2d(calcium_imag_shifted_binned,
                                                                                                  position_binned_valid,y_grid,
                                                                                                  x_grid,nbins_cal,nbins_pos,
                                                                                                  smoothing_size)

        return mutual_info_shifted, modulation_index_shifted, mutual_info_shifted_NN, mutual_info_skaggs_shifted,\
               place_field_shifted, place_field_smoothed_shifted,mutual_info_distribution,mutual_info_distribution_bezzi,mutual_info_shifted_regression


