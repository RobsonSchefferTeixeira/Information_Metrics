import numpy as np
import os
import spatial_metrics.helper_functions as hf
from joblib import Parallel, delayed
import warnings

class PlaceCell:
    def __init__(self,**kwargs):
           
        kwargs.setdefault('animal_id', None)
        kwargs.setdefault('day', None)
        kwargs.setdefault('neuron', None)
        kwargs.setdefault('trial', None)
        kwargs.setdefault('dataset', None)
        kwargs.setdefault('video_srate', 30.)
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

        valid_kwargs = ['animal_id','day','neuron','dataset','trial','video_srate',
                        'min_time_spent','min_visits','min_speed_threshold','smoothing_size',
                        'x_bin_size','y_bin_size','shift_time','num_cores','percentile_threshold','min_num_of_pixels',
                        'num_surrogates','saving_path','saving','saving_string','environment_edges']

        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            setattr(self, k, v)
        self.__dict__['input_parameters'] = kwargs
        
        
    def main(self,I_timestamps,track_timevector,x_coordinates,y_coordinates):

        if len(I_timestamps) == 0:
            warnings.warn("Signal doesn't contain spike times")
            inputdict = np.nan
            
        else:

            speed = hf.get_speed(x_coordinates, y_coordinates, track_timevector)

            x_grid, y_grid, x_center_bins, y_center_bins, x_center_bins_repeated, y_center_bins_repeated = hf.get_position_grid(
                x_coordinates, y_coordinates, self.x_bin_size, self.y_bin_size,
                environment_edges=self.environment_edges)

            position_binned = hf.get_binned_2Dposition(x_coordinates, y_coordinates, x_grid, y_grid)

            visits_bins, new_visits_times = hf.get_visits(x_coordinates, y_coordinates, position_binned,
                                                          x_center_bins, y_center_bins)

            time_spent_inside_bins = hf.get_position_time_spent(position_binned, self.video_srate)

            I_keep = self.get_valid_timepoints(speed, visits_bins, time_spent_inside_bins,
                                               self.min_speed_threshold, self.min_visits, self.min_time_spent)

            x_coordinates_valid = x_coordinates[I_keep].copy()
            y_coordinates_valid = y_coordinates[I_keep].copy()
            track_timevector_valid = track_timevector[I_keep].copy()
            visits_bins_valid = visits_bins[I_keep].copy()
            position_binned_valid = position_binned[I_keep].copy()
            # I_timestamps_valid = np.intersect1d(I_timestamps, np.where(I_keep)[0])

            I_keep_spk = np.where(I_keep)[0]
            I_timestamps_valid = np.where(np.in1d(I_keep_spk, I_timestamps))[0]

            spike_rate_occupancy = self.get_spike_occupancy(I_timestamps_valid,x_coordinates_valid,y_coordinates_valid,
                                                            x_grid,y_grid)

            position_occupancy_valid = hf.get_occupancy(x_coordinates_valid, y_coordinates_valid, x_grid, y_grid,
                                                  self.video_srate)

            position_occupancy = hf.get_occupancy(x_coordinates,y_coordinates, x_grid, y_grid,
                                                  self.video_srate)

            visits_occupancy = hf.get_visits_occupancy(x_coordinates, y_coordinates, new_visits_times, x_grid, y_grid,
                                                       self.min_visits)

            place_field,place_field_smoothed = self.get_place_field(spike_rate_occupancy,position_occupancy_valid,self.smoothing_size)

            I_sec,I_spk = self.get_spatial_metrics(place_field,position_occupancy_valid)

            results = self.parallelize_surrogate(I_timestamps,I_keep,x_coordinates_valid,y_coordinates_valid,track_timevector,position_occupancy_valid,x_grid,y_grid,self.video_srate,self.smoothing_size,self.shift_time,self.num_cores,self.num_surrogates)

            I_sec_permutation = []
            I_spk_permutation = []
            place_field_shuffled = []
            place_field_smoothed_shuffled = []
            for perm in range(self.num_surrogates):
                I_sec_permutation.append(results[perm][0])
                I_spk_permutation.append(results[perm][1])
                place_field_shuffled.append(results[perm][2])
                place_field_smoothed_shuffled.append(results[perm][3])
            I_sec_permutation = np.array(I_sec_permutation)
            I_spk_permutation = np.array(I_spk_permutation)
            place_field_shuffled = np.array(place_field_shuffled)
            place_field_smoothed_shuffled = np.array(place_field_smoothed_shuffled)

            I_sec_zscored,I_sec_centered = self.get_mutual_information_zscored(I_sec,I_sec_permutation)
            I_spk_zscored,I_spk_centered = self.get_mutual_information_zscored(I_spk,I_spk_permutation)

            num_of_islands, islands_x_max, islands_y_max, pixels_place_cell_absolute, pixels_place_cell_relative = \
                hf.field_coordinates_using_shuffled(place_field_smoothed, place_field_smoothed_shuffled,
                                                    visits_occupancy, percentile_threshold=self.percentile_threshold,
                                                    min_num_of_pixels=self.min_num_of_pixels)

            sparsity = hf.get_sparsity(place_field, position_occupancy)

            x_peaks_location = x_coordinates_valid[I_timestamps_valid]
            y_peaks_location = y_coordinates_valid[I_timestamps_valid]

            inputdict = dict()
            inputdict['spike_rate_occupancy'] = spike_rate_occupancy
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
            inputdict['x_peaks_location'] = x_peaks_location
            inputdict['y_peaks_location'] = y_peaks_location
            inputdict['numb_events'] = I_timestamps_valid.shape[0]
            inputdict['I_sec'] = I_sec
            inputdict['I_spk'] = I_spk
            inputdict['I_spk_permutation'] = I_spk_permutation
            inputdict['I_sec_permutation'] = I_sec_permutation
            inputdict['I_sec_zscored'] = I_sec_zscored
            inputdict['I_spk_zscored'] = I_spk_zscored
            inputdict['I_sec_centered'] = I_sec_centered
            inputdict['I_spk_centered'] = I_spk_centered
            inputdict['sparsity'] = sparsity
            inputdict['num_of_islands'] = num_of_islands
            inputdict['islands_x_max'] = islands_x_max
            inputdict['islands_y_max'] = islands_y_max
            inputdict['place_cell_extension_absolute'] = pixels_place_cell_absolute
            inputdict['place_cell_extension_relative'] = pixels_place_cell_relative
            inputdict['sparsity'] = sparsity
            inputdict['input_parameters'] = self.__dict__['input_parameters']
            
            filename = hf.filename_constructor(self.saving_string,self.animal_id,self.dataset,self.day,self.neuron,self.trial)
        
        if self.saving == True:
            hf.caller_saving(inputdict,filename,self.saving_path)
            print(filename + ' saved')
        else:
            print('File not saved')
        return inputdict


    def get_mutual_information_zscored(self,mutual_info_original,mutual_info_shuffled):
        mutual_info_centered = mutual_info_original-np.nanmean(mutual_info_shuffled)
        mutual_info_zscored = (mutual_info_original-np.nanmean(mutual_info_shuffled))/np.nanstd(mutual_info_shuffled)
        
        return mutual_info_zscored,mutual_info_centered


    def get_valid_timepoints(self,speed, visits_bins, time_spent_inside_bins, min_speed_threshold,
                             min_visits, min_time_spent):
        # min speed
        I_speed_thres = speed >= min_speed_threshold

        # min visits
        I_visits_times_thres = visits_bins >= min_visits

        # min time spent
        I_time_spent_thres = time_spent_inside_bins >= min_time_spent

        I_keep = I_speed_thres * I_visits_times_thres * I_time_spent_thres
        return I_keep


    def get_place_field(self,spike_rate_occupancy,position_occupancy,smoothing_size):

        place_field = spike_rate_occupancy/position_occupancy

        place_field_to_smooth = np.copy(place_field)
        place_field_to_smooth[np.isnan(place_field_to_smooth)] = 0
        place_field_smoothed = hf.gaussian_smooth_2d(place_field_to_smooth,smoothing_size)

        return place_field,place_field_smoothed


    def get_spiketimes_binarized(self,I_timestamps,xy_timevector,video_srate):
        # this way only works when two spikes don't fall in the same bin
        # spike_timevector = np.zeros(timevector.shape[0])
        # spike_timevector[I_timestamps] = 1
        eps = np.finfo(np.float64).eps
        xy_timevector_hist = np.append(xy_timevector,xy_timevector[-1] + eps)
        spike_timevector = np.histogram(xy_timevector[I_timestamps],xy_timevector_hist)[0]
    
        # xy_timevector_hist = np.append(xy_timevector,xy_timevector[-1]+(1/video_srate))
        # spike_timevector = np.histogram(xy_timevector[I_timestamps],xy_timevector_hist)[0]

        return spike_timevector

    def get_surrogate(self,I_timestamps,xy_timevector,video_srate,shift_time):
        eps = np.finfo(np.float64).eps
        xy_timevector_hist = np.append(xy_timevector,xy_timevector[-1] + eps)
        spike_timevector = np.histogram(xy_timevector[I_timestamps],xy_timevector_hist)[0]

        I_break = np.random.choice(np.linspace(-shift_time*video_srate,video_srate*shift_time),1)[0].astype(int)
        input_vector_shuffled = np.concatenate([spike_timevector[I_break:], spike_timevector[0:I_break]])

        timestamps_shuffled = np.repeat(xy_timevector, input_vector_shuffled)

        I_timestamps_shuffled = self.searchsorted2(xy_timevector, timestamps_shuffled)
        I_keep = np.abs(xy_timevector[I_timestamps_shuffled]-timestamps_shuffled)<0.1
        I_timestamps_shuffled = I_timestamps_shuffled[I_keep]

        return I_timestamps_shuffled


    def searchsorted2(self,known_array, test_array):
        index_sorted = np.argsort(known_array)
        known_array_sorted = known_array[index_sorted]
        known_array_middles = known_array_sorted[1:] - np.diff(known_array_sorted.astype('f'))/2
        idx1 = np.searchsorted(known_array_middles, test_array)
        indices = index_sorted[idx1]
        return indices

    def get_spatial_metrics(self,place_field,position_occupancy):

        place_field[np.isinf(place_field) | np.isnan(place_field)] = 0
        non_zero_vals = place_field != 0

        noccup_ratio = position_occupancy/np.nansum(position_occupancy)
        overall_frate = np.nansum(place_field[non_zero_vals]*noccup_ratio[non_zero_vals])


        # bits per second
        I_sec = np.nansum(place_field[non_zero_vals]*noccup_ratio[non_zero_vals]*np.log2(place_field[non_zero_vals]/overall_frate))
        # bits per spike
        I_spk = I_sec/overall_frate
        # I_spk = np.nansum((spike_rate_occupancy[non_zero_vals]/overall_frate)*noccup_ratio[non_zero_vals]*np.log2(spike_rate_occupancy[non_zero_vals]/overall_frate))

        return I_sec,I_spk



    def get_spike_occupancy(self,I_timestamps,x_coordinates,y_coordinates,x_grid,y_grid):

        # calculate mean calcium per pixel
        spike_rate_occupancy = np.nan*np.zeros((y_grid.shape[0]-1,x_grid.shape[0]-1)) 
        for xx in range(0,x_grid.shape[0]-1):
            for yy in range(0,y_grid.shape[0]-1):

                check_x_occupancy = np.logical_and(x_coordinates >= x_grid[xx],x_coordinates < (x_grid[xx+1]))
                check_y_occupancy = np.logical_and(y_coordinates >= y_grid[yy],y_coordinates < (y_grid[yy+1]))
                I_location = np.where(np.logical_and(check_x_occupancy,check_y_occupancy))[0]
                # spike_rate_occupancy[yy,xx] = np.sum(np.in1d(I_timestamps,I_location))/I_location.shape[0]
                spike_rate_occupancy[yy,xx] = np.sum(np.in1d(I_timestamps,I_location))
        return spike_rate_occupancy



    def parallelize_surrogate(self,I_timestamps,I_keep,x_coordinates_valid,y_coordinates_valid,track_timevector,position_occupancy_valid,
                              x_grid,y_grid,video_srate,smoothing_size,
                              shift_time,num_cores,num_surrogates):

        results = Parallel(n_jobs=num_cores,verbose = 1)(delayed(self.get_spatial_metrics_surrogate)(I_timestamps,I_keep,x_coordinates_valid,
                                                                                                     y_coordinates_valid,track_timevector,
                                                                                                     position_occupancy_valid,
                                                                                                     x_grid,y_grid,video_srate,
                                                                                                     smoothing_size,shift_time)
                                                         for permi in range(num_surrogates))
        return np.array(results)


    def get_spatial_metrics_surrogate(self,I_timestamps,I_keep,x_coordinates_valid,y_coordinates_valid,track_timevector,
                                      position_occupancy_valid,x_grid,y_grid,video_srate,smoothing_size,shift_time):

        I_timestamps_shuffled = self.get_surrogate(I_timestamps,track_timevector,video_srate,shift_time)

        I_keep_spk = np.where(I_keep)[0]
        I_timestamps_shuffled_valid = np.where(np.in1d(I_keep_spk, I_timestamps_shuffled))[0]

        spike_rate_occupancy_shuffled = self.get_spike_occupancy(I_timestamps_shuffled_valid,x_coordinates_valid,
                                                                 y_coordinates_valid,x_grid,y_grid)

        place_field_shuffled, place_field_smoothed_shuffled = self.get_place_field(spike_rate_occupancy_shuffled,
                                                                                   position_occupancy_valid, smoothing_size)

        I_sec_shuffled,I_spk_shuffled = self.get_spatial_metrics(place_field_shuffled,position_occupancy_valid)

        return I_sec_shuffled,I_spk_shuffled,place_field_shuffled,place_field_smoothed_shuffled


    

            
            



    
    