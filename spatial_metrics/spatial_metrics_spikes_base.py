import numpy as np
import os
import sys
from scipy import stats as stats
import spatial_metrics.helper_functions as hf
import spatial_metrics.detect_peaks as dp
from joblib import Parallel, delayed

class PlaceCell:
    def __init__(self,**kwargs):
           
        kwargs.setdefault('animal_id', None)
        kwargs.setdefault('day', None)
        kwargs.setdefault('neuron', None)
        kwargs.setdefault('trial', None)
        kwargs.setdefault('dataset', None)
        kwargs.setdefault('video_srate', 30.)
        kwargs.setdefault('mintimespent', 0.1)
        kwargs.setdefault('minvisits', 1)
        kwargs.setdefault('speed_threshold', 2.5)
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

        valid_kwargs = ['animal_id','day','neuron','dataset','trial','video_srate',
                        'mintimespent','minvisits','speed_threshold','smoothing_size',
                        'x_bin_size','y_bin_size','shift_time','num_cores',
                        'num_surrogates','saving_path','saving','saving_string','environment_edges']
        
        
        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            setattr(self, k, v)
            
        self.__dict__['input_parameters'] = kwargs
        
        
    def main(self,I_timestamps,xy_timevector,x_coordinates,y_coordinates):

        if len(I_timestamps) == 0:
            warnings.warn("Signal doesn't contain spike times")
            inputdict = np.nan
            
        else:

            speed = self.get_speed(x_coordinates,y_coordinates,xy_timevector)

            x_coordinates_valid,y_coordinates_valid = self.get_valid_timepoints(speed,x_coordinates,y_coordinates,self.speed_threshold)
    
            x_grid,y_grid,x_center_bins,y_center_bins,x_center_bins_repeated,y_center_bins_repeated = self.get_position_grid(x_coordinates,                           y_coordinates, self.x_bin_size, self.y_bin_size,environment_edges = self.environment_edges)

            spike_rate_occupancy = self.get_spike_occupancy(I_timestamps,x_coordinates_valid,y_coordinates_valid,x_grid,y_grid)

            position_occupancy = self.get_occupancy(x_coordinates_valid,y_coordinates_valid,x_grid,y_grid,self.video_srate)

            visits_occupancy = self.get_visits(x_coordinates_valid,y_coordinates_valid,x_grid,y_grid,x_center_bins,y_center_bins)

            place_field,place_field_smoothed = self.validate_place_field(spike_rate_occupancy,position_occupancy,                                                     visits_occupancy,self.mintimespent, self.minvisits,self.smoothing_size)

            I_sec,I_spk = self.get_spatial_metrics(place_field,position_occupancy)

            sparsity = self.get_sparsity(place_field_smoothed,position_occupancy)

 
            results = self.parallelize_surrogate(I_timestamps,x_coordinates,y_coordinates,xy_timevector,position_occupancy,visits_occupancy,x_grid,y_grid,self.video_srate,self.mintimespent,self.minvisits,self.smoothing_size,self.shift_time,self.num_cores,self.num_surrogates)

            I_sec_permutation = results[:,0]
            I_spk_permutation = results[:,1]

            I_sec_zscored,I_sec_centered = self.get_mutual_information_zscored(I_sec,I_sec_permutation)
            I_spk_zscored,I_spk_centered = self.get_mutual_information_zscored(I_spk,I_spk_permutation)
            
            
            spatial_map_smoothed_threshold = np.copy(place_field_smoothed)
            I_threshold = 2*np.nanstd(spatial_map_smoothed_threshold)

            total_visited_pixels = np.nansum(visits_occupancy != 0)
            pixels_above = np.nansum(spatial_map_smoothed_threshold > I_threshold)
            pixels_total = spatial_map_smoothed_threshold.shape[0]*spatial_map_smoothed_threshold.shape[1]

            pixels_place_cell_relative = pixels_above/total_visited_pixels
            pixels_place_cell_absolute = pixels_above/pixels_total

            place_field_above_to_island = np.copy(spatial_map_smoothed_threshold)
            place_field_above_to_island[place_field_above_to_island < I_threshold] = 0
            place_field_above_to_island[place_field_above_to_island >= I_threshold] = 1

            if np.any(place_field_above_to_island==1):
                sys.setrecursionlimit(10000)
                num_of_islands = self.number_of_islands(np.copy(place_field_above_to_island))

            else:
                num_of_islands = 0
            

            inputdict = dict()
            inputdict['spike_rate_occupancy'] = spike_rate_occupancy
            inputdict['place_field'] = place_field
            inputdict['place_field_smoothed'] = place_field_smoothed        
            inputdict['ocuppancy_map'] = position_occupancy
            inputdict['visits_map'] = visits_occupancy
            inputdict['x_grid'] = x_grid
            inputdict['y_grid'] = y_grid
            inputdict['x_center_bins'] = x_center_bins
            inputdict['y_center_bins'] = y_center_bins         
            inputdict['numb_events'] = I_timestamps.shape[0]
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
            inputdict['place_cell_extension_absolute'] = pixels_place_cell_absolute
            inputdict['place_cell_extension_relative'] = pixels_place_cell_relative

            inputdict['input_parameters'] = self.__dict__['input_parameters']
            
            filename = self.filename_constructor(self.saving_string,self.animal_id,self.dataset,self.day,self.neuron,self.trial)
        
        if self.saving == True:
            self.caller_saving(inputdict,filename,self.saving_path)
            print(filename + ' saved')
        else:
            print('File not saved')
        
        return inputdict

    

    def get_mutual_information_zscored(self,mutual_info_original,mutual_info_shuffled):
        mutual_info_centered = mutual_info_original-np.nanmean(mutual_info_shuffled)
        mutual_info_zscored = (mutual_info_original-np.nanmean(mutual_info_shuffled))/np.nanstd(mutual_info_shuffled)
        
        return mutual_info_zscored,mutual_info_centered

    
    def filename_constructor(self,saving_string,animal_id,dataset,day,neuron,trial):

        first_string =  saving_string
        animal_id_string = '.' + animal_id
        dataset_string = '.Dataset.' + dataset
        day_string = '.Day.' + str(day)
        neuron_string = '.Neuron.' + str(neuron)
        trial_string = '.Trial.' + str(trial)

        filename_checklist = np.array([first_string,animal_id, dataset, day, neuron, trial])
        inlcude_this = np.where(filename_checklist != None)[0]

        filename_backbone = [first_string, animal_id_string,dataset_string, day_string, neuron_string, trial_string]

        filename = ''.join([filename_backbone[i] for i in inlcude_this])
               
        return filename
    
    def caller_saving(self,inputdict,filename,saving_path):
        os.chdir(saving_path)
        output = open(filename, 'wb') 
        np.save(output,inputdict)
        output.close()
     

    def get_sparsity(self,place_field,position_occupancy):
        
        position_occupancy_norm = np.nansum(position_occupancy/np.nansum(position_occupancy))
        sparsity = np.nanmean(position_occupancy_norm*place_field)**2/np.nanmean(position_occupancy_norm*place_field**2)

        return sparsity
    
   

    def get_speed(self,x_coordinates,y_coordinates,xy_timevector):

        speed = np.sqrt(np.diff(x_coordinates)**2 + np.diff(y_coordinates)**2)
        speed = hf.smooth(speed/np.diff(xy_timevector),window_len=10)
        speed = np.hstack([speed,0])
        return speed

    def get_valid_timepoints(self,speed,x_coordinates,y_coordinates,speed_threshold):

        x_coordinates_valid = np.copy(x_coordinates)
        y_coordinates_valid = np.copy(y_coordinates)
        I_speed_non_valid = speed <= speed_threshold
        x_coordinates_valid[I_speed_non_valid] = np.nan
        y_coordinates_valid[I_speed_non_valid] = np.nan
        
        return x_coordinates_valid,y_coordinates_valid


    def validate_place_field(self,spike_rate_occupancy,position_occupancy,visits_occupancy,mintimespent, minvisits,smoothing_size):

        place_field = spike_rate_occupancy/position_occupancy

        Valid=(position_occupancy>=mintimespent)*(visits_occupancy>=minvisits)*1.
        Valid[Valid == 0] = np.nan
        place_field = place_field*Valid

        place_field_to_smooth = np.copy(place_field)
        place_field_to_smooth[np.isnan(place_field_to_smooth)] = 0
        place_field_smoothed = hf.gaussian_smooth_2d(place_field_to_smooth,smoothing_size)

        return place_field,place_field_smoothed


    def get_binned_2Dposition(self,x_coordinates,y_coordinates,x_grid,y_grid):

        # calculate position occupancy
        position_binned = np.zeros(x_coordinates.shape)*np.nan
        count = 0
        for xx in range(0,x_grid.shape[0]-1):
            for yy in range(0,y_grid.shape[0]-1):
                if xx == x_grid.shape[0]-2:
                    check_x_ocuppancy = np.logical_and(x_coordinates >= x_grid[xx],x_coordinates <= (x_grid[xx+1]))
                else:
                    check_x_ocuppancy = np.logical_and(x_coordinates >= x_grid[xx],x_coordinates < (x_grid[xx+1]))

                if yy == y_grid.shape[0]-2:
                    check_y_ocuppancy = np.logical_and(y_coordinates >= y_grid[yy],y_coordinates <= (y_grid[yy+1]))
                else:   
                    check_y_ocuppancy = np.logical_and(y_coordinates >= y_grid[yy],y_coordinates < (y_grid[yy+1]))

                
                position_binned[np.logical_and(check_x_ocuppancy,check_y_ocuppancy)] = count
                count += 1

        return position_binned

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

                check_x_ocuppancy = np.logical_and(x_coordinates >= x_grid[xx],x_coordinates < (x_grid[xx+1]))
                check_y_ocuppancy = np.logical_and(y_coordinates >= y_grid[yy],y_coordinates < (y_grid[yy+1]))
                I_location = np.where(np.logical_and(check_x_ocuppancy,check_y_ocuppancy))[0]
                # spike_rate_occupancy[yy,xx] = np.sum(np.in1d(I_timestamps,I_location))/I_location.shape[0]
                spike_rate_occupancy[yy,xx] = np.sum(np.in1d(I_timestamps,I_location))
        return spike_rate_occupancy



    def get_position_grid(self,x_coordinates,y_coordinates,x_bin_size,y_bin_size,environment_edges=None):

        # x_bin_size and y_bin_size in cm
        # environment_edges = [[x1, x2], [y1, y2]]
        
        if environment_edges==None:
            x_min = np.nanmin(x_coordinates)
            x_max = np.nanmax(x_coordinates)
            y_min = np.nanmin(y_coordinates)
            y_max = np.nanmax(y_coordinates)
            
            environment_edges = [[x_min,x_max],[y_min,y_max]]


        x_grid = np.arange(environment_edges[0][0]- x_bin_size,environment_edges[0][1] + x_bin_size,x_bin_size)

        y_grid = np.arange(environment_edges[1][0]- y_bin_size,environment_edges[1][1] + y_bin_size,y_bin_size)

        x_center_bins = x_grid[0:-1] + x_bin_size/2
        y_center_bins = y_grid[0:-1] + y_bin_size/2

        x_center_bins_repeated = np.repeat(x_center_bins,y_center_bins.shape[0])
        y_center_bins_repeated = np.tile(y_center_bins,x_center_bins.shape[0])


        return x_grid,y_grid,x_center_bins,y_center_bins,x_center_bins_repeated,y_center_bins_repeated



    def get_occupancy(self,x_coordinates,y_coordinates,x_grid,y_grid,video_srate):
        # calculate position occupancy
        position_occupancy = np.zeros((y_grid.shape[0]-1,x_grid.shape[0]-1))
        for xx in range(0,x_grid.shape[0]-1):
            for yy in range(0,y_grid.shape[0]-1):

                check_x_ocuppancy = np.logical_and(x_coordinates >= x_grid[xx],x_coordinates < (x_grid[xx+1]))
                check_y_ocuppancy = np.logical_and(y_coordinates >= y_grid[yy],y_coordinates < (y_grid[yy+1]))

                position_occupancy[yy,xx] = np.sum(np.logical_and(check_x_ocuppancy,check_y_ocuppancy))/video_srate

        return position_occupancy





    def get_visits(self,x_coordinates,y_coordinates,x_grid,y_grid,x_center_bins,y_center_bins):

        I_x_coord = []
        I_y_coord = []

        for xx in range(0,x_coordinates.shape[0]):
            I_x_coord.append(np.argmin(np.abs(x_coordinates[xx] - x_center_bins)))
            I_y_coord.append(np.argmin(np.abs(y_coordinates[xx] - y_center_bins)))

        I_x_coord = np.array(I_x_coord)
        I_y_coord = np.array(I_y_coord)

        dx = np.diff(np.hstack([I_x_coord[0]-1,I_x_coord]))
        dy = np.diff(np.hstack([I_y_coord[0]-1,I_y_coord]))

        newvisitstimes = (-1*(dy == 0))*(dx==0)+1
        newvisitstimes2 = (np.logical_or((dy != 0), (dx!=0))*1)

        I_visit = np.where(newvisitstimes>0)[0]

        # calculate visits

        x_coordinate_visit = x_coordinates[I_visit]
        y_coordinate_visit = y_coordinates[I_visit]

        visits_occupancy = np.zeros((y_grid.shape[0]-1,x_grid.shape[0]-1))        
        for xx in range(0,x_grid.shape[0]-1):
            for yy in range(0,y_grid.shape[0]-1):

                check_x_ocuppancy = np.logical_and(x_coordinate_visit >= x_grid[xx],x_coordinate_visit < (x_grid[xx+1]))
                check_y_ocuppancy = np.logical_and(y_coordinate_visit >= y_grid[yy],y_coordinate_visit < (y_grid[yy+1]))

                visits_occupancy[yy,xx] = np.sum(np.logical_and(check_x_ocuppancy,check_y_ocuppancy))

        return visits_occupancy



    def parallelize_surrogate(self,I_timestamps,x_coordinates,y_coordinates,xy_timevector,position_occupancy,visits_occupancy,x_grid,y_grid,video_srate,mintimespent,minvisits,smoothing_size,shift_time,num_cores,num_surrogates):

        results = Parallel(n_jobs=num_cores,verbose = 1)(delayed(self.get_spatial_metrics_surrogate)(I_timestamps,x_coordinates,y_coordinates,xy_timevector,position_occupancy,visits_occupancy,x_grid,y_grid,video_srate,mintimespent,minvisits,smoothing_size,shift_time) for permi in range(num_surrogates))
        return np.array(results)


    def get_spatial_metrics_surrogate(self,I_timestamps,x_coordinates,y_coordinates,xy_timevector,position_occupancy,visits_occupancy,x_grid,y_grid,video_srate,mintimespent,minvisits,smoothing_size,shift_time):
        I_timestamps_shuffled = self.get_surrogate(I_timestamps,xy_timevector,video_srate,shift_time)

        spike_rate_occupancy_shuffled = self.get_spike_occupancy(I_timestamps_shuffled,x_coordinates,y_coordinates,x_grid,y_grid)
        place_field_shuffled,place_field_smoothed_shuffled = self.validate_place_field(spike_rate_occupancy_shuffled,position_occupancy,visits_occupancy, mintimespent,minvisits,smoothing_size)
        I_sec_shuffled,I_spk_shuffled = self.get_spatial_metrics(place_field_shuffled,position_occupancy)

        return I_sec_shuffled,I_spk_shuffled


    
    
    def number_of_islands(self,input_array):


        row = input_array.shape[0]
        col = input_array.shape[1]
        count = 0

        for i in range(row):
            for j in range(col):
                if input_array[i,j] == 1:
                    self.dfs(input_array,row,col,i,j)
                    count+=1
        return count

    def dfs(self,input_array,row,col,i,j):

        if input_array[i,j] == 0:
            return 
        input_array[i,j] = 0

        if i != 0:
            self.dfs(input_array,row,col,i-1,j)

        if i != row-1:
            self.dfs(input_array,row,col,i+1,j)

        if j != 0:
            self.dfs(input_array,row,col,i,j-1)

        if j != col - 1:
            self.dfs(input_array,row,col,i,j+1)
            
            
            



    
    