import ray
import numpy as np
from scipy.io import loadmat
import pandas as pd
import os
from scipy import stats as stats
import helper_functions as hf
from joblib import Parallel, delayed
        
ray.init(num_cpus=os.cpu_count(), ignore_reinit_error=True,log_to_driver=False) 

class PlaceCell:
    def __init__(self,**kwargs):
           
        kwargs.setdefault('Session', [])
        kwargs.setdefault('day', [])
        kwargs.setdefault('shank', [])
        kwargs.setdefault('neuron', [])
        kwargs.setdefault('dataset', [])
        kwargs.setdefault('video_srate', 30.)
        kwargs.setdefault('mintimespent', 0.1)
        kwargs.setdefault('minvisits', 1)
        kwargs.setdefault('speed_threshold', 2.5)
        kwargs.setdefault('nbins_pos_x', 20)
        kwargs.setdefault('nbins_pos_y', 20)
        kwargs.setdefault('num_cores', 10)
        kwargs.setdefault('num_surrogates', 200)
        kwargs.setdefault('surrogate_window', 10)
        kwargs.setdefault('saving_path', os.getcwd())
        kwargs.setdefault('saving', False)
        kwargs.setdefault('saving_string', [])

        valid_kwargs = ['Session','day','shank','neuron','dataset', 'video_srate','mintimespent','minvisits','speed_threshold',
                        'nbins_pos_x','nbins_pos_y','num_cores','surrogate_window',
                        'num_surrogates','saving_path','saving','saving_string','surrogate_window']
        
        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            setattr(self, k, v)
            
        self.__dict__['input_parameters'] = kwargs
        
        
    def main(self,I_timestamps,timevector,x_coordinates,y_coordinates):

        if len(I_timestamps) == 0:
            
            inputdict = dict()
            inputdict['spike_rate_occupancy'] = np.nan
            inputdict['place_field'] = np.nan
            inputdict['place_field_smoothed'] = np.nan        
            inputdict['ocuppancyMap'] = np.nan
            inputdict['visitsMap'] = np.nan
            inputdict['x_grid'] = np.nan
            inputdict['y_grid'] = np.nan
            inputdict['x_center_bins'] = np.nan
            inputdict['y_center_bins'] = np.nan
            inputdict['numb_events'] = np.nan
            inputdict['I_sec'] = np.nan
            inputdict['I_spk'] = np.nan
            inputdict['I_spk_permutation'] = np.nan
            inputdict['I_sec_permutation'] = np.nan
            inputdict['sparsity'] = np.nan
            inputdict['input_parameters'] = self.__dict__['input_parameters']
            
        else:

            speed = self.get_speed(x_coordinates,y_coordinates,timevector)

            x_coordinates_valid,y_coordinates_valid = self.get_valid_timepoints(speed,x_coordinates,y_coordinates,self.speed_threshold)
    
            x_grid,y_grid,x_center_bins,y_center_bins = self.get_position_grid(x_coordinates,y_coordinates,self.nbins_pos_x,self.nbins_pos_y)

            spike_rate_occupancy = self.get_spike_occupancy(I_timestamps,x_coordinates_valid,y_coordinates_valid,x_grid,y_grid)

            position_occupancy = self.get_occupancy(x_coordinates_valid,y_coordinates_valid,x_grid,y_grid,self.video_srate)

            visits_occupancy = self.get_visits(x_coordinates_valid,y_coordinates_valid,x_grid,y_grid,x_center_bins,y_center_bins)

            place_field,place_field_smoothed = self.placeField(spike_rate_occupancy,position_occupancy,visits_occupancy,self.mintimespent, self.minvisits)

            I_sec,I_spk = self.get_spatial_metrics(place_field,position_occupancy)

            sparsity = self.get_sparsity(place_field_smoothed,position_occupancy)


            #spike_timevector = self.get_spiketimes_binarized(I_timestamps,timevector)

            results = self.parallelize_surrogate(I_timestamps,timevector,x_coordinates_valid,y_coordinates_valid,position_occupancy,visits_occupancy,x_grid,y_grid,self.num_cores,self.num_surrogates)

            I_sec_permutation = results[:,0]
            I_spk_permutation = results[:,1]


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
            inputdict['sparsity'] = sparsity
            inputdict['input_parameters'] = self.__dict__['input_parameters']
            
            
        if self.saving == True:
            filename = self.Session + '.' +  self.dataset + '.' + self.saving_string + '.PlaceField.ModulationIndex.Day.' + str(self.day) + '.Shank.' + str(self.shank) + '.Neuron.' + str(self.neuron)

            self.caller_saving(inputdict,filename,self.saving_path)

        else:

            filename = self.Session + '.' +  self.dataset + '.' + self.saving_string + '.PlaceField.ModulationIndex.Day.' + str(self.day) + '.Shank.' + str(self.shank) + '.Neuron.' + str(self.neuron)

            print('File not saved!')
        
        
        return inputdict
    
            
    def caller_saving(self,inputdict,filename,saving_path):
        print('Saving data file...')
        os.chdir(saving_path)
        output = open(filename, 'wb') 
        np.save(output,inputdict)
        output.close()
     

    def get_sparsity(self,place_field,position_occupancy):
        
        position_occupancy_norm = np.nansum(position_occupancy/np.nansum(position_occupancy))
        sparsity = np.nanmean(position_occupancy_norm*place_field)**2/np.nanmean(position_occupancy_norm*place_field**2)

        
        return sparsity
    
   

    def get_speed(self,x_coordinates,y_coordinates,timevector):

        speed = np.sqrt(np.diff(x_coordinates)**2 + np.diff(y_coordinates)**2)
        speed = hf.smooth(speed/np.diff(timevector),window_len=10)
        speed = np.hstack([speed,0])
        return speed

    def get_valid_timepoints(self,speed,x_coordinates,y_coordinates,speed_threshold):

        x_coordinates_valid = np.copy(x_coordinates)
        y_coordinates_valid = np.copy(y_coordinates)
        I_speed_non_valid = speed <= speed_threshold
        x_coordinates_valid[I_speed_non_valid] = np.nan
        y_coordinates_valid[I_speed_non_valid] = np.nan
        
        return x_coordinates_valid,y_coordinates_valid


    def placeField(self,spike_rate_occupancy,position_occupancy,visits_occupancy,mintimespent, minvisits):

        place_field = spike_rate_occupancy/position_occupancy

        Valid=(position_occupancy>=mintimespent)*(visits_occupancy>=minvisits)*1.
        Valid[Valid == 0] = np.nan
        place_field = place_field*Valid

        place_field_to_smooth = np.copy(place_field)
        place_field_to_smooth[np.isnan(place_field_to_smooth)] = 0
        place_field_smoothed = hf.gaussian_smooth_2d(place_field_to_smooth,2)

        return place_field,place_field_smoothed


    def get_binned_2Dposition(self,x_coordinates,y_coordinates,x_grid,y_grid):

        # calculate position occupancy
        position_binned = np.zeros(x_coordinates.shape) 
        count = 0
        for xx in range(0,x_grid.shape[0]-1):
            for yy in range(0,y_grid.shape[0]-1):

                check_x_ocuppancy = np.logical_and(x_coordinates >= x_grid[xx],x_coordinates < (x_grid[xx+1]))
                check_y_ocuppancy = np.logical_and(y_coordinates >= y_grid[yy],y_coordinates < (y_grid[yy+1]))
                position_binned[np.logical_and(check_x_ocuppancy,check_y_ocuppancy)] = count
                count += 1

        return position_binned


#     def gaussian_smooth_2d(self,input_matrix,s_in):
#         import numpy as np
#         from scipy import signal as sig

#         gaussian2d = self.gaussian2d_kernel(s_in)
#         smoothed_matrix = sig.convolve2d(input_matrix,gaussian2d,mode='same')

#         return smoothed_matrix

#     def gaussian2d_kernel(self,s):
#         import numpy as np
#         x_vec = np.arange(-100,101,1)
#         y_vec = np.arange(-100,101,1)
#     #     s = 2
#         gaussian_kernel = np.zeros([y_vec.shape[0],x_vec.shape[0]])
#         x_count = 0
#         for xx in x_vec:
#             y_count = 0
#             for yy in y_vec:
#                 gaussian_kernel[y_count,x_count] = np.exp(-((xx**2 + yy**2)/(2*(s**2))))

#                 y_count += 1
#             x_count += 1

#         return gaussian_kernel



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


#    def get_surrogate(self,input_vector,srate,shuffling_max_time):
#        eps = np.finfo(float).eps
#        I_break = np.random.choice(np.arange(-shuffling_max_time*srate,srate*shuffling_max_time),1)[0].astype(int)
#        input_vector_shuffled = np.concatenate([input_vector[I_break:], input_vector[0:I_break]])

#        return input_vector_shuffled



    def get_surrogate(self,I_timestamps,xy_timevector,srate,shuffling_max_time):

        xy_timevector_hist = np.append(xy_timevector,xy_timevector[-1]+(1/srate))
        spike_timevector = np.histogram(xy_timevector[I_timestamps],xy_timevector_hist)[0]

        I_break = np.random.choice(np.linspace(-shuffling_max_time*srate,srate*shuffling_max_time),1)[0].astype(int)
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



    def get_position_grid(self,x_coordinates,y_coordinates,nbins_pos_x,nbins_pos_y):
        # here someone should also be able to set the enviroment edges

        x_range = (np.nanmax(x_coordinates) - np.nanmin(x_coordinates))
        x_grid_window = x_range/nbins_pos_x
        x_grid = np.arange(np.nanmin(x_coordinates),np.nanmax(x_coordinates) +x_grid_window/2,x_grid_window)

        y_range = (np.nanmax(y_coordinates) - np.nanmin(y_coordinates))
        y_grid_window = y_range/nbins_pos_y
        y_grid = np.arange(np.nanmin(y_coordinates),np.nanmax(y_coordinates)+y_grid_window/2,y_grid_window)

        x_center_bins = x_grid[0:-1] + x_grid_window/2
        y_center_bins = y_grid[0:-1] + y_grid_window/2

        return x_grid,y_grid,x_center_bins,y_center_bins


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


    
    
    
    
#     def parallelize_surrogate(self,spike_timevector,x_coordinates,y_coordinates,position_occupancy,
#                                        visits_occupancy,x_grid,y_grid,num_cores,num_surrogates):

#         results = Parallel(n_jobs=num_cores,verbose = 10)(delayed(self.get_spatial_metrics_surrogate) 
#                                             (spike_timevector,x_coordinates,y_coordinates,position_occupancy,
#                                             visits_occupancy,x_grid,y_grid) for permi in range(num_surrogates))

#         return np.array(results)
    


    
#     def get_spatial_metrics_surrogate(self,spike_timevector,x_coordinates,y_coordinates,position_occupancy,
#                                        visits_occupancy,x_grid,y_grid):

#         spike_timevector_shuffled = self.get_surrogate(spike_timevector,self.video_srate,self.surrogate_window)
#         I_timestamps_shuffled = np.where(spike_timevector_shuffled)[0]

#         spike_rate_occupancy_shuffled = self.get_spike_occupancy(I_timestamps_shuffled,x_coordinates,y_coordinates,x_grid,y_grid)
#         place_field_shuffled,place_field_smoothed_shuffled = self.placeField(spike_rate_occupancy_shuffled,position_occupancy,visits_occupancy,self.mintimespent, self.minvisits)

#         I_sec_shuffled,I_spk_shuffled = self.spatial_metrics(place_field_shuffled,position_occupancy)


#         return I_sec_shuffled,I_spk_shuffled
    
    
    
    
    
    @ray.remote
    def get_spatial_metrics_surrogate(self,I_timestamps,timevector,x_coordinates,y_coordinates,position_occupancy,
                                       visits_occupancy,x_grid,y_grid):

        I_timestamps_shuffled = self.get_surrogate(I_timestamps,timevector,self.video_srate,self.surrogate_window)
        spike_rate_occupancy_shuffled = self.get_spike_occupancy(I_timestamps_shuffled,x_coordinates,y_coordinates,x_grid,y_grid)
        place_field_shuffled,place_field_smoothed_shuffled = self.placeField(spike_rate_occupancy_shuffled,position_occupancy,visits_occupancy,self.mintimespent, self.minvisits)

        I_sec_shuffled,I_spk_shuffled = self.get_spatial_metrics(place_field_shuffled,position_occupancy)


        return I_sec_shuffled,I_spk_shuffled
    
    def parallelize_surrogate(self,I_timestamps,timevector,x_coordinates,y_coordinates,position_occupancy,
                                       visits_occupancy,x_grid,y_grid,num_cores,num_surrogates):
        
        results = ray.get([self.get_spatial_metrics_surrogate.remote(self,I_timestamps,timevector,x_coordinates,y_coordinates,position_occupancy,visits_occupancy,x_grid,y_grid) for _ in range(num_surrogates)])

        return np.array(results)
    



    
    