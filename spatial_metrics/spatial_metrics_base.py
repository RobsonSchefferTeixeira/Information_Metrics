import numpy as np
import os
import sys
from scipy import stats as stats
import spatial_metrics.helper_functions as hf
import spatial_metrics.detect_peaks as dp
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_classif
import warnings

class PlaceCell:
    def __init__(self,**kwargs):
        
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
        kwargs.setdefault('nbins_cal', 10)
        kwargs.setdefault('field_threshold', 2)
        

                    
                    
        valid_kwargs = ['animal_id','day','neuron','dataset','trial','mean_video_srate',
                        'min_time_spent','min_visits','min_speed_threshold','smoothing_size',
                        'x_bin_size','y_bin_size','shift_time','num_cores','field_threshold',
                        'num_surrogates','saving_path','saving','saving_string','environment_edges','nbins_cal']
        
        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            setattr(self, k, v)
            
        self.__dict__['input_parameters'] = kwargs
        
    def main(self,calcium_imag,track_timevector,x_coordinates,y_coordinates):

        
        if np.all(np.isnan(calcium_imag)):
            warnings.warn("Signal contains only NaN's")
            inputdict = np.nan
            filename = self.filename_constructor(self.saving_string,self.animal_id,self.dataset,self.day,self.neuron,self.trial)
        else:
            speed = self.get_speed(x_coordinates,y_coordinates,track_timevector)

            
            
            x_grid,y_grid,x_center_bins,y_center_bins,x_center_bins_repeated,y_center_bins_repeated = self.get_position_grid(                                         x_coordinates,y_coordinates,self.x_bin_size,self.y_bin_size,environment_edges = self.environment_edges)

            position_binned = self.get_binned_2Dposition(x_coordinates,y_coordinates,x_grid,y_grid)
            visits_bins,new_visits_times = self.get_visits(x_coordinates,y_coordinates,position_binned,x_center_bins,y_center_bins)
            time_spent_inside_bins = self.get_position_time_spent(position_binned,self.mean_video_srate)
            
            I_keep = self.get_valid_timepoints(calcium_imag,speed,visits_bins,time_spent_inside_bins,                                                                 self.min_speed_threshold,self.min_visits,self.min_time_spent)

            calcium_imag_valid = calcium_imag[I_keep].copy()
            x_coordinates_valid = x_coordinates[I_keep].copy()
            y_coordinates_valid = y_coordinates[I_keep].copy()
            track_timevector_valid = track_timevector[I_keep].copy()
            visits_bins_valid = visits_bins[I_keep].copy()
            position_binned_valid = position_binned[I_keep].copy()
                       
            position_occupancy = self.get_occupancy(x_coordinates_valid,y_coordinates_valid,x_grid,y_grid,self.mean_video_srate)
            visits_occupancy = self.get_visits_occupancy(x_coordinates,y_coordinates,new_visits_times,x_grid,y_grid,self.min_visits)

            place_field,place_field_smoothed = self.get_place_field(calcium_imag_valid,x_coordinates_valid,y_coordinates_valid,                                       x_grid,y_grid,self.smoothing_size)

            # mutual_info_original = self.get_mutual_information(calcium_imag_valid,position_binned)
            calcium_imag_valid_binned = self.get_binned_signal(calcium_imag_valid,self.nbins_cal)
            nbins_pos = (x_grid.shape[0]-1)*(y_grid.shape[0]-1)
            entropy1 = self.get_entropy(position_binned_valid,nbins_pos)
            entropy2 = self.get_entropy(calcium_imag_valid_binned,self.nbins_cal)
            joint_entropy = self.get_joint_entropy(position_binned_valid,calcium_imag_valid_binned,nbins_pos,self.nbins_cal)
            
            mutual_info_original = self.get_mutual_information(entropy1,entropy2,joint_entropy)
            mutual_info_kullback_leibler_original = self.get_kullback_leibler_normalized(calcium_imag_valid,position_binned_valid)
            mutual_info_NN_original = self.get_mutual_information_NN(calcium_imag_valid,position_binned_valid)
            mutual_info_skaggs_original = self.get_mutual_info_skaggs(calcium_imag_valid,position_binned_valid)

            
            results = self.parallelize_surrogate(calcium_imag,I_keep,position_binned_valid,self.mean_video_srate,self.shift_time,self.nbins_cal,                       nbins_pos,x_coordinates_valid,y_coordinates_valid,x_grid,y_grid,self.smoothing_size,self.num_cores,self.num_surrogates)
            
            place_field_shuffled = []
            place_field_smoothed_shuffled = []
            mutual_info_shuffled = []
            mutual_info_NN_shuffled = []
            mutual_info_kullback_leibler_shuffled = []
            mutual_info_skaggs_shuffled = []
            for perm in range(self.num_surrogates):
                mutual_info_shuffled.append(results[perm][0])
                mutual_info_kullback_leibler_shuffled.append(results[perm][1])
                mutual_info_NN_shuffled.append(results[perm][2])
                mutual_info_skaggs_shuffled.append(results[perm][3])
                place_field_shuffled.append(results[perm][4])
                place_field_smoothed_shuffled.append(results[perm][5])
                
            mutual_info_NN_shuffled = np.array(mutual_info_NN_shuffled)
            mutual_info_shuffled = np.array(mutual_info_shuffled)
            mutual_info_kullback_leibler_shuffled = np.array(mutual_info_kullback_leibler_shuffled)
            mutual_info_skaggs_shuffled = np.array(mutual_info_skaggs_shuffled)
            place_field_shuffled = np.array(place_field_shuffled)
            place_field_smoothed_shuffled = np.array(place_field_smoothed_shuffled)
                    
            mutual_info_zscored,mutual_info_centered = self.get_mutual_information_zscored(mutual_info_original,mutual_info_shuffled)
            mutual_info_kullback_leibler_zscored,mutual_info_kullback_leibler_centered = self.get_mutual_information_zscored(                                         mutual_info_kullback_leibler_original,mutual_info_kullback_leibler_shuffled)
            
            mutual_info_NN_zscored,mutual_info_NN_centered = self.get_mutual_information_zscored(mutual_info_NN_original,mutual_info_NN_shuffled)
            
            mutual_info_skaggs_zscored,mutual_info_skaggs_centered = self.get_mutual_information_zscored(                                                             mutual_info_skaggs_original,mutual_info_skaggs_shuffled)


            num_of_islands,islands_x_max,islands_y_max = hf.field_coordinates(place_field,smoothing_size=self.smoothing_size,                                         field_threshold = self.field_threshold)
                
            I_peaks = dp.detect_peaks(calcium_imag_valid,mpd=0.5*self.mean_video_srate,mph=1.*np.nanstd(calcium_imag_valid))
            peaks_amplitude = calcium_imag_valid[I_peaks]
            x_peaks_location = x_coordinates_valid[I_peaks]
            y_peaks_location = y_coordinates_valid[I_peaks]
                

            total_visited_pixels = np.nansum(visits_occupancy != 0)
            pixels_above = np.nansum(place_field_smoothed > self.field_threshold)
            pixels_total = place_field_smoothed.shape[0]*place_field_smoothed.shape[1]

            pixels_place_cell_relative = pixels_above/total_visited_pixels
            pixels_place_cell_absolute = pixels_above/pixels_total
   
            inputdict = dict()
            inputdict['place_field'] = place_field
            inputdict['place_field_smoothed'] = place_field_smoothed
            
            inputdict['place_field_shuffled'] = place_field_shuffled
            inputdict['place_field_smoothed_shuffled'] = place_field_smoothed_shuffled

            
            inputdict['ocuppancy_map'] = position_occupancy
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
            
            inputdict['place_cell_extension_absolute'] = pixels_place_cell_absolute
            inputdict['place_cell_extension_relative'] = pixels_place_cell_relative
            
            inputdict['mutual_info_original'] = mutual_info_original
            inputdict['mutual_info_shuffled'] = mutual_info_shuffled
            inputdict['mutual_info_zscored'] = mutual_info_zscored
            inputdict['mutual_info_centered'] = mutual_info_centered
            
            inputdict['mutual_info_kullback_leibler_original'] = mutual_info_kullback_leibler_original     
            inputdict['mutual_info_kullback_leibler_shuffled'] = mutual_info_kullback_leibler_shuffled     
            inputdict['mutual_info_kullback_leibler_zscored'] = mutual_info_kullback_leibler_zscored     
            inputdict['mutual_info_kullback_leibler_centered'] = mutual_info_kullback_leibler_centered     
            
            inputdict['mutual_info_NN_original'] = mutual_info_NN_original     
            inputdict['mutual_info_NN_shuffled'] = mutual_info_NN_shuffled     
            inputdict['mutual_info_NN_zscored'] = mutual_info_NN_zscored     
            inputdict['mutual_info_NN_centered'] = mutual_info_NN_centered           
          
            inputdict['mutual_info_skaggs_original'] = mutual_info_skaggs_original     
            inputdict['mutual_info_skaggs_shuffled'] = mutual_info_skaggs_shuffled     
            inputdict['mutual_info_skaggs_zscored'] = mutual_info_skaggs_zscored     
            inputdict['mutual_info_skaggs_centered'] = mutual_info_skaggs_centered     
            
            
            inputdict['input_parameters'] = self.__dict__['input_parameters']

            filename = hf.filename_constructor(self.saving_string,self.animal_id,self.dataset,self.day,self.neuron,self.trial)
            
        if self.saving == True:
            hf.caller_saving(inputdict,filename,self.saving_path)
            print(filename + ' saved')

        else:
            print(filename + ' not saved')


        return inputdict
    
    
#     def filename_constructor(self,saving_string,animal_id,dataset,day,neuron,trial):

#         first_string =  saving_string
#         animal_id_string = '.' + animal_id
#         dataset_string = '.Dataset.' + dataset
#         day_string = '.Day.' + str(day)
#         neuron_string = '.Neuron.' + str(neuron)
#         trial_string = '.Trial.' + str(trial)

#         filename_checklist = np.array([first_string,animal_id, dataset, day, neuron, trial])
#         inlcude_this = np.where(filename_checklist != None)[0]

#         filename_backbone = [first_string, animal_id_string,dataset_string, day_string, neuron_string, trial_string]

#         filename = ''.join([filename_backbone[i] for i in inlcude_this])
               
#         return filename
    
            
#     def caller_saving(self,inputdict,filename,saving_path):
#         os.chdir(saving_path)
#         output = open(filename, 'wb') 
#         np.save(output,inputdict)
#         output.close()
#         print('File saved.')
     

    def get_sparsity(self,place_field,position_occupancy):
        
        position_occupancy_norm = np.nansum(position_occupancy/np.nansum(position_occupancy))
        sparsity = np.nanmean(position_occupancy_norm*place_field)**2/np.nanmean(position_occupancy_norm*place_field**2)

        
        return sparsity
    

    def get_speed(self,x_coordinates,y_coordinates,timevector):

        speed = np.sqrt(np.diff(x_coordinates)**2 + np.diff(y_coordinates)**2)
        speed = hf.smooth(speed/np.diff(timevector),window_len=10)
        speed = np.hstack([speed,0])
        return speed
    
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


    
    def get_occupancy(self,x_coordinates,y_coordinates,x_grid,y_grid,mean_video_srate):
        # calculate position occupancy
        position_occupancy = np.zeros((y_grid.shape[0]-1,x_grid.shape[0]-1))
        for xx in range(0,x_grid.shape[0]-1):
            for yy in range(0,y_grid.shape[0]-1):

                check_x_ocuppancy = np.logical_and(x_coordinates>= x_grid[xx],x_coordinates < (x_grid[xx+1]))
                check_y_ocuppancy = np.logical_and(y_coordinates >= y_grid[yy],y_coordinates < (y_grid[yy+1]))

                position_occupancy[yy,xx] = np.sum(np.logical_and(check_x_ocuppancy,check_y_ocuppancy))/mean_video_srate

        position_occupancy[position_occupancy==0] = np.nan
        return position_occupancy
  

    
    def get_visits(self,x_coordinates,y_coordinates,position_binned,x_center_bins,y_center_bins):

        I_x_coord = []
        I_y_coord = []

        for xx in range(0,x_coordinates.shape[0]):
            if np.isnan(x_coordinates[xx]):
                I_x_coord.append(np.nan)
                I_y_coord.append(np.nan)
            else:
                I_x_coord.append(np.nanargmin(np.abs(x_coordinates[xx] - x_center_bins)))
                I_y_coord.append(np.nanargmin(np.abs(y_coordinates[xx] - y_center_bins)))

        I_x_coord = np.array(I_x_coord)
        I_y_coord = np.array(I_y_coord)

        dx = np.diff(np.hstack([I_x_coord[0]-1,I_x_coord]))
        dy = np.diff(np.hstack([I_y_coord[0]-1,I_y_coord]))


        new_visits_times = (np.logical_or(((dy != 0) & (~np.isnan(dy))), ((dx!=0) & (~np.isnan(dx)))))


        visits_id, visits_counts = np.unique(position_binned[new_visits_times],return_counts=True)

        visits_bins = np.zeros(position_binned.shape)*np.nan
        for ids in range(visits_id.shape[0]):
            if ~np.isnan(visits_id[ids]):
                I_pos = position_binned == visits_id[ids]
                visits_bins[I_pos] = visits_counts[ids]

        return visits_bins,new_visits_times*1



    def get_visits_occupancy(self,x_coordinates,y_coordinates,new_visits_times,x_grid,y_grid,min_visits = 1):
    
        I_visit = np.where(new_visits_times > 0)[0]

        x_coordinate_visit = x_coordinates[I_visit]
        y_coordinate_visit = y_coordinates[I_visit]

        visits_occupancy = np.zeros((y_grid.shape[0]-1,x_grid.shape[0]-1))*np.nan
        for xx in range(0,x_grid.shape[0]-1):
            for yy in range(0,y_grid.shape[0]-1):

                check_x_ocuppancy = np.logical_and(x_coordinate_visit >= x_grid[xx],x_coordinate_visit < (x_grid[xx+1]))
                check_y_ocuppancy = np.logical_and(y_coordinate_visit >= y_grid[yy],y_coordinate_visit < (y_grid[yy+1]))

                visits_occupancy[yy,xx] = np.sum(np.logical_and(check_x_ocuppancy,check_y_ocuppancy))

        visits_occupancy[visits_occupancy<min_visits] = np.nan

        return visits_occupancy

    def get_position_time_spent(self,position_binned,mean_video_srate):

        positions_id, positions_counts = np.unique(position_binned,return_counts=True)

        time_spent_inside_bins = np.zeros(position_binned.shape)*np.nan
        for ids in range(positions_id.shape[0]):
            if ~np.isnan(positions_id[ids]):
                I_pos = position_binned == positions_id[ids]
                time_spent_inside_bins[I_pos] = positions_counts[ids]/mean_video_srate

        return time_spent_inside_bins


   

    def get_valid_timepoints(self,calcium_imag,speed,visits_bins,time_spent_inside_bins,min_speed_threshold,min_visits,min_time_spent):

        # min speed
        I_speed_thres = speed >= min_speed_threshold

        # min visits
        I_visits_times_thres = visits_bins >= min_visits

        # min time spent
        I_time_spent_thres = time_spent_inside_bins >= min_time_spent

        # valid calcium points
        I_valid_calcium = ~np.isnan(calcium_imag)

        I_keep = I_speed_thres*I_visits_times_thres*I_time_spent_thres*I_valid_calcium

        # calcium_imag_valid = calcium_imag_valid[I_keep]
        # x_coordinates_valid = x_coordinates_valid[I_keep]
        # y_coordinates_valid = y_coordinates_valid[I_keep]
        # track_timevector_valid = track_timevector_valid[I_keep]

        return I_keep


    def get_place_field(self,calcium_imag,x_coordinates,y_coordinates,x_grid,y_grid,smoothing_size):

        # calculate mean calcium per pixel
        place_field = np.nan*np.zeros((y_grid.shape[0]-1,x_grid.shape[0]-1)) 
        for xx in range(0,x_grid.shape[0]-1):
            for yy in range(0,y_grid.shape[0]-1):

                check_x_ocuppancy = np.logical_and(x_coordinates >= x_grid[xx],x_coordinates < (x_grid[xx+1]))
                check_y_ocuppancy = np.logical_and(y_coordinates >= y_grid[yy],y_coordinates < (y_grid[yy+1]))

                place_field[yy,xx] = np.nanmean(calcium_imag[np.logical_and(check_x_ocuppancy,check_y_ocuppancy)])


        place_field_to_smooth = np.copy(place_field)
        place_field_to_smooth[np.isnan(place_field_to_smooth)] = 0 
        place_field_smoothed = hf.gaussian_smooth_2d(place_field_to_smooth,smoothing_size)

        return place_field,place_field_smoothed
    
   
    
#     def get_spatial_statistics(self,calcium_imag,x_coordinates,y_coordinates,environment_edges,x_bin_size,y_bin_size):

#         placefield_nbins_pos_x = (environment_edges[0][1] - environment_edges[0][0])/x_bin_size
#         placefield_nbins_pos_y = (environment_edges[1][1] - environment_edges[1][0])/y_bin_size
#         results = stats.binned_statistic_2d(x_coordinates, y_coordinates, calcium_imag, statistic = 'mean',
#                                             bins =[placefield_nbins_pos_x,placefield_nbins_pos_y], range = environment_edges,
#                                             expand_binnumbers=False)
#         x_grid = results[1]
#         y_grid = results[2]
#         calcium_mean_occupancy = results[0].T

#         x_center_bins = x_grid[0:-1] + np.diff(x_grid)
#         y_center_bins = y_grid[0:-1] + np.diff(y_grid)
        
#         position_binned = results[3]
#         return calcium_mean_occupancy,position_binned,x_grid,y_grid,x_center_bins,y_center_bins
    

    
    
    def get_mutual_information_NN(self,calcium_imag,position_binned):
        mutual_info_NN_original = mutual_info_classif(calcium_imag.reshape(-1,1),position_binned,discrete_features=False,n_neighbors=5)[0]

        return mutual_info_NN_original
    
    


    
    def get_binned_signal(self,calcium_imag,nbins_cal):

        calcium_imag_bins = np.linspace(np.nanmin(calcium_imag),np.nanmax(calcium_imag),nbins_cal+1)
        calcium_imag_binned = np.zeros(calcium_imag.shape[0])
        for jj in range(calcium_imag_bins.shape[0]-1):
            I_amp = (calcium_imag > calcium_imag_bins[jj]) & (calcium_imag <= calcium_imag_bins[jj+1])
            calcium_imag_binned[I_amp] = jj

        return calcium_imag_binned

       
    def get_joint_entropy(self,bin_vector1,bin_vector2,nbins_1,nbins_2):

        eps = np.finfo(float).eps

        bin_vector1 = np.copy(bin_vector1)
        bin_vector2 = np.copy(bin_vector2)

        jointprobs = np.zeros([nbins_1,nbins_2])
        
        for i1 in range(nbins_1):
            for i2 in range(nbins_2):
                jointprobs[i1,i2] = np.nansum((bin_vector1==i1) & (bin_vector2==i2))

        jointprobs = jointprobs/np.nansum(jointprobs)
        joint_entropy = -np.nansum(jointprobs*np.log2(jointprobs+eps));

        return joint_entropy
    
    
    
    def get_entropy(self,binned_input,nbins):

        eps = np.finfo(float).eps

        hdat = np.histogram(binned_input,nbins)[0]
        hdat = hdat/np.nansum(hdat)
        entropy = -np.nansum(hdat*np.log2(hdat+eps))

        return entropy
    
    def get_mutual_information(self,entropy1,entropy2,joint_entropy):
        mutual_info = entropy1 + entropy2 - joint_entropy
        return mutual_info
    

    def get_mutual_information_zscored(self,mutual_info_original,mutual_info_shuffled):
        mutual_info_centered = mutual_info_original-np.nanmean(mutual_info_shuffled)
        mutual_info_zscored = (mutual_info_original-np.nanmean(mutual_info_shuffled))/np.nanstd(mutual_info_shuffled)
        
        return mutual_info_zscored,mutual_info_centered

 
    def parallelize_surrogate(self,calcium_imag,I_keep,position_binned_valid,mean_video_srate,shift_time,nbins_cal,nbins_pos,x_coordinates_valid,         y_coordinates_valid,x_grid,y_grid,smoothing_size,num_cores,num_surrogates):
        results = Parallel(n_jobs=num_cores)(delayed(self.get_mutual_info_surrogate)                                                         (calcium_imag,I_keep,position_binned_valid,mean_video_srate,shift_time,nbins_cal,nbins_pos,x_coordinates_valid,         y_coordinates_valid,x_grid,y_grid,smoothing_size) for permi in range(num_surrogates))
        
        return results
    

    def get_surrogate(self,input_vector,mean_video_srate,shift_time):
        eps = np.finfo(float).eps
        
        I_break = np.random.choice(np.arange(-shift_time*mean_video_srate,mean_video_srate*shift_time),1)[0].astype(int)
        # I_break = np.random.choice(np.arange(0,input_vector.shape[0]),1)[0].astype(int)
    
        input_vector_shuffled = np.concatenate([input_vector[I_break:], input_vector[0:I_break]])

        return input_vector_shuffled

    def get_mutual_info_surrogate(self,calcium_imag,I_keep,position_binned_valid,mean_video_srate,shift_time,nbins_cal,nbins_pos,x_coordinates_valid,         y_coordinates_valid,x_grid,y_grid,smoothing_size):
        
        calcium_imag_shuffled = self.get_surrogate(calcium_imag,mean_video_srate,shift_time)
        calcium_imag_shuffled_valid = calcium_imag_shuffled[I_keep].copy()
        
        calcium_imag_shuffled_binned = self.get_binned_signal(calcium_imag_shuffled_valid,nbins_cal)
        entropy1 = self.get_entropy(position_binned_valid,nbins_pos)
        entropy2 = self.get_entropy(calcium_imag_shuffled_binned,nbins_cal)
        joint_entropy = self.get_joint_entropy(position_binned_valid,calcium_imag_shuffled_binned,nbins_pos,nbins_cal)
        mutual_info_shuffled = self.get_mutual_information(entropy1,entropy2,joint_entropy)
        
        mutual_info_shuffled_NN = self.get_mutual_information_NN(calcium_imag_shuffled_valid,position_binned_valid)
        
        modulation_index_shuffled = self.get_kullback_leibler_normalized(calcium_imag_shuffled_valid,position_binned_valid)
        
        mutual_info_skaggs_shuffled = self.get_mutual_info_skaggs(calcium_imag_shuffled_valid,position_binned_valid)

        place_field_shuffled,place_field_smoothed_shuffled = self.get_place_field(calcium_imag_shuffled_valid,x_coordinates_valid,y_coordinates_valid,             x_grid,y_grid,smoothing_size)
        
        return mutual_info_shuffled,modulation_index_shuffled,mutual_info_shuffled_NN,mutual_info_skaggs_shuffled,                                                 place_field_shuffled,place_field_smoothed_shuffled


    def get_kullback_leibler_normalized(self,calcium_imag,position_binned):

        position_bins = np.unique(position_binned)
        nbin = position_bins.shape[0]
        
        mean_calcium_activity = []
        for pos in position_bins:
            I_pos = np.where(pos == position_binned)[0]
            mean_calcium_activity.append(np.nanmean(calcium_imag[I_pos]))
        mean_calcium_activity = np.array(mean_calcium_activity)
   
        observed_distr = -np.nansum((mean_calcium_activity/np.nansum(mean_calcium_activity))*np.log((mean_calcium_activity/np.nansum(mean_calcium_activity))))
        test_distr = np.log(nbin)
        modulation_index = (test_distr - observed_distr) / test_distr
        return modulation_index


    

    def get_mutual_info_skaggs(self,calcium_imag,position_binned):

        overall_mean_amplitude = np.nanmean(calcium_imag)

        position_bins = np.unique(position_binned)
        nbin = position_bins.shape[0]

        bin_probability = []
        mean_calcium_activity = []
        for pos in position_bins:
            I_pos = np.where(pos == position_binned)[0]
            bin_probability.append(I_pos.shape[0]/position_binned.shape[0])
            mean_calcium_activity.append(np.nanmean(calcium_imag[I_pos]))
        mean_calcium_activity = np.array(mean_calcium_activity)
        bin_probability = np.array(bin_probability)
        
        mutual_info_skaggs = np.nansum((bin_probability*(mean_calcium_activity/overall_mean_amplitude))*np.log2(mean_calcium_activity/overall_mean_amplitude))
        
        # spatial info in bits per deltaF/F s^-1
        
        return mutual_info_skaggs

    
    
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




    
    

