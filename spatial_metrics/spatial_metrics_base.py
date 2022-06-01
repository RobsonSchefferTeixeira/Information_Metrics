import numpy as np
import os
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
        kwargs.setdefault('mintimespent', 0.1)
        kwargs.setdefault('minvisits', 1)
        kwargs.setdefault('speed_threshold', 2.5)
        kwargs.setdefault('x_bin_size', 1)
        kwargs.setdefault('y_bin_size', 1)
        kwargs.setdefault('environment_edges', None)
        kwargs.setdefault('shuffling_shift', 10)
        kwargs.setdefault('num_cores', 1)
        kwargs.setdefault('num_surrogates', 200)
        kwargs.setdefault('saving_path', os.getcwd())
        kwargs.setdefault('saving', False)
        kwargs.setdefault('saving_string', 'SpatialMetrics')

        valid_kwargs = ['animal_id','day','neuron','dataset','trial','mean_video_srate',
                        'mintimespent','minvisits','speed_threshold',
                        'x_bin_size','y_bin_size','shuffling_shift','num_cores',
                        'num_surrogates','saving_path','saving','saving_string','environment_edges']
        
        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            setattr(self, k, v)
            
        self.__dict__['input_parameters'] = kwargs
        
    def main(self,calcium_imag,track_timevector,x_coordinates,y_coordinates):

        
        if np.all(np.isnan(calcium_imag)):
            warnings.warn("Signal contains only NaN's")
            inputdict = np.nan
           
        else:
            speed = self.get_speed(x_coordinates,y_coordinates,track_timevector,self.mean_video_srate)

            I_peaks = dp.detect_peaks(calcium_imag,mpd=0.5*self.mean_video_srate,mph=1.*np.nanstd(calcium_imag))

            x_coordinates_valid, y_coordinates_valid, calcium_imag_valid, track_timevector_valid = self.get_valid_timepoints(calcium_imag,x_coordinates,y_coordinates,track_timevector,self.speed_threshold,self.mean_video_srate)

            calcium_mean_occupancy,position_binned,x_grid,y_grid,x_center_bins,y_center_bins = self.get_spatial_statistics(calcium_imag_valid,x_coordinates_valid,y_coordinates_valid,self.environment_edges,self.x_bin_size,self.y_bin_size)

            mutual_info_original = self.get_mutual_information(calcium_imag_valid,position_binned)

            mutual_info_shuffled = self.parallelize_surrogate(calcium_imag_valid,position_binned,                                                          self.mean_video_srate,self.num_cores,self.num_surrogates,self.shuffling_shift)

            mutual_info_zscored,mutual_info_centered = self.get_mutual_information_zscored(mutual_info_original,mutual_info_shuffled)

            position_occupancy = self.get_occupancy(x_coordinates_valid,y_coordinates_valid,x_grid,y_grid,self.mean_video_srate)

            visits_occupancy = self.get_visits(x_coordinates_valid,y_coordinates_valid,x_grid,y_grid,x_center_bins,y_center_bins)

            place_field,place_field_smoothed = self.validate_place_field(calcium_mean_occupancy,position_occupancy,visits_occupancy,
                                                                         self.mintimespent,self.minvisits)

            spatial_map_smoothed_threshold = np.copy(place_field_smoothed)
            I_threshold = 2*np.nanstd(spatial_map_smoothed_threshold)

            total_visited_pixels = np.nansum(visits_occupancy != 0)
            pixels_above = np.nansum(spatial_map_smoothed_threshold > I_threshold)
            pixels_total = spatial_map_smoothed_threshold.shape[0]*spatial_map_smoothed_threshold.shape[1]

            pixels_place_cell_relative = pixels_above/total_visited_pixels
            pixels_place_cell_absolute = pixels_above/pixels_total

            calcium_mean_occupancy_above_to_island = np.copy(spatial_map_smoothed_threshold)
            calcium_mean_occupancy_above_to_island[calcium_mean_occupancy_above_to_island < I_threshold] = 0
            calcium_mean_occupancy_above_to_island[calcium_mean_occupancy_above_to_island >= I_threshold] = 1

            islandsnum = self.numIslands(np.copy(calcium_mean_occupancy_above_to_island))



            inputdict = dict()
            inputdict['signal_map'] = calcium_mean_occupancy
            inputdict['place_field'] = place_field
            inputdict['place_field_smoothed'] = place_field_smoothed        
            inputdict['ocuppancy_map'] = position_occupancy
            inputdict['visits_map'] = visits_occupancy
            inputdict['x_grid'] = x_grid
            inputdict['y_grid'] = y_grid
            inputdict['x_center_bins'] = x_center_bins
            inputdict['y_center_bins'] = y_center_bins
            inputdict['numb_events'] = I_peaks.shape[0]
            inputdict['events_index'] = I_peaks
            inputdict['mutual_info_original'] = mutual_info_original
            inputdict['mutual_info_shuffled'] = mutual_info_shuffled
            inputdict['mutual_info_zscored'] = mutual_info_zscored
            inputdict['mutual_info_centered'] = mutual_info_centered
            inputdict['num_of_islands'] = islandsnum
            inputdict['place_cell_extension_absolute'] = pixels_place_cell_absolute
            inputdict['place_cell_extension_relative'] = pixels_place_cell_relative
            inputdict['input_parameters'] = self.__dict__['input_parameters']

            filename = self.filename_constructor(self.saving_string,self.animal_id,self.dataset,self.day,self.neuron,self.trial)
            
        if self.saving == True:
            self.caller_saving(inputdict,filename,self.saving_path)
        else:
            print('File not saved!')


        return inputdict
    
    
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
        print('File saved.')
     

    def get_sparsity(self,place_field,position_occupancy):
        
        position_occupancy_norm = np.nansum(position_occupancy/np.nansum(position_occupancy))
        sparsity = np.nanmean(position_occupancy_norm*place_field)**2/np.nanmean(position_occupancy_norm*place_field**2)

        
        return sparsity
    
    def get_speed(self,x_coordinates,y_coordinates,track_timevector,mean_video_srate):
        smooth_coeff = 0.5
        speed = np.sqrt(np.diff(x_coordinates)**2 + np.diff(y_coordinates)**2)
#         speed = hf.smooth(speed/np.diff(track_timevector),window_len=hf.round_up_to_even(int(smooth_coeff*mean_video_srate)))
        speed = np.hstack([speed,0])
        return speed


    def get_occupancy(self,x_coordinates_speed,y_coordinates_speed,x_grid,y_grid,mean_video_srate):
        # calculate position occupancy
        position_occupancy = np.zeros((y_grid.shape[0]-1,x_grid.shape[0]-1))
        for xx in range(0,x_grid.shape[0]-1):
            for yy in range(0,y_grid.shape[0]-1):

                check_x_ocuppancy = np.logical_and(x_coordinates_speed >= x_grid[xx],x_coordinates_speed < (x_grid[xx+1]))
                check_y_ocuppancy = np.logical_and(y_coordinates_speed >= y_grid[yy],y_coordinates_speed < (y_grid[yy+1]))

                position_occupancy[yy,xx] = np.sum(np.logical_and(check_x_ocuppancy,check_y_ocuppancy))/mean_video_srate

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

        
    def get_valid_timepoints(self,mean_calcium_to_behavior,x_coordinates,y_coordinates,track_timevector,speed_threshold,mean_video_srate):
        
        I_keep = ~np.isnan(mean_calcium_to_behavior)
        mean_calcium_to_behavior = mean_calcium_to_behavior[I_keep]
        x_coordinates = x_coordinates[I_keep]
        y_coordinates = y_coordinates[I_keep]
        track_timevector = track_timevector[I_keep]
        
        speed = self.get_speed(x_coordinates,y_coordinates,track_timevector,mean_video_srate)

        I_speed_thres = speed > speed_threshold

        mean_calcium_to_behavior_valid = mean_calcium_to_behavior[I_speed_thres].copy()
        x_coordinates_valid = x_coordinates[I_speed_thres].copy()
        y_coordinates_valid = y_coordinates[I_speed_thres].copy()
        track_timevector_valid = track_timevector[I_speed_thres].copy()
        
        return x_coordinates_valid, y_coordinates_valid, mean_calcium_to_behavior_valid, track_timevector_valid
  

    def validate_place_field(self,calcium_mean_occupancy,position_occupancy,visits_occupancy,mintimespent,minvisits):

        valid_bins=(position_occupancy>=mintimespent)*(visits_occupancy>=minvisits)*1.
        valid_bins[valid_bins == 0] = np.nan
        place_field = calcium_mean_occupancy*valid_bins

        place_field_to_smooth = np.copy(place_field)
        place_field_to_smooth[np.isnan(place_field_to_smooth)] = 0 
        place_field_smoothed = hf.gaussian_smooth_2d(place_field_to_smooth,2)

        return place_field,place_field_smoothed
                          
    def get_spatial_statistics(self,calcium_imag,x_coordinates,y_coordinates,environment_edges,x_bin_size,y_bin_size):

        placefield_nbins_pos_x = (environment_edges[0][1] - environment_edges[0][0])/x_bin_size
        placefield_nbins_pos_y = (environment_edges[1][1] - environment_edges[1][0])/y_bin_size
        results = stats.binned_statistic_2d(x_coordinates, y_coordinates, calcium_imag, statistic = 'mean',
                                            bins =[placefield_nbins_pos_x,placefield_nbins_pos_y], range = environment_edges,
                                            expand_binnumbers=False)
        x_grid = results[1]
        y_grid = results[2]
        calcium_mean_occupancy = results[0].T

        x_center_bins = x_grid[0:-1] + np.diff(x_grid)
        y_center_bins = y_grid[0:-1] + np.diff(y_grid)
        
        position_binned = results[3]
        return calcium_mean_occupancy,position_binned,x_grid,y_grid,x_center_bins,y_center_bins

    def get_mutual_information(self,calcium_imag,position_binned):
        mutual_info_original = mutual_info_classif(calcium_imag.reshape(-1,1),position_binned)[0]
        return mutual_info_original


    def get_mutual_information_zscored(self,mutual_info_original,mutual_info_shuffled):
        mutual_info_centered = mutual_info_original-np.nanmean(mutual_info_shuffled)
        mutual_info_zscored = (mutual_info_original-np.nanmean(mutual_info_shuffled))/np.nanstd(mutual_info_shuffled)
        
        return mutual_info_zscored,mutual_info_centered

 
    def parallelize_surrogate(self,calcium_imag,position_binned,mean_video_srate,num_cores,num_surrogates,shuffling_shift):
        mutual_info_shuffled = Parallel(n_jobs=num_cores)(delayed(self.get_mutual_info_surrogate)                                                         (calcium_imag,position_binned,mean_video_srate,shuffling_shift) for permi in range(num_surrogates))
        
        return np.array(mutual_info_shuffled)
    

    def get_surrogate(self,input_vector,mean_video_srate,shuffling_shift):
        eps = np.finfo(float).eps
        I_break = np.random.choice(np.arange(-shuffling_shift*mean_video_srate,mean_video_srate*shuffling_shift),1)[0].astype(int)
        input_vector_shuffled = np.concatenate([input_vector[I_break:], input_vector[0:I_break]])

        return input_vector_shuffled

    def get_mutual_info_surrogate(self,calcium_imag,position_binned,mean_video_srate,shuffling_shift):
        calcium_imag_shuffled = self.get_surrogate(calcium_imag,mean_video_srate,shuffling_shift)
        mutual_info_shuffled = self.get_mutual_information(calcium_imag_shuffled,position_binned)
        
        return mutual_info_shuffled



    def numIslands(self,input_array):


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

            
            
            
class PlaceCellBinarized(PlaceCell):



    def get_mutual_information(self,calcium_imag,position_binned):
        
        # I've translated this code to Python. 
        # Originally I took it from https://github.com/etterguillaume/CaImDecoding/blob/master/extract_1D_information.m

        # I'm calling the input variable as calcium_imag just for the sake of class inheritance, but a better name
        # would be binarized_signal
        bin_vector = np.unique(position_binned)

        # Create bin vectors
        prob_being_active = np.nansum(calcium_imag)/calcium_imag.shape[0] # Expressed in probability of firing (<1)

         # Compute joint probabilities (of cell being active while being in a state bin)
        likelihood = []
        occupancy_vector = []

        MI = 0
        for i in range(bin_vector.shape[0]):
            position_idx = position_binned == bin_vector[i]

            if np.sum(position_idx)>0:
                occupancy_vector.append(position_idx.shape[0]/calcium_imag.shape[0])

                activity_in_bin_idx = np.where((calcium_imag == 1) & position_idx)[0]
                inactivity_in_bin_idx = np.where((calcium_imag == 0) & position_idx)[0]
                likelihood.append(activity_in_bin_idx.shape[0]/np.sum(position_idx))

                joint_prob_active = activity_in_bin_idx.shape[0]/calcium_imag.shape[0]
                joint_prob_inactive = inactivity_in_bin_idx.shape[0]/calcium_imag.shape[0]
                prob_in_bin = np.sum(position_idx)/calcium_imag.shape[0]

                if joint_prob_active > 0:
                    MI = MI + joint_prob_active*np.log2(joint_prob_active/(prob_in_bin*prob_being_active))

                if joint_prob_inactive > 0:
                    MI = MI + joint_prob_inactive*np.log2(joint_prob_inactive/(prob_in_bin*(1-prob_being_active)))
        occupancy_vector = np.array(occupancy_vector)
        likelihood = np.array(likelihood)

        posterior = likelihood*occupancy_vector/prob_being_active


        return MI