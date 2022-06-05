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
        kwargs.setdefault('smoothing_size', 2)
        kwargs.setdefault('shift_time', 10)
        kwargs.setdefault('num_cores', 1)
        kwargs.setdefault('num_surrogates', 200)
        kwargs.setdefault('saving_path', os.getcwd())
        kwargs.setdefault('saving', False)
        kwargs.setdefault('saving_string', 'SpatialMetrics')

        valid_kwargs = ['animal_id','day','neuron','dataset','trial','mean_video_srate',
                        'mintimespent','minvisits','speed_threshold','smoothing_size',
                        'x_bin_size','y_bin_size','shift_time','num_cores',
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
            speed = self.get_speed(x_coordinates,y_coordinates,track_timevector)


            x_coordinates_valid, y_coordinates_valid, calcium_imag_valid, track_timevector_valid = self.get_valid_timepoints(calcium_imag,x_coordinates,y_coordinates,track_timevector,speed,self.speed_threshold)

#            calcium_mean_occupancy,position_binned,x_grid,y_grid,x_center_bins,y_center_bins = self.get_spatial_statistics(calcium_imag_valid,x_coordinates_valid,y_coordinates_valid,self.environment_edges,self.x_bin_size,self.y_bin_size)

            x_grid,y_grid,x_center_bins,y_center_bins,x_center_bins_repeated,y_center_bins_repeated = self.get_position_grid(x_coordinates,y_coordinates,self.x_bin_size,self.y_bin_size,environment_edges = self.environment_edges)

            position_binned = self.get_binned_2Dposition(x_coordinates_valid,y_coordinates_valid,x_grid,y_grid)

            calcium_mean_occupancy = self.get_calcium_occupancy(calcium_imag_valid,x_coordinates_valid,y_coordinates_valid,x_grid,y_grid)

            mutual_info_original = self.get_mutual_information(calcium_imag_valid,position_binned)

            mutual_info_shuffled = self.parallelize_surrogate(calcium_imag_valid,position_binned,                                                          self.mean_video_srate,self.num_cores,self.num_surrogates,self.shift_time)

            mutual_info_zscored,mutual_info_centered = self.get_mutual_information_zscored(mutual_info_original,mutual_info_shuffled)

            position_occupancy = self.get_occupancy(x_coordinates_valid,y_coordinates_valid,x_grid,y_grid,self.mean_video_srate)

            visits_occupancy = self.get_visits(x_coordinates_valid,y_coordinates_valid,x_grid,y_grid,x_center_bins,y_center_bins)

            place_field,place_field_smoothed = self.validate_place_field(calcium_mean_occupancy,position_occupancy,visits_occupancy,
                                                                         self.mintimespent,self.minvisits,self.smoothing_size)

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

            num_of_islands = self.number_of_islands(np.copy(calcium_mean_occupancy_above_to_island))

            I_peaks = dp.detect_peaks(calcium_imag_valid,mpd=0.5*self.mean_video_srate,mph=1.*np.nanstd(calcium_imag_valid))
            peaks_amplitude = calcium_imag_valid[I_peaks]
            x_peaks_location = x_coordinates_valid[I_peaks]
            y_peaks_location = y_coordinates_valid[I_peaks]
                
                
                
                
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
            inputdict['x_peaks_location'] = x_peaks_location
            inputdict['y_peaks_location'] = y_peaks_location
            inputdict['events_amplitude'] = peaks_amplitude
            inputdict['mutual_info_original'] = mutual_info_original
            inputdict['mutual_info_shuffled'] = mutual_info_shuffled
            inputdict['mutual_info_zscored'] = mutual_info_zscored
            inputdict['mutual_info_centered'] = mutual_info_centered
            inputdict['num_of_islands'] = num_of_islands
            inputdict['place_cell_extension_absolute'] = pixels_place_cell_absolute
            inputdict['place_cell_extension_relative'] = pixels_place_cell_relative
            inputdict['input_parameters'] = self.__dict__['input_parameters']

            filename = self.filename_constructor(self.saving_string,self.animal_id,self.dataset,self.day,self.neuron,self.trial)
            
        if self.saving == True:
            self.caller_saving(inputdict,filename,self.saving_path)
            print(filename + ' saved')

        else:
            print(filename + ' not saved')


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

        return position_occupancy
    
    def get_calcium_occupancy(self,calcium_imag,x_coordinates,y_coordinates,x_grid,y_grid):

        # calculate mean calcium per pixel
        calcium_mean_occupancy = np.nan*np.zeros((y_grid.shape[0]-1,x_grid.shape[0]-1)) 
        for xx in range(0,x_grid.shape[0]-1):
            for yy in range(0,y_grid.shape[0]-1):

                check_x_ocuppancy = np.logical_and(x_coordinates >= x_grid[xx],x_coordinates < (x_grid[xx+1]))
                check_y_ocuppancy = np.logical_and(y_coordinates >= y_grid[yy],y_coordinates < (y_grid[yy+1]))

                calcium_mean_occupancy[yy,xx] = np.nanmean(calcium_imag[np.logical_and(check_x_ocuppancy,check_y_ocuppancy)])

        return calcium_mean_occupancy


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

        
    def get_valid_timepoints(self,calcium_imag,x_coordinates,y_coordinates,track_timevector,speed,speed_threshold):
        
        calcium_imag_valid = calcium_imag.copy()
        x_coordinates_valid = x_coordinates.copy()
        y_coordinates_valid = y_coordinates.copy()
        track_timevector_valid = track_timevector.copy()
        
        
        I_keep = ~np.isnan(calcium_imag)
        calcium_imag_valid = calcium_imag_valid[I_keep]
        x_coordinates_valid = x_coordinates_valid[I_keep]
        y_coordinates_valid = y_coordinates_valid[I_keep]
        track_timevector_valid = track_timevector_valid[I_keep]
        
        I_speed_thres = speed > speed_threshold

        calcium_imag_valid = calcium_imag_valid[I_speed_thres].copy()
        x_coordinates_valid = x_coordinates_valid[I_speed_thres].copy()
        y_coordinates_valid = y_coordinates_valid[I_speed_thres].copy()
        track_timevector_valid = track_timevector_valid[I_speed_thres].copy()
        
        return x_coordinates_valid, y_coordinates_valid, calcium_imag_valid, track_timevector_valid
  

    def validate_place_field(self,calcium_mean_occupancy,position_occupancy,visits_occupancy,mintimespent,minvisits,smoothing_size):

        valid_bins=(position_occupancy>=mintimespent)*(visits_occupancy>=minvisits)*1.
        valid_bins[valid_bins == 0] = np.nan
        place_field = calcium_mean_occupancy*valid_bins

        place_field_to_smooth = np.copy(place_field)
        place_field_to_smooth[np.isnan(place_field_to_smooth)] = 0 
        place_field_smoothed = hf.gaussian_smooth_2d(place_field_to_smooth,smoothing_size)

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

 
    def parallelize_surrogate(self,calcium_imag,position_binned,mean_video_srate,num_cores,num_surrogates,shift_time):
        mutual_info_shuffled = Parallel(n_jobs=num_cores)(delayed(self.get_mutual_info_surrogate)                                                         (calcium_imag,position_binned,mean_video_srate,shift_time) for permi in range(num_surrogates))
        
        return np.array(mutual_info_shuffled)
    

    def get_surrogate(self,input_vector,mean_video_srate,shift_time):
        eps = np.finfo(float).eps
        I_break = np.random.choice(np.arange(-shift_time*mean_video_srate,mean_video_srate*shift_time),1)[0].astype(int)
        input_vector_shuffled = np.concatenate([input_vector[I_break:], input_vector[0:I_break]])

        return input_vector_shuffled

    def get_mutual_info_surrogate(self,calcium_imag,position_binned,mean_video_srate,shift_time):
        calcium_imag_shuffled = self.get_surrogate(calcium_imag,mean_video_srate,shift_time)
        mutual_info_shuffled = self.get_mutual_information(calcium_imag_shuffled,position_binned)
        
        return mutual_info_shuffled



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